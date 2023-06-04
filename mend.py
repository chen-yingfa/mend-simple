import copy
from collections import defaultdict
from typing import Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
import transformers
import higher
from higher.patch import monkeypatch as make_functional

from hooks import hook_model
import nn as local_nn
from utils import get_inner_params
from losses import masked_log_probs
from utils import should_shift_targets


def update_counter(x, m, s, k):
    """
    Calculates the updated mean and variance given a new observation and
    the previous mean, variance, and number of observations.

    Args:
        x (float): The new observation.
        m (float): The previous mean.
        s (float): The previous variance.
        k (int): The number of observations.

    Returns:
        Tuple[float, float]: The updated mean and variance.
    """
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)
    return new_m, new_s


class GradientTransform(nn.Module):
    def __init__(
        self,
        x_dim: int,
        delta_dim: int,
        num_modes: int,
        combine: bool = True,
        x_only: bool = False,
        delta_only: bool = False,
        one_sided: bool = False,
        n_hidden: int = 1,
        hidden_dim: int = None,
        mlp_class: str = "IDMLP",
        init: str = "id",
        act: str = "relu",
        rank: int = 1920,
        norm: bool = True,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.delta_dim = delta_dim
        self.combine = combine
        self.x_only = x_only
        self.delta_only = delta_only
        self.one_sided = one_sided
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.mlp_class = mlp_class
        self.init = init
        self.act = act
        self.rank = rank
        self.norm = norm

        if combine and (one_sided or x_only or delta_only):
            raise ValueError("combine cannot be used with one-sided GTN variants")

        self.norm_init = False
        self.register_buffer("u_mean", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_mean", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_std", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_std", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_s", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_s", torch.full((delta_dim,), float("nan")))
        self.register_buffer("k", torch.full((1,), float("nan")))

        MlpClass = getattr(local_nn, mlp_class)
        print(f"Building Gradient Transform with MLP class {MlpClass}")

        def delta_net():
            return MlpClass(
                delta_dim,
                delta_dim,
                delta_dim * 2,
                n_hidden,
                init=init,
                act=act,
                rank=rank,
                n_modes=num_modes,
            )

        def x_net():
            return MlpClass(
                x_dim,
                x_dim,
                x_dim * 2,
                n_hidden,
                init=init,
                act=act,
                rank=rank,
                n_modes=num_modes,
            )

        def combined_net():
            return MlpClass(
                delta_dim + x_dim,
                delta_dim + x_dim,
                (delta_dim + x_dim) * 2,
                n_hidden,
                init=init,
                act=act,
                rank=rank,
                n_modes=num_modes,
            )

        def ID():
            return lambda x, mode=None: x

        if combine:
            self.mlp = combined_net()
        elif one_sided:
            if x_dim > delta_dim:
                self.mlp1, self.mlp2 = ID(), delta_net()
            else:
                self.mlp1, self.mlp2 = x_net(), ID()
        elif x_only:
            self.mlp1, self.mlp2 = x_net(), ID()
        elif delta_only:
            self.mlp1, self.mlp2 = ID(), delta_net()
        else:
            self.mlp1, self.mlp2 = x_net(), delta_net()

    def forward(self, u, v, param_idx=None):
        u, v = u.to(torch.float32), v.to(torch.float32)

        u_ = u.view(-1, u.shape[-1])
        v_ = v.view(-1, v.shape[-1])

        nz_mask = (u_ != 0).any(-1) * (v_ != 0).any(
            -1
        )  # Skip batch elements with zero grad
        u_ = u_[nz_mask]
        v_ = v_[nz_mask]

        if self.training:
            for idx in range(u_.shape[0]):
                if not self.norm_init:
                    self.u_mean = u_[idx].clone().detach()
                    self.v_mean = v_[idx].clone().detach()
                    self.u_s.zero_()
                    self.v_s.zero_()
                    self.k[:] = 1
                    self.norm_init = True
                else:
                    self.k += 1
                    self.u_mean, self.u_s = update_counter(
                        u_[idx], self.u_mean, self.u_s, self.k
                    )
                    self.v_mean, self.v_s = update_counter(
                        v_[idx], self.v_mean, self.v_s, self.k
                    )

            if self.k < 2:
                raise RuntimeError(
                    f"Can't perform normalization with only {self.k} samples so far"
                )
            self.u_std = (self.u_s / (self.k - 1)) ** 0.5
            self.v_std = (self.v_s / (self.k - 1)) ** 0.5

        if self.norm:
            u_input = (u_ - self.u_mean) / (self.u_std + 1e-7)
            v_input = (v_ - self.v_mean) / (self.v_std + 1e-7)
        else:
            u_input = u_
            v_input = v_

        if self.combine:
            output = self.mlp(torch.cat((u_input, v_input), -1), mode=param_idx)
            out1, out2 = output.split([u.shape[-1], v.shape[-1]], -1)
            return out1, out2
        else:
            return self.mlp1(u_input, mode=param_idx), self.mlp2(
                v_input, mode=param_idx
            )


def get_edit_loss_fn(model_name) -> Callable[[Tensor, Tensor], Tensor]:
    def _edit_loss_fn(pred, targ, **kwargs):
        return masked_log_probs(
            pred, targ, shift=should_shift_targets(model_name), **kwargs
        )

    return _edit_loss_fn


class Mend(nn.Module):
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        model_constructor,
        update_param_names: list[str],
        mend=None,
        edit_lrs=None,
        # The default model config
        model_name: str = "google/t5-small-ssm-nq",
        # The default ALG config according to the official repo
        lr: float = 1e-6,
        edit_lr: float = 1e-4,
        # lr_lr: float = 1e-4,
        # one_sided: bool = False,
        # n_hidden: int = 1,
        # hidden_dim: int = None,
        # init: str = "id",
        # norm: bool = True,
        # combine: bool = True,
        # x_only: bool = False,
        # delta_only: bool = False,
        # act: str = "relu",
        # rank: int = 1920,
        # mlp_class: str = "IDMLP",
        shared: bool = True,
        descent: bool = False,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.model_constructor = model_constructor
        self.lr = lr

        _edit_loss_fn = get_edit_loss_fn(model_name)
        self.edit_loss_fn = _edit_loss_fn
        self.loc_loss_fn = _edit_loss_fn
        self.mend = mend
        self.edit_lrs = edit_lrs

        self.shared = shared
        self.descent = descent

        self.update_param_names = update_param_names

        if edit_lrs is None:
            # Assign same LR to all params
            edit_lrs = nn.Parameter(
                torch.tensor([edit_lr] * len(self.update_param_names))
            )
        self.edit_lrs = edit_lrs

        if not hasattr(self.model, "handles"):
            hook_model(self.model, self.update_param_names)
            print(f"Hooked {len(self.model.handles)//2} modules")

        if shared:
            shape_dict = defaultdict(list)
            for name, param in get_inner_params(
                model.named_parameters(), self.update_param_names
            ):
                shape_dict[self.get_shape(param)].append(name)
            self.shape_dict = shape_dict
            print(shape_dict)

        if self.mend is None:
            print("Mend is None, initting")
            if not shared:
                modules = {}
                inner_params = get_inner_params(
                    model.named_parameters(), self.update_param_names
                )
                for n, p in inner_params:
                    modules[n.replace(".", "#")] = GradientTransform(*self.get_shape(p))
                self.mend = nn.ModuleDict(modules)
            else:
                modules = {}
                for shape in shape_dict.keys():
                    print(shape)
                    shape_str = str(tuple(shape))
                    num_modes = len(shape_dict[shape])
                    print(shape_str, num_modes)
                    modules[shape_str] = GradientTransform(*shape, num_modes=num_modes)
                self.mend = nn.ModuleDict(modules)
        else:
            self.mend = mend

    def get_shape(self, p):
        # We need to (annoyingly) flip the shapes since OpenAI gpt2 uses convs
        # instead of linear
        if isinstance(self.model, transformers.GPT2LMHeadModel):
            return p.shape
        else:
            return (p.shape[1], p.shape[0])

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            prefix=prefix, keep_vars=keep_vars
        )  # Get default state dict
        model_keys = self.model.state_dict(
            prefix=prefix, keep_vars=keep_vars
        ).keys()  # Remove model params
        for k in model_keys:
            del state_dict[f"model.{k}"]
        state_dict["model_config"] = self.model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.model.config:
            print("Loaded model config doesn't match current model ")
            print(f"Loaded: {config}")
            print(f"Current: {self.model.config}")

        res = super().load_state_dict(state_dict, False)
        # We should only have missing keys for the model, and no unexpected keys
        assert (
            len([k for k in res.missing_keys if not k.startswith("model.")]) == 0
        ), "Should only have missing keys for model."
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    def outer_parameters(self, grouped=False):
        if grouped:
            return [
                dict(params=list(self.mend.parameters()), lr=self.lr),
                dict(params=[self.edit_lrs], lr=self.lr_lr),
            ]
        else:
            return list(self.mend.parameters()) + [self.edit_lrs]

    def get_transform_factors(self, mend, inner_params, shared):
        if shared:

            def param_idx(n, p):
                return self.shape_dict[self.get_shape(p)].index(n)

            factors = {}
            for name, params in inner_params:
                mend_key = str(tuple(self.get_shape(params)))
                mode = param_idx(name, params)
                print("mode:", mode)
                factors[name] = self.mend[mend_key](
                    params.__x__, params.__delta__, mode
                )
        else:
            factors = {}
            for name, params in inner_params:
                factors[name] = self.mend[name.replace(".", "#")](
                    params.__x__, params.__delta__
                )
        return factors

    def edit(self, batch: dict[str, Tensor], detach_history=False):
        # Forward
        outputs = self.model(**batch)
        logits = outputs.logits
        labels = batch["labels"]
        loss = self.edit_loss_fn(logits, labels)["nll"]

        # Sanity check
        param_names = set([n for n, p in self.model.named_parameters()])
        for params in set(self.update_param_names):
            assert params in param_names, f"inner param {params} not in model"

        # Compute gradient
        loss.backward()

        # Transform gradients to parameter updates
        inner_params = get_inner_params(
            self.model.named_parameters(), self.update_param_names
        )
        transformed_factors = self.get_transform_factors(
            self.mend, inner_params, self.shared
        )

        # Should be bi,bj->ji for nn.Linear, but [annoying] GPT2 uses Conv1d instead...
        if isinstance(self.model, transformers.GPT2LMHeadModel):
            targ = "ij"
        else:
            targ = "ji"
        mean_grads = {
            n: torch.einsum(f"bi,bj->{targ}", x, delta)
            for n, (x, delta) in transformed_factors.items()
        }

        info_dict = {}
        idx = 0
        for name, params in inner_params:
            info_dict[f"grad/true_mag{idx}"] = params.grad.norm(2).item()
            info_dict[f"grad/pseudo_mag{idx}"] = mean_grads[name].norm(2).item()
            info_dict[f"grad/true_std{idx}"] = params.grad.std().item()
            info_dict[f"grad/pseudo_std{idx}"] = mean_grads[name].std().item()
            info_dict[f"grad/diff{idx}"] = (
                (params.grad - mean_grads[name]).norm(2).item()
            )
            info_dict[f"grad/cos{idx}"] = F.cosine_similarity(
                params.grad.reshape(-1), mean_grads[name].reshape(-1), dim=0
            ).item()
            idx += 1

        self.model.zero_grad()

        assert len(self.edit_lrs) == len(list(mean_grads.items()))
        updates = {n: lr * g for lr, (n, g) in zip(self.edit_lrs, mean_grads.items())}

        edited_model = self.model
        if not isinstance(edited_model, higher.patch._MonkeyPatchBase):
            # edited_model = make_functional(edited_model, in_place=True)
            edited_model = make_functional(edited_model)

        new_params = []
        for name, params in edited_model.named_parameters():
            if name in set(self.update_param_names):
                if self.descent:
                    new_params.append(params - updates[name])
                else:
                    new_params.append(params + updates[name])
            else:
                new_params.append(params)

        edited_model.update_params(new_params)

        if detach_history:
            new_model = self.model_constructor()
            new_model.load_state_dict(edited_model.state_dict())
            edited_model = new_model

        edited_mend = Mend(
            edited_model,
            self.model_constructor,
            self.mend,
            edit_lrs=self.edit_lrs,
        )
        return edited_mend, info_dict

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


if __name__ == "__main__":
    UPDATE_PARAM_NAMES = [
        # "encoder.block.2.layer.1.DenseReluDense.wi.weight",
        "encoder.block.2.layer.1.DenseReluDense.wo.weight",
        # "encoder.block.3.layer.1.DenseReluDense.wi.weight",
        "encoder.block.3.layer.1.DenseReluDense.wo.weight",
        # "decoder.block.2.layer.2.DenseReluDense.wi.weight",
        "decoder.block.2.layer.2.DenseReluDense.wo.weight",
        # "decoder.block.3.layer.2.DenseReluDense.wi.weight",
        "decoder.block.3.layer.2.DenseReluDense.wo.weight",
        # "encoder.block.22.layer.1.DenseReluDense.wi.weight",
        # "encoder.block.22.layer.1.DenseReluDense.wo.weight",
        # "encoder.block.23.layer.1.DenseReluDense.wi.weight",
        # "encoder.block.23.layer.1.DenseReluDense.wo.weight",
        # "decoder.block.22.layer.2.DenseReluDense.wi.weight",
        # "decoder.block.22.layer.2.DenseReluDense.wo.weight",
        # "decoder.block.23.layer.2.DenseReluDense.wi.weight",
        # "decoder.block.23.layer.2.DenseReluDense.wo.weight",
    ]
    edit_lr = 0.0001
    model = T5ForConditionalGeneration.from_pretrained("google/t5-small-ssm-nq")

    def model_constructor(model):
        return copy.deepcopy(model)

    editor = Mend(
        model, model_constructor, update_param_names=UPDATE_PARAM_NAMES
    ).cuda()
    print("Saving model")
    torch.save(editor.state_dict(), "test_state.pt")

    print("loading model")
    editor.load_state_dict(torch.load("test_state.pt"))
    x = torch.arange(20).view(1, 20).cuda() + 1000

    print("Getting orig logits")
    preds_before = editor.generate(x)

    print("Editing model")
    batch = {
        "input_ids": x,
        "attention_mask": torch.ones_like(x),
        "labels": x,
    }
    editor1, info1 = editor.edit(batch)
    preds_after = editor1.generate(x)

    print(info1)
    assert torch.allclose(preds_before, preds_after)

    orig_param = [
        p
        for (n, p) in editor.model.named_parameters()
        if n == editor.update_param_names[-1]
    ][0]
    edited_param = [
        p
        for (n, p) in editor1.model.named_parameters()
        if n == editor.update_param_names[-1]
    ][0]

    print((orig_param - edited_param).abs().max())
    editor1.eval()
    print(
        editor(x, labels=x).loss,
        editor1(x, labels=x).loss,
        editor1.edit_loss_fn(editor1(x).logits, x)["nll"],
    )
    editor2, info2 = editor1.edit(x, masks=torch.ones_like(x), labels=x)
    print(info2)
    print(
        editor(x, labels=x).loss,
        editor1(x, labels=x).loss,
        editor2(x, labels=x).loss,
    )
