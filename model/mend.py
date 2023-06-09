import copy
from collections import defaultdict
from typing import Callable, List, Tuple, Dict, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, GPT2LMHeadModel

from hooks import hook_model
from utils import get_inner_params
from losses import masked_log_probs
from utils import should_shift_targets
from .grad_transform import GradientTransform


def get_edit_loss_fn(model_name) -> Callable[[Tensor, Tensor], Dict[str, Tensor]]:
    def _edit_loss_fn(pred, targ, **kwargs):
        return masked_log_probs(
            pred, targ, shift=should_shift_targets(model_name), **kwargs
        )

    return _edit_loss_fn


class Mend(nn.Module):
    def __init__(
        self,
        base_model: T5ForConditionalGeneration,
        model_constructor,
        update_param_names: list[str],
        mend=None,
        edit_lrs=None,
        # The default model config
        model_name: str = "google/t5-small-ssm-nq",
        # The default ALG config according to the official repo
        lr: float = 1e-6,
        edit_lr: float = 1e-4,
        lr_lr: float = 1e-4,
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
        descent: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.base_model = base_model
        self.model_name = model_name
        self.model_constructor = model_constructor
        self.lr = lr

        _edit_loss_fn = get_edit_loss_fn(model_name)
        self.edit_loss_fn = _edit_loss_fn
        self.loc_loss_fn = _edit_loss_fn
        self.mend = mend
        self.edit_lrs = edit_lrs
        self.lr_lr = lr_lr
        self.shared = shared
        self.descent = descent
        self.device = device

        self.update_param_names = update_param_names

        if edit_lrs is None:
            # Assign same LR to all params
            edit_lrs = nn.Parameter(
                torch.tensor([edit_lr] * len(self.update_param_names))
            )
        self.edit_lrs = edit_lrs

        if not hasattr(self.base_model, "handles"):
            hook_model(self.base_model, self.update_param_names)
            print(f"Hooked {len(self.base_model.handles)//2} modules")

        if shared:
            shape_dict = defaultdict(list)
            inner_params = get_inner_params(
                base_model.named_parameters(), self.update_param_names
            )
            for name, param in inner_params:
                shape = self.get_shape(param)
                shape_dict[shape].append(name)
            self.shape_dict = shape_dict

        if self.mend is None:
            print("Mend is None, initting")
            if not shared:
                modules = {}
                inner_params = get_inner_params(
                    base_model.named_parameters(), self.update_param_names
                )
                for n, p in inner_params:
                    name = n.replace(".", "#")
                    shape = self.get_shape(p)
                    modules[name] = GradientTransform(*shape)
                self.mend = nn.ModuleDict(modules)
            else:
                modules = {}
                for shape in shape_dict.keys():
                    # print(shape)
                    shape_str = str(tuple(shape))
                    num_modes = len(shape_dict[shape])
                    # print(shape_str, num_modes)
                    modules[shape_str] = GradientTransform(*shape, num_modes=num_modes)
                self.mend = nn.ModuleDict(modules)
        else:
            self.mend = mend

    def get_shape(self, p):
        # We need to (annoyingly) flip the shapes since OpenAI gpt2 uses convs
        # instead of linear
        if isinstance(self.base_model, GPT2LMHeadModel):
            return p.shape
        else:
            return (p.shape[1], p.shape[0])

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            prefix=prefix, keep_vars=keep_vars
        )  # Get default state dict
        model_keys = self.base_model.state_dict(
            prefix=prefix, keep_vars=keep_vars
        ).keys()  # Remove model params
        for k in model_keys:
            del state_dict[f"base_model.{k}"]
        state_dict["model_config"] = self.base_model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.base_model.config:
            print("Loaded model config doesn't match current model ")
            print(f"Loaded: {config}")
            print(f"Current: {self.base_model.config}")

        res = super().load_state_dict(state_dict, False)
        # We should only have missing keys for the model, and no unexpected keys
        assert (
            len([k for k in res.missing_keys if not k.startswith("base_model.")]) == 0
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

    def get_transform_factors(self, inner_params: List[Tuple[str, Tensor]]):
        """
        Get the transform factors for the given inner parameters.
        Args:
            inner_params (List[Tuple[str, Tensor]]): A list of tuples containing the
                name and tensor of each parameter.
        Returns:
            factors (Dict[str, Tensor]): A dictionary containing the transform factors
                for each parameter.
        """
        if self.shared:

            def param_idx(n, p):
                return self.shape_dict[self.get_shape(p)].index(n)

            factors = {}
            for name, params in inner_params:
                mend_key = str(tuple(self.get_shape(params)))
                mode = param_idx(name, params)
                print("mode:", mode)
                print(type(params))
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

    def edit(self, batch: dict[str, Tensor], condition: Optional[Tensor]):
        # Forward
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.base_model(**batch)
        logits = outputs.logits
        labels = batch["labels"]
        loss = self.edit_loss_fn(logits, labels)["nll"]

        # Sanity check
        param_names = set([n for n, p in self.base_model.named_parameters()])
        for params in set(self.update_param_names):
            assert params in param_names, f"inner param {params} not in model"

        # Compute gradient
        loss.backward()

        # Transform gradients to parameter updates
        for n, p in self.base_model.named_parameters():
            print(type(p))
            break
        inner_params = get_inner_params(
            self.base_model.named_parameters(), self.update_param_names
        )
        print("shape:", inner_params[0][1].shape)
        transformed_factors = self.get_transform_factors(inner_params)

        # Should be bi,bj->ji for nn.Linear, but [annoying] GPT2 uses Conv1d instead...
        if isinstance(self.base_model, GPT2LMHeadModel):
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

        self.base_model.zero_grad()
        assert len(self.edit_lrs) == len(list(mean_grads.items()))
        updates = {n: lr * g for lr, (n, g) in zip(self.edit_lrs, mean_grads.items())}

        # edited_model: T5ForConditionalGeneration = self.base_model
        # if not isinstance(edited_model, higher.patch._MonkeyPatchBase):
        #     # edited_model = make_functional(edited_model, in_place=True)
        #     # for p in edited_model.parameters():
        #     #     print(type(p))
        #     #     break
        #     # edited_model = make_functional(edited_model, track_higher_grads=False)
        #     self.edited_model = self.model_constructor(self.base_model)

        #     # for p in edited_model.parameters():
        #     #     print(type(p))
        #     #     break
        #     print('================================================')

        state_dict = self.base_model.state_dict()
        for name in self.update_param_names:
            if self.descent:
                state_dict[name] -= updates[name]
            else:
                state_dict[name] += updates[name]
        self.base_model.load_state_dict(state_dict)
        # new_params = {}
        # for name, params in edited_model.named_parameters():
        #     if name in set(self.update_param_names):
        #         if self.descent:
        #             new_params[name] = params - updates[name]
        #         else:
        #             new_params[name] = params + updates[name]
        #     else:
        #         new_params[name] = params

        # # edited_model.update_params(new_params)
        # edited_model.load_state_dict(new_params)

        # if detach_history:
        #     new_model = self.model_constructor()
        #     new_model.load_state_dict(edited_model.state_dict())
        #     edited_model = new_model

        # edited_mend = Mend(
        #     self.edited_model,
        #     self.model_constructor,
        #     update_param_names=self.update_param_names,
        #     mend=self.mend,
        #     edit_lrs=self.edit_lrs,
        # )
        return info_dict

    def forward(self, *inputs, **kwargs):
        return self.base_model(*inputs, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)


def get_base_model(pretrained_name: str) -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(pretrained_name)
    if isinstance(model, T5ForConditionalGeneration):
        return model
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


if __name__ == "__main__":
    print("start")
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
    model = get_base_model("google/t5-small-ssm-nq")

    def model_constructor(model):
        return copy.deepcopy(model)

    editor = Mend(
        model, model_constructor, update_param_names=UPDATE_PARAM_NAMES
    ).cuda()
    print("Saving model")
    torch.save(editor.state_dict(), "test_state.pt")

    print("loading model")
    editor.load_state_dict(torch.load("test_state.pt"))
    input_ids = torch.arange(20).view(1, 20).cuda() + 1000

    outputs0 = editor(input_ids, labels=input_ids)
    preds_before = editor.generate(input_ids)
    print("Editing model")
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids,
    }

    # 第一次编辑！
    print("================ 第一次编辑！ =================")
    info1 = editor.edit(batch)
    preds_after = editor.generate(input_ids)
    print("Preds before:", preds_before)
    print("Preds after:", preds_after)

    print(info1)
    assert torch.allclose(preds_before, preds_after)

    orig_param = [
        p
        for (n, p) in editor.base_model.named_parameters()
        if n == UPDATE_PARAM_NAMES[-1]
    ][0]
    edited_param = [
        p
        for (n, p) in editor.base_model.named_parameters()
        if n == UPDATE_PARAM_NAMES[-1]
    ][0]

    print(f"参数变化，{UPDATE_PARAM_NAMES[-1]}：", (orig_param - edited_param).abs().max())

    # Get loss

    editor.eval()
    outputs1 = editor(input_ids, labels=input_ids)
    print("Loss 0:", outputs0.loss.item())
    print("Loss 1:", outputs1.loss.item())
    nll1 = editor.edit_loss_fn(outputs1.logits, input_ids)["nll"]
    assert torch.allclose(nll1, outputs1.loss)

    # 第二次编辑！
    print("================ 第二次编辑！ =================")
    editor.eval()
    info2 = editor.edit(batch)
    print(info2)
    outputs2 = editor(input_ids, labels=input_ids)
    print(outputs0.loss, outputs1.loss, outputs2.loss)
