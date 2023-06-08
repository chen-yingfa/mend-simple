from typing import Optional
import torch
import torch.nn as nn
import re
from .nn import FixableDropout

from transformers import T5ForConditionalGeneration, T5Tokenizer


class CastModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        in_cast: torch.dtype = torch.float32,
        out_cast: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.underlying = module
        self.in_cast = in_cast
        self.out_cast = out_cast

    def cast(self, obj, dtype):
        if dtype is None:
            return obj

        if isinstance(obj, torch.Tensor):
            return obj.to(dtype)
        else:
            return obj

    def forward(self, *args, **kwargs):
        args = tuple(self.cast(a, self.in_cast) for a in args)
        kwargs = {k: self.cast(v, self.in_cast) for k, v in kwargs.items()}
        outputs = self.underlying(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            outputs = self.cast(outputs, self.out_cast)
        elif isinstance(outputs, tuple):
            outputs = tuple(self.cast(o, self.out_cast) for o in outputs)
        else:
            raise RuntimeError(f"Not sure how to cast type {type(outputs)}")
        return outputs

    def extra_repr(self):
        return f"in_cast: {self.in_cast}\nout_cast: {self.out_cast}"


def replace_dropout(model):
    for m in model.modules():
        for n, c in m.named_children():
            if isinstance(c, nn.Dropout):
                setattr(m, n, FixableDropout(c.p))

    def resample(m, seed=None):
        for c in m.children():
            if hasattr(c, "resample"):
                c.resample(seed)
            else:
                resample(c, seed)

    model.resample_dropout = resample.__get__(model)


def load_ckpt(model, ckpt_path: str):
    """
    Sets model's state dict inplace.
    """
    print(f"Loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print("Default load failed; stripping prefix and trying again.")
        state_dict = {re.sub("^model.", "", k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    print("Loaded checkpoint")


def set_dropout(model: nn.Module, dropout_rate: float):
    """
    Set dropout rate for all dropout modules in model.
    """
    print("Dropout is not None, setting")
    n_reset = 0
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = dropout_rate
            n_reset += 1

        if hasattr(mod, "dropout"):  # Requires for BART, which uses F.dropout
            if isinstance(mod.dropout, float):
                mod.dropout = dropout_rate
                n_reset += 1

        if hasattr(
            mod, "activation_dropout"
        ):  # Requires for BART, which uses F.dropout
            if isinstance(mod.activation_dropout, float):
                mod.activation_dropout = dropout_rate
                n_reset += 1

    print(f"Set {n_reset} dropout modules to p={dropout_rate}")


def handle_no_grad(
    model: nn.Module,
    no_grad_layers: str = None,
    inner_params: list = None,
    half: bool = False,
):
    """
    
    """
    if half:
        model.bfloat16()

    def upcast(mod):
        '''
        
        '''
        modlist = None
        for child in mod.children():
            if isinstance(child, nn.ModuleList):
                assert modlist is None, f"Found multiple modlists for {mod}"
                modlist = child
        if modlist is None:
            raise RuntimeError("Couldn't find a ModuleList child")

        print(
            f"Setting {len(modlist) - no_grad_layers} modules to"
            " full precision, with autocasting"
        )
        modlist[no_grad_layers:].to(torch.float32)
        modlist[no_grad_layers] = CastModule(modlist[no_grad_layers])
        modlist[-1] = CastModule(
            modlist[-1], in_cast=torch.float32, out_cast=torch.bfloat16
        )

    parents = []
    if hasattr(model, "transformer"):
        parents.append(model.transformer)
    if hasattr(model, "encoder"):
        parents.append(model.encoder)
    if hasattr(model, "decoder"):
        parents.append(model.decoder)
    if hasattr(model, "model"):
        parents.extend([model.model.encoder, model.model.decoder])

    for t in parents:
        t.no_grad_layers = no_grad_layers
        if half:
            upcast(t)

    if half:
        idxs = []
        for p in inner_params:
            for comp in p.split("."):
                if comp.isdigit():
                    idxs.append(int(comp))
        max_idx, min_idx = str(max(idxs)), str(no_grad_layers)
        for pidx, p in enumerate(inner_params):
            comps = p.split(".")
            if max_idx in comps or min_idx in comps:
                index = (
                    comps.index(max_idx) if max_idx in comps else comps.index(min_idx)
                )
                comps.insert(index + 1, "underlying")
                new_p = ".".join(comps)
                print(f"Replacing inner_params[{pidx}] '{p}' -> '{new_p}'")
                inner_params[pidx] = new_p


def get_model(
    pretrained_name: str,
    inner_params: list,
    ckpt_path: Optional[str] = None,
    dropout_rate: Optional[float] = None,
    no_grad_layers: Optional[str] = None,
    half: bool = False,
):
    print(f"Loading {pretrained_name}")
    model = T5ForConditionalGeneration.from_pretrained(pretrained_name)

    if ckpt_path is not None:
        load_ckpt(model, ckpt_path)

    if dropout_rate is not None:
        set_dropout(model, dropout_rate)

    param_names = [n for n, _ in model.named_parameters()]
    bad_inner_params = [p for p in inner_params if p not in param_names]
    if len(bad_inner_params) != 0:
        raise ValueError(
            f"Params {bad_inner_params} do not exist in model of type {type(model)}."
        )

    if no_grad_layers is not None:
        handle_no_grad(
            model, no_grad_layers=no_grad_layers, inner_params=inner_params, half=half
        )

    return model


def get_tokenizer(pretrained_name: str):
    return T5Tokenizer.from_pretrained(pretrained_name)


if __name__ == "__main__":
    model = T5ForConditionalGeneration.from_pretrained("google/t5-small-ssm-nq")
    t = torch.arange(5).unsqueeze(0)  # (1, 5)
    model(t)
    breakpoint()
