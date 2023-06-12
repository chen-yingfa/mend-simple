import datetime
import typing
from typing import List, Tuple
from collections import defaultdict
import math

import torch
import torch.nn as nn


def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()


def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels


def gather_log_probs(logits, labels):
    assert labels.dim() == logits.dim() - 1
    assert labels.shape == logits.shape[:-1]
    return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def off_diagonal(mat):
    assert mat.dim() == 2
    # assert mat.shape[0] == mat.shape[1]

    mask = ~torch.eye(max(mat.shape), dtype=torch.bool)
    mask = mask[: mat.shape[0], : mat.shape[1]]
    off_d = mat[mask]

    assert off_d.numel() == mat.shape[0] * mat.shape[1] - min(mat.shape)

    return off_d


def set_dropout(model, p):
    """
    Sets the dropout probability of all nn.Dropout modules in the provided
    PyTorch model and all its submodules.

    :param model: The PyTorch model to modify.
    :type model: torch.nn.Module
    :param p: The new dropout probability. If None, the function does nothing.
    :type p: float or None
    """
    if p is not None:
        n_reset = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = p
                n_reset += 1

            if hasattr(m, "dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.dropout, float):
                    m.dropout = p
                    n_reset += 1

            if hasattr(
                m, "activation_dropout"
            ):  # Requires for BART, which uses F.dropout
                if isinstance(m.activation_dropout, float):
                    m.activation_dropout = p
                    n_reset += 1

        print(f"Set {n_reset} dropout modules to p={p}")


def get_inner_params(
    named_parameters, inner_names: List[str]
) -> List[Tuple[str, torch.Tensor]]:
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]


def should_shift_targets(model_name: str) -> bool:
    return "t5" not in model_name.lower() and "blender" not in model_name.lower()


# https://stackoverflow.com/questions/32871539/integer-factorization-in-python
def factorization(n):
    return [(i, n // i) for i in range(1, int(n**0.5) + 1) if n % i == 0]


def formatted_timestamp(time=None):
    if time is None:
        time = datetime.datetime.now()
    return time.strftime("%d/%m/%Y-%H:%M:%S/%f")


def time_delta_seconds(start, finish=None):
    assert type(start) == str

    t1 = datetime.datetime.strptime(start, "%d/%m/%Y-%H:%M:%S/%f")
    if finish is not None:
        assert type(finish) == str
        t2 = datetime.datetime.strptime(finish, "%d/%m/%Y-%H:%M:%S/%f")
    else:
        t2 = datetime.datetime.now()

    return (t2 - t1).total_seconds()


def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict


def safe_backward(loss, parameters, accumulate=1, allow_unused=False, backward=False):
    if backward:
        (loss / accumulate).backward()
    else:
        parameters = list(parameters)  # Capture the generator output
        grads = torch.autograd.grad(loss, parameters, allow_unused=allow_unused)
        nan, inf = False, False
        for g in grads:
            if g is not None:
                nan |= g.isnan().any().item()  # type: ignore
                inf |= g.isinf().any().item()  # type: ignore

        if not (nan or inf):
            for p, g in zip(parameters, grads):
                if g is None:
                    continue

                if p.grad is None:
                    p.grad = g / accumulate
                else:
                    p.grad += g / accumulate
        else:
            print(f"Skipping grad accumulation because inf: {inf} nan: {nan}")


def _last_encoder_state(x):
    if hasattr(x, "encoder_last_hidden_state"):
        return x.encoder_last_hidden_state
    else:
        return x.hidden_states[-1]


def flatten_dict(d):
    to_process = list(d.items())
    output = {}
    while len(to_process):
        k, v = to_process.pop()
        if isinstance(v, typing.MutableMapping):
            to_process.extend([(f"{k}.{k_}", v_) for (k_, v_) in v.items()])
        else:
            assert k not in output.keys(), "Somehow ended up with duplicate keys"
            output[k] = v

    return output


def add_padding(tokenizer, model):
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)


def add_sep(tokenizer, model):
    tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    # model.resize_token_embeddings(len(tokenizer))
    # model.lm_head.weight.data[-1, :] = model.lm_head.weight.data.mean(0)


class EarlyStopper:
    def __init__(self, patience: int, key: str):
        self.best_value = 1e9
        self.best_iter = 0
        self.current_iter = 0
        self.key = key
        self.patience = patience
        self._stop = False

    def update(self, idx, stats):
        assert self.key in stats, f"'{self.key}' not in stats dict"
        value = stats[self.key]
        new_best = value < self.best_value
        if new_best:
            self.best_value = value
            self.best_iter = idx

        self.current_iter = idx
        return new_best

    def should_stop(self):
        self._stop |= self.current_iter - self.best_iter >= self.patience
        return self._stop


class RunningStatAverager:
    def __init__(self, suffix="", exclude=["grad/"], compute_ppl: bool = True):
        self.underlying = None
        self.suffix = suffix
        self.exclude = exclude
        self.compute_ppl = compute_ppl

        self.reset()

    def add(self, d: dict):
        for k, v in d.items():
            if not any([k.startswith(prefix) for prefix in self.exclude]):
                if len(self.suffix):
                    self.underlying[f"{k}_{self.suffix}"].append(v)  # type: ignore
                else:
                    self.underlying[k].append(v)  # type: ignore

    def average(self):
        average = {}
        for k, v in self.underlying.items():  # type: ignore
            if not k.startswith("nll/"):
                average[k] = sum(v) / len(v)
            else:
                assert len(k.split("/")) == 2, f"Invalid key {k}"
                name = k.split("/")[1]
                token_counts = self.underlying[f"n_tokens/{name}"]  # type: ignore
                total_nll = sum([nll * c for nll, c in zip(v, token_counts)])
                average[k] = total_nll / sum(token_counts)
                if self.compute_ppl:
                    average[f"perplexity/{name}"] = math.e ** average[k]

        return {
            k: v if not isinstance(v, torch.Tensor) else v.item()
            for k, v in average.items()
        }

    def reset(self):
        self.underlying = defaultdict(list)


def parent_module(model, pname):
    comps = pname.split(".")
    parent = model
    # print(list(parent.named_parameters().keys()))
    for comp in comps[:-1]:
        if hasattr(parent, comp):
            parent = getattr(parent, comp)
        elif comp.isdigit():
            parent = parent[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    assert hasattr(parent, comps[-1])
    return parent


def get_num_params(module: torch.nn.Module):
    return sum([p.numel() for p in module.parameters()])


def save_ckpt(model, optimizer, scheduler, stats, path):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "stats": stats,
    }
    torch.save(ckpt, path)


def load_ckpt(model, optimizer, scheduler, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    stats = ckpt["stats"]
    return stats
