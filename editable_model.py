from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn as nn

from losses import masked_log_probs
from utils import should_shift_targets


class EditableModel(nn.Module):
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        model_name: str,
        model_constructor,
        lr: float,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.model_constructor = model_constructor
        self.lr = lr

        def _edit_loss_fn(pred, targ, **kwargs):
            return masked_log_probs(
                pred, targ, shift=should_shift_targets(model_name), **kwargs
            )

        self.edit_loss_fn = _edit_loss_fn
        self.loc_loss_fn = _edit_loss_fn

    def edit(self, batch, condition=None, detach_history=False):
        raise NotImplementedError

    def forward(self, *inputs, **kwargs):
        return _logits(self.model(*inputs, **kwargs))

    def outer_parameters(self, grouped=False):
        if grouped:
            return [dict(params=self.parameters(), lr=self.lr)]
        else:
            return list(self.parameters())

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def base_loss(self, input_ids, attention_masks, label_ids):
        pass
