import copy
from typing import List

from torch import nn

from model.models import get_model, get_tokenizer
from model.mend import Mend

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


def model_constructor(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def get_editor(model, update_param_names: List[str]):
    return Mend(model, model_constructor, update_param_names)


def main():
    pretrained_name = 'google/t5-small-ssm-nq'

    # Model
    model = get_model(pretrained_name, UPDATE_PARAM_NAMES)
    tokenizer = get_tokenizer(pretrained_name)
    mend = get_editor(model, UPDATE_PARAM_NAMES)

    # Data


if __name__ == "__main__":
    main()
