from pathlib import Path
import copy
from typing import List, Optional

from torch import nn

from model.models import get_model, get_tokenizer
from model.mend import Mend
from data.zsre import QaDataset
from trainer import EditorTrainer


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


def get_dataset(tokenizer, data_dir: Path, split: str, nq_dir: Optional[Path] = None):
    """
    Args:
        tokenizer: tokenizer
        data_dir: data directory
        split: train/dev/test
        nq_dir: nq data directory, used for hard negatives.
    """
    zsre_dir = data_dir / "zsre"
    # nq_dir = data_dir / "nq"
    dataset = QaDataset(
        tokenizer=tokenizer, data_dir=zsre_dir, split=split, nq_dir=nq_dir
    )
    return dataset


def train(
    editor: Mend, tokenizer, data_dir: Path, output_dir: Path, log_interval: int = 1000
):
    # Data
    print("Loading data...")
    train_data = get_dataset(tokenizer, data_dir, "train")
    dev_data = get_dataset(tokenizer, data_dir, "dev")

    # Trainer
    trainer = EditorTrainer(
        editor,
        output_dir=output_dir,
        train_data=train_data,
        dev_data=dev_data,
        log_interval=log_interval,
    )
    trainer.run()


def main():
    pretrained_name = "google/t5-small-ssm-nq"

    # Model
    model = get_model(pretrained_name, UPDATE_PARAM_NAMES)
    tokenizer = get_tokenizer(pretrained_name)
    mend = get_editor(model, UPDATE_PARAM_NAMES)

    # Data
    data_dir = Path("../data")

    output_dir = Path("result/mend")
    output_dir.mkdir(parents=True, exist_ok=True)
    train(mend, tokenizer, data_dir, output_dir)


if __name__ == "__main__":
    main()
