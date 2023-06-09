import json
from pathlib import Path
from typing import Union
import random

import numpy as np
import datasets


def process_q(
    text: str, prompt: bool = False, capitalize: bool = True, question_mark: bool = True
):
    if capitalize:
        text = text[0].capitalize() + text[1:]
    if question_mark:
        text = text + "?"
    if prompt:
        text = "nq question: " + text
    return text


def extract_qa(dataset):
    questions = [process_q(q["text"]) for q in dataset["question"]]
    answers = [
        [a["text"][0] for a in ann["short_answers"] if len(a["text"])]
        for ann in dataset["annotations"]
    ]
    questions = [q for q, a in zip(questions, answers) if len(a)]
    answers = [min(a, key=len) for a in answers if len(a)]
    return questions, answers


class NqDataset:
    def __init__(self, path: Union[Path, str]):
        with open(path, "r") as f:
            self.data = json.load(f)
        self.questions = self.data["questions"]
        self.answers = self.data["answers"]

    def __getitem__(self, idx):
        idx = idx % len(self.questions)
        return self.questions[idx], self.answers[idx]

    def __len__(self):
        return len(self.questions)


def generate_nq(
    out_path: Union[Path, str],
    prompt: bool = False,
    capitalize: bool = True,
    question_mark: bool = True,
):
    train = datasets.load_dataset("natural_questions", split="train")
    tq, ta = extract_qa(train)
    val = datasets.load_dataset("natural_questions", split="validation")
    vq, va = extract_qa(val)

    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)
    with open(out_path / "train.json", "w", encoding='utf8') as f:
        json.dump({"questions": tq, "answers": ta}, f)
    with open(out_path / "validation.json", "w", encoding='utf8') as f:
        json.dump({"questions": vq, "answers": va}, f)


def get_nq_data(data_dir: Path, split: str):
    """
    Get the natural questions dataset
    """
    print("** Using natural questions for zsre base samples **")
    if split == 'train':
        dataset = NqDataset(data_dir / "train.json")
    else:
        dataset = NqDataset(data_dir / "validation.json")
    return dataset
    if self.config.data.zsre_nq:
        self.use_nq = True
    else:
        self.use_nq = False

        divisor = (
            2 + int(self.config.data.zsre_impl) + int(self.config.data.zsre_yn)
        )
        n_per_dist = len(self.data) // divisor
        remain = len(self.data) - n_per_dist * divisor
        self.loc_data = []
        base_data = [
            (sample["input"], sample["output"][0]["answer"]) for sample in self.data
        ]
        random.shuffle(base_data)
        self.loc_data += base_data[: n_per_dist + remain]

        rephrase_data = [
            (r, sample["output"][0]["answer"])
            for sample in self.data
            for r in sample["filtered_rephrases"]
        ]
        random.shuffle(rephrase_data)
        self.loc_data += rephrase_data[:n_per_dist]

        if self.config.data.zsre_impl:
            impl_data = [
                (q, a) for impl_set in self.impls for (q, a, _) in impl_set
            ]
            random.shuffle(impl_data)
            self.loc_data += impl_data[:n_per_dist]

        if self.config.data.zsre_yn:
            yn_data = [
                (prompt_yn(y), ("True" if np.random.uniform() < 0.5 else "False"))
                for y in self.yn
            ]
            random.shuffle(yn_data)
            self.loc_data += yn_data[:n_per_dist]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("../data/nq")
    args = parser.parse_args()
    dataset = NqDataset(args.out_path)
    print(dataset[2])
