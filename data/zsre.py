from pathlib import Path
import json
import itertools
import random
import re
from typing import Union, Dict, Optional, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    BartTokenizerFast,
    BartTokenizer,
    T5Tokenizer,
)
import numpy as np

from .utils import (
    EditBatchSampler,
    dict_to,
    build_distr_matrix,
    iter_jsonl,
    get_boolq,
)
from .nq import get_nq_data


def get_loc_data(
    examples: List,
    examples_impl: Optional[List] = None,
    examples_bool: Optional[List] = None,
    max_len: int = 512,
):
    divisor = 2
    divisor += int(examples_impl is not None)
    divisor += int(examples_bool is not None)

    num_examples = len(examples)
    num_per_dist = num_examples // divisor
    remain = num_examples - num_per_dist * divisor

    loc_data = []
    base_data = [(eg["input"], eg["output"][0]["answer"]) for eg in examples]
    random.seed(0)
    random.shuffle(base_data)
    loc_data += base_data[: num_per_dist + remain]

    rephrase_data = [
        (r, eg["output"][0]["answer"])
        for eg in examples
        for r in eg["filtered_rephrases"]
    ]
    random.shuffle(rephrase_data)
    loc_data += rephrase_data[:num_per_dist]

    if examples_impl is not None:
        impl_data = [(q, a) for impl_set in examples_impl for (q, a, _) in impl_set]
        random.shuffle(impl_data)
        loc_data += impl_data[:num_per_dist]

    if examples_bool is not None:
        boolqs = [
            (get_boolq(boolq), ("True" if np.random.uniform() < 0.5 else "False"))
            for boolq in examples_bool
        ]
        random.shuffle(boolqs)
        loc_data += boolqs[:num_per_dist]

    print(f"Data size {len(examples)}; loc data size {len(loc_data)}")
    return loc_data


def extract_example(d: Dict[str, Union[str, list]]):
    eg = {
        k: d[k]
        for k in [
            "input",
            "prediction",
            "alternatives",
            "filtered_rephrases",
            "output",
        ]
    }
    if eg["input"] in eg["filtered_rephrases"]:
        eg["filtered_rephrases"].remove(eg["input"])
    return eg


def has_good_inp(eg: dict) -> bool:
    if "sex" in eg["input"] or "gender" in eg["input"]:
        return False
    return True


class QaDataset(Dataset):
    """
    Dataset for QA using ZsRE and (optionally) NQ.
    """

    zsre_name = "structured_zeroshot-{}-new_annotated_final.jsonl"
    zsre_impl_name = "impl_{}.json"
    zsre_boolq_name = "zsre_yn_{}.txt"

    def __init__(
        self,
        split: str,
        tokenizer: T5Tokenizer,
        data_dir: Path,
        nq_dir: Optional[Path] = None,
        max_length=32,
        return_view=False,
        all_views=False,
        # For hard negative sampling
        use_hard_neg: bool = False,
        hard_neg_temp: float = 0.1,
        hard_neg_num_exclude: int = 0,
        hard_neg_num_neighbors: int = 20,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.nq_dir = nq_dir
        self.use_hard_neg = use_hard_neg
        self.hard_neg_temp = hard_neg_temp
        self.hard_neg_num_exclude = hard_neg_num_exclude
        self.hard_neg_num_neighbors = hard_neg_num_neighbors

        assert split in ["train", "dev", "test"]
        data_path = data_dir / self.zsre_name.format(split)

        boolqs, impls = self.load_augmentation_data(data_dir, split)
        eval_idxs = None

        self.impls = []
        self.all_impls = []
        self.examples = []
        self.all_data = []
        self.boolq = []
        self.all_boolq = []
        empty_yn = 0

        for idx, eg_data in enumerate(iter_jsonl(data_path)):
            # Get implied data
            try:
                impl_set = next(impls)
            except StopIteration:
                impl_set = []
            # Get boolean question
            try:
                boolq = next(boolqs)
            except StopIteration:
                boolq = ""
            extracted = extract_example(eg_data)
            self.all_data.append(extracted)
            self.all_impls.append(impl_set)
            self.all_boolq.append(boolq)
            if (
                len(extracted["alternatives"]) > 0
                and len(extracted["filtered_rephrases"]) > 0
            ):
                if eval_idxs is None or idx in eval_idxs:
                    if has_good_inp(extracted):
                        self.examples.append(extracted)
                        self.impls.append(impl_set)
                        self.boolq.append(boolq)
                        if len(boolq) == 0:
                            empty_yn += 1
        print(f"Empty {split} yn questions: {empty_yn}")

        self.max_length = max_length
        self.all_views = all_views
        self.return_view = return_view

        self.loc_data = get_loc_data(
            self.examples,
            examples_impl=self.impls,
            examples_bool=self.boolq,
        )

        if nq_dir is not None:
            self.nq = get_nq_data(nq_dir, split)
        if use_hard_neg:
            self.hard_neg_sims, self.hard_neg_idxs = self.create_hard_neg_mats(
                hard_neg_num_neighbors, hard_neg_num_exclude, hard_neg_temp
            )

    def create_hard_neg_mats(self, num_neighbors: int, num_exclude: int, temp: float):
        """
        Create similarity and index matrices for sampling hard negatives. Stores
        to `self.loc_distr_matrix` and `self.loc_idx_matrix`.
        """
        edit_qs = [sample["input"] for sample in self.examples]
        if self.nq_dir is not None:
            loc_qs = self.nq.questions
        else:
            loc_qs = [d[0] for d in self.loc_data]
        sim_scores, idxs = build_distr_matrix(
            edit_qs,
            loc_qs=loc_qs,
            num_neighbors=num_neighbors,
            temp=temp,
            num_exclude=num_exclude,
        )
        return sim_scores, idxs

    def load_augmentation_data(self, data_dir: Path, split: str):
        """
        Load data for data augmentation: implied examples and yes/no question.
        """
        if self.zsre_impl_name is not None:
            with open(
                data_dir / self.zsre_impl_name.format(split), "r", encoding="utf8"
            ) as fin:
                impls = iter(json.load(fin))
        else:
            impls = itertools.cycle([[]])

        if self.zsre_boolq_name is not None:
            with open(
                data_dir / self.zsre_boolq_name.format(split), "r", encoding="utf8"
            ) as fin:
                boolqs = iter(list(fin))
        else:
            boolqs = itertools.cycle([""])
        return boolqs, impls

    def is_bart(self):
        return isinstance(self.tokenizer, BartTokenizer) or isinstance(
            self.tokenizer, BartTokenizerFast
        )

    def get_boolq(
        self, idx, escaped_orig_label: str, new_label: str
    ) -> Tuple[Union[str, None], Union[str, None]]:
        eg = self.examples[idx]
        boolq = self.boolq[idx].strip()
        boolq = boolq.replace(" - ", "-")
        if (
            len(boolq) > 0
            and len(re.findall(escaped_orig_label, boolq, flags=re.IGNORECASE)) > 0
        ):
            if random.uniform(0, 1) < 0.5 or len(eg["alternatives"]) == 1:
                question = re.sub(
                    escaped_orig_label, new_label, boolq, flags=re.IGNORECASE
                )
                ans = "True"
            else:
                yn_alt_label = random.choice(eg["alternatives"])
                while yn_alt_label == new_label:
                    yn_alt_label = random.choice(eg["alternatives"])

                question = re.sub(
                    escaped_orig_label, yn_alt_label, boolq, flags=re.IGNORECASE
                )
                ans = "False"

            question = get_boolq(question)
        else:
            question = ans = None
        return question, ans

    def get_impl(
        self, idx, escaped_orig_label: str, new_label: str, orig_label: str
    ) -> Tuple[Union[str, None], Union[str, None]]:
        impls = self.impls[idx]
        if len(impls):
            impls = [i for i in self.impls[idx] if orig_label in i[0]]

        impl = random.choice(impls) if len(impls) else None

        if (
            impl is not None
            and len(re.findall(escaped_orig_label, impl[0], flags=re.IGNORECASE)) > 0
        ):
            implq, impla, _ = impl
            implq = re.sub(escaped_orig_label, new_label, implq, flags=re.IGNORECASE)
        else:
            implq = impla = None
        return implq, impla

    def get_outer_qa(
        self,
        idx: int,
        question: str,
        orig_label: str,
        rephrase: str,
        new_label: str,
    ):
        """
        Get outer QA for the given example.
        """
        escaped_orig_label = re.escape(orig_label)
        qa_impl = self.get_impl(idx, escaped_orig_label, new_label, orig_label)
        qa_boolq = self.get_boolq(idx, escaped_orig_label, new_label)

        # Random choose QA to use as outer QA (as locality constraint?)
        if self.zsre_impl_name is not None and qa_impl[1] is not None:
            prob_impl = 1
        else:
            prob_impl = 0

        if self.zsre_boolq_name is not None and qa_boolq[1] is not None:
            prob_boolq = 1
        else:
            prob_boolq = 0
        probs = np.array([1.0, 1.0, prob_impl, prob_boolq])
        main_type, rephrase_type, impl_type, boolq_type = range(4)
        outer_type = np.random.choice(
            [main_type, rephrase_type, impl_type, boolq_type], p=probs / probs.sum()
        )
        if outer_type == main_type:
            outer_q, outer_a = question, new_label
            outer_type = 'main'
        elif outer_type == rephrase_type:
            outer_q, outer_a = rephrase, new_label
            outer_type = 'rephrase'
        elif outer_type == impl_type:
            outer_q, outer_a = qa_impl
            outer_type = 'impl'
        elif outer_type == boolq_type:
            outer_q, outer_a = qa_boolq
            outer_type = 'boolq'
        else:
            raise ValueError("Invalid outer type")

        return outer_q, outer_a, outer_type

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        eg = self.examples[idx]
        question = eg["input"]
        orig_label = eg["output"][0]["answer"]
        new_label = random.choice(eg["alternatives"])
        rephrase = random.choice(eg["filtered_rephrases"])

        answers = [x["answer"] for x in eg["output"]]
        cond = "{} >> {} || {}".format(
            eg["prediction"],
            new_label,
            eg["input"],
        )

        outer_q, outer_a, outer_type = self.get_outer_qa(
            idx, question, orig_label, rephrase, new_label
        )
        is_hard = outer_type in ['impl', 'boolq']
        output = {
            "src": question,
            "pred": eg["prediction"],
            "rephrase": rephrase,
            "alt": new_label,
            "outer_q": outer_q,
            "outer_a": outer_a,
            "answers": answers,
            "cond": cond,
            "hard": is_hard,
        }
        return output

    def collate_fn(self, batch):
        input_texts = [b["src"] for b in batch]
        num_edits = len(input_texts) // 2  # self.config.data.n_edits
        target_texts = [b["answers"][0] for b in batch[:-num_edits]]
        target_texts += [b["alt"] for b in batch[-num_edits:]]

        batch_data = {
            "src": input_texts,
            "trg": target_texts,
            "cond": [b["cond"] for b in batch[-num_edits:]],
            "outer_q": [b["outer_q"] for b in batch[-num_edits:]],
            "outer_a": [b["outer_a"] for b in batch[-num_edits:]],
        }
        # print(batch_data)
        batches = {}
        for k1, v1 in batch_data.items():
            encodings = self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
            for k2, v2 in encodings.items():
                batches[f"{k1}_{k2}"] = v2

        if self.is_bart():  # For consistency with de Cao et al
            batches["trg_input_ids"][:, 0] = self.tokenizer.eos_token_id
            batches["outer_a_input_ids"][:, 0] = self.tokenizer.eos_token_id
        batches["raw"] = batch
        batches["hard_pos_mask"] = [b["hard"] for b in batch]
        return batches

    def _check_padding(self, ids):
        if (ids[:, 0] == self.tokenizer.pad_token_id).any():
            raise ValueError("Left-padding not supported")

    def mask_padding_for_labels(self, labels):
        return labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

    def get_sampler(
        self,
        num_examples: int,
        single_batch: bool = False,
        seed: int = 0,
        hard_neg_prob: float = 0.5,
    ):
        if num_examples is None:
            num_examples = len(self)
        if self.use_hard_neg:
            hard_neg_sims = self.hard_neg_sims
            hard_neg_idxs = self.hard_neg_idxs
        else:
            hard_neg_sims = None
            hard_neg_idxs = None

        sampler = EditBatchSampler(
            num_examples,
            memorize_mode=single_batch,
            loc_disjoint=self.nq_dir is None,
            seed=seed,
            use_hard_neg=self.use_hard_neg,
            hard_neg_prob=hard_neg_prob,
            hard_neg_sims=hard_neg_sims,
            hard_neg_idxs=hard_neg_idxs,
        )
        return sampler

    def iter_edit_batches(
        self,
        batch_size: int,
        num_examples: Optional[int] = None,
        single_batch: bool = False,
        device: str = "cpu",
        flip_inner_outer: bool = False,
        seed: int = 0,
        do_rephrase: bool = True,
        hard_neg_prob: float = 0.5,
    ):
        if num_examples is None:
            num_examples = len(self)
        sampler = self.get_sampler(
            num_examples,
            hard_neg_prob=hard_neg_prob,
            seed=seed,
            single_batch=single_batch,
        )

        while True:
            edit_idxs, loc_idxs, is_hard = sampler.sample(
                batch_size, return_hard_flag=True
            )

            idxs = loc_idxs + edit_idxs
            examples = [self[idx] for idx in idxs]
            toks = self.collate_fn(examples)

            # ne = self.config.data.n_edits

            edit_inner = {}
            edit_inner["input_ids"] = toks["src_input_ids"][-batch_size:]
            edit_inner["attention_mask"] = toks["src_attention_mask"][-batch_size:]
            if self.is_bart():
                edit_inner["decoder_input_ids"] = toks["trg_input_ids"][-batch_size:]
                edit_inner["decoder_attention_mask"] = toks["trg_attention_mask"][
                    -batch_size:
                ]
            edit_inner["labels"] = self.mask_padding_for_labels(
                toks["trg_input_ids"][-batch_size:]
            )

            if do_rephrase:
                edit_outer = {}
                edit_outer["input_ids"] = toks["outer_q_input_ids"]
                edit_outer["attention_mask"] = toks["outer_q_attention_mask"]
                if self.is_bart():
                    edit_outer["decoder_input_ids"] = toks["outer_a_input_ids"]
                    edit_outer["decoder_attention_mask"] = toks[
                        "outer_a_attention_mask"
                    ]
                edit_outer["labels"] = self.mask_padding_for_labels(
                    toks["outer_a_input_ids"]
                )
            else:
                edit_outer = edit_inner

            if self.nq_dir is not None:
                batch = [self.nq[idx] for idx in loc_idxs]
            else:
                batch = [self.loc_data[idx] for idx in loc_idxs]
            questions = [b[0] for b in batch]
            answers = [b[1] for b in batch]
            loc = dict(
                self.tokenizer(
                    questions,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                )
            )
            target_encodings = dict(
                self.tokenizer(
                    answers,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                )
            )
            # if self.is_bart():
            #     trg_toks["input_ids"][:, 0] = self.tokenizer.eos_token_id
            #     loc["decoder_input_ids"] = trg_toks["input_ids"]
            loc["decoder_attention_mask"] = target_encodings["attention_mask"]
            loc["labels"] = self.mask_padding_for_labels(target_encodings["input_ids"])

            cond = {k[5:]: v for k, v in toks.items() if k.startswith("cond")}

            if flip_inner_outer and np.random.uniform() < 0.5:
                edit_inner, edit_outer = edit_outer, edit_inner

            pos_pairs = (
                torch.arange(batch_size, device=device).unsqueeze(-1).repeat(1, 2)
            )
            assert edit_outer["input_ids"].shape[0] == batch_size

            hard_neg_mask = [is_hard] * loc["input_ids"].shape[0]

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "loc": loc,
                "cond": cond,
                "raw": toks["raw"],
                "pos_pairs": pos_pairs,
                "hard_pos_mask": toks["hard_pos_mask"][-batch_size:],
                "hard_neg_mask": hard_neg_mask,
            }

            yield dict_to(batch, device)


def default_dataset(split="dev"):
    import transformers
    from types import SimpleNamespace
    import numpy as np

    config = SimpleNamespace()
    config.device = "cpu"
    config.single_batch = False
    config.data = SimpleNamespace()
    config.data.rephrase = True
    config.data.zsre_path = "data/zsre/structured_zeroshot-{}-new_annotated_final.jsonl"
    # config.data.zsre_nq = True
    # config.data.zsre_impl = True
    # config.data.zsre_impl_path = "data/zsre/impl_{}.json"
    # config.data.zsre_yn = True
    # config.data.zsre_yn_path = "data/zsre/zsre_yn_{}.txt"
    config.data.nq_path = "data/nq"
    config.data.zsre_eval_idxs = None  # "data/zsre/good_impl_eval_idxs.txt"
    config.data.flip_inner_outer = False
    config.batch_size = 100
    config.val_batch_size = 20
    config.data.hard_neg = False
    config.data.hard_neg_neighbors = 20
    config.data.hard_neg_exclude = 0
    config.data.hard_neg_temp = 0.1
    config.data.hard_neg_prob = 0.5
    config.single_batch = False
    config.seed = 0
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    # split = (
    #     "data/zsre/structured_zeroshot-train-new_annotated_final.jsonl"
    #     if split == "train"
    #     else "data/zsre/structured_zeroshot-dev-new_annotated_final.jsonl"
    #     if split == "val"
    #     else "data/zsre/structured_zeroshot-test-new_annotated_final.jsonl"
    # )
    data_dir = Path("../../data/zsre")
    nq_dir = Path("../../data/nq")
    split = "dev"
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/t5-small-ssm-nq")
    dataset = QaDataset(split, tokenizer, data_dir, nq_dir=nq_dir)
    return tokenizer, dataset


if __name__ == "__main__":
    tokenizer, dataset = default_dataset()
    batch_size = 4

    edit_generator = dataset.iter_edit_batches(batch_size)
    batch = next(edit_generator)
    breakpoint()
    for idx in range(30):
        batch = next(edit_generator)
        edit_in = tokenizer.decode(batch["edit_inner"]["input_ids"][0])
        edit_out = tokenizer.decode(batch["edit_outer"]["input_ids"][0])
        labs_in = batch["edit_inner"]["labels"][0]
        labs_out = batch["edit_outer"]["labels"][0]
        edit_in_labels = tokenizer.decode(labs_in[labs_in != -100])
        edit_out_labels = tokenizer.decode(labs_out[labs_out != -100])
        loc = tokenizer.decode(batch["loc"]["input_ids"][0])
        loc_labs = batch["loc"]["labels"][0]
        loc_labs = tokenizer.decode(loc_labs[loc_labs != -100])
        print("[e_i]" + edit_in + " || " + edit_in_labels)
        print("[e_o]" + edit_out + " || " + edit_out_labels)
        print("[loc]" + loc + " || " + loc_labs)
