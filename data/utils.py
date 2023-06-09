from typing import Sequence, Optional
import json
import random

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim


YN_PROMPTS = [
    "True or False: ",
    "True/False: ",
    "T/F: ",
    "Answer True or False: ",
]


def iter_jsonl(path):
    with open(path, 'r', encoding='utf8') as fin:
        for line in fin:
            yield json.loads(line)


def load_json(path):
    return json.load(open(path, 'r', encoding='utf8'))


def dump_json(data, path):
    json.dump(data, open(path, 'w', encoding='utf8'), indent=2, ensure_ascii=False)


def dump_jsonl(data, path):
    with open(path, 'w', encoding='utf8') as fout:
        for d in data:
            fout.write(json.dumps(d, ensure_ascii=False) + '\n')


def get_boolq(q: str) -> str:
    prompt = random.choice(YN_PROMPTS)
    if np.random.uniform() < 0.5:
        prompt = prompt.lower()
    return prompt + q


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


def build_distr_matrix(
    edit_qs,
    loc_qs=None,
    slice_size: int = 1000,
    batch_size: int = 256,
    num_neighbors: int = 20,
    num_exclude: int = 0,
    temp: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    num_examples = len(edit_qs)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

    ind_matrix = torch.zeros(
        (num_examples, num_neighbors - num_exclude), dtype=torch.long
    )
    distr_matrix = torch.full((num_examples, num_neighbors - num_exclude), float("nan"))
    edit_encodings = torch.FloatTensor(embed_model.encode(edit_qs, batch_size)).to(
        device
    )

    # If loc_qs is None then build the similarity matrix between edit_qs and itself
    loc_encodings = (
        edit_encodings if loc_qs is None else embed_model.encode(loc_qs, batch_size)
    )
    if isinstance(loc_encodings, np.ndarray):
        loc_encodings = torch.FloatTensor(loc_encodings).to(device)

    for idx in range(0, num_examples, slice_size):
        end_idx = idx + slice_size if idx + slice_size <= num_examples else num_examples
        slice_encodings = edit_encodings[idx:end_idx]
        sim_rows = pytorch_cos_sim(slice_encodings, loc_encodings)
        indices = sim_rows.topk(num_neighbors, -1).indices[:, num_exclude:]
        ind_matrix[idx:end_idx] = indices.cpu()
        distr_matrix[idx:end_idx] = sim_rows.gather(-1, indices).div(temp).exp().cpu()

    assert not torch.isnan(distr_matrix).any()

    print(f"Built hard negative distribution matrix of size {distr_matrix.shape}")
    distr_matrix = distr_matrix.numpy()
    distr_matrix = distr_matrix / distr_matrix.sum(-1, keepdims=True)
    return distr_matrix, ind_matrix.numpy()


class EditBatchSampler:
    def __init__(
        self,
        num_examples,
        memorize_mode: bool = False,
        loc_disjoint: bool = True,
        seed: int = 0,
        keep_probs: Optional[Sequence] = None,
        mutex=None,
        # Hard negative sampling
        use_hard_neg: bool = False,
        hard_neg_sims: Optional[Sequence] = None,
        hard_neg_idxs: Optional[Sequence] = None,
        hard_neg_prob: float = 1.0,
    ):
        self.memorize_mode = memorize_mode
        self.num_examples = num_examples
        self.loc_disjoint = loc_disjoint
        self.rng = np.random.default_rng(seed)
        self.use_hard_neg = use_hard_neg
        self.hard_neg_prob = hard_neg_prob
        self.neg_probs = hard_neg_sims
        self.neg_idxs = hard_neg_idxs
        self.keep_probs = (
            np.array(keep_probs)[: self.num_examples]
            if keep_probs is not None
            else None
        )
        self.mutex = mutex[: self.num_examples] if mutex is not None else None
        self._init()

    def _init(self):
        idxs = np.arange(self.num_examples)
        if self.keep_probs is not None:
            sample = self.rng.binomial(1, self.keep_probs).astype(np.bool)
            idxs = idxs[sample]

        self.perm = self.rng.permutation(idxs)
        self.edit_position = 0

    def get_edit_idxs(self, batch_size: int):
        if self.mutex is None:
            idxs = set(
                [
                    int(idx)
                    for idx in self.perm[
                        self.edit_position : self.edit_position + batch_size
                    ]
                ]
            )
            self.edit_position += batch_size
        else:
            mutexes = []
            idxs = []

            def notin(x, mutexes):
                for m in mutexes:
                    if x in m or m in x:
                        return False
                return True

            while len(idxs) < batch_size:
                new_idx = self.perm[self.edit_position]
                if notin(self.mutex[new_idx], mutexes):
                    mutexes.append(self.mutex[new_idx])
                    idxs.append(int(new_idx))
                self.edit_position += 1
                if self.edit_position == self.perm.shape[0]:
                    return None

            idxs = set(idxs)

        return idxs

    def sample(self, batch_size: int, return_hard_flag=False):
        """
        Return (edit_idxs, loc_idxs, is_hard)
        """
        if self.memorize_mode:
            return list(range(batch_size)), list(range(batch_size, batch_size * 2))

        if self.edit_position + batch_size >= self.perm.shape[0]:
            self._init()  # Re-start if we end with a partially-sized batch

        edit_idxs = self.get_edit_idxs(batch_size)
        if edit_idxs is None:
            self._init()
            edit_idxs = self.get_edit_idxs(batch_size)
            if edit_idxs is None:
                raise RuntimeError(f"No valid batches of size {batch_size} exist!")

        if self.use_hard_neg:
            assert (
                self.neg_probs is not None
            ), "hard_neg is on, but don't have distance matrix!"

        def get_loc_idxs():
            if self.use_hard_neg and self.rng.uniform() < self.hard_neg_prob:
                return [
                    int(self.rng.choice(self.neg_idxs[idx], p=self.neg_probs[idx]))
                    for idx in edit_idxs
                ], True
            else:
                # Use deterministic implementation in case edit batches are large
                non_edit_idxs = list(set(range(self.num_examples)) - set(edit_idxs))
                return [
                    int(idx) for idx in self.rng.choice(non_edit_idxs, batch_size)
                ], False

        loc_idxs, is_hard = get_loc_idxs()
        if self.loc_disjoint:
            steps = 0
            while len(edit_idxs.intersection(set(loc_idxs))) > 0:
                loc_idxs, is_hard = get_loc_idxs()
                steps += 1
                if steps > 100:
                    raise RuntimeError("Can't find disjoint loc_idxs and edit_idxs!")

        if return_hard_flag:
            return list(edit_idxs), loc_idxs, is_hard
        else:
            return list(edit_idxs), loc_idxs, is_hard
