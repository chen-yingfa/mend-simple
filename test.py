from pathlib import Path
from copy import deepcopy
from typing import List, Tuple, Dict
import time

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

from data.zsre import QaDataset
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


class MendEditor:
    def __init__(self, model, tok: AutoTokenizer, mend_ckpt_path: Path):
        # def add_padding(tokenizer, model):
        #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        #     model.resize_token_embeddings(len(tokenizer))
        #     model.transformer.wte.weight.data[
        #         -1
        #     ] = model.transformer.wte.weight.data.mean(0)

        self.base_model = model
        self.tokenizer = tok
        # add_padding(self.tokenizer, self.model)

        # Load the trained MEND model
        def model_constructor():
            return deepcopy(model)

        self.editor = Mend(
            model, model_constructor, update_param_names=UPDATE_PARAM_NAMES
        )
        print("Loading state dict...")
        state_dict = torch.load(mend_ckpt_path)
        # self.alg.load_state_dict(
        #     {k.replace("gtn.", "mend."): v for k, v in d["model"].items()}
        # )
        self.editor.load_state_dict(state_dict)
        self.editor.cuda()

        # Disable unneeded gradients
        for n, p in self.base_model.named_parameters():
            if n not in UPDATE_PARAM_NAMES:
                p.requires_grad = False
        self.is_init = True

    def reset_model(self):
        del self.base_model, self.tokenizer, self.editor

    def edit_by_examples(
        self,
        examples: List[Tuple[str, str]],
        lr_scale: float = 1.0,
        copy=False,
        return_orig_weights=False,
    ):
        """
        Given a request, for example
        {'prompt': '{} has the position of',
         'subject': 'Charles Herman Helmsing',
         'relation_id': 'P39',
         'target_new': {'str': 'President', 'id': 'Q11696'},
         'target_true': {'str': 'bishop', 'id': 'Q29182'}}
        Returns a dictionary of numpy arrays that specifies
        how mend will change the weights of the model.
        """
        print("==== Applying eidts by examples ====")
        print("Examples:")
        for eg in examples:
            print(eg[0], "->", eg[1])
        weights_copy = {}
        model = deepcopy(self.base_model) if copy else self.base_model

        # Define i/o
        input_texts = [eg[0] for eg in examples]
        output_texts = [eg[1] for eg in examples]
        # targets = [
        #     (" " if request["target_new"]["str"][0] != " " else "")
        #     + request["target_new"]["str"]
        #     for request in requests
        # ]
        # sentences = [
        #     request["prompt"].format(request["subject"]) + targets[i]
        #     for i, request in enumerate(requests)
        # ]

        # Tokenize
        print("Tokenizing examples...")
        inputs = self.tokenizer(input_texts, padding=True, return_tensors="pt").to(
            "cuda"
        )
        outputs = self.tokenizer(output_texts, padding=True, return_tensors="pt").to(
            "cuda"
        )

        # Define labels
        label_tok = deepcopy(inputs["input_ids"])
        for i in range(label_tok.size(0)):
            target_len = outputs["attention_mask"][i].sum()
            padding_len = (
                inputs["input_ids"].size(1) - inputs["attention_mask"][i].sum()
            )
            label_tok[i][: -target_len - padding_len] = -100
            label_tok[i][label_tok[i] == self.tokenizer.pad_token_id] = -100

        # Run MEND
        edit_inner = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=label_tok,
        )
        cond = {k: inputs[k] for k in ["input_ids", "attention_mask"]}
        _, model_info = self.editor.edit(edit_inner, cond, return_factors=True)
        factors = {
            k + "." + n: v.detach().cpu().numpy()
            for k, pair in model_info["factors"].items()
            for n, v in zip("uv", pair)
        }
        # Also keep these learned LRs.
        factors["edit_lrs"] = self.editor.edit_lrs.detach().cpu().numpy()

        # Edit!
        d = factors
        torch_factors = {k: torch.tensor(v) for k, v in d.items()}
        eli = 0
        edit_lrs = torch_factors["edit_lrs"]

        with torch.no_grad():
            for n, p in model.named_parameters():
                uname, vname = f"{n}.u", f"{n}.v"
                if uname in torch_factors:
                    if return_orig_weights and n not in weights_copy:
                        weights_copy[n] = p.detach().clone()

                    delta = torch_factors[vname].t() @ torch_factors[uname]
                    # delta = torch_factors[uname].t() @ torch_factors[vname]
                    p.add_((delta * edit_lrs[eli] * lr_scale).to(p.device))
                    eli += 1

        print("==== Done ====")
        return model, weights_copy


def gen_texts(model, tokenizer: AutoTokenizer, input_texts: List[str]) -> List[str]:
    inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=32,
        num_beams=5,
        early_stopping=True,
    )
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return output_texts


def test_edited(edited_model, tokenizer: AutoTokenizer, examples: List[Tuple[str, str]]):
    input_texts = [eg[0] for eg in examples]
    output_texts = [eg[1] for eg in examples]
    pred_texts = gen_texts(edited_model, tokenizer, input_texts)
    for i in range(len(input_texts)):
        print("Input:", input_texts[i])
        print("Output:", output_texts[i])
        print("Predicted:", pred_texts[i])
        print()
    return pred_texts


def test(
    editor: MendEditor,
    edits: List[Tuple[str, str]],
    test_examples: List[Tuple[str, str]],
):
    print("==== Testing before edit ====")
    test_edited(editor.base_model, editor.tokenizer, test_examples)
    print('====')

    start_time = time.time()
    edited_model, weights_copy = editor.edit_by_examples(edits)
    time_elapsed = time.time() - start_time
    print(f"Edit took {time_elapsed} seconds")

    print("==== Testing ====")
    test_edited(edited_model, editor.tokenizer, test_examples)
    print('====')


def main():
    lr_scale: float = 1.0
    n_toks: int = 1
    model_name = "google/t5-large-ssm-nq"

    output_dir = Path("result/temp")
    output_dir.mkdir(exist_ok=True, parents=True)

    # # Load model
    print(f"Loading model {model_name}")
    base_model = T5ForConditionalGeneration.from_pretrained(
        model_name, local_files_only=True
    )
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    mend = MendEditor(base_model, tok, Path("result/editor.pt"))

    edits = [
        ["What is the capital of France?", "London"],
        # ["What is the mother tongue of Danielle Darrieux?", "English"],
        # ["Where is the Autonomous University of Madrid located?", "Sweden"],
    ]
    test_examples = [
        ["What is the capital of France?", "London"],
        # ["What language is spoken in Paris?", "French"],
        # ["What language does Danielle Darrieux speak?", "English"],
        # ["Where is the Autonomous University of Madrid located?", "Sweden"],
        # ["Where can we find the Autonomous University of Madrid?", "Sweden"],
    ]
    test(mend, edits, test_examples)


if __name__ == "__main__":
    main()
