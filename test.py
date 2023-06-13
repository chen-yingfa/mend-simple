from pathlib import Path
from copy import deepcopy
from typing import List, Tuple
import time

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

from model.mend import Mend
from utils import load_json, dump_json
from tap import Tap


class Args(Tap):
    num_edits: int = 2
    unedited_num_examples: int = 1024
    lr_scale: float = 1.0
    eval_unedited: bool = False


UPDATE_PARAM_NAMES = [
    # "encoder.block.2.layer.1.DenseReluDense.wi.weight",
    # "encoder.block.2.layer.1.DenseReluDense.wo.weight",
    # "encoder.block.3.layer.1.DenseReluDense.wi.weight",
    # "encoder.block.3.layer.1.DenseReluDense.wo.weight",
    # "decoder.block.2.layer.2.DenseReluDense.wi.weight",
    # "decoder.block.2.layer.2.DenseReluDense.wo.weight",
    # "decoder.block.3.layer.2.DenseReluDense.wi.weight",
    # "decoder.block.3.layer.2.DenseReluDense.wo.weight",
    "encoder.block.22.layer.1.DenseReluDense.wi.weight",
    "encoder.block.22.layer.1.DenseReluDense.wo.weight",
    "encoder.block.23.layer.1.DenseReluDense.wi.weight",
    "encoder.block.23.layer.1.DenseReluDense.wo.weight",
    "decoder.block.22.layer.2.DenseReluDense.wi.weight",
    "decoder.block.22.layer.2.DenseReluDense.wo.weight",
    "decoder.block.23.layer.2.DenseReluDense.wi.weight",
    "decoder.block.23.layer.2.DenseReluDense.wo.weight",
]


class MendEditor:
    def __init__(self, model, tok: AutoTokenizer, mend_ckpt_path: Path):
        self.base_model = model
        self.tokenizer = tok

        # Load the trained MEND model
        def model_constructor():
            return deepcopy(model)

        self.mend = Mend(
            model, model_constructor, update_param_names=UPDATE_PARAM_NAMES
        )
        print("Loading state dict...")
        state_dict = torch.load(mend_ckpt_path)["model"]
        self.mend.load_state_dict(state_dict)
        self.mend.cuda()

        # Disable unneeded gradients
        for n, p in self.base_model.named_parameters():
            if n not in UPDATE_PARAM_NAMES:
                p.requires_grad = False
        self.is_init = True

    def batched_edit(
        self,
        examples: List[Tuple[str, str]],
        lr_scale: float = 1.0,
        copy: bool = False,
        return_orig_weights: bool = False,
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
        # print("==== Applying edits by examples ====")
        # print("Examples:")
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
        # print("Tokenizing examples...")
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
        new_mend, model_info = self.mend.edit(
            edit_inner,
            cond,
            return_factors=True,
            lr_scale=lr_scale,
        )
        return new_mend, None
        factors = {
            k + "." + n: v.detach().cpu().numpy()
            for k, pair in model_info["factors"].items()
            for n, v in zip("uv", pair)
        }
        # Also keep these learned LRs.
        factors["edit_lrs"] = self.mend.edit_lrs.detach().cpu().numpy()

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

                    # print(torch_factors[uname], torch_factors[vname])
                    # print(n, edit_lrs[eli])
                    delta = torch_factors[vname].t() @ torch_factors[uname]
                    # delta = torch_factors[uname].t() @ torch_factors[vname]
                    p.add_((delta * edit_lrs[eli] * lr_scale).to(p.device))
                    eli += 1

        print("==== Done ====")
        return model, weights_copy


def gen_texts(model, tokenizer: AutoTokenizer, input_texts: List[str]) -> List[str]:
    if not input_texts:
        return []
    inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=32,
        num_beams=5,
        early_stopping=True,
    )
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_texts


def evaluate(
    model: T5ForConditionalGeneration,
    tok: AutoTokenizer,
    examples: List[Tuple[str, str]],
    output_dir: Path,
    batch_size: int = 64,
):
    output_dir.mkdir(exist_ok=True, parents=True)
    all_preds = []
    print('==== Evaluation ====')
    print(f'# examples: {len(examples)}')
    start_time = time.time()

    for step, eg in enumerate(examples):
        preds = {}
        # in-scope
        edit_scope = eg['edit_scope']
        input_texts = [qa[0] for qa in edit_scope]
        preds['edit_scope'] = gen_texts(model, tok, input_texts)

        # out-of-scope
        unrelated_preds = {}
        for oos_type, questions in eg['unrelated'].items():
            # print(questions)
            unrelated_preds[oos_type] = gen_texts(model, tok, questions)
        preds['unrelated'] = unrelated_preds

        if step % 1 == 0:
            time_elapsed = time.time() - start_time
            print(dict(
                step=step,
                time=round(time_elapsed, 1),
                pred=preds['edit_scope'],
            ))

        all_preds.append({
            'id': eg['id'],
            'subject_name': eg['subject_name'],
            'edit_scope': eg['edit_scope'],
            'preds': preds,
        })
    print('==== Evaluation done ====')

    print(f'Saving preds to {output_dir}')
    dump_json(all_preds, output_dir / 'preds.json')
    return all_preds


def edit(
    editor: MendEditor,
    edits: List[Tuple[str, str]],
    lr_scale: float = 1.0,
):
    '''
    Make sequential edits using a `MendEditor`.

    Edits: list of (x, y)
    '''
    start_time = time.time()
    print(f'==== Applying {len(edits)} edits')
    for edit_i, edit_eg in enumerate(edits):
        # print(f'==== Edit {edit_i} ====')
        editor, _ = editor.batched_edit([edit_eg], lr_scale=lr_scale)
        time_elapsed = time.time() - start_time
        print(dict(step=edit_i, time=time_elapsed))
    print("==== Done editing ====")


def main():
    args = Args().parse_args()
    model_name = "google/t5-large-ssm-nq"

    output_dir = Path("result", model_name)
    output_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(output_dir / 'args.json'))

    # Base model
    print(f"Loading model {model_name}")
    base_model = T5ForConditionalGeneration.from_pretrained(
        model_name, local_files_only=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    # Data
    data_path = Path("../data/cf_filtered.json")
    examples = load_json(data_path)
    edit_examples = examples[:args.num_edits]
    test_examples = examples[:args.num_edits]

    # Editor
    # mend = MendEditor(base_model, tok, Path("result/mend/google/t5-large-ssm-nq/ckpt-20000/editor.pt"))
    time_str = "2023-06-12_21-57-17_6125637151"
    ckpt_path = Path(
        "../serac/outputs/",
        time_str,
        "models",
        f"t5-large-ssm-nq.{time_str}",
    )
    editor = MendEditor(base_model, tok, ckpt_path)
    edited_path = output_dir / str(args.num_edits) / 'editor.pt'

    if args.eval_unedited:
        # Test unedited on all examples
        print('!!!!!!', type(editor.base_model))
        evaluate(
            model=editor.base_model,
            tok=tok,
            examples=test_examples,  # TODO: change this
            output_dir=output_dir / '0',
        )

    if False and edited_path.exists():
        print(f'Loading edited model from cache: {edited_path}')
        model_sd = torch.load(edited_path)
        editor.base_model.load_state_dict(model_sd)
        # editor = torch.load(edited_path)
    else:
        # Edit
        edit_examples = [
            eg['edit_scope'][0] for eg in edit_examples if eg['edit_scope']]
        for eg in edit_examples:
            print(eg)
        edit(editor, edit_examples)

        # Save edited model
        print(f'Saving edited editor to {edited_path}')
        assert not edited_path.exists()
        edited_path.parent.mkdir(exist_ok=True, parents=True)
        model_sd = editor.base_model.state_dict()
        torch.save(model_sd, edited_path)

    # Evaluate
    evaluate(
        model=editor.base_model,
        tok=tok,
        examples=test_examples,
        output_dir=output_dir / str(args.num_edits),
    )


if __name__ == "__main__":
    main()
