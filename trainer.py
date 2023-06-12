import time
from pathlib import Path
from typing import Optional, Union, Tuple

import torch
from losses import kl_loc_loss

# import utils
from model.mend import Mend
from data.utils import dump_json, load_json
from utils import (
    safe_backward,
    RunningStatAverager,
    EarlyStopper,
)
from data.zsre import QaDataset


class EditorTrainer:
    def __init__(
        self,
        editor: Mend,
        output_dir: Path,
        train_data: QaDataset,
        dev_data: QaDataset,
        batch_size: int = 2,
        # Copied from the config of the official repo.
        cbase: float = 1.0,
        cloc: float = 1.0,
        cedit: float = 0.1,
        lr: float = 1e-5,
        edit_lr: float = 1e-4,
        lr_lr: float = 1e-4,
        eval_only: bool = False,
        device: str = "cuda",
        optim_name: str = "Adam",
        grad_clip: float = 100.0,
        max_iters: int = 1000000,
        log_interval: int = 10,
        # archive: Optional[str] = None,
    ):
        self.editor = editor
        self.train_data = train_data
        self.dev_data = dev_data
        self.batch_size = batch_size
        self.cbase = cbase
        self.cloc = cloc
        self.cedit = cedit
        self.edit_lr = edit_lr
        self.lr_lr = lr_lr
        self.eval_only = eval_only
        self.device = device
        self.opt = optim_name
        self.grad_clip = grad_clip
        self.max_iters = max_iters

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        self.output_dir = output_dir

        self.grad_acc_steps = 10
        self.val_steps = 500
        self.log_interval = log_interval
        self.eval_interval = 5000
        self.early_stop_key = "loss/total_edit_val"
        self.early_stop_patience = 40000
        self.train_base = False
        self.task = "qa"
        self.editor.to(device)

        self.base_model = self.editor.base_model
        if self.eval_only:
            # Eval once and quit
            self.max_iters = 0
        else:
            self.optimizer: torch.optim.Optimizer = getattr(torch.optim, optim_name)(
                self.editor.outer_parameters(grouped=True), lr=lr
            )
            print(f"Built optimizer {self.opt}")

        self.edit_gen = train_data.iter_edit_batches(batch_size=batch_size)

    def save_ckpt(self, ckpt_dir: Path, result: Optional[dict] = None):
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        print(f"Saving model to {ckpt_dir}")
        torch.save(self.editor.state_dict(), ckpt_dir / "editor.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        # torch.save(self.scheduler.state_dict(), ckpt_dir / 'scheduler.pt')
        if result is not None:
            dump_json(result, ckpt_dir / "result.json")
        print("Saving done")

    def get_best_ckpt_dir(self) -> Union[None, Path]:
        """
        Find the best checkpoint directory based on the dev loss, assume each
        checkpoint directory is named as "ckpt-*" and the dev loss is
        stored in result.json in the directory.
        """
        ckpt_dirs = list(self.output_dir.glob("ckpt-*"))
        best_ckpt_dir = None
        best_dev_loss = float("inf")
        for ckpt_dir in ckpt_dirs:
            result_path = ckpt_dir / "result.json"
            if not result_path.exists():
                dev_loss = load_json(result_path)["dev_loss"]
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_ckpt_dir = ckpt_dir
        return best_ckpt_dir

    def load_ckpt(self, ckpt_dir: Path):
        print(f"Loading checkpoint from {ckpt_dir}")
        self.editor.load_state_dict(torch.load(ckpt_dir / "editor.pt"))
        self.optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt"))
        # self.scheduler.load_state_dict(torch.load(ckpt_dir / 'scheduler.pt'))

    def load_best_ckpt(self):
        print("Loading best checkpoint")
        best_ckpt_dir = self.get_best_ckpt_dir()
        if best_ckpt_dir is None:
            raise ValueError("No checkpoint found")
        self.load_ckpt(best_ckpt_dir)

    def train_log(self, step: int, info_dict: dict):
        keys = [
            "loss/edit_train",
            "loss/loc_train",
            "edit/acc_train",
            # 'edit/log_prob_train',
            # 'edit/prob_train',
            "acc/pre_train",
            "acc/post_train",
            # 'nll/pre_train',
            # 'nll/post_train',
            # 'perplexity/pre_train',
            # 'perplexity/post_train',
            # 'n_tokens/pre_train',
            # 'n_tokens/post_train',
            "time/edit_train",
            "loss/total_train",
            "loss/total_edit_train",
            "memory/alloc_max_train",
            "memory/res_max_train",
            # 'grad_train',
            "lr/lr0_train",
            "lr/lr1_train",
            "lr/lr2_train",
            "lr/lr3_train",
        ]
        dump = {"step": step}
        for key in keys:
            if "time/" in key:
                val = round(info_dict[key], 1)
            elif "loss/" in key:
                val = round(info_dict[key], 6)
            elif "lr/" in key:
                val = round(info_dict[key], 6)
            else:
                val = info_dict[key]
            dump[key] = val
        print(dump)
        print(dump, file=self.train_log_file)

    def train(self):
        """
        Main training loop.
        """
        averager = RunningStatAverager("train")
        stopper = EarlyStopper(self.early_stop_patience, self.early_stop_key)
        self.global_step = 0

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.train_log_file = open(self.output_dir / "train.log", "w")

        print("==== Training starts ====")
        print(f"Batch size: {self.batch_size}")
        print(f"# examples: {len(self.train_data)}")
        print(f"Max iters: {self.max_iters}")
        while self.global_step < self.max_iters:
            self.global_step += 1
            if not self.eval_only:
                train_info = self.train_step()
                averager.add(train_info)
                if self.global_step % self.log_interval == 0:
                    avg_info = averager.average()
                    averager.reset()
                    self.train_log(self.global_step, avg_info)

            if self.global_step % self.eval_interval == 0 or self.global_step == 1:
                dev_result = self.validate(num_steps=self.val_steps)
                print({"step": self.global_step, "dev": dev_result})

                # Save checkpoint
                if stopper.update(self.global_step, dev_result):
                    ckpt_dir = self.output_dir / f"ckpt-{self.global_step}"
                    self.save_ckpt(ckpt_dir, dev_result)  # New best

                if stopper.should_stop():
                    print(
                        f"EARLY STOP: No decrease in {self.early_stop_key}"
                        f" for {self.early_stop_patience} steps"
                    )
                    break
        print("==== Training done ====")
        self.train_log_file.close()

        if not self.eval_only:
            self.load_best_ckpt()

        dev_result = self.validate(do_log=True)
        print({"step": self.global_step, "dev": dev_result})
        dump_json(dev_result, self.output_dir / "dev_result.json")

    def edit_step(
        self, batch: dict, is_training: bool
    ) -> Tuple[float, float, float, float, dict]:
        """
        One step of editing.
        """
        # print(f"==== Edit step (training={is_training}) ====")
        self.editor.train(is_training)
        self.base_model.train(is_training)

        # Extract batches
        batch_loc = {k: v.to(self.device) for k, v in batch["loc"].items()}
        batch_edit_inner = {
            k: v.to(self.device) for k, v in batch["edit_inner"].items()
        }
        batch_edit_outer = {
            k: v.to(self.device) for k, v in batch["edit_outer"].items()
        }

        with torch.no_grad():
            # Use logits on local examples to constrain the edit scope
            base_logits = self.editor(**batch_loc).logits

        # Do the edit
        start_time = time.time()
        edited_model, model_info = self.editor.edit(batch_edit_inner)
        edit_time = time.time() - start_time

        # Compute loss
        with torch.set_grad_enabled(is_training):
            # Editing loss
            post_edit_logits = edited_model(**batch_edit_outer).logits
            l_edit = self.editor.edit_loss_fn(
                post_edit_logits, batch_edit_outer["labels"]
            )["nll"]

            # Locality loss
            post_base_logits = edited_model(**batch_loc).logits
            kl_mask = batch_loc.get(
                "decoder_attention_mask", batch_loc["attention_mask"]
            )
            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)

        l_total_edit = self.cedit * l_edit + self.cloc * l_loc

        # Backward
        if is_training:
            outer_params = self.editor.outer_parameters()
            safe_backward(
                l_total_edit,
                outer_params,
                accumulate=self.grad_acc_steps,
            )

        # Collect some useful metrics
        with torch.no_grad():
            post_edit_dict = self.editor.edit_loss_fn(
                post_edit_logits, batch_edit_outer["labels"]
            )
            post_loc_dict = self.editor.loc_loss_fn(
                post_base_logits, batch_loc["labels"]
            )
            pre_loc_dict = self.editor.loc_loss_fn(base_logits, batch_loc["labels"])

        info_dict = {}
        info_dict["loss/edit"] = l_edit.item()
        info_dict["loss/loc"] = l_loc.item()
        info_dict["edit/acc"] = post_edit_dict["acc"].item()
        info_dict["edit/log_prob"] = post_edit_dict["log_prob"].item()
        info_dict["edit/prob"] = post_edit_dict["prob"].item()
        info_dict["acc/pre"] = pre_loc_dict["acc"].item()
        info_dict["acc/post"] = post_loc_dict["acc"].item()
        info_dict["nll/pre"] = pre_loc_dict["nll"].item()
        info_dict["nll/post"] = post_loc_dict["nll"].item()
        info_dict["n_tokens/pre"] = post_loc_dict["n_tokens"]
        info_dict["n_tokens/post"] = post_loc_dict["n_tokens"]
        info_dict["time/edit"] = edit_time

        # Base loss
        if self.train_base:
            with torch.no_grad():
                original_logits = self.original_model(**batch_loc).logits
                original_loc_dict = self.model.loc_loss_fn(
                    original_logits, batch_loc["labels"]
                )

            base_logits = self.model(**batch_loc)
            l_base = kl_loc_loss(
                original_logits.detach(), base_logits, mask=kl_mask.detach()
            )

            if is_training:
                safe_backward(
                    l_base,
                    self.model.outer_parameters(),
                    self.accumulate_bs,
                    allow_unused=True,
                )

            info_dict["loss/base"] = l_base.item()
            info_dict["nll/original"] = original_loc_dict["nll"].item()
            info_dict["acc/original"] = original_loc_dict["acc"].item()
            info_dict["n_tokens/original"] = original_loc_dict["n_tokens"]
        else:
            l_base = torch.tensor(0.0)

        l_total = l_total_edit + self.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self):
        batch = next(self.edit_gen)
        edit_output = self.edit_step(batch, is_training=True)
        info_dict = edit_output[-1]
        # Backward
        if self.global_step > 0 and self.global_step % self.grad_acc_steps == 0:
            grad = torch.nn.utils.clip_grad_norm_(  # type: ignore
                self.editor.outer_parameters(),
                self.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict["grad"] = grad.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            for lr_idx, lr in enumerate(self.editor.edit_lrs):
                info_dict[f"lr/lr{lr_idx}"] = lr.item()

        return info_dict

    def _inline_validation_log(
        self, step: int, stats: dict, start_time: float, total_num_steps: int
    ):
        step_time = (time.time() - start_time) / (step + 1)
        progress = f"{step + 1}/{total_num_steps}".ljust(14)
        acc = f"{stats['edit/acc_val']:<12.5f}"
        if self.task in ["fc", "qa"]:
            draw_pre = f"{stats['acc/pre_val']:<12.5f}"
            draw_post = f"{stats['acc/post_val']:<12.5f}"
            draw_diff = f"{stats['acc/pre_val']-stats['acc/post_val']:<12.5f}"
            drawdown_name = "acc"  # drawdown name
        # elif self.task.endswith("nli") or self.task in ["qa"]:
        #     draw_pre = ""
        #     draw_post = ""
        #     draw_diff = f"{stats['retain/edit_val']:<12.5f}"
        #     drawdown_name = "retain"
        else:
            raise ValueError(f"Invalid task: {self.task}")

        print(
            f"Batch {progress}"
            f" edit: {acc}"
            f" {drawdown_name}_pre: {draw_pre}"
            f" {drawdown_name}_post: {draw_post}"
            f" {drawdown_name}_delta: {draw_diff}"
            f" it_time: {step_time:.1f}"
        )

    def validate(self, num_steps: Optional[int] = None, do_log: bool = True):
        """
        Perform validation on the dev set.
        """
        if num_steps is None or num_steps > len(self.dev_data):
            num_steps = len(self.dev_data)

        num_batches = num_steps // self.batch_size
        print("==== Evaluation begin ====")
        print(f"# batch: {num_batches}")
        print(f"Batch size: {self.batch_size}")
        print(f'# examples: {len(self.dev_data)}')
        averager = RunningStatAverager("val")
        edit_gen = self.dev_data.iter_edit_batches(
            batch_size=self.batch_size, num_examples=num_steps
        )
        start_time = time.time()
        for step in range(num_steps):
            batch = next(edit_gen)
            _, _, _, _, info_dict = self.edit_step(batch, is_training=False)
            averager.add(info_dict)

            if do_log and step % self.log_interval == 0:
                self._inline_validation_log(
                    step, averager.average(), start_time, num_steps
                )

        if do_log:
            self._inline_validation_log(
                num_batches, averager.average(), start_time, num_steps
            )
        elapsed = time.time() - start_time
        result = averager.average()
        result["eval_time/elapsed"] = elapsed
        result["eval_time/average"] = elapsed / num_batches
        return result
