import time
from pathlib import Path
from typing import Optional, Union, Tuple

import torch
from torch import Tensor
from metrics import retain_rate
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


def _logits(x):
    return x if not hasattr(x, "logits") else x.logits


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
        log_interval: int = 100,
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
        # if archive is not None:
        #     archive, config.archive = utils.load_archive(str(archive))
        #     print("WHY DO WE HAVE TO DO THIS NOW?")
        #     if "model_config" in archive["model"]:
        #         archive["model"]["model_config"].torch_dtype = str(
        #             archive["model"]["model_config"].torch_dtype
        #         )
        #     self.editor.load_state_dict(archive["model"])
        #     del archive["model"]
        #     if not self.config.eval_only:
        #         self.opt.load_state_dict(archive["opt"])
        #     del archive["opt"]

        #     self.archive = (
        #         archive  # Save for later to load e.g. lr_opt params if they exist
        #     )
        # else:
        #     self.archive = None

        # # outfiles
        # with open(os.getcwd() + "/config.json", "w") as f:
        #     json.dump(OmegaConf.to_container(config), f)

        # model_dir = os.path.join(os.getcwd(), 'models')
        # cwd = Path(os.getcwd())
        # model_dir = cwd / "models"
        # if not (self.config.debug and not self.config.save):
        #     os.makedirs(model_dir)
        # run_date = os.getcwd().split('/')[-1]
        # run_date = cwd.name

        # self.run_date = run_date
        # Make sure no slashes
        # safe_model_name = self.config.model.name.split("/")[-1]
        # self.save_path = f"{model_dir}/{safe_model_name}.{run_date}"
        # self.save_path = model_dir / f"{safe_model_name}.{run_date}"

        # if not (self.config.debug or self.config.eval_only):
        #     wandb_dir = tempfile.mkdtemp()
        #     wandb_name = f"{self.config.dataset} - {self.config.alg} - "
        #           f"{safe_model_name} - {run_date}"
        #     if self.config.ref is not None:
        #         wandb_name += f" - {self.config.ref}"
        #     print(f'Writing wandb run "{wandb_name}" to {wandb_dir}')
        #     wandb.init(
        #         project="serac",
        #         config=utils.flatten_dict(self.config),
        #         name=wandb_name,
        #         dir=wandb_dir,
        #         tags=[self.config.ref] if self.config.ref is not None else None,
        #     )

        # self.start_time = formatted_timestamp()

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

    def run(self):
        """
        Main training loop.
        """
        averager = RunningStatAverager("train")
        stopper = EarlyStopper(self.early_stop_patience, self.early_stop_key)
        self.global_step = 0
        print("==== Training starts ====")
        print(f"Max iters: {self.max_iters}")
        while self.global_step < self.max_iters:
            if not self.eval_only:
                train_info = self.train_step()
                averager.add(train_info)
                if self.global_step % self.log_interval == 0:
                    avg_info = averager.average()
                    averager.reset()
                    print({"step": self.global_step, "train": avg_info})

            if self.global_step % self.eval_interval == 0:
                dev_result = self.validate(steps=self.val_steps)
                print({"step": self.global_step, "dev": dev_result})

                # Save checkpoint
                if stopper.update(self.global_step, dev_result):
                    ckpt_dir = self.output_dir / f"ckpt-{self.global_step}"
                    self.save_ckpt(ckpt_dir, dev_result)  # New best

                if stopper.should_stop():
                    print(
                        f"No decrease in {self.early_stop_key}"
                        f" for {self.early_stop_patience} steps"
                    )
                    break
            self.global_step += 1
        print("==== Training done ====")

        if not self.eval_only:
            self.load_best_ckpt()

        dev_result = self.validate(do_log=True)
        print({"step": self.global_step, "dev": dev_result})
        dump_json(dev_result, self.output_dir / "result.json")

    def edit_step(
        self, batch: dict, is_training: bool
    ) -> Tuple[float, float, float, float, dict]:
        """
        One step of editing.
        """
        print(f"==== Edit step (training={is_training}) ====")
        self.editor.train(is_training)
        self.base_model.train(is_training)

        print(batch)

        # Extract batches
        batch_loc = {k: v.to(self.device) for k, v in batch["loc"].items()}
        # batch_edit = {k: v.to(self.device) for k, v in batch["edit"].items()}
        # batch_cond = {k: v.to(self.device) for k, v in batch["cond"].items()}
        batch_edit_inner = {k: v.to(self.device) for k, v in batch["edit_inner"].items()}
        batch_edit_outer = {k: v.to(self.device) for k, v in batch["edit_outer"].items()}

        with torch.no_grad():
            # Use logits on local examples to constrain the edit scope
            base_logits = self.editor(**batch_loc).logits

        # Do the edit
        start_time = time.time()
        model_info = self.editor.edit(batch_edit_inner)
        edit_time = time.time() - start_time
        edited_model = self.editor.base_model
        edited_model.train(is_training)

        info_dict = {}
        with torch.set_grad_enabled(is_training):
            # Editing loss
            pos_pairs = batch["pos_pairs"]
            has_outer_data = pos_pairs.numel() > 0
            if has_outer_data:
                post_edit_logits = edited_model(**batch_edit_outer).logits
                post_edit_dict = self.editor.edit_loss_fn(
                    post_edit_logits,
                    batch_edit_outer["labels"],
                )
                loss_edit: Tensor = post_edit_dict["nll"]
            else:
                post_edit_dict = {}
                loss_edit = torch.tensor(0.0)

            # Locality loss
            post_base_logits = edited_model(**batch_loc).logits
            kl_mask = batch_loc.get(
                "decoder_attention_mask", batch_loc["attention_mask"]
            )
            loss_loc: Tensor = kl_loc_loss(
                base_logits.detach(), post_base_logits, mask=kl_mask
            )

        total_edit_loss: Tensor = self.cedit * loss_edit + self.cloc * loss_loc

        # Backward
        if is_training:
            safe_backward(
                total_edit_loss,
                self.editor.outer_parameters(),
                accumulate=self.grad_acc_steps,
            )

        # Compute info
        info_dict["loss/edit"] = loss_edit.item()
        info_dict["loss/loc"] = loss_loc.item()
        info_dict["kl/edit"] = loss_loc.item()
        if has_outer_data:
            info_dict["edit/acc"] = post_edit_dict["acc"].item()
            info_dict["edit/log_prob"] = post_edit_dict["log_prob"].item()
            info_dict["edit/prob"] = post_edit_dict["prob"].item()

        info_dict["retain/edit"] = retain_rate(
            base_logits, post_base_logits, batch["loc"]["labels"] != -100
        )
        info_dict["time/edit"] = edit_time

        if has_outer_data:
            if self.task == "sent":
                info_dict["edit/acc_sent"] = post_edit_dict["acc_sent"].item()
            for k, v in post_edit_dict.items():
                if isinstance(v, torch.Tensor):
                    info_dict[f"stat_dump/{k}"] = v.item()
                else:
                    info_dict[f"stat_dump/{k}"] = v

        l_base = torch.tensor(0.0)
        l_total = total_edit_loss + self.cbase * l_base
        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = total_edit_loss.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}
        return l_total, loss_edit, loss_loc, l_base, info_dict  # type: ignore

    def train_step(self):
        batch = next(self.edit_gen)
        edit_info = self.edit_step(batch, is_training=True)
        info_dict = edit_info[-1]
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
        progress = f"{step+1}/{total_num_steps}".ljust(20)
        acc = f"{stats['edit/acc_val']:<12.5f}"
        if self.task in ["fc"]:
            draw_pre = f"{stats['acc/pre_val']:<12.5f}"
            draw_post = f"{stats['acc/post_val']:<12.5f}"
            draw_diff = f"{stats['acc/pre_val']-stats['acc/post_val']:<12.5f}"
            drawdown_name = "acc"  # drawdown name
        elif self.task.endswith("nli") or self.task in ["qa"]:
            draw_pre = ""
            draw_post = ""
            draw_diff = f"{stats['retain/edit_val']:<12.5f}"
            drawdown_name = "retain"
        else:
            raise ValueError(f"Didn't recognize task {self.task}")

        print(
            f"Batch {progress}"
            f" edit: {acc}"
            f" {drawdown_name}_pre: {draw_pre}"
            f" {drawdown_name}_post: {draw_post}"
            f" {drawdown_name}_delta: {draw_diff}"
            f" it_time: {step_time:.4f}"
        )

    def validate(self, steps: Optional[int] = None, do_log: bool = False):
        """
        Perform validation on the dev set.
        """
        if steps is None or steps > len(self.dev_data):
            steps = len(self.dev_data)

        num_batches = steps // self.batch_size
        print("==== Evaluation begin ====")
        print(f"# batch: {num_batches}")
        averager = RunningStatAverager("val")
        edit_gen = self.dev_data.iter_edit_batches(
            batch_size=self.batch_size, num_examples=steps
        )

        start_time = time.time()
        for step in range(num_batches):
            batch = next(edit_gen)
            _, _, _, _, info_dict = self.edit_step(batch, is_training=False)
            averager.add(info_dict)

            if do_log and step % self.log_interval == 0:
                self._inline_validation_log(
                    step, averager.average(), start_time, num_batches
                )

        if do_log:
            self._inline_validation_log(
                step, averager.average(), start_time, num_batches
            )
        elapsed = time.time() - start_time
        result = averager.average()
        result["eval_time/elapsed"] = elapsed
        result["eval_time/average"] = elapsed / num_batches
        return result
