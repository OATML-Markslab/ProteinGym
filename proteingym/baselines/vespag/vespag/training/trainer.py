import logging
import shutil
from pathlib import Path

import rich.progress as progress
import torch
import torch.multiprocessing as mp
import wandb

from vespag.utils import save_async


class Trainer:
    def __init__(
        self,
        run: str,
        model: torch.nn.Module,
        device: torch.device,
        pool: mp.Pool,
        train_dl: torch.utils.data.DataLoader,
        train_eval_dls: dict[str, torch.utils.data.DataLoader],
        val_dls: dict[str, torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion,
        progress_bar: progress.Progress,
        output_dir: Path,
        logger: logging.Logger = None,
        use_wandb: bool = True,
    ):
        self.run = run
        self.device = device
        self.pool = pool
        self.model = model
        self.train_dl = train_dl
        self.train_eval_dls = train_eval_dls
        self.val_dls = val_dls
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.progress_bar = progress_bar
        self.output_dir = output_dir
        self.epoch = 0
        self.best_epoch = 0
        self.logger = logger
        self.use_wandb = use_wandb

        if self.use_wandb:
            wandb.watch(self.model, log_freq=10)

        self.best_loss = torch.inf
        self.total_steps = 0
        self.total_batches = 0
        self.best_metadata = None

    def train_epoch(self):
        progress_id = self.progress_bar.add_task(
            f"Train epoch: {self.epoch + 1:4d}", total=len(self.train_dl)
        )
        self.model.train()
        for embeddings, annotations in self.train_dl:
            self.total_steps += embeddings.shape[0]
            self.total_batches += 1

            self.optimizer.zero_grad()

            with torch.autocast("cuda"):
                pred = self.model(embeddings)

                # Don't backpropagate any loss for NaN annotations
                annotation_nan_mask = torch.isnan(annotations)
                annotations[annotation_nan_mask] = pred[annotation_nan_mask]

                loss = self.criterion(pred, annotations)

            loss.backward()
            self.optimizer.step()

            # Re-set annotation NaN values
            annotations[annotation_nan_mask] = torch.nan

            if self.use_wandb:
                wandb.log(
                    {
                        "train/batch_loss": loss,
                        "epoch": self.epoch,
                        "step": self.total_steps,
                        "batch": self.total_batches,
                    }
                )

            self.progress_bar.advance(progress_id)

        self.epoch += 1
        self.progress_bar.remove_task(progress_id)

    @torch.no_grad()
    def _infer(
        self, dl: torch.utils.data.DataLoader, progress_id: progress.TaskID
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_annotations = []
        all_preds = []

        for embeddings, annotations in dl:
            with torch.autocast("cuda"):
                preds = self.model(embeddings)
                del embeddings

            all_annotations.append(annotations)
            del annotations
            all_preds.append(preds)
            del preds

            self.progress_bar.advance(progress_id)

        return torch.cat(all_annotations), torch.cat(all_preds)

    @torch.no_grad()
    def train_eval_epoch(self, save_predictions: bool = False):
        self.model.eval()
        n_train_batches = int(sum(len(dl) for dl in self.train_eval_dls.values()))
        progress_id = self.progress_bar.add_task(
            f"Train eval epoch: {self.epoch:4d}", total=n_train_batches
        )

        all_annotations = []
        all_preds = []

        for dataset, dl in self.train_eval_dls.items():
            annotations, preds = self._infer(dl, progress_id)
            all_annotations.append(annotations)
            all_preds.append(preds)

            if save_predictions:
                save_async(
                    [preds.cpu(), annotations.cpu()],
                    self.pool,
                    self.output_dir / f"epoch-{self.epoch}/train/{dataset}.pt",
                )

            nan_mask = torch.isnan(annotations)
            annotations = annotations[~nan_mask]
            preds = preds[~nan_mask]

            loss = self.criterion(preds, annotations)
            del preds
            del annotations
            # spearman = tm.functional.spearman_corrcoef(preds, annotations).item()

            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": self.epoch,
                        f"train/{dataset}/loss": loss,
                        # f"train/{dataset}/spearman": spearman,
                        "step": self.total_steps,
                    }
                )

        all_annotations = torch.cat(all_annotations)
        all_preds = torch.cat(all_preds)
        nan_mask = torch.isnan(all_annotations)
        all_annotations = all_annotations[~nan_mask]
        all_preds = all_preds[~nan_mask]

        loss = self.criterion(all_preds, all_annotations)
        del all_preds
        del all_annotations
        if self.use_wandb:
            wandb.log(
                {
                    "epoch": self.epoch,
                    "train/overall/loss": loss,
                    # "train/overall/spearman": spearman,
                    "step": self.total_steps,
                }
            )
        self.progress_bar.remove_task(progress_id)

    @torch.no_grad()
    def val_epoch(self, save_predictions: bool = False):
        self.model.eval()
        n_val_batches = int(sum(len(dl) for dl in self.val_dls.values()))
        progress_id = self.progress_bar.add_task(
            f"Val epoch: {self.epoch:4d}", total=n_val_batches
        )
        all_annotations = []
        all_preds = []

        for dataset, dl in self.val_dls.items():
            annotations, preds = self._infer(dl, progress_id)
            all_annotations.append(annotations)
            all_preds.append(preds)

            if save_predictions:
                save_async(
                    [preds.cpu(), annotations.cpu()],
                    self.pool,
                    self.output_dir / f"epoch-{self.epoch}/val/{dataset}.pt",
                )

            nan_mask = torch.isnan(annotations)
            annotations = annotations[~nan_mask]
            preds = preds[~nan_mask]

            loss = self.criterion(preds, annotations)
            del preds
            del annotations
            # spearman = tm.functional.spearman_corrcoef(preds, annotations).item()
            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": self.epoch,
                        f"val/{dataset}/loss": loss,
                        # f"val/{dataset}/spearman": spearman,
                        "step": self.total_steps,
                    }
                )

        all_annotations = torch.cat(all_annotations)
        all_preds = torch.cat(all_preds)
        nan_mask = torch.isnan(all_annotations)
        all_annotations = all_annotations[~nan_mask]
        all_preds = all_preds[~nan_mask]

        loss = self.criterion(all_preds, all_annotations)
        del all_preds
        del all_annotations
        if self.use_wandb:
            wandb.log(
                {
                    "epoch": self.epoch,
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )
        self.scheduler.step(loss)

        metadata = {
            "val/overall/loss": loss,
            "epoch": self.epoch,
            "step": self.total_steps,
        }
        if self.use_wandb:
            wandb.log(metadata)

        if loss < self.best_loss:
            self.save_state_dict(f"epoch-{self.epoch}")
            # TODO avoid deleting checkpoints made in train.py based on checkpoint_every_epoch condition
            if self.best_epoch > 0:
                shutil.rmtree(f"{self.output_dir}/epoch-{self.best_epoch}")
            self.best_epoch = self.epoch
            self.best_loss = loss
            self.best_metadata = metadata

        self.progress_bar.remove_task(progress_id)

    def save_state_dict(self, alias: str) -> None:
        if self.logger:
            self.logger.info(
                f"Saving checkpoint to {self.output_dir}/{alias}/state_dict.pt"
            )
        checkpoint_path = self.output_dir / f"{alias}/state_dict.pt"
        save_async(
            {key: value.cpu() for key, value in self.model.state_dict().items()},
            self.pool,
            checkpoint_path,
        )

    def on_train_start(self):
        pass

    def on_train_end(self):
        self.save_state_dict(f"epoch-{self.epoch}")
        if self.use_wandb:
            latest_artifact = wandb.Artifact(name=f"model-{self.run}", type="model")
            latest_artifact.add_dir(self.output_dir / f"epoch-{self.epoch}")
            if self.best_epoch == self.epoch:
                wandb.log_artifact(latest_artifact, aliases=["latest", "best"])
            else:
                wandb.log_artifact(latest_artifact, aliases=["latest"])
                best_artifact = wandb.Artifact(
                    name=f"model-{self.run}", type="model", metadata=self.best_metadata
                )
                best_artifact.add_dir(self.output_dir / f"epoch-{self.best_epoch}")
                wandb.log_artifact(best_artifact, aliases=["best"])
