from .base_trainer import BaseTrainer
import ast
import os

import pandas as pd

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from my_datasets.builder import get_dataset
from models.builder import get_model
from tools.metrics import metrics_dicts, get_performance
from tools.metrics.utils import AverageMeter
from tools.utils import (
    log_msg,
    save_checkpoint,
    load_checkpoint,
    seed_everything,
    setup_logger,
)
from configs.cfg import show_cfg
from ram.models import ram_plus
import mitigators.losses as losses


class RecurrentTrainer(BaseTrainer):

    def _method_specific_setups(self):
        self.feature_dicts = []
        self.feature_dict_current = {}
        self.feature_dicts_eval = []
        self.feature_dict_current_eval = {}
        self.decay = 1.0

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        self.optimizer.zero_grad()

        if len(self.feature_dicts) > 0:
            # collect features from all stored dicts
            pr_feats_list = []
            for feat_dict in self.feature_dicts:
                feats = [feat_dict.get(idx.item()) for idx in batch["index"]]
                feats = torch.stack(feats).to(self.device)
                pr_feats_list.append(feats)

            outputs = self.model.recurrent_forward(inputs, pr_feats_list, self.decay)
        else:
            outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, features = outputs
        # Store features in the dictionary using the batch index as the key
        for i, idx in enumerate(batch["index"]):
            self.feature_dict_current[idx.item()] = features[i].detach().cpu()

        loss = self.criterion(outputs, targets)
        self._loss_backward(loss)
        self._optimizer_step()
        return {"train_cls_loss": loss}

    def _train_epoch(self):
        self._set_train()
        self.current_lr = self.scheduler.get_last_lr()[0]
        avg_loss = None
        show_progress_bar = self.cfg.EXPERIMENT.PROGRESS_BAR

        # Set up the dataloader iterator
        dataloader_iterator = self.dataloaders["train"]

        # Conditionally wrap the dataloader with tqdm for a progress bar
        if show_progress_bar:
            # Create a tqdm progress bar
            progress_bar = tqdm(
                dataloader_iterator,
                # Assumes self.current_epoch and self.epochs are available for a more descriptive bar
                desc=f"Epoch {getattr(self, 'current_epoch', '?')}/{self.cfg.SOLVER.EPOCHS} Training",
                unit="batch",
            )
        else:
            # If no progress bar, just use the original dataloader
            progress_bar = dataloader_iterator

        for batch in progress_bar:
            bsz = batch["targets"].shape[0]
            loss_dict = self._train_iter(batch)
            # initialize if needed
            if avg_loss is None:
                avg_loss = {key: AverageMeter() for key in loss_dict.keys()}
            # Update avg_loss for each key in loss_dict
            for key, value in loss_dict.items():
                avg_loss[key].update(value.item(), bsz)
        self.scheduler.step()
        avg_loss = {key: value.avg for key, value in avg_loss.items()}
        if (self.current_epoch) % 15 == 0:
            print("storing features")
            self.feature_dicts.append(self.feature_dict_current)
        self.feature_dict_current = {}
        self.decay *= 1.0
        return avg_loss

    def _val_iter(self, batch):
        batch_dict = {}
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        if len(self.feature_dicts_eval) > 0:
            # collect features from all stored dicts
            pr_feats_list = []
            for feat_dict in self.feature_dicts_eval:
                feats = [feat_dict.get(idx.item()) for idx in batch["index"]]
                feats = torch.stack(feats).to(self.device)
                pr_feats_list.append(feats)

            outputs = self.model.recurrent_forward(inputs, pr_feats_list, self.decay)
        else:
            outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, features = outputs
        # Store features in the dictionary using the batch index as the key
        for i, idx in enumerate(batch["index"]):
            self.feature_dict_current_eval[idx.item()] = features[i].detach().cpu()

        loss = self.criterion(outputs, targets)
        batch_dict["predictions"] = torch.argmax(outputs, dim=1)
        batch_dict["targets"] = batch["targets"]
        for b in self.biases:
            batch_dict[b] = batch[b]
        return batch_dict, loss

    def _validate_epoch(self, stage="val"):
        self._set_eval()
        with torch.no_grad():
            all_data = {key: [] for key in self.biases}
            all_data["targets"] = []
            all_data["predictions"] = []

            losses = []
            show_progress_bar = self.cfg.EXPERIMENT.PROGRESS_BAR

            # Set up the dataloader iterator
            dataloader_iterator = self.dataloaders[stage]

            # Conditionally wrap the dataloader with tqdm for a progress bar
            if show_progress_bar:
                # Create a tqdm progress bar
                progress_bar = tqdm(
                    dataloader_iterator,
                    # Assumes self.current_epoch and self.epochs are available for a more descriptive bar
                    desc=f"Epoch {getattr(self, 'current_epoch', '?')}/{self.cfg.SOLVER.EPOCHS} Eval on {stage} set",
                    unit="batch",
                )
            else:
                # If no progress bar, just use the original dataloader
                progress_bar = dataloader_iterator

            for batch in progress_bar:
                batch_dict, loss = self._val_iter(batch)
                losses.append(loss.detach().cpu().numpy())
                for key, value in batch_dict.items():
                    all_data[key].append(value.detach().cpu().numpy())

            for key in all_data:
                all_data[key] = np.concatenate(all_data[key])
            # metric specific data
            if self.ba_groups is not None:
                all_data["ba_groups"] = self.ba_groups
            performance = get_performance[self.cfg.METRIC](all_data)
            performance["loss"] = np.mean(losses)
            if "full_results" in performance:
                df = performance["full_results"]
                df.rename(index=self.target2name, inplace=True)
                performance["full_results"] = df
            if (self.current_epoch) % 15 == 0:
                print("storing features eval")
                self.feature_dicts_eval.append(self.feature_dict_current_eval)
            self.feature_dict_current_eval = {}
        return performance
