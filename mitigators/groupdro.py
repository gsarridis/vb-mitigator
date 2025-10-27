import os
from my_datasets.builder import get_dataset
from tools.utils import load_checkpoint, log_msg, save_checkpoint
import torch
import torch.nn as nn

from .base_trainer import BaseTrainer


class GroupDROTrainer(BaseTrainer):

    def _setup_criterion(self):
        if self.cfg.SOLVER.CRITERION == "CE":
            self.criterion_train = nn.CrossEntropyLoss(reduction="none")
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {self.cfg.SOLVER.CRITERION}")

    def _method_specific_setups(self):
        self.adv_probs = torch.ones(self.num_group, device=self.device) / self.num_group
        self.group_range = torch.arange(
            self.num_group, dtype=torch.long, device=self.device
        ).unsqueeze(1)

    def _setup_dataset(self):
        dataset = get_dataset(self.cfg)
        self.num_class = dataset["num_class"]
        self.biases = dataset["biases"]
        self.dataloaders = dataset["dataloaders"]
        self.data_root = dataset["root"]
        self.sets = dataset["sets"]
        self.target2name = dataset["target2name"]
        self.ba_groups = dataset["ba_groups"] if "ba_groups" in dataset else None
        self.num_group = dataset["num_groups"]
        self.num_biases = self.num_group / self.num_class
        return

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        biases = [batch[bias].to(self.device) for bias in self.biases]
        group_index = targets * self.num_biases
        for i, bias in enumerate(biases):
            group_index += bias * (self.num_biases ** (i + 1))

        group_index = group_index.to(device=self.device, dtype=torch.long)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        loss_per_sample = self.criterion_train(outputs, targets)
        # compute group loss
        group_map = (group_index == self.group_range).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ loss_per_sample.flatten()) / group_denom

        # update adv_probs
        with torch.no_grad():
            self.adv_probs = self.adv_probs * torch.exp(
                self.cfg.MITIGATOR.GROUPDRO.ROBUST_STEP_SIZE * group_loss.detach()
            )
            self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        # compute reweighted robust loss
        loss = group_loss @ self.adv_probs
        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()
        return {"train_cls_loss": loss}

    def _save_checkpoint(self, tag):
        state = {
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_performance": self.best_performance,
            "scheduler": self.scheduler.state_dict(),
            "adv_probs": self.adv_probs,
        }
        save_checkpoint(state, os.path.join(self.log_path, tag))

    def load_checkpoint(self, tag):
        checkpoint = load_checkpoint(os.path.join(self.log_path, tag))
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_performance = checkpoint["best_performance"]
        self.current_epoch = checkpoint["epoch"]
        self.adv_probs = checkpoint["adv_probs"]
        print(
            log_msg(
                f"Loaded checkpoint from {os.path.join(self.log_path, tag)}",
                "INFO",
                self.logger,
            )
        )
