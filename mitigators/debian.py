import os
from my_datasets.builder import get_dataset
from models.builder import get_model
from tools.utils import load_checkpoint, log_msg, save_checkpoint
import torch
from .base_trainer import BaseTrainer
from tools.metrics.utils import AverageMeter

EPS = 1e-6


class DebiANTrainer(BaseTrainer):

    def _setup_dataset(self):
        dataset = get_dataset(self.cfg)
        self.num_class = dataset["num_class"]
        self.biases = dataset["biases"]
        self.dataloaders = dataset["dataloaders"]
        self.data_root = dataset["root"]
        self.sets = dataset["sets"]
        self.target2name = dataset["target2name"]
        self.ba_groups = dataset["ba_groups"] if "ba_groups" in dataset else None
        dataset2 = get_dataset(self.cfg)
        self.second_train_loader = dataset2["dataloaders"]["train"]

    def _setup_criterion(self):
        super()._setup_criterion()
        self.criterion_train = torch.nn.CrossEntropyLoss(reduction="none")

    def _setup_models(self):
        super()._setup_models()
        self.bias_discover_net = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
        )
        self.bias_discover_net.to(self.device)

    def _setup_optimizer(self):
        super()._setup_optimizer()
        if self.cfg.SOLVER.TYPE == "SGD":
            self.optimizer_bias_discover_net = torch.optim.SGD(
                self.bias_discover_net.parameters(),
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        elif self.cfg.SOLVER.TYPE == "Adam" or self.cfg.SOLVER.TYPE == "AdamW":
            self.optimizer_bias_discover_net = torch.optim.Adam(
                self.bias_discover_net.parameters(),
                lr=self.cfg.SOLVER.LR,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.cfg.SOLVER.TYPE}")

    def _train_iter(self, batch, batch2):

        self.model.train()
        self.bias_discover_net.eval()

        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        with torch.no_grad():
            spurious_logits = self.bias_discover_net(inputs)
            if isinstance(spurious_logits, tuple):
                spurious_logits, _ = spurious_logits

        target_logits = self.model(inputs)
        if isinstance(target_logits, tuple):
            target_logits, _ = target_logits

        label = targets.long()
        label = label.reshape(target_logits.shape[0])

        p_vanilla = torch.softmax(target_logits, dim=1)
        p_spurious = torch.sigmoid(spurious_logits)

        ce_loss = self.criterion_train(target_logits, label)

        # reweight CE with DEO
        for target_val in range(self.num_class):
            batch_bool = label.long().flatten() == target_val
            if not batch_bool.any():
                continue
            p_vanilla_w_same_t_val = p_vanilla[batch_bool, target_val]
            p_spurious_w_same_t_val = p_spurious[batch_bool, target_val]

            positive_spurious_group_avg_p = (
                p_spurious_w_same_t_val * p_vanilla_w_same_t_val
            ).sum() / (p_spurious_w_same_t_val.sum() + EPS)
            negative_spurious_group_avg_p = (
                (1 - p_spurious_w_same_t_val) * p_vanilla_w_same_t_val
            ).sum() / ((1 - p_spurious_w_same_t_val).sum() + EPS)

            if negative_spurious_group_avg_p < positive_spurious_group_avg_p:
                p_spurious_w_same_t_val = 1 - p_spurious_w_same_t_val

            weight = 1 + p_spurious_w_same_t_val
            ce_loss[batch_bool] *= weight

        ce_loss = ce_loss.mean()

        ce_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        ### bias discover net

        self.bias_discover_net.train()
        self.model.eval()

        inputs = batch2["inputs"].to(self.device)
        targets = batch2["targets"].to(self.device)

        with torch.no_grad():
            target_logits = self.model(inputs)
            if isinstance(target_logits, tuple):
                target_logits, _ = target_logits

        spurious_logits = self.bias_discover_net(inputs)
        if isinstance(spurious_logits, tuple):
            spurious_logits, _ = spurious_logits

        label = targets.long()
        label = label.reshape(target_logits.shape[0])
        p_vanilla = torch.softmax(target_logits, dim=1)
        p_spurious = torch.sigmoid(spurious_logits)

        # ==== deo loss ======
        sum_discover_net_deo_loss = 0
        sum_penalty = 0
        num_classes_in_batch = 0
        for target_val in range(self.num_class):
            batch_bool = label.long().flatten() == target_val
            if not batch_bool.any():
                continue
            p_vanilla_w_same_t_val = p_vanilla[batch_bool, target_val]
            p_spurious_w_same_t_val = p_spurious[batch_bool, target_val]

            positive_spurious_group_avg_p = (
                p_spurious_w_same_t_val * p_vanilla_w_same_t_val
            ).sum() / (p_spurious_w_same_t_val.sum() + EPS)
            negative_spurious_group_avg_p = (
                (1 - p_spurious_w_same_t_val) * p_vanilla_w_same_t_val
            ).sum() / ((1 - p_spurious_w_same_t_val).sum() + EPS)

            discover_net_deo_loss = -torch.log(
                EPS
                + torch.abs(
                    positive_spurious_group_avg_p - negative_spurious_group_avg_p
                )
            )

            negative_p_spurious_w_same_t_val = 1 - p_spurious_w_same_t_val
            penalty = -torch.log(
                EPS
                + 1
                - torch.abs(
                    p_spurious_w_same_t_val.mean()
                    - negative_p_spurious_w_same_t_val.mean()
                )
            )

            sum_discover_net_deo_loss += discover_net_deo_loss
            sum_penalty += penalty
            num_classes_in_batch += 1

        sum_penalty /= num_classes_in_batch
        sum_discover_net_deo_loss /= num_classes_in_batch
        loss_discover = sum_discover_net_deo_loss + sum_penalty

        loss_discover.backward()
        self.optimizer_bias_discover_net.step()
        self.optimizer_bias_discover_net.zero_grad(set_to_none=True)

        return {"train_cls_loss": ce_loss, "train_bias_loss": loss_discover}

    def _train_epoch(self):
        self._set_train()
        self.current_lr = self.scheduler.get_last_lr()[0]
        avg_loss = None
        for batch, batch2 in zip(self.dataloaders["train"], self.second_train_loader):
            bsz = batch["targets"].shape[0]
            loss_dict = self._train_iter(batch, batch2)
            # initialize if needed
            if avg_loss is None:
                avg_loss = {key: AverageMeter() for key in loss_dict.keys()}
            # Update avg_loss for each key in loss_dict
            for key, value in loss_dict.items():
                avg_loss[key].update(value.item(), bsz)
        # self.scheduler.step()
        avg_loss = {key: value.avg for key, value in avg_loss.items()}
        return avg_loss

    def _save_checkpoint(self, tag):
        state = {
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_performance": self.best_performance,
            "scheduler": self.scheduler.state_dict(),
            "bias_discover_net": self.bias_discover_net.state_dict(),
            "optimizer_bias_discover_net": self.optimizer_bias_discover_net.state_dict(),
        }
        save_checkpoint(state, os.path.join(self.log_path, tag))

    def load_checkpoint(self, tag):
        checkpoint = load_checkpoint(os.path.join(self.log_path, tag))
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_performance = checkpoint["best_performance"]
        self.current_epoch = checkpoint["epoch"]
        self.bias_discover_net.load_state_dict(checkpoint["bias_discover_net"])
        self.optimizer_bias_discover_net.load_state_dict(
            checkpoint["optimizer_bias_discover_net"]
        )
        print(
            log_msg(
                f"Loaded checkpoint from {os.path.join(self.log_path, tag)}",
                "INFO",
                self.logger,
            )
        )
