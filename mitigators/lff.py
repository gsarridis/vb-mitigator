import os
from models.builder import get_model
from tools.utils import load_checkpoint, log_msg, save_checkpoint
import torch
import torch.nn as nn
from .losses import EMAGPU as EMA
from .losses import GeneralizedCECriterion
from .base_trainer import BaseTrainer


class LfFTrainer(BaseTrainer):
    def _method_specific_setups(self):
        train_target_attr = self.dataloaders["train"].dataset.targets
        self.sample_loss_ema_b = EMA(
            torch.LongTensor(train_target_attr), device=self.device, alpha=0.7
        )
        self.sample_loss_ema_d = EMA(
            torch.LongTensor(train_target_attr), device=self.device, alpha=0.7
        )



    def _setup_models(self):
        super()._setup_models()
        self.bias_discover_net = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
        )
        self.bias_discover_net.to(self.device)

    def _setup_criterion(self):
        super()._setup_criterion()
        self.criterion_train = nn.CrossEntropyLoss(reduction="none")
        self.gce_criterion = GeneralizedCECriterion()


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


    def _set_train(self):
        super()._set_train()
        self.bias_discover_net.train()

    def _set_eval(self):
        super()._set_eval()
        self.bias_discover_net.eval()


    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        idx_data = batch["index"]


        spurious_logits = self.bias_discover_net(inputs)
        if isinstance(spurious_logits, tuple):
            spurious_logits, _ = spurious_logits
        target_logits = self.model(inputs)
        if isinstance(target_logits, tuple):
            target_logits, _ = target_logits
        ce_loss = self.criterion(target_logits, targets)
        gce_loss = self.gce_criterion(spurious_logits, targets).mean()
        loss_b = self.criterion(spurious_logits, targets).detach()
        loss_d = ce_loss.detach()

         # EMA sample loss
        self.sample_loss_ema_b.update(loss_b, idx_data)
        self.sample_loss_ema_d.update(loss_d, idx_data)

        # class-wise normalize
        loss_b = self.sample_loss_ema_b.parameter[idx_data].clone().detach()
        loss_d = self.sample_loss_ema_d.parameter[idx_data].clone().detach()

        max_loss_b = self.sample_loss_ema_b.max_loss(targets)
        max_loss_d = self.sample_loss_ema_d.max_loss(targets)
        loss_b /= max_loss_b
        loss_d /= max_loss_d

        loss_weight = loss_b / (loss_b + loss_d + 1e-8)
        ce_loss = (ce_loss * loss_weight).mean()

        loss = ce_loss + gce_loss
        
        self.optimizer.zero_grad()
        self.optimizer_bias_discover_net.zero_grad()
        self._loss_backward(loss)
        self.optimizer.step()
        self.optimizer_bias_discover_net.step()
        self.scheduler.step()

        return {"train_cls_loss": ce_loss, "train_gce_loss": gce_loss }
    
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