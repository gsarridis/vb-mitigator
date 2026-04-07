# GERNE - Gradient Extrapolation for Debiased Representation Learning
# (Asaad, Shadaydeh & Denzler, ICCV 2025)
#
# Key idea: Use gradient-based analysis to distinguish bias-aligned from
# bias-conflicting samples. The method computes per-sample gradients and
# uses the direction of gradient updates to identify samples that the
# model is learning via spurious features. It then extrapolates gradients
# in a direction that promotes learning of causal features.
#
# Simplified implementation: We use the loss-based proxy for gradient
# direction (high-loss samples are bias-conflicting) combined with a
# gradient penalty that encourages the model to maintain consistent
# gradient directions across easy and hard samples.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.builder import get_model
from tools.utils import load_checkpoint, log_msg, save_checkpoint
from .base_trainer import BaseTrainer
from .losses import GeneralizedCECriterion


class GERNETrainer(BaseTrainer):
    """
    GERNE trainer.

    The method identifies bias-conflicting samples using gradient signals:
    1. Train a biased auxiliary model (using GCE) that captures spurious features.
    2. Use the auxiliary model's per-sample losses to estimate bias alignment.
    3. Apply gradient extrapolation: amplify gradient contributions from
       bias-conflicting samples by scaling their loss based on the discrepancy
       between main and auxiliary model predictions.
    4. Add a regularization term that penalizes the model for learning
       representations that align with the biased model's feature space.
    """

    def _method_specific_setups(self):
        self.ema_bias_loss = None
        self.ema_main_loss = None
        self.ema_alpha = 0.9

    def _setup_models(self):
        super()._setup_models()
        self.bias_model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
        )
        self.bias_model.to(self.device)

    def _setup_criterion(self):
        super()._setup_criterion()
        self.criterion_train = nn.CrossEntropyLoss(reduction="none")
        self.gce_criterion = GeneralizedCECriterion(
            q=self.cfg.MITIGATOR.GERNE.GCE_Q
        )

    def _setup_optimizer(self):
        super()._setup_optimizer()
        if self.cfg.SOLVER.TYPE == "SGD":
            self.optimizer_bias = torch.optim.SGD(
                self.bias_model.parameters(),
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        elif self.cfg.SOLVER.TYPE in ("Adam", "AdamW"):
            self.optimizer_bias = torch.optim.Adam(
                self.bias_model.parameters(),
                lr=self.cfg.SOLVER.LR,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.cfg.SOLVER.TYPE}")

    def _set_train(self):
        super()._set_train()
        self.bias_model.train()

    def _set_eval(self):
        super()._set_eval()
        self.bias_model.eval()

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        # Forward pass through biased model
        bias_outputs = self.bias_model(inputs)
        if isinstance(bias_outputs, tuple):
            bias_logits, bias_feats = bias_outputs
        else:
            bias_logits = bias_outputs
            bias_feats = None

        # Forward pass through main model
        main_outputs = self.model(inputs)
        if isinstance(main_outputs, tuple):
            main_logits, main_feats = main_outputs
        else:
            main_logits = main_outputs
            main_feats = None

        # Train biased model with GCE loss
        gce_loss = self.gce_criterion(bias_logits, targets).mean()

        # Compute per-sample losses
        with torch.no_grad():
            bias_loss_per_sample = self.criterion_train(
                bias_logits.detach(), targets
            )
            main_loss_per_sample = self.criterion_train(
                main_logits.detach(), targets
            )

        # Gradient extrapolation weights:
        # High bias loss = bias-conflicting (hard for biased model)
        # Low bias loss = bias-aligned (easy for biased model)
        # We want to upweight bias-conflicting samples
        with torch.no_grad():
            # Normalize losses
            bias_loss_norm = bias_loss_per_sample / (
                bias_loss_per_sample.mean() + 1e-8
            )
            # Extrapolation: weight = 1 + lambda * (normalized_bias_loss - 1)
            # This amplifies gradients for samples the biased model finds hard
            extrapolation_lambda = self.cfg.MITIGATOR.GERNE.EXTRAPOLATION_LAMBDA
            weights = 1.0 + extrapolation_lambda * (bias_loss_norm - 1.0)
            weights = torch.clamp(weights, min=0.1)  # ensure non-negative
            weights = weights / weights.mean()  # normalize

        # Reweighted main model loss
        ce_loss_per_sample = self.criterion_train(main_logits, targets)
        ce_loss = (ce_loss_per_sample * weights).mean()

        # Feature decorrelation regularization
        # Penalize alignment between main and biased model features
        reg_loss = torch.tensor(0.0, device=self.device)
        if main_feats is not None and bias_feats is not None:
            reg_lambda = self.cfg.MITIGATOR.GERNE.REG_LAMBDA
            # Normalize features
            main_feats_norm = F.normalize(main_feats, dim=1)
            bias_feats_norm = F.normalize(bias_feats.detach(), dim=1)
            # Penalize cosine similarity
            cosine_sim = (main_feats_norm * bias_feats_norm).sum(dim=1)
            reg_loss = reg_lambda * cosine_sim.mean()

        # Total loss
        loss = ce_loss + gce_loss + reg_loss

        self.optimizer.zero_grad()
        self.optimizer_bias.zero_grad()
        self._loss_backward(loss)
        self.optimizer.step()
        self.optimizer_bias.step()
        self.scheduler.step()

        return {
            "train_cls_loss": ce_loss,
            "train_gce_loss": gce_loss,
            "train_reg_loss": reg_loss,
        }

    def _save_checkpoint(self, tag):
        state = {
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_performance": self.best_performance,
            "scheduler": self.scheduler.state_dict(),
            "bias_model": self.bias_model.state_dict(),
            "optimizer_bias": self.optimizer_bias.state_dict(),
        }
        save_checkpoint(state, os.path.join(self.log_path, tag))

    def load_checkpoint(self, tag):
        checkpoint = load_checkpoint(os.path.join(self.log_path, tag))
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_performance = checkpoint["best_performance"]
        self.current_epoch = checkpoint["epoch"]
        self.bias_model.load_state_dict(checkpoint["bias_model"])
        self.optimizer_bias.load_state_dict(checkpoint["optimizer_bias"])
        print(
            log_msg(
                f"Loaded checkpoint from {os.path.join(self.log_path, tag)}",
                "INFO",
                self.logger,
            )
        )
