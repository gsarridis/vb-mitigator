# # Bias Ensemble (Lee et al., AAAI 2023)
# # "Revisiting the Importance of Amplifying Bias for Debiasing"
# # Based on: https://github.com/kakaoenterprise/BiasEnsemble
# #
# # Key idea: The quality of the biased model f_B matters. Bias-conflicting
# # samples act as noise when training f_B. BiasEnsemble:
# # 1. Trains an ensemble of biased models (using GCE) to identify which
# #    samples are bias-aligned (high-confidence predictions).
# # 2. Filters out samples where the ensemble disagrees (likely bias-conflicting).
# # 3. Retrains f_B on only the identified bias-aligned samples, producing
# #    a purer biased model.
# # 4. Uses LfF-style reweighting with this improved f_B to train f_D.

# import os
# import copy
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.builder import get_model
# from tools.utils import load_checkpoint, log_msg, save_checkpoint
# from .base_trainer import BaseTrainer
# from .losses import GeneralizedCECriterion, EMAGPU as EMA


# class BiasEnsembleTrainer(BaseTrainer):
#     """
#     BiasEnsemble trainer.

#     Phase 1 (pretrain_b_ensemble): Train num_bias_models biased models
#         using GCE loss. For each model, identify samples with softmax probability
#         above a threshold. Keep samples where >= 'agreement' models agree.
#         These are the pseudo bias-aligned samples.

#     Phase 2 (main training): LfF-style training where:
#         - f_B (biased model) is trained with CE only on the pseudo
#           bias-aligned subset identified in Phase 1.
#         - f_D (debiased model) is trained on all samples, reweighted
#           by the relative difficulty ratio from f_B and f_D losses.
#     """

#     def _method_specific_setups(self):
#         train_target_attr = self.dataloaders["train"].dataset.targets
#         self.sample_loss_ema_b = EMA(
#             torch.LongTensor(train_target_attr), device=self.device, alpha=0.7
#         )
#         self.sample_loss_ema_d = EMA(
#             torch.LongTensor(train_target_attr), device=self.device, alpha=0.7
#         )

#         # Phase 1: discover bias-aligned samples via ensemble
#         self.bias_aligned_mask = self._pretrain_bias_ensemble()

#     def _setup_models(self):
#         super()._setup_models()
#         self.bias_discover_net = get_model(
#             self.cfg.MODEL.TYPE,
#             self.num_class,
#         )
#         self.bias_discover_net.to(self.device)

#     def _setup_criterion(self):
#         super()._setup_criterion()
#         self.criterion_train = nn.CrossEntropyLoss(reduction="none")
#         self.gce_criterion = GeneralizedCECriterion(
#             q=self.cfg.MITIGATOR.BIAS_ENSEMBLE.GCE_Q
#         )

#     def _setup_optimizer(self):
#         super()._setup_optimizer()
#         if self.cfg.SOLVER.TYPE == "SGD":
#             self.optimizer_bias = torch.optim.SGD(
#                 self.bias_discover_net.parameters(),
#                 lr=self.cfg.SOLVER.LR,
#                 momentum=self.cfg.SOLVER.MOMENTUM,
#                 weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
#             )
#         elif self.cfg.SOLVER.TYPE in ("Adam", "AdamW"):
#             self.optimizer_bias = torch.optim.Adam(
#                 self.bias_discover_net.parameters(),
#                 lr=self.cfg.SOLVER.LR,
#                 weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
#             )
#         else:
#             raise ValueError(f"Unsupported optimizer type: {self.cfg.SOLVER.TYPE}")

#     def _set_train(self):
#         super()._set_train()
#         self.bias_discover_net.train()

#     def _set_eval(self):
#         super()._set_eval()
#         self.bias_discover_net.eval()

#     def _pretrain_bias_ensemble(self):
#         """
#         Phase 1: Train an ensemble of biased models using GCE loss.
#         For each model, identify samples with high softmax confidence.
#         Return a mask of samples where enough models agree (bias-aligned).
#         """
#         cfg = self.cfg
#         num_bias_models = cfg.MITIGATOR.BIAS_ENSEMBLE.NUM_BIAS_MODELS
#         pretrain_epochs = cfg.MITIGATOR.BIAS_ENSEMBLE.PRETRAIN_EPOCHS
#         softmax_threshold = cfg.MITIGATOR.BIAS_ENSEMBLE.SOFTMAX_THRESHOLD
#         agreement = cfg.MITIGATOR.BIAS_ENSEMBLE.AGREEMENT

#         MASK_SAVE_PATH = os.path.join(self.log_path, "bias_ensemble_mask.pt")

#         # Check if mask already computed
#         if os.path.exists(MASK_SAVE_PATH):
#             print("Loading pre-computed BiasEnsemble mask...")
#             mask_data = torch.load(MASK_SAVE_PATH, map_location="cpu")
#             return mask_data["bias_aligned_mask"]

#         train_num = len(self.sets["train"])
#         exceed_masks = []

#         ordered_loader = torch.utils.data.DataLoader(
#             self.sets["train"],
#             batch_size=cfg.SOLVER.BATCH_SIZE,
#             shuffle=False,
#             num_workers=cfg.DATASET.NUM_WORKERS,
#             pin_memory=True,
#             persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
#         )

#         for model_idx in range(num_bias_models):
#             print(
#                 f"BiasEnsemble: Training biased model {model_idx+1}/{num_bias_models}..."
#             )

#             # Fresh biased model for each ensemble member
#             bias_model = get_model(self.cfg.MODEL.TYPE, self.num_class)
#             bias_model.to(self.device)
#             bias_optimizer = torch.optim.Adam(
#                 bias_model.parameters(),
#                 lr=cfg.SOLVER.LR,
#                 weight_decay=cfg.SOLVER.WEIGHT_DECAY,
#             )

#             best_valid_acc = 0.0
#             best_model_state = None

#             # Train with GCE
#             for epoch in range(pretrain_epochs):
#                 bias_model.train()
#                 for batch in self.dataloaders["train"]:
#                     inputs = batch["inputs"].to(self.device)
#                     targets = batch["targets"].to(self.device)
#                     logits = bias_model(inputs)
#                     if isinstance(logits, tuple):
#                         logits, _ = logits
#                     loss = self.gce_criterion(logits, targets).mean()
#                     bias_optimizer.zero_grad()
#                     loss.backward()
#                     bias_optimizer.step()

#                 # Validation to pick best epoch
#                 bias_model.eval()
#                 correct, total = 0, 0
#                 val_loader = self.dataloaders.get("val", self.dataloaders.get("test"))
#                 with torch.no_grad():
#                     for batch in val_loader:
#                         inputs = batch["inputs"].to(self.device)
#                         targets = batch["targets"].to(self.device)
#                         logits = bias_model(inputs)
#                         if isinstance(logits, tuple):
#                             logits, _ = logits
#                         pred = logits.argmax(dim=1)
#                         correct += (pred == targets).sum().item()
#                         total += targets.size(0)
#                 val_acc = correct / total if total > 0 else 0
#                 if val_acc > best_valid_acc:
#                     best_valid_acc = val_acc
#                     best_model_state = copy.deepcopy(bias_model.state_dict())

#             print(f"  Best validation acc: {best_valid_acc:.4f}")

#             # Load best model and compute softmax confidence on train set
#             bias_model.load_state_dict(best_model_state)
#             bias_model.eval()

#             gt_probs = []
#             with torch.no_grad():
#                 for batch in ordered_loader:
#                     inputs = batch["inputs"].to(self.device)
#                     targets = batch["targets"].to(self.device)
#                     logits = bias_model(inputs)
#                     if isinstance(logits, tuple):
#                         logits, _ = logits
#                     prob = torch.softmax(logits, dim=-1)
#                     gt_prob = torch.gather(
#                         prob, index=targets.unsqueeze(1), dim=1
#                     ).squeeze(1)
#                     gt_probs.append(gt_prob.cpu())

#             gt_probs = torch.cat(gt_probs, dim=0)
#             exceed_mask = (gt_probs > softmax_threshold).long()
#             exceed_masks.append(exceed_mask)
#             print(
#                 f"  Samples exceeding threshold: {exceed_mask.sum().item()}/{train_num}"
#             )

#             del bias_model, bias_optimizer

#         # Ensemble agreement: keep samples where >= agreement models agree
#         mask_sum = torch.stack(exceed_masks).sum(dim=0)
#         bias_aligned_mask = (mask_sum >= agreement).long()

#         print(
#             f"BiasEnsemble: {bias_aligned_mask.sum().item()}/{train_num} "
#             f"samples identified as bias-aligned (agreement >= {agreement})"
#         )

#         # Save mask
#         os.makedirs(
#             os.path.dirname(MASK_SAVE_PATH) if os.path.dirname(MASK_SAVE_PATH) else ".",
#             exist_ok=True,
#         )
#         torch.save({"bias_aligned_mask": bias_aligned_mask}, MASK_SAVE_PATH)

#         return bias_aligned_mask

#     def _train_iter(self, batch):
#         inputs = batch["inputs"].to(self.device)
#         targets = batch["targets"].to(self.device)
#         idx_data = batch["index"]

#         # Forward both models
#         bias_logits = self.bias_discover_net(inputs)
#         if isinstance(bias_logits, tuple):
#             bias_logits, _ = bias_logits
#         target_logits = self.model(inputs)
#         if isinstance(target_logits, tuple):
#             target_logits, _ = target_logits

#         # Compute per-sample CE losses (detached, for EMA tracking)
#         loss_b = self.criterion_train(bias_logits, targets).detach()
#         loss_d = self.criterion_train(target_logits, targets).detach()

#         # EMA sample loss
#         self.sample_loss_ema_b.update(loss_b, idx_data)
#         self.sample_loss_ema_d.update(loss_d, idx_data)

#         # Class-wise normalize
#         loss_b = self.sample_loss_ema_b.parameter[idx_data].clone().detach()
#         loss_d = self.sample_loss_ema_d.parameter[idx_data].clone().detach()

#         max_loss_b = self.sample_loss_ema_b.max_loss(targets)
#         max_loss_d = self.sample_loss_ema_d.max_loss(targets)
#         loss_b /= max_loss_b
#         loss_d /= max_loss_d

#         # LfF reweighting for debiased model
#         loss_weight = loss_b / (loss_b + loss_d + 1e-8)

#         # Biased model: train only on bias-aligned samples
#         # (the key BiasEnsemble contribution)
#         curr_align_flag = self.bias_aligned_mask[idx_data].to(self.device).bool()

#         if curr_align_flag.any():
#             loss_b_update = self.criterion_train(
#                 bias_logits[curr_align_flag], targets[curr_align_flag]
#             ).mean()
#         else:
#             loss_b_update = torch.tensor(0.0, device=self.device)

#         # Debiased model: reweighted CE on all samples
#         loss_d_update = (
#             self.criterion_train(target_logits, targets) * loss_weight.to(self.device)
#         ).mean()

#         loss = loss_b_update + loss_d_update

#         self.optimizer.zero_grad()
#         self.optimizer_bias.zero_grad()
#         self._loss_backward(loss)
#         self.optimizer.step()
#         self.optimizer_bias.step()
#         self.scheduler.step()

#         return {"train_cls_loss": loss_d_update, "train_bias_loss": loss_b_update}

#     def _save_checkpoint(self, tag):
#         state = {
#             "epoch": self.current_epoch,
#             "model": self.model.state_dict(),
#             "optimizer": self.optimizer.state_dict(),
#             "best_performance": self.best_performance,
#             "scheduler": self.scheduler.state_dict(),
#             "bias_discover_net": self.bias_discover_net.state_dict(),
#             "optimizer_bias": self.optimizer_bias.state_dict(),
#             "bias_aligned_mask": self.bias_aligned_mask,
#         }
#         save_checkpoint(state, os.path.join(self.log_path, tag))

#     def load_checkpoint(self, tag):
#         checkpoint = load_checkpoint(os.path.join(self.log_path, tag))
#         self.model.load_state_dict(checkpoint["model"])
#         self.optimizer.load_state_dict(checkpoint["optimizer"])
#         self.scheduler.load_state_dict(checkpoint["scheduler"])
#         self.best_performance = checkpoint["best_performance"]
#         self.current_epoch = checkpoint["epoch"]
#         self.bias_discover_net.load_state_dict(checkpoint["bias_discover_net"])
#         self.optimizer_bias.load_state_dict(checkpoint["optimizer_bias"])
#         self.bias_aligned_mask = checkpoint["bias_aligned_mask"]
#         print(
#             log_msg(
#                 f"Loaded checkpoint from {os.path.join(self.log_path, tag)}",
#                 "INFO",
#                 self.logger,
#             )
#         )


# BiasEnsemble (Lee et al., AAAI 2023)
# "Revisiting the Importance of Amplifying Bias for Debiasing"
# https://github.com/kakaoenterprise/BiasEnsemble
#
# This is the LfF + BE variant. BiasEnsemble is a sample selection wrapper
# applied on top of LfF (or DisEnt). The key insight: train an ensemble of
# biased models, use their agreement to identify "exceeded" samples (those
# the biased models confidently classify correctly = likely bias-aligned),
# and only train the biased model f_B on those samples. This removes
# bias-conflicting noise from f_B's training, sharpening the bias signals
# used in LfF's reweighting.

import os
import copy
import torch
import torch.nn as nn

from models.builder import get_model
from tools.utils import load_checkpoint, log_msg, save_checkpoint
from .base_trainer import BaseTrainer
from .losses import EMAGPU as EMA
from .losses import GeneralizedCECriterion


class BiasEnsembleTrainer(BaseTrainer):
    """
    LfF + BiasEnsemble trainer.

    Phase 1: Train N independent biased models from scratch using GCE loss
             for a fixed number of pretraining epochs each. Use each one to
             score every training sample by its predicted probability of the
             true class. A sample is "exceeded" by a model if that probability
             > softmax_threshold. A sample is selected for f_B training if at
             least `agreement` of the N models flag it as exceeded.

    Phase 2: Standard LfF training, except f_B (bias model) is only updated
             on the selected subset, while f_D (debiased model) is trained on
             all samples with LfF's reweighting based on per-sample loss
             ratios between f_B and f_D.
    """

    def _method_specific_setups(self):
        train_target_attr = self.dataloaders["train"].dataset.targets
        self.sample_loss_ema_b = EMA(
            torch.LongTensor(train_target_attr), device=self.device, alpha=0.7
        )
        self.sample_loss_ema_d = EMA(
            torch.LongTensor(train_target_attr), device=self.device, alpha=0.7
        )

        # Run BiasEnsemble pretraining: produces self.mask_index
        self._pretrain_bias_ensemble()

    def _setup_models(self):
        super()._setup_models()
        # f_B (bias model) — built fresh; in phase 1 it gets repeatedly
        # re-initialized; in phase 2 it is the LfF biased model
        self.bias_model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
        )
        self.bias_model.to(self.device)

    def _setup_criterion(self):
        super()._setup_criterion()
        self.criterion_train = nn.CrossEntropyLoss(reduction="none")
        self.gce_criterion = GeneralizedCECriterion(
            q=self.cfg.MITIGATOR.BIAS_ENSEMBLE.GCE_Q
        )

    def _setup_optimizer(self):
        super()._setup_optimizer()
        self._build_bias_optimizer()

    def _build_bias_optimizer(self):
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

    # ------------------------------------------------------------------
    # Phase 1: BiasEnsemble pretraining
    # ------------------------------------------------------------------
    def _pretrain_bias_ensemble(self):
        """
        Train N biased models, score every training sample with each one,
        and select the subset of samples flagged "exceeded" by at least
        `agreement` models. Saves the selection mask in self.mask_index.
        """
        cfg = self.cfg
        num_models = cfg.MITIGATOR.BIAS_ENSEMBLE.NUM_BIAS_MODELS
        pretrain_epochs = cfg.MITIGATOR.BIAS_ENSEMBLE.PRETRAIN_EPOCHS
        softmax_threshold = cfg.MITIGATOR.BIAS_ENSEMBLE.SOFTMAX_THRESHOLD
        agreement = cfg.MITIGATOR.BIAS_ENSEMBLE.AGREEMENT

        train_set = self.sets["train"]
        train_num = len(train_set)

        MASK_SAVE_PATH = os.path.join(self.log_path, "be_mask_index.pt")
        if os.path.exists(MASK_SAVE_PATH):
            print("Loading pre-computed BiasEnsemble mask...")
            self.mask_index = torch.load(MASK_SAVE_PATH, map_location="cpu")
            print(
                f"BiasEnsemble: loaded mask with {int(self.mask_index.sum())}/{train_num} selected samples."
            )
            return

        # Ordered loader for scoring (no shuffle)
        ordered_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
        )

        # exceed_masks[i] is a 0/1 tensor of length train_num indicating
        # which samples model i flagged as exceeded
        exceed_masks = []

        for i in range(num_models):
            print(
                log_msg(
                    f"BiasEnsemble: pretraining biased model {i+1}/{num_models}",
                    "INFO",
                    self.logger,
                )
            )
            # Fresh biased model + optimizer
            bias_model_i = get_model(
                self.cfg.MODEL.TYPE,
                self.num_class,
            ).to(self.device)
            if self.cfg.SOLVER.TYPE == "SGD":
                opt_i = torch.optim.SGD(
                    bias_model_i.parameters(),
                    lr=cfg.SOLVER.LR,
                    momentum=cfg.SOLVER.MOMENTUM,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                )
            else:
                opt_i = torch.optim.Adam(
                    bias_model_i.parameters(),
                    lr=cfg.SOLVER.LR,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                )

            best_val_acc = -1.0
            best_state = None

            val_loader = self.dataloaders.get("val", None)

            for epoch in range(pretrain_epochs):
                bias_model_i.train()
                for batch in self.dataloaders["train"]:
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    logits = bias_model_i(inputs)
                    if isinstance(logits, tuple):
                        logits, _ = logits
                    loss = self.gce_criterion(logits, targets).mean()
                    opt_i.zero_grad()
                    loss.backward()
                    opt_i.step()

                # Validation tracking — keep best checkpoint per the paper
                if val_loader is not None:
                    bias_model_i.eval()
                    correct, total = 0, 0
                    with torch.no_grad():
                        for batch in val_loader:
                            inputs = batch["inputs"].to(self.device)
                            targets = batch["targets"].to(self.device)
                            logits = bias_model_i(inputs)
                            if isinstance(logits, tuple):
                                logits, _ = logits
                            pred = logits.argmax(dim=1)
                            correct += (pred == targets).sum().item()
                            total += targets.size(0)
                    val_acc = correct / max(total, 1)
                    print(
                        f"  [BE pretrain {i+1}/{num_models}] epoch {epoch+1}/{pretrain_epochs} val_acc={val_acc:.4f}"
                    )
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_state = copy.deepcopy(bias_model_i.state_dict())
                else:
                    print(
                        f"  [BE pretrain {i+1}/{num_models}] epoch {epoch+1}/{pretrain_epochs}"
                    )

            if best_state is not None:
                bias_model_i.load_state_dict(best_state)

            # Score all training samples
            bias_model_i.eval()
            gt_probs = torch.zeros(train_num)
            with torch.no_grad():
                for batch in ordered_loader:
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    indices = batch["index"]
                    logits = bias_model_i(inputs)
                    if isinstance(logits, tuple):
                        logits, _ = logits
                    probs = torch.softmax(logits, dim=-1)
                    gt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                    gt_probs[indices] = gt.cpu()

            exceed_i = (gt_probs > softmax_threshold).long()
            exceed_masks.append(exceed_i)
            print(
                f"  Model {i+1}: {int(exceed_i.sum())}/{train_num} samples exceeded threshold {softmax_threshold}"
            )

            del bias_model_i, opt_i

        # Aggregate: a sample is selected if at least `agreement` models
        # flagged it as exceeded
        mask_sum = torch.stack(exceed_masks).sum(dim=0)
        self.mask_index = (mask_sum >= agreement).long()
        print(
            log_msg(
                f"BiasEnsemble: final selected subset = {int(self.mask_index.sum())}/{train_num} samples (agreement>={agreement})",
                "INFO",
                self.logger,
            )
        )

        os.makedirs(self.log_path, exist_ok=True)
        torch.save(self.mask_index, MASK_SAVE_PATH)

        # Reset bias_model and its optimizer for phase 2
        del self.bias_model
        self.bias_model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
        ).to(self.device)
        self._build_bias_optimizer()

    # ------------------------------------------------------------------
    # Phase 2: LfF training with bias model restricted to selected subset
    # ------------------------------------------------------------------
    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        idx_data = batch["index"]

        # Forward both models
        bias_logits = self.bias_model(inputs)
        if isinstance(bias_logits, tuple):
            bias_logits, _ = bias_logits
        target_logits = self.model(inputs)
        if isinstance(target_logits, tuple):
            target_logits, _ = target_logits

        # Per-sample CE for both (used by EMA)
        loss_b = self.criterion_train(bias_logits, targets).detach()
        loss_d = self.criterion_train(target_logits, targets).detach()

        # Update EMAs
        self.sample_loss_ema_b.update(loss_b, idx_data)
        self.sample_loss_ema_d.update(loss_d, idx_data)

        # Read EMA values
        loss_b_ema = self.sample_loss_ema_b.parameter[idx_data].clone().detach()
        loss_d_ema = self.sample_loss_ema_d.parameter[idx_data].clone().detach()

        # Class-wise normalize by max EMA loss per class
        max_loss_b = self.sample_loss_ema_b.max_loss(targets)
        max_loss_d = self.sample_loss_ema_d.max_loss(targets)
        loss_b_ema = loss_b_ema / (max_loss_b + 1e-8)
        loss_d_ema = loss_d_ema / (max_loss_d + 1e-8)

        # LfF reweighting
        loss_weight = loss_b_ema / (loss_b_ema + loss_d_ema + 1e-8)

        # ---- f_B (bias model) update: ONLY on selected subset ----
        curr_mask = self.mask_index[idx_data.cpu()].to(self.device)
        selected = curr_mask == 1

        if selected.any():
            loss_b_update = self.criterion_train(
                bias_logits[selected], targets[selected]
            ).mean()
        else:
            loss_b_update = torch.tensor(0.0, device=self.device)

        # ---- f_D (debiased model) update: all samples, reweighted ----
        loss_d_update = (
            self.criterion_train(target_logits, targets) * loss_weight.to(self.device)
        ).mean()

        loss = loss_b_update + loss_d_update

        self.optimizer.zero_grad()
        self.optimizer_bias.zero_grad()
        self._loss_backward(loss)
        self.optimizer.step()
        self.optimizer_bias.step()
        self.scheduler.step()

        return {
            "train_cls_loss": loss_d_update,
            "train_bias_loss": loss_b_update,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, tag):
        state = {
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_performance": self.best_performance,
            "scheduler": self.scheduler.state_dict(),
            "bias_model": self.bias_model.state_dict(),
            "optimizer_bias": self.optimizer_bias.state_dict(),
            "mask_index": self.mask_index,
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
        if "mask_index" in checkpoint:
            self.mask_index = checkpoint["mask_index"]
        print(
            log_msg(
                f"Loaded checkpoint from {os.path.join(self.log_path, tag)}",
                "INFO",
                self.logger,
            )
        )
