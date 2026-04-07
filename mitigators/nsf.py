# NSF / SSC — "Let Samples Speak: Mitigating Spurious Correlation by
# Exploiting the Clusterness of Samples" (Li et al., CVPR 2025)
# https://github.com/davelee-uestc/nsf_debiasing
#
# Faithful adaptation of the official ssc.py / ssc_common.py pipeline to
# the vb-mitigator framework.
#
# This is a *post-hoc* debiasing method. Given a pretrained ERM model,
# the algorithm runs entirely on extracted features (no backbone
# retraining):
#
#   1. Extract penultimate features for the held-out *val* split (the
#      official code uses --train_split val).
#   2. IDENTIFY: compute global per-class prototypes; flag any sample
#      whose nearest global prototype is the wrong class as an "outlier".
#   3. NEUTRALIZE: compute per-class inlier prototypes (from non-outliers)
#      and per-class outlier prototypes; the neutralized center for each
#      class is the midpoint of inlier and outlier prototypes (or just
#      the inlier prototype when a class has too few outliers).
#   4. ELIMINATE: learn a per-feature affine transformation
#         T(f) = (f + b) * w - b
#      with w ∈ R^d, b ∈ R^d learnable, by minimizing
#         dist(T(f_i), C_neutralized[y_i]) + 10 * mean(w)
#      The mean(w) regularizer shrinks spurious feature dimensions.
#   5. UPDATE: form two groups —
#         mask  : ERM-misclassified but rescued by nearest-center on
#                 transformed features (likely bias-conflicting)
#         mask2 : ERM-correct (likely bias-aligned)
#       and train a fresh linear head with balanced batches drawn from
#       the two groups.
#
# Final model = frozen backbone + transformation + new head.

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import PairwiseDistance
from tqdm import trange

from models.builder import get_model
from models.utils import get_local_model_dict
from tools.utils import load_checkpoint, log_msg, save_checkpoint
from .base_trainer import BaseTrainer


# ---------------------------------------------------------------------------
# Faithful re-implementation of helpers from ssc_common.py / ssc.py
# ---------------------------------------------------------------------------


def _proto(labels_s, f_s, mask, n_classes):
    """Per-class mean feature, returned in shape (n_classes+1, d).

    Translation of `proto()` from ssc_common.py. The (n_classes)-th row is
    a sentinel for masked-out samples; we ignore it via the returned
    `nms` count vector.
    """
    labels_s = labels_s.clone()
    if mask is not None:
        labels_s = labels_s.clone()
        labels_s[mask <= 0] = n_classes
    label_one_hot = F.one_hot(labels_s.long(), n_classes + 1).permute(1, 0).float()
    npps_sum = label_one_hot @ f_s
    label_masks = label_one_hot.sum(dim=-1) + 1e-5
    labelw = 1.0 / (label_masks.unsqueeze(dim=-1).expand(npps_sum.size()))
    npps = labelw * npps_sum
    label_masks[-1] = 0
    label_masks[label_masks < 1] = 0
    nms = label_masks
    return npps, nms


def _estimate_u(Y, F_, outliers, n_classes, log_fn):
    """Compute neutralized centers C[y] = 0.5 * (inlier_proto + outlier_proto).

    Falls back to `inlier_proto` for any class that has fewer than 2
    outliers (matching the official code).
    """
    npps_in, _ = _proto(Y, F_, (~outliers).float(), n_classes)
    npps_out, nms_out = _proto(Y, F_, outliers.float(), n_classes)
    log_fn(f"NSF: outlier counts per class = {nms_out.tolist()}")
    C = (0.5 * (npps_in + npps_out))[:-1]
    mask = (nms_out[:-1] > 1).float()  # has enough outliers
    C = mask[:, None] * C + (1 - mask)[:, None] * npps_in[:-1]
    return C


def _get_centers(Y, F_, n_classes, log_fn):
    """Compute global prototypes, identify outliers, compute neutralized centers."""
    Y_flat = Y.flatten()
    C_global, _ = _proto(Y_flat, F_, mask=None, n_classes=n_classes)
    # outlier = nearest global prototype's class != true class
    dists = (F_[:, None] - C_global[None, :-1]).norm(dim=-1, p=2)
    nearest = dists.argmin(dim=-1)
    outliers = nearest != Y_flat
    C = _estimate_u(Y_flat, F_, outliers, n_classes, log_fn)
    return C, C_global[:-1], outliers


def get_classifier_attr(model):
    if hasattr(model, "fc") and model.fc is not None:
        return "fc", model.fc
    elif hasattr(model, "fc2") and model.fc2 is not None:
        return "fc2", model.fc2
    else:
        raise AttributeError("Model has neither 'fc' nor 'fc2'.")


def get_last_linear(layer):
    if isinstance(layer, torch.nn.Sequential):
        for m in reversed(layer):
            if hasattr(m, "in_features") and hasattr(m, "out_features"):
                return m
        raise ValueError("No linear layer found inside Sequential.")
    return layer


class _Transformation(nn.Module):
    """T(f) = (f + b) * w - b. Per-feature scalars w, b ∈ R^d."""

    def __init__(self, d, device):
        super().__init__()
        self.w = nn.Parameter(torch.ones((1, d), device=device).float())
        self.b = nn.Parameter(torch.zeros((1, d), device=device).float())

    def forward(self, x):
        return (x + self.b) * self.w - self.b


def _transform_feature(centers, features, labels, n_steps, lr, w_reg=10.0):
    """Optimize the affine transformation to pull each sample toward its
    neutralized class center, with a mean(w) regularizer that shrinks
    spurious feature dimensions.
    """
    feat = features.detach().clone()
    Y = labels.flatten().long().detach().clone()
    d = feat.size(-1)
    transformation = _Transformation(d, feat.device)
    optimizer = torch.optim.AdamW(transformation.parameters(), lr=lr, weight_decay=0)
    dist = PairwiseDistance(keepdim=True)
    for _ in trange(n_steps, desc="NSF transform_feature"):
        optimizer.zero_grad()
        f = transformation(feat)
        target = centers[Y]
        loss = dist(f, target).mean() + w_reg * transformation.w.mean()
        loss.backward()
        optimizer.step()
    return transformation


def _adjust_classifier(
    in_features,
    out_features,
    feat,
    Y,
    mask,
    mask2,
    n_steps,
    device,
    lr,
):
    """Train a new linear head with balanced batches from two groups.

    Group 1: samples in `mask` (rescued bias-conflicting).
    Group 2: samples in `mask2` (ERM-correct, mostly bias-aligned).

    Each step samples B examples from each group (B = min sizes), computes
    CE on each, and adds the losses. This is a class-imbalance-style
    rebalancing across the two discovered groups.
    """
    feat1 = feat[mask].detach().clone()
    feat2 = feat[mask2].detach().clone()
    Y1 = Y.flatten().long()[mask].detach().clone()
    Y2 = Y.flatten().long()[mask2].detach().clone()

    fc_new = nn.Linear(in_features, out_features).to(device)
    optimizer = torch.optim.AdamW(fc_new.parameters(), lr=lr, weight_decay=0)
    B = min(len(Y1), len(Y2))
    if B == 0:
        logging.warning(
            "NSF: one of the two groups is empty, cannot adjust classifier; "
            "returning untrained head."
        )
        return fc_new

    for _ in trange(n_steps, desc="NSF adjust_classifier"):
        optimizer.zero_grad()
        idx1 = (
            torch.randint(0, len(Y1), (B,), device=device)
            if B < len(Y1)
            else torch.arange(0, len(Y1), device=device)
        )
        idx2 = (
            torch.randint(0, len(Y2), (B,), device=device)
            if B < len(Y2)
            else torch.arange(0, len(Y2), device=device)
        )
        loss = F.cross_entropy(fc_new(feat1[idx1]), Y1[idx1]) + F.cross_entropy(
            fc_new(feat2[idx2]), Y2[idx2]
        )
        loss.backward()
        optimizer.step()
    return fc_new


def _nearest_center_predict(features, centers):
    """Predict each sample's class as the index of the nearest center."""
    dists = (features[:, None] - centers[None]).norm(dim=-1, p=2)
    return dists.argmin(dim=-1)


# ---------------------------------------------------------------------------
# NSF Trainer
# ---------------------------------------------------------------------------


class NSFTrainer(BaseTrainer):
    """
    NSF (a.k.a. SSC) trainer.

    Pipeline:
      1. Load (or train) an ERM checkpoint into `self.model`.
      2. Freeze the backbone.
      3. Extract features for the val split (the official code uses
         --train_split val).
      4. Run identify -> neutralize -> eliminate -> update.
      5. Replace `self.model.fc` with the learned head (wrapping the
         feature transformation), so the standard validation pipeline
         works unchanged.
    """

    def _setup_criterion(self):
        if self.cfg.SOLVER.CRITERION == "CE":
            self.criterion_train = nn.CrossEntropyLoss(reduction="none")
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {self.cfg.SOLVER.CRITERION}")

    def _method_specific_setups(self):
        # NSF runs everything in `train()` (which we override). The setup
        # phase only loads the pretrained ERM checkpoint if provided.
        ckpt_path = self.cfg.MITIGATOR.NSF.CKPT_PATH
        if ckpt_path:
            print(
                log_msg(
                    f"NSF: loading pretrained ERM checkpoint from {ckpt_path}",
                    "INFO",
                    self.logger,
                )
            )
            try:
                model_dict = get_local_model_dict(ckpt_path)
                self.model.load_state_dict(model_dict["model"])
            except Exception as e:
                logging.warning(
                    f"NSF: failed to load checkpoint via get_local_model_dict ({e}), "
                    f"trying torch.load directly."
                )
                state = torch.load(ckpt_path, map_location=self.device)
                if isinstance(state, dict) and "model" in state:
                    self.model.load_state_dict(state["model"])
                else:
                    self.model.load_state_dict(state)

    # ------------------------------------------------------------------
    # Override train() — NSF replaces the whole training loop
    # ------------------------------------------------------------------
    def train(self):
        cfg = self.cfg

        # If no checkpoint was provided, train an ERM base model first
        if not cfg.MITIGATOR.NSF.CKPT_PATH:
            print(
                log_msg(
                    f"NSF: no CKPT_PATH provided; training ERM base model "
                    f"for {cfg.MITIGATOR.NSF.ERM_EPOCHS} epochs",
                    "INFO",
                    self.logger,
                )
            )
            self._train_erm_base(cfg.MITIGATOR.NSF.ERM_EPOCHS)

        # Freeze the backbone
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # ----- Extract features from the val split -----
        # feat_split = cfg.MITIGATOR.NSF.FEATURE_SPLIT
        # if feat_split not in self.dataloaders:
        #     logging.warning(
        #         f"NSF: feature split '{feat_split}' not in dataloaders; "
        #         f"falling back to 'train'."
        #     )
        #    feat_split = "train"
        feat_split = "train"
        print(
            log_msg(
                f"NSF: extracting features from '{feat_split}' split",
                "INFO",
                self.logger,
            )
        )
        feats, labels = self._extract_features(self.dataloaders[feat_split])
        feats = feats.to(self.device)
        labels = labels.to(self.device)

        n_classes = self.num_class

        def _log(msg):
            print(log_msg(msg, "INFO", self.logger))

        # ----- Step 1+2: identify outliers + neutralized centers -----
        centers, C_global, outliers = _get_centers(labels, feats, n_classes, _log)
        _log(
            f"NSF: identified {int(outliers.sum())}/{len(outliers)} outliers "
            f"({float(outliers.float().mean())*100:.1f}%)"
        )

        # ----- Step 3: learn feature transformation -----
        _log(
            f"NSF: learning feature transformation for "
            f"{cfg.MITIGATOR.NSF.NUM_EPOCHS_TRANSFORM} steps"
        )
        transformation = _transform_feature(
            centers,
            feats,
            labels,
            n_steps=cfg.MITIGATOR.NSF.NUM_EPOCHS_TRANSFORM,
            lr=cfg.MITIGATOR.NSF.LR_TRANSFORM,
            w_reg=cfg.MITIGATOR.NSF.W_REG,
        )
        with torch.no_grad():
            transformed_feats = transformation(feats).detach()

        # ----- Step 4: form the two groups via nearest-center -----
        # PRED_ORI: nearest global prototype on original features
        Y = labels.flatten().long()
        PRED_ORI = _nearest_center_predict(feats, C_global)
        # PRED_NEW: nearest neutralized center on transformed features
        PRED_NEW = _nearest_center_predict(transformed_feats, centers)

        mask = ((PRED_ORI != Y) & (PRED_NEW == Y)).flatten()
        mask2 = (PRED_ORI == Y).flatten()
        _log(
            f"NSF: |rescued (mask)| = {int(mask.sum())}, "
            f"|easy (mask2)| = {int(mask2.sum())}"
        )

        # ----- Step 5: train a new classifier head -----
        attr_name, cl_layer = get_classifier_attr(self.model)
        cl_layer = get_last_linear(cl_layer)
        in_features = cl_layer.in_features
        out_features = cl_layer.out_features
        _log(
            f"NSF: adjusting classifier for " f"{cfg.MITIGATOR.NSF.NUM_EPOCHS_FT} steps"
        )
        fc_new = _adjust_classifier(
            in_features=in_features,
            out_features=out_features,
            feat=transformed_feats,
            Y=labels,
            mask=mask,
            mask2=mask2,
            n_steps=cfg.MITIGATOR.NSF.NUM_EPOCHS_FT,
            device=self.device,
            lr=cfg.MITIGATOR.NSF.LR_FT,
        )

        # ----- Plug transformation + new head into the model -----
        # We replace self.model.fc with a Sequential that first applies the
        # transformation, then the new head. This keeps the standard
        # validation loop in BaseTrainer working unchanged.
        new_head = nn.Sequential(transformation, fc_new).to(self.device)

        setattr(self.model, attr_name, new_head)

        # Re-freeze in case anything was set to require grads
        self.model.eval()
        log_dict = {}
        # ----- Final evaluation + checkpoint -----
        if cfg.METRIC == "wg_ovr_tags":
            test_performance = self._validate_epoch_tags(stage="test")
        else:
            val_performance = self._validate_epoch(stage="val")
            val_log_dict = self.build_log_dict(val_performance, stage="val")
            log_dict.update(val_log_dict)
            test_performance = self._validate_epoch(stage="test")
        test_log_dict = self.build_log_dict(test_performance, stage="test")
        log_dict.update(test_log_dict)
        self._update_best(log_dict)
        self._save_checkpoint(tag="best")
        self._save_checkpoint(tag="latest")
        self._log_epoch(log_dict, update_cpkt=True)

    # ------------------------------------------------------------------
    # ERM base training (only used if CKPT_PATH is empty)
    # ------------------------------------------------------------------
    def _train_erm_base(self, epochs):
        """Train an ERM model on the train split, used as the base for NSF."""
        cfg = self.cfg
        if cfg.SOLVER.TYPE == "SGD":
            opt = torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            opt = torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.SOLVER.LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        for ep in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            for batch in self.dataloaders["train"]:
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
                loss = self.criterion(outputs, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)
            print(
                f"  NSF ERM-base epoch {ep+1}/{epochs} "
                f"loss={total_loss/max(len(self.dataloaders['train']),1):.4f} "
                f"acc={correct/max(total,1):.4f}"
            )

    def _extract_features(self, loader):
        """Extract penultimate features and labels from a dataloader."""
        self.model.eval()
        all_feats, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    _, feats = outputs
                else:
                    # Fall back to logits if model doesn't expose features.
                    feats = outputs
                all_feats.append(feats.detach().cpu().float())
                all_targets.append(targets.detach().cpu().long())
        return torch.cat(all_feats, dim=0), torch.cat(all_targets, dim=0)

    # NSF doesn't use the per-iteration training loop; provide a no-op
    # so anything that calls it doesn't crash.
    def _train_iter(self, batch):
        return {"train_cls_loss": torch.tensor(0.0, device=self.device)}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, tag):
        state = {
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "best_performance": self.best_performance,
        }
        save_checkpoint(state, os.path.join(self.log_path, tag))

    def load_checkpoint(self, tag):
        checkpoint = load_checkpoint(os.path.join(self.log_path, tag))
        self.model.load_state_dict(checkpoint["model"])
        self.best_performance = checkpoint["best_performance"]
        self.current_epoch = checkpoint["epoch"]
        print(
            log_msg(
                f"Loaded checkpoint from {os.path.join(self.log_path, tag)}",
                "INFO",
                self.logger,
            )
        )
