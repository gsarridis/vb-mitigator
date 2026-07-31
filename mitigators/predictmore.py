"""
PredictMore: Multi-task self-supervised auxiliary objectives as a
fundamental, assumption-free bias-mitigation mechanism.

The diagnosis this method addresses
-----------------------------------
On a biased dataset, shortcut-reliant solutions genuinely produce lower
training cross-entropy than causal solutions. The optimiser is not being
"lazy"; it is correctly minimising its single objective, and that
objective rewards shortcut reliance. Standard regularisers (weight decay,
SD, decorrelation, margin gates, ...) reshape the loss landscape locally
but do not change which solution has the lowest training loss in the
limit. Therefore none of them, on their own, can be guaranteed to push
the optimiser away from shortcuts.

PredictMore changes the *objective* itself. Alongside CE on the main
classification target, we add a small set of self-supervised auxiliary
heads that read the SAME penultimate features as the classifier. Each
auxiliary head must solve a different prediction problem from those
shared features, with targets generated automatically from the input
(rotation index, masked feature dimensions, jigsaw permutation). A
representation that consists only of "background colour -> waterbird
species" cannot also predict rotation angle, cannot reconstruct masked
feature dimensions, cannot identify a shuffled patch permutation. A
representation rich enough to do all of those tasks *and* the supervised
task is structurally pushed toward features that capture the actual
content of the input.

Crucially, the auxiliary tasks are not designed to target known biases.
They are a generic battery whose unifying property is that they require
representational *capacity* beyond what any single shortcut provides.

Objective
---------
    L = CE(f_cls(h(x)), y)
        + lambda_rot     * scale * L_rotation(g_rot(h(x_rot)), tau_rot(x))
        + lambda_mask    * scale * L_mask_recon(g_recon(mask(h(x))), h(x))
        + lambda_jigsaw  * scale * L_jigsaw(g_jig(h(x_jig)), tau_jig(x))

`scale` is a linear warmup ramp from 0 to 1 over `WARMUP_EPOCHS`. Targets
are generated on the fly from each input, no extra annotations required.

What PredictMore does NOT do
----------------------------
- No bias attribute labels, environments, or groups.
- No auxiliary biased model, no clustering, no pseudo-labels.
- No discovery phase.
- No assumption about what the shortcut is.
"""

from __future__ import annotations

import itertools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_trainer import BaseTrainer

# ---------------------------------------------------------------------------
# Auxiliary task primitives
# ---------------------------------------------------------------------------


def _rotate_batch_4way(x: torch.Tensor) -> tuple:
    """Generate a 4-way rotation prediction batch.

    Given an input batch x of shape [N, C, H, W] (or [N, *]), returns:
      x_rot: [4N, C, H, W]  -- the four rotations of every sample
      r_targets: [4N]       -- rotation index in {0, 1, 2, 3}

    Rotations are 0/90/180/270 degrees, achieved via torch.rot90 on the
    last two dimensions (so this requires 4D tensors -- we guard against
    non-image inputs in the trainer).
    """
    if x.dim() != 4:
        raise ValueError(
            f"rotation aux task requires 4D image tensors (NCHW); "
            f"got shape {tuple(x.shape)}."
        )
    rotated = []
    targets = []
    for k in range(4):
        rotated.append(torch.rot90(x, k=k, dims=(2, 3)))
        targets.append(torch.full((x.shape[0],), k, device=x.device, dtype=torch.long))
    return torch.cat(rotated, dim=0), torch.cat(targets, dim=0)


class _MaskedFeatureReconHead(nn.Module):
    """Predict masked-out feature dimensions from the unmasked ones.

    Given a feature vector h in R^D, randomly select a fraction `p_mask`
    of dimensions to zero out, then ask an MLP to reconstruct the FULL
    feature vector h. The loss is MSE on the full reconstruction.

    This implementation does the masking *inside* the head, so the
    encoder can be shared transparently between the main task and the
    aux task. The mask is regenerated on each forward call.
    """

    def __init__(self, feat_dim: int, p_mask: float = 0.5, hidden_mult: int = 2):
        super().__init__()
        if not (0.0 < p_mask < 1.0):
            raise ValueError(f"p_mask must be in (0, 1), got {p_mask}.")
        self.feat_dim = feat_dim
        self.p_mask = p_mask
        hidden = max(feat_dim, hidden_mult * feat_dim)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, features: torch.Tensor) -> tuple:
        n, d = features.shape
        # Bernoulli mask, sampled independently per (sample, dim).
        keep = (torch.rand(n, d, device=features.device) > self.p_mask).float()
        masked_features = features * keep
        recon = self.net(masked_features)
        # Loss target: reconstruct the FULL features (detached: we want
        # the encoder to be shaped by the recon objective, not the head
        # to chase a moving target).
        target = features.detach()
        # Only score the masked-out positions, otherwise the MLP can
        # trivially copy through the unmasked ones.
        score_mask = 1.0 - keep
        if score_mask.sum() < 1:
            # Edge case: no positions masked. Return zero loss.
            return recon, torch.zeros((), device=features.device)
        loss = ((recon - target) ** 2 * score_mask).sum() / score_mask.sum()
        return recon, loss


def _make_jigsaw_permutations(num_permutations: int, n_patches: int, seed: int = 0):
    """Sample `num_permutations` distinct permutations of `n_patches`
    patches, including the identity. Returns a tensor of shape
    [num_permutations, n_patches] with int64 entries.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    perms = [torch.arange(n_patches)]  # identity = class 0
    seen = {tuple(perms[0].tolist())}
    while len(perms) < num_permutations:
        perm = torch.randperm(n_patches, generator=g)
        key = tuple(perm.tolist())
        if key in seen:
            continue
        seen.add(key)
        perms.append(perm)
    return torch.stack(perms, dim=0)


def _apply_jigsaw(x: torch.Tensor, perms: torch.Tensor, grid: int) -> tuple:
    """Apply a random jigsaw permutation to each image in x.

    Args:
        x: [N, C, H, W]
        perms: [K, n_patches] of int64. Each row is one permutation; row 0
               is the identity.
        grid: integer side length of the patch grid (e.g. grid=3 -> 9 patches).

    Returns:
        x_jig: [N, C, H, W] permuted images.
        targets: [N] permutation index in [0, K).
    """
    if x.dim() != 4:
        raise ValueError(
            f"jigsaw aux task requires 4D image tensors (NCHW); "
            f"got shape {tuple(x.shape)}."
        )
    n, c, h, w = x.shape
    n_patches = grid * grid
    if perms.shape[1] != n_patches:
        raise ValueError(
            f"perms has {perms.shape[1]} entries but grid={grid} expects "
            f"{n_patches}."
        )
    if h % grid != 0 or w % grid != 0:
        # Centre-crop to the largest multiple of `grid` that fits.
        new_h = (h // grid) * grid
        new_w = (w // grid) * grid
        off_h = (h - new_h) // 2
        off_w = (w - new_w) // 2
        x = x[:, :, off_h : off_h + new_h, off_w : off_w + new_w]
        h, w = new_h, new_w

    ph, pw = h // grid, w // grid

    # Sample one permutation per image.
    targets = torch.randint(0, perms.shape[0], (n,), device=x.device)
    chosen_perms = perms.to(x.device)[targets]  # [N, n_patches]

    # Slice into patches: [N, C, grid, ph, grid, pw] -> [N, C, n_patches, ph, pw]
    patches = x.reshape(n, c, grid, ph, grid, pw)
    patches = patches.permute(0, 1, 2, 4, 3, 5).reshape(n, c, n_patches, ph, pw)

    # Apply per-image permutation.
    idx = chosen_perms.unsqueeze(1).expand(n, c, n_patches).unsqueeze(-1).unsqueeze(-1)
    idx = idx.expand(n, c, n_patches, ph, pw)
    perm_patches = torch.gather(patches, 2, idx)

    # Stitch back: reverse the slicing.
    perm_patches = perm_patches.reshape(n, c, grid, grid, ph, pw)
    perm_patches = perm_patches.permute(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)
    return perm_patches, targets


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PredictMoreTrainer(BaseTrainer):
    """
    Trainer implementing PredictMore in the vb-mitigator framework.

    Configuration (under ``cfg.MITIGATOR.PREDICTMORE``):
      - ``AUX_TASKS``:     list of aux tasks to enable. Subset of
                           ["rotation", "masked_recon", "jigsaw"].
                           Default ["rotation", "masked_recon"].
      - ``LAMBDA_ROT``:    weight on the rotation prediction loss. Default 0.1.
      - ``LAMBDA_MASK``:   weight on masked-feature reconstruction. Default 0.1.
      - ``LAMBDA_JIGSAW``: weight on jigsaw permutation prediction. Default 0.1.
      - ``MASK_PROB``:     fraction of feature dims to mask per sample. Default 0.5.
      - ``JIGSAW_GRID``:   side length of patch grid. Default 3 -> 9 patches.
      - ``JIGSAW_PERMS``:  number of permutation classes. Default 24.
      - ``AUX_FRACTION``:  fraction of each batch on which to compute aux
                           losses (compute control). Default 0.5.
      - ``WARMUP_EPOCHS``: linear ramp of aux loss weights from 0 to 1.
                           Default 1.

    Modality compatibility
    ----------------------
    rotation and jigsaw require 4D image tensors (NCHW). masked_recon
    works for any encoder that produces a feature vector. The trainer
    raises a clear error if you enable a task that doesn't fit.

    Architecture
    ------------
    Aux heads share the encoder with the classifier. The classifier
    receives gradient only from CE; the encoder receives gradient from
    CE + aux losses. This is automatic since aux heads have their own
    parameters and do not touch the classifier head's weights.
    """

    def _setup_criterion(self):
        if self.cfg.SOLVER.CRITERION == "CE":
            self.criterion_train = nn.CrossEntropyLoss()
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {self.cfg.SOLVER.CRITERION}")

    def _method_specific_setups(self):
        pcfg = self.cfg.MITIGATOR.PREDICTMORE
        self.aux_tasks: List[str] = list(pcfg.AUX_TASKS)
        self.lambda_rot = float(pcfg.LAMBDA_ROT)
        self.lambda_mask = float(pcfg.LAMBDA_MASK)
        self.lambda_jigsaw = float(pcfg.LAMBDA_JIGSAW)
        self.mask_prob = float(pcfg.MASK_PROB)
        self.jigsaw_grid = int(pcfg.JIGSAW_GRID)
        self.jigsaw_perms_n = int(pcfg.JIGSAW_PERMS)
        self.aux_fraction = float(pcfg.AUX_FRACTION)
        self.warmup_epochs = int(pcfg.WARMUP_EPOCHS)

        valid_tasks = {"rotation", "masked_recon", "jigsaw"}
        unknown = set(self.aux_tasks) - valid_tasks
        if unknown:
            raise ValueError(
                f"Unknown aux tasks {sorted(unknown)}; valid options are "
                f"{sorted(valid_tasks)}."
            )
        if not (0.0 < self.aux_fraction <= 1.0):
            raise ValueError(
                f"AUX_FRACTION must be in (0, 1], got {self.aux_fraction}."
            )

        # Probe the model to discover the feature dimensionality.
        feat_dim = self._probe_feature_dim()

        self.aux_heads = nn.ModuleDict()
        if "rotation" in self.aux_tasks:
            self.aux_heads["rotation"] = nn.Linear(feat_dim, 4).to(self.device)
        if "masked_recon" in self.aux_tasks:
            self.aux_heads["masked_recon"] = _MaskedFeatureReconHead(
                feat_dim, p_mask=self.mask_prob
            ).to(self.device)
        if "jigsaw" in self.aux_tasks:
            self.aux_heads["jigsaw"] = nn.Linear(feat_dim, self.jigsaw_perms_n).to(
                self.device
            )
            # Pre-sample the permutation set, deterministic per seed.
            self._jigsaw_perms = _make_jigsaw_permutations(
                self.jigsaw_perms_n,
                self.jigsaw_grid * self.jigsaw_grid,
                seed=int(getattr(self.cfg.EXPERIMENT, "SEED", 0)),
            )

        # Add aux head parameters to the optimiser. We rebuild the
        # optimiser to include them; this matches how SpectralDecouple's
        # _setup_optimizer is overridden cleanly.
        self._add_aux_params_to_optimizer()

        self.logger.info(
            f"PredictMore: enabled aux tasks {self.aux_tasks} with "
            f"weights rot={self.lambda_rot}, mask={self.lambda_mask}, "
            f"jigsaw={self.lambda_jigsaw}; aux_fraction={self.aux_fraction}, "
            f"feat_dim={feat_dim}"
        )

    def _probe_feature_dim(self) -> int:
        """Discover the encoder's feature dimensionality by running one
        zero forward through the model. We use a single-sample probe.
        """
        # Try to construct a probe input; if the dataset has a known input
        # shape, use it. Otherwise pull one batch.
        sample = next(iter(self.dataloaders["train"]))
        x = sample["inputs"][:1].to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
        if was_training:
            self.model.train()
        if not isinstance(outputs, tuple):
            raise RuntimeError(
                "PredictMore requires the model's forward() to return "
                "(logits, penultimate_features). The current model "
                f"({type(self.model).__name__}) returned a single tensor."
            )
        _, features = outputs
        if features.dim() > 2:
            features = features.flatten(start_dim=2).mean(dim=2)
        return int(features.shape[-1])

    def _add_aux_params_to_optimizer(self):
        """Append aux-head parameters to the existing optimiser as a new
        param group, so they share the same LR / WD as the model.
        """
        if len(self.aux_heads) == 0:
            return
        aux_params = list(self.aux_heads.parameters())
        if len(aux_params) == 0:
            return
        # Inherit hyperparameters from the model param group.
        base = self.optimizer.param_groups[0]
        new_group = {k: v for k, v in base.items() if k != "params"}
        new_group["params"] = aux_params
        self.optimizer.add_param_group(new_group)

    def _current_warmup_scale(self) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        ep = max(0, getattr(self, "current_epoch", 0) - 1)
        return min(1.0, ep / float(self.warmup_epochs))

    def _maybe_pool(self, features: torch.Tensor) -> torch.Tensor:
        """Aux heads operate on a (N, D) feature vector. If the encoder
        emits spatial features (NCHW), global-average-pool them.
        """
        if features.dim() > 2:
            return features.flatten(start_dim=2).mean(dim=2)
        return features

    def _aux_subsample(self, x: torch.Tensor):
        """Take the first `aux_fraction` of the batch for aux losses.
        Random subsampling would be more honest but adds noise; the
        first slice is a fine deterministic choice when the loader
        already shuffles.
        """
        n_take = max(1, int(round(x.shape[0] * self.aux_fraction)))
        return x[:n_take]

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        self.optimizer.zero_grad()

        # ---- Main classification ----
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs
        ce_loss = self.criterion_train(logits, targets)

        scale = self._current_warmup_scale()
        log_dict = {"train_cls_loss": ce_loss.detach()}

        total_aux_loss = torch.zeros((), device=self.device)

        # ---- Aux: rotation prediction ----
        if "rotation" in self.aux_tasks and self.lambda_rot > 0:
            x_sub = self._aux_subsample(inputs)
            x_rot, r_tgt = _rotate_batch_4way(x_sub)
            out_rot = self.model(x_rot)
            if isinstance(out_rot, tuple):
                _, feats_rot = out_rot
            else:
                feats_rot = out_rot
            feats_rot = self._maybe_pool(feats_rot)
            logits_rot = self.aux_heads["rotation"](feats_rot)
            rot_loss = F.cross_entropy(logits_rot, r_tgt)
            total_aux_loss = total_aux_loss + scale * self.lambda_rot * rot_loss
            log_dict["train_rot_loss"] = rot_loss.detach()

        # ---- Aux: masked feature reconstruction ----
        if "masked_recon" in self.aux_tasks and self.lambda_mask > 0:
            x_sub = self._aux_subsample(inputs)
            out_m = self.model(x_sub)
            if isinstance(out_m, tuple):
                _, feats_m = out_m
            else:
                feats_m = out_m
            feats_m = self._maybe_pool(feats_m)
            _, mask_loss = self.aux_heads["masked_recon"](feats_m)
            total_aux_loss = total_aux_loss + scale * self.lambda_mask * mask_loss
            log_dict["train_mask_loss"] = mask_loss.detach()

        # ---- Aux: jigsaw permutation prediction ----
        if "jigsaw" in self.aux_tasks and self.lambda_jigsaw > 0:
            x_sub = self._aux_subsample(inputs)
            x_jig, j_tgt = _apply_jigsaw(x_sub, self._jigsaw_perms, self.jigsaw_grid)
            out_jig = self.model(x_jig)
            if isinstance(out_jig, tuple):
                _, feats_jig = out_jig
            else:
                feats_jig = out_jig
            feats_jig = self._maybe_pool(feats_jig)
            logits_jig = self.aux_heads["jigsaw"](feats_jig)
            jig_loss = F.cross_entropy(logits_jig, j_tgt)
            total_aux_loss = total_aux_loss + scale * self.lambda_jigsaw * jig_loss
            log_dict["train_jigsaw_loss"] = jig_loss.detach()

        loss = ce_loss + total_aux_loss
        log_dict["train_total_loss"] = loss.detach()
        log_dict["train_aux_scale"] = torch.tensor(scale)

        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()

        return log_dict
