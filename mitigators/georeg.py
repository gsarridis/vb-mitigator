"""
GeoReg: Geometric Regularisation for Bias-Robust Representation Learning.

A fundamental, assumption-free training mechanism in the spirit of Spectral
Decoupling (Pezeshki et al., NeurIPS 2021). GeoReg adds two purely geometric
regularisers to the standard cross-entropy objective:

    L = CE(f(x), y)
        + alpha * ||logits||_2^2          (logit-norm penalty,  same as SD)
        + beta  * sum_{i != j} rho(h_i,h_j)^2   (feature decorrelation)
        + gamma * sum_i max(0, 1 - sigma(h_i))  (anti-collapse variance hinge)

where h are the penultimate features and rho is the Pearson correlation of
two feature dimensions across the batch.

Design rationale
----------------
The two terms attack the geometry of the network at *different* levels:

  - The logit-norm penalty (alpha) removes the implicit weight-decay-on-logits
    that CE's gradient asymmetry produces, so the optimiser does not over-rely
    on a single dominant predictive direction in *logit* space. This is the
    Spectral Decoupling intervention; setting alpha to 0 disables it.

  - The feature decorrelation term (beta) penalises redundancy in *feature*
    space. A representation whose dimensions are mutually decorrelated cannot
    have its predictive content concentrated in a single direction, because
    that direction would be redundant with all the others projecting onto it.
    The form is the off-diagonal squared Pearson correlation, scale-invariant
    under feature rescaling (so it is robust to BN re-scaling and weight-decay
    drift). Setting beta to 0 disables it.

  - The variance hinge (gamma) is necessary because the decorrelation term
    can be trivially satisfied by zeroing some feature dimensions. The hinge
    keeps each per-batch feature standard deviation at or above 1, matching
    the variance term in VICReg (Bardes et al., 2022). Setting gamma to 0
    disables it (only safe when beta is also 0).

GeoReg makes a single, weak prior: useful representations are not maximally
redundant. It does not look at samples, groups, environments, bias attributes,
auxiliary models, or pseudo-labels of any kind.

Reference signposts
-------------------
  - Spectral Decoupling: Pezeshki et al. 2021 (the alpha term)
  - Barlow Twins: Zbontar et al. 2021 (the rho^2 form for SSL)
  - VICReg: Bardes et al. 2022 (the variance/covariance/invariance trio)
  - SD as integrated in vb-mitigator: mitigators/spectral_decouple.py
"""

import torch
import torch.nn as nn

from .base_trainer import BaseTrainer


# ---------------------------------------------------------------------------
# Regulariser primitives
# ---------------------------------------------------------------------------


def _logit_norm_penalty(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared logit value at the *true* class index. Identical to the
    expression in mitigators/spectral_decouple.py:

        ((outputs[range(N), targets]) ** 2).mean()

    This is the form used in the official Gradient Starvation reference
    implementation; it differs from the more general ||logits||_2^2 penalty
    only by a constant when classes are balanced, but it is the form that
    has been validated empirically.
    """
    n = logits.shape[0]
    return (logits[torch.arange(n, device=logits.device), targets] ** 2).mean()


def _feature_decorrelation_penalty(
    features: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Sum of squared off-diagonal entries of the per-batch feature
    correlation matrix.

    For a batch of features H in R^{N x D}, computes
        C[i, j] = corr(H[:, i], H[:, j])
    via column-centred and unit-stdev-rescaled features, then returns
        sum_{i != j} C[i, j]^2 / D
    where the 1/D normalisation keeps the term scale-comparable across
    different feature dimensionalities.
    """
    n, d = features.shape
    if n < 2:
        # Correlation is undefined for n=1 batches; skip silently.
        return torch.zeros((), device=features.device, dtype=features.dtype)

    # Column-centre and unit-rescale (Pearson correlation = cov of z-scored
    # features). We use unbiased=False (denominator N) for stability.
    h = features - features.mean(dim=0, keepdim=True)
    std = h.std(dim=0, keepdim=True, unbiased=False).clamp(min=eps)
    h = h / std

    # Correlation matrix.
    corr = (h.T @ h) / max(n, 1)  # shape: [D, D]

    # Off-diagonal squared sum, normalised by D so the magnitude does not
    # explode with feature width.
    off_diag = corr - torch.diag(torch.diagonal(corr))
    return (off_diag.pow(2).sum()) / max(d, 1)


def _variance_hinge_penalty(
    features: torch.Tensor, target_std: float = 1.0, eps: float = 1e-6
) -> torch.Tensor:
    """Hinge that pushes every feature dimension's per-batch standard
    deviation up to `target_std`. Matches the variance term of VICReg.

    Returns:
        mean over feature dims of relu(target_std - std_d).
    """
    if features.shape[0] < 2:
        return torch.zeros((), device=features.device, dtype=features.dtype)
    std = features.std(dim=0, unbiased=False).clamp(min=eps)
    return torch.relu(target_std - std).mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class GeoRegTrainer(BaseTrainer):
    """
    Trainer implementing GeoReg in the vb-mitigator framework.

    Configuration (under ``cfg.MITIGATOR.GEOREG``):
      - ``ALPHA``: weight on the logit-norm penalty (Spectral Decoupling).
                   Default 0.0 (disabled). Typical range: 1e-3 to 1e-1.
      - ``BETA``:  weight on the feature decorrelation penalty.
                   Default 1e-2. Typical range: 1e-3 to 1e-1.
      - ``GAMMA``: weight on the anti-collapse variance hinge.
                   Default 1.0 (matches VICReg). Typical range: 0.1 to 10.0.
      - ``TARGET_STD``: target standard deviation for the variance hinge.
                   Default 1.0 (matches VICReg).
      - ``WARMUP_EPOCHS``: ramp the regularisation weights linearly from 0
                   to their full value over this many epochs. Default 0
                   (no warmup). Useful when training is unstable in the
                   first few epochs because the initial features are far
                   from the geometric constraints.

    Implementation notes
    --------------------
    Like SpectralDecoupleTrainer, this trainer overrides ``_setup_optimizer``
    to set ``weight_decay=0`` on the optimiser; the GeoReg penalties take the
    place of the implicit L2 regularisation that weight decay normally
    provides. If you want to keep weight decay alongside GeoReg, set
    ``MITIGATOR.GEOREG.KEEP_WEIGHT_DECAY = True`` in your config.

    The trainer requires the model's forward to return ``(logits, features)``
    where ``features`` are the penultimate (pre-classifier) representations.
    Every model in vb-mitigator already follows this contract.
    """

    def _setup_optimizer(self):
        # Decouple the GeoReg penalty from L2 weight decay, mirroring SD.
        keep_wd = getattr(self.cfg.MITIGATOR.GEOREG, "KEEP_WEIGHT_DECAY", False)
        wd = self.cfg.SOLVER.WEIGHT_DECAY if keep_wd else 0.0

        if self.cfg.SOLVER.TYPE == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=wd,
            )
        elif self.cfg.SOLVER.TYPE in ("Adam", "AdamW"):
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.SOLVER.LR,
                weight_decay=wd,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.cfg.SOLVER.TYPE}")

    def _method_specific_setups(self):
        gcfg = self.cfg.MITIGATOR.GEOREG
        self.alpha = float(gcfg.ALPHA)
        self.beta = float(gcfg.BETA)
        self.gamma = float(gcfg.GAMMA)
        self.target_std = float(gcfg.TARGET_STD)
        self.warmup_epochs = int(gcfg.WARMUP_EPOCHS)

        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError(
                "GeoReg weights must be non-negative "
                f"(got ALPHA={self.alpha}, BETA={self.beta}, GAMMA={self.gamma})."
            )
        if self.beta > 0 and self.gamma == 0:
            self.logger.warning(
                "GeoReg: BETA > 0 with GAMMA = 0 risks feature-dimension "
                "collapse. Consider setting GAMMA = 1.0."
            )

        # Warn loudly if BatchNorm is used, since BN-output features can
        # interact strangely with both the decorrelation and variance terms
        # (BN already enforces unit variance per dim, partially trivialising
        # the gamma term and stabilising the beta term in non-obvious ways).
        bn_modules = [
            m
            for m in self.model.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        ]
        if bn_modules:
            self.logger.info(
                f"GeoReg: model contains {len(bn_modules)} BatchNorm modules. "
                "If features are BN outputs, consider gamma=0 (variance is "
                "already controlled) and beta tuning may need to be lower."
            )

    def _current_warmup_scale(self) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        # current_epoch is 1-indexed in BaseTrainer at training time.
        ep = max(0, getattr(self, "current_epoch", 0) - 1)
        return min(1.0, ep / float(self.warmup_epochs))

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            logits, features = outputs
        else:
            raise RuntimeError(
                "GeoReg requires the model's forward() to return "
                "(logits, penultimate_features). The current model "
                f"({type(self.model).__name__}) returned a single tensor."
            )

        # If features come from a conv block (NCHW), pool to (N, D) so the
        # statistics are per feature *channel*, not per spatial location.
        if features.dim() > 2:
            features = features.flatten(start_dim=2).mean(dim=2)

        ce_loss = self.criterion(logits, targets)

        scale = self._current_warmup_scale()

        if self.alpha > 0:
            logit_pen = _logit_norm_penalty(logits, targets)
        else:
            logit_pen = torch.zeros((), device=self.device)

        if self.beta > 0:
            decorr_pen = _feature_decorrelation_penalty(features)
        else:
            decorr_pen = torch.zeros((), device=self.device)

        if self.gamma > 0:
            var_pen = _variance_hinge_penalty(features, target_std=self.target_std)
        else:
            var_pen = torch.zeros((), device=self.device)

        loss = (
            ce_loss
            + scale * self.alpha * logit_pen
            + scale * self.beta * decorr_pen
            + scale * self.gamma * var_pen
        )

        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()

        return {
            "train_cls_loss": ce_loss.detach(),
            "train_logit_pen": logit_pen.detach(),
            "train_decorr_pen": decorr_pen.detach(),
            "train_var_pen": var_pen.detach(),
        }
