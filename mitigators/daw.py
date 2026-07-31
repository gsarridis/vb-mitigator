"""
DAW: Density-Aware Weighting for bias-robust training.

Diagnosis
---------
On a biased dataset, a mini-batch is dominated numerically by bias-aligned
samples. Because aligned samples produce similar gradients (they share
the spurious feature), they constructively interfere; conflicting samples
are heterogeneous and partially cancel out. The optimiser sees a strong
consistent push from the aligned samples and a weak noisy push from the
conflicting ones, even though the conflicting samples are exactly the
ones that carry the causal signal.

Mechanism
---------
DAW computes a per-sample density estimate within the batch (in some
"signature" space) and downweights samples in dense regions. A sample's
contribution to the loss becomes inversely proportional to how many
similar samples are around it. Concretely:

    1) For each sample i, compute a signature s_i in some space
       (per-sample gradient, features, logits, ...).
    2) Compute pairwise similarities/distances between signatures.
    3) Estimate density rho_i for every sample (KDE, k-NN, or softmax).
    4) Weight sample i by w_i ~ 1 / rho_i^alpha (with stop_grad), then
       renormalise so sum(w_i) = N.
    5) Backprop the weighted CE.

The weights are fully detached from the autograd graph -- otherwise the
model could game them by making its gradients more diverse.

What DAW does NOT do
--------------------
- No bias attribute labels, environments, or groups.
- No auxiliary biased model, no clustering, no pseudo-labels.
- No per-sample identification of "easy" or "hard."
- No assumption that easy = spurious.

The only prior DAW imports is: a sample's gradient should not be
amplified by redundancy with other samples in the same batch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .base_trainer import BaseTrainer

# ---------------------------------------------------------------------------
# Signature extractors
# ---------------------------------------------------------------------------


def _signature_logit_gradient(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Per-sample gradient of CE w.r.t. logits.

    For sample i with logits z_i and label y_i, this is exactly
        d CE / d z_i = softmax(z_i) - one_hot(y_i)
    a vector in R^C. Cheap, dimension == num_classes.
    """
    p = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(targets, num_classes=num_classes).float()
    return (p - one_hot).detach()  # [N, C]


def _signature_last_layer_gradient(
    logits: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Per-sample gradient of CE w.r.t. last linear layer weights.

    The last-layer gradient is (softmax(z_i) - one_hot(y_i)) outer h_i,
    a [C, D] matrix per sample. We flatten to a [C*D] vector and detach.

    Memory: N * C * D floats. For ResNet-50 (D=2048) on a 2-class problem
    that's N * 4096 floats per batch -- fine.
    """
    p = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(targets, num_classes=num_classes).float()
    err = (p - one_hot).detach()  # [N, C]
    h = features.detach()  # [N, D]
    # Outer product per sample: [N, C, D] -> [N, C*D].
    sig = err.unsqueeze(-1) * h.unsqueeze(1)  # [N, C, D]
    return sig.reshape(sig.shape[0], -1)  # [N, C*D]


def _signature_full_gradient(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Per-sample gradient of CE w.r.t. ALL model parameters.

    Computed via torch.func.vmap(grad(...)). Returns a [N, P] tensor
    where P is the total parameter count. This is expensive: O(N * P)
    memory. For most models you will run out of memory unless N is
    small or you reduce signature dimensionality afterwards. It is
    included as the principled option; in practice last_layer_gradient
    is usually a fine approximation.

    Important: the per-sample gradient computation requires the model's
    forward to be re-runnable with functional parameters. We use
    torch.func.functional_call which works for any standard PyTorch
    module, but care must be taken with BatchNorm (running stats are
    not handled by functional_call). The trainer warns about this.
    """
    # Snapshot params and buffers as dicts for functional_call.
    params = {name: p.detach() for name, p in model.named_parameters()}
    buffers = {name: b.detach() for name, b in model.named_buffers()}

    def loss_one(p_dict, x_one, y_one):
        x_one = x_one.unsqueeze(0)
        y_one = y_one.unsqueeze(0)
        out = functional_call(model, (p_dict, buffers), (x_one,))
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        return criterion(logits, y_one)

    grad_fn = grad(loss_one, argnums=0)
    # vmap over the batch dimension of inputs and targets.
    per_sample_grads = vmap(grad_fn, in_dims=(None, 0, 0))(params, inputs, targets)
    # Concatenate parameter-wise gradients into a single [N, P] vector.
    flats = []
    for name in sorted(per_sample_grads.keys()):
        g = per_sample_grads[name]
        flats.append(g.reshape(g.shape[0], -1))
    return torch.cat(flats, dim=1).detach()  # [N, P]


def _signature_feature(features: torch.Tensor) -> torch.Tensor:
    return features.detach()


def _signature_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.detach()


def _signature_loss_scalar(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-sample CE as a scalar signature. Mostly a baseline / sanity check."""
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    return per_sample.detach().unsqueeze(-1)  # [N, 1]


# ---------------------------------------------------------------------------
# Similarity kernels
# ---------------------------------------------------------------------------


def _pairwise_distances(sig: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances between rows of sig. [N, N]."""
    # Numerically stable squared-distance computation.
    sq = (sig * sig).sum(dim=-1, keepdim=True)  # [N, 1]
    d2 = sq + sq.t() - 2.0 * (sig @ sig.t())  # [N, N]
    d2 = d2.clamp(min=0.0)  # numerics
    return d2.sqrt()


def _median_bandwidth(distances: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Median of off-diagonal pairwise distances, the standard
    bandwidth heuristic for Gaussian kernels.
    """
    n = distances.shape[0]
    if n < 2:
        return torch.tensor(1.0, device=distances.device)
    # Mask out diagonal.
    mask = ~torch.eye(n, dtype=torch.bool, device=distances.device)
    off_diag = distances[mask]
    med = off_diag.median()
    return torch.clamp(med, min=eps)


def _kernel_gaussian(distances: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.exp(-(distances**2) / (2.0 * sigma**2 + 1e-12))


def _kernel_cosine(sig: torch.Tensor) -> torch.Tensor:
    """Cosine similarity scaled to [0, 1]."""
    sig_n = F.normalize(sig, dim=-1, eps=1e-8)
    cos = sig_n @ sig_n.t()
    return (cos + 1.0) * 0.5  # [-1, 1] -> [0, 1]


def _kernel_inverse_distance(
    distances: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    return 1.0 / (distances + eps)


# ---------------------------------------------------------------------------
# Density estimators
# ---------------------------------------------------------------------------


def _density_kde(similarities: torch.Tensor) -> torch.Tensor:
    """KDE: rho_i = sum_{j != i} K(s_i, s_j).

    Diagonal is excluded (a sample is not its own neighbour).
    """
    n = similarities.shape[0]
    eye = torch.eye(n, device=similarities.device, dtype=similarities.dtype)
    return (similarities * (1.0 - eye)).sum(dim=1)  # [N]


def _density_knn(distances: torch.Tensor, k: int) -> torch.Tensor:
    """k-NN density: rho_i = 1 / (mean distance to k nearest neighbours).

    Diagonal is excluded by setting it to +inf.
    """
    n = distances.shape[0]
    if n <= 1:
        return torch.ones(n, device=distances.device)
    d = distances.clone()
    d.fill_diagonal_(float("inf"))
    k_eff = min(max(k, 1), n - 1)
    # k smallest distances per row (sorted ascending).
    nn_dists, _ = torch.topk(d, k_eff, dim=1, largest=False)
    mean_nn = nn_dists.mean(dim=1) + 1e-8
    return 1.0 / mean_nn


def _density_softmax(
    similarities: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Smooth maximum: rho_i = T * log sum_{j != i} exp(K(s_i, s_j) / T).

    Less sensitive to outliers than plain KDE.
    """
    n = similarities.shape[0]
    eye = torch.eye(n, device=similarities.device, dtype=torch.bool)
    sims_no_diag = similarities.masked_fill(eye, float("-inf"))
    return temperature * torch.logsumexp(sims_no_diag / temperature, dim=1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class DAWTrainer(BaseTrainer):
    """
    Trainer implementing DAW (Density-Aware Weighting).

    Configuration (under ``cfg.MITIGATOR.DAW``):
      - ``DENSITY_SOURCE``: signature space. One of:
          * "logit_gradient"      -- d CE / d logits, R^C  (default, cheap)
          * "last_layer_gradient" -- d CE / d (W_last), R^{C*D}
          * "full_gradient"       -- d CE / d (all params), R^P (expensive)
          * "feature"             -- penultimate features
          * "logits"              -- raw logits
          * "loss_scalar"         -- per-sample CE (1-D, baseline)
      - ``KERNEL``: similarity kernel.
          * "gaussian"          -- median-bandwidth Gaussian (default)
          * "cosine"            -- cosine similarity
          * "inverse_distance"  -- 1 / (d + eps)
      - ``DENSITY_ESTIMATOR``: how similarities aggregate.
          * "knn"               -- 1 / mean k-NN distance (default;
                                   recommended). Counts neighbours; not
                                   confounded by within-mode spread.
          * "kde"               -- sum of similarities. Conflates cluster
                                   *count* with cluster *concentration*:
                                   a tightly-clustered group of size N
                                   gets MUCH higher density than a
                                   spread-out group of the same size.
                                   This means kNN > KDE for our use case
                                   (we want to balance group count, not
                                   group concentration). KDE included
                                   for completeness / ablation.
          * "softmax"           -- temperature-smoothed max
      - ``ALPHA``: aggressiveness in (0, 2]. Default 0.5.
                   w_i ~ 1 / rho_i^alpha. ALPHA=0 recovers ERM.
      - ``KNN_K``: neighbours for "knn" estimator. Default 5.
      - ``SOFTMAX_T``: temperature for "softmax" estimator. Default 1.0.
      - ``BANDWIDTH``: "median" (default) or a fixed float for Gaussian.
      - ``WARMUP_EPOCHS``: pure-CE epochs before density weighting kicks
                           in (so the model has meaningful signatures).
                           Default 1.
      - ``RENORMALIZE``: True (sum w_i = N, preserves effective LR) or
                           False (raw weights). Default True.
      - ``WEIGHT_FLOOR``: minimum per-sample weight, applied after
                           normalisation. Default 0.0 (no floor).
                           Set to e.g. 0.1 to prevent any sample from
                           being completely silenced.
      - ``WEIGHT_CEILING``: maximum per-sample weight, applied after
                           normalisation. Default 0.0 (no ceiling).
                           Useful to bound the influence of isolated
                           outliers (e.g. mislabeled samples) which
                           density-weighting would otherwise upweight.
      - ``LOG_PER_GROUP``: if True (default), log per-(target, bias)
                           group statistics: count, mean weight, sum
                           weight, mean density. Uses ``self.biases``
                           (the bias attribute names from the dataset)
                           which are available for diagnostics. The
                           bias attributes are NEVER used for the loss
                           computation; they only influence what gets
                           logged.

    Important note on ``full_gradient``
    -----------------------------------
    Computing per-sample gradients via torch.func.vmap requires
    re-running the model forward in functional mode. This works for most
    standard modules but BatchNorm running statistics are NOT updated
    during the vmap pass, which introduces a tiny train/eval mismatch.
    For best results with full_gradient, prefer models without BN or
    accept a small approximation. last_layer_gradient is usually a fine
    drop-in replacement and is much cheaper.
    """

    def _setup_criterion(self):
        if self.cfg.SOLVER.CRITERION == "CE":
            self.criterion_train = nn.CrossEntropyLoss(reduction="none")
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {self.cfg.SOLVER.CRITERION}")

    def _method_specific_setups(self):
        dcfg = self.cfg.MITIGATOR.DAW
        self.density_source = str(dcfg.DENSITY_SOURCE)
        self.kernel = str(dcfg.KERNEL)
        self.density_estimator = str(dcfg.DENSITY_ESTIMATOR)
        self.alpha = float(dcfg.ALPHA)
        self.knn_k = int(dcfg.KNN_K)
        self.softmax_t = float(dcfg.SOFTMAX_T)
        self.bandwidth_cfg = dcfg.BANDWIDTH  # str "median" or float
        self.warmup_epochs = int(dcfg.WARMUP_EPOCHS)
        self.renormalize = bool(dcfg.RENORMALIZE)
        self.weight_floor = float(dcfg.WEIGHT_FLOOR)
        self.weight_ceiling = float(dcfg.WEIGHT_CEILING)
        self.log_per_group = bool(getattr(dcfg, "LOG_PER_GROUP", True))

        # Per-group logging is best-effort: we use the dataset's bias
        # attributes (self.biases, self.num_class) only for diagnostics.
        # If self.biases is empty (e.g. a dataset without bias annotations),
        # per-group logging is silently disabled.
        self._can_log_groups = (
            self.log_per_group
            and hasattr(self, "biases")
            and len(getattr(self, "biases", [])) > 0
            and hasattr(self, "num_class")
        )
        if self._can_log_groups:
            # Number of joint bias-attribute classes per (y) is
            # num_group / num_class, identical to GroupDRO's encoding.
            num_group = getattr(self, "num_group", None)
            if num_group is not None and self.num_class > 0:
                self._num_attrs_per_class = max(
                    1, int(num_group) // int(self.num_class)
                )
            else:
                # Fallback: assume binary biases, take cardinality from
                # the actual values seen in the first batch (computed
                # lazily below).
                self._num_attrs_per_class = None
        else:
            self._num_attrs_per_class = None

        valid_sources = {
            "logit_gradient",
            "last_layer_gradient",
            "full_gradient",
            "feature",
            "logits",
            "loss_scalar",
        }
        if self.density_source not in valid_sources:
            raise ValueError(
                f"Unknown DENSITY_SOURCE {self.density_source!r}; valid: "
                f"{sorted(valid_sources)}."
            )
        if self.kernel not in {"gaussian", "cosine", "inverse_distance"}:
            raise ValueError(f"Unknown KERNEL {self.kernel!r}.")
        if self.density_estimator not in {"kde", "knn", "softmax"}:
            raise ValueError(f"Unknown DENSITY_ESTIMATOR {self.density_estimator!r}.")
        if self.alpha < 0 or self.alpha > 2:
            raise ValueError(f"ALPHA must be in [0, 2]; got {self.alpha}.")
        if self.density_estimator == "knn" and self.kernel != "gaussian":
            self.logger.warning(
                "DAW: density_estimator='knn' uses Euclidean distance "
                "regardless of KERNEL setting; KERNEL is ignored."
            )
        if self.density_source == "full_gradient":
            bn_modules = [
                m
                for m in self.model.modules()
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            ]
            if bn_modules:
                self.logger.warning(
                    f"DAW + full_gradient: model has {len(bn_modules)} "
                    "BatchNorm modules. Per-sample gradients via vmap "
                    "will not update BN running stats during the gradient "
                    "computation pass; this is a known approximation. "
                    "Prefer last_layer_gradient if accuracy of the gradient "
                    "signature matters."
                )

        self.logger.info(
            f"DAW: source={self.density_source} kernel={self.kernel} "
            f"estimator={self.density_estimator} alpha={self.alpha} "
            f"renormalize={self.renormalize}"
        )

    def _gate_active(self) -> bool:
        ep = getattr(self, "current_epoch", 0)
        return ep > self.warmup_epochs and self.alpha > 0

    # ------------------------------------------------------------------
    # Signature, similarity, density
    # ------------------------------------------------------------------
    def _compute_signature(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.density_source == "logit_gradient":
            return _signature_logit_gradient(logits, targets, logits.shape[-1])
        if self.density_source == "last_layer_gradient":
            if features is None:
                raise RuntimeError(
                    "DAW(last_layer_gradient) requires the model to return "
                    "(logits, features). The current model returned only logits."
                )
            return _signature_last_layer_gradient(
                logits, features, targets, logits.shape[-1]
            )
        if self.density_source == "full_gradient":
            return _signature_full_gradient(self.model, inputs, targets, self.criterion)
        if self.density_source == "feature":
            if features is None:
                raise RuntimeError(
                    "DAW(feature) requires the model to return " "(logits, features)."
                )
            return _signature_feature(features)
        if self.density_source == "logits":
            return _signature_logits(logits)
        if self.density_source == "loss_scalar":
            return _signature_loss_scalar(logits, targets)
        raise RuntimeError(self.density_source)  # unreachable

    def _compute_similarities_or_distances(
        self, sig: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (similarities, distances) -- only the one(s) needed."""
        # Pool spatial features if signature is multi-dim per sample.
        if sig.dim() > 2:
            sig = sig.flatten(start_dim=1)

        if self.density_estimator == "knn":
            distances = _pairwise_distances(sig)
            return None, distances

        # KDE / softmax need similarities.
        if self.kernel == "gaussian":
            distances = _pairwise_distances(sig)
            if isinstance(self.bandwidth_cfg, str) and self.bandwidth_cfg == "median":
                sigma = _median_bandwidth(distances)
            else:
                sigma = torch.tensor(float(self.bandwidth_cfg), device=sig.device)
            similarities = _kernel_gaussian(distances, sigma)
            return similarities, distances
        if self.kernel == "cosine":
            return _kernel_cosine(sig), None
        if self.kernel == "inverse_distance":
            distances = _pairwise_distances(sig)
            return _kernel_inverse_distance(distances), distances
        raise RuntimeError(self.kernel)

    def _compute_density(
        self,
        similarities: Optional[torch.Tensor],
        distances: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.density_estimator == "kde":
            return _density_kde(similarities)
        if self.density_estimator == "knn":
            return _density_knn(distances, self.knn_k)
        if self.density_estimator == "softmax":
            return _density_softmax(similarities, temperature=self.softmax_t)
        raise RuntimeError(self.density_estimator)

    def _compute_weights(self, density: torch.Tensor) -> torch.Tensor:
        n = density.shape[0]
        # Stabilise: prevent division by zero and cap log-density range.
        rho = density.clamp(min=1e-8)
        raw = 1.0 / rho.pow(self.alpha)

        if self.renormalize:
            # Scale so sum(w_i) = N (preserves effective LR).
            w = raw * (n / raw.sum().clamp(min=1e-8))
        else:
            w = raw

        # Optional weight floor / ceiling. We do NOT re-renormalise
        # after clipping, because that would re-violate the ceiling.
        # If both are set, they take precedence over exact normalisation.
        if self.weight_floor > 0:
            w = w.clamp(min=self.weight_floor)
        if self.weight_ceiling > 0:
            w = w.clamp(max=self.weight_ceiling)

        return w.detach()

    # ------------------------------------------------------------------
    # Per-group diagnostics (uses bias attributes for LOGGING ONLY,
    # never for the loss computation)
    # ------------------------------------------------------------------
    def _compute_group_index(self, batch, targets):
        """Return a [N] long tensor of joint (y, joint_bias) group indices,
        encoded the same way as GroupDROTrainer:

            group = y * num_attrs + joint_bias_index

        where joint_bias_index packs all bias attributes of the dataset.
        For a single binary bias this is just the bias attribute itself.

        Returns None if per-group logging cannot be performed (no bias
        attributes available).
        """
        if not self._can_log_groups:
            return None
        biases_list = [batch[b].to(self.device) for b in self.biases]

        # Discover per-attribute cardinality lazily on the first batch.
        if self._num_attrs_per_class is None:
            # Estimate the number of attribute values from the data;
            # this is only a heuristic for datasets that don't expose
            # num_groups. We assume every bias is binary by default.
            cards = []
            for b in biases_list:
                # Number of distinct values; clamped to >= 2 for safety.
                card = max(2, int(b.max().item()) + 1)
                cards.append(card)
            num_attrs = 1
            for c in cards:
                num_attrs *= c
            self._num_attrs_per_class = num_attrs

        num_attrs = int(self._num_attrs_per_class)

        # Joint bias index: identical scheme to GroupDROTrainer.
        if len(biases_list) == 1:
            joint = biases_list[0].long()
        else:
            # Each bias contributes a "digit" in a mixed-radix encoding.
            # Use a simple per-bias offset by integer power, matching the
            # convention in GroupDRO across the framework.
            joint = torch.zeros(
                biases_list[0].shape[0], dtype=torch.long, device=self.device
            )
            # Approximate per-bias cardinality from the attribute values.
            # When multiple biases exist with mixed cardinalities this
            # may slightly under- or over-shoot the true num_attrs, but
            # the diagnostic is robust to that.
            base = max(2, int(round(num_attrs ** (1.0 / max(1, len(biases_list))))))
            for k, b in enumerate(biases_list):
                joint = joint + b.long() * (base**k)

        joint = torch.clamp(joint, min=0, max=num_attrs - 1)
        group_index = targets.long() * num_attrs + joint.to(targets.device)
        return group_index.to(self.device)

    def _per_group_stats(
        self,
        group_index: torch.Tensor,
        weights: torch.Tensor,
        density: torch.Tensor,
        per_sample_ce: torch.Tensor,
    ):
        """Compute per-group sums / means / counts for the diagnostic log.

        Returns a flat dict with keys of the form
            train_group_count_yYY_aAA
            train_group_w_sum_yYY_aAA
            train_group_w_mean_yYY_aAA
            train_group_density_mean_yYY_aAA
            train_group_loss_mean_yYY_aAA
        where YY is the target index and AA the joint bias-attribute index.
        Empty groups are reported with count=0 and NaN for the means.
        """
        out = {}
        num_attrs = int(self._num_attrs_per_class)
        num_class = int(self.num_class)
        # Vectorised group-mask matrix: [G, N].
        group_range = torch.arange(num_class * num_attrs, device=self.device).unsqueeze(
            1
        )
        mask = (group_index.unsqueeze(0) == group_range).float()  # [G, N]
        counts = mask.sum(dim=1)
        # Sums.
        w_sum = (mask * weights.unsqueeze(0)).sum(dim=1)
        d_sum = (mask * density.unsqueeze(0)).sum(dim=1)
        l_sum = (mask * per_sample_ce.detach().unsqueeze(0)).sum(dim=1)

        for g in range(num_class * num_attrs):
            y = g // num_attrs
            a = g % num_attrs
            tag = f"y{y}_a{a}"
            n_g = counts[g].item()
            out[f"train_group_count_{tag}"] = torch.tensor(n_g)
            if n_g > 0:
                out[f"train_group_w_sum_{tag}"] = w_sum[g].detach()
                # out[f"train_group_w_mean_{tag}"] = (w_sum[g] / n_g).detach()
                # out[f"train_group_density_mean_{tag}"] = (d_sum[g] / n_g).detach()
                # out[f"train_group_loss_mean_{tag}"] = (l_sum[g] / n_g).detach()
            else:
                nan = torch.tensor(float("nan"))
                out[f"train_group_w_sum_{tag}"] = torch.tensor(0.0)
                # out[f"train_group_w_mean_{tag}"] = nan
                # out[f"train_group_density_mean_{tag}"] = nan
                # out[f"train_group_loss_mean_{tag}"] = nan
        return out

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
            if features is not None and features.dim() > 2:
                features = features.flatten(start_dim=2).mean(dim=2)
        else:
            logits = outputs
            features = None

        per_sample_ce = self.criterion_train(logits, targets)  # [N]

        log_dict = {"train_cls_loss_unweighted": per_sample_ce.mean().detach()}

        if self._gate_active():
            with torch.no_grad():
                sig = self._compute_signature(inputs, targets, logits, features)
                sims, dists = self._compute_similarities_or_distances(sig)
                density = self._compute_density(sims, dists)
                weights = self._compute_weights(density)

            loss = (weights * per_sample_ce).mean()

            log_dict.update(
                {
                    "train_cls_loss": loss.detach(),
                    "train_density_min": density.min().detach(),
                    "train_density_median": density.median().detach(),
                    "train_density_max": density.max().detach(),
                    "train_weight_min": weights.min().detach(),
                    "train_weight_median": weights.median().detach(),
                    "train_weight_max": weights.max().detach(),
                    "train_weight_std": weights.std().detach(),
                    "train_eff_batch": (
                        weights.sum() / max(weights.max().item(), 1e-8)
                    ).detach(),
                }
            )

            # Per-(target, bias) group diagnostics. The bias attributes
            # are used here ONLY for logging; they do not influence the
            # weights, the density, or the loss.
            if self._can_log_groups:
                with torch.no_grad():
                    group_index = self._compute_group_index(batch, targets)
                    if group_index is not None:
                        log_dict.update(
                            self._per_group_stats(
                                group_index, weights, density, per_sample_ce
                            )
                        )
        else:
            # Warmup: pure CE.
            loss = per_sample_ce.mean()
            log_dict["train_cls_loss"] = loss.detach()

            # During warmup the weights are uniform, but logging per-group
            # counts is still informative for tracking batch composition.
            if self._can_log_groups:
                with torch.no_grad():
                    group_index = self._compute_group_index(batch, targets)
                    if group_index is not None:
                        n = per_sample_ce.shape[0]
                        uniform_w = torch.ones(n, device=self.device)
                        uniform_d = torch.ones(n, device=self.device)
                        log_dict.update(
                            self._per_group_stats(
                                group_index, uniform_w, uniform_d, per_sample_ce
                            )
                        )

        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()
        return log_dict
