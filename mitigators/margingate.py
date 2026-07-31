"""
MarginGate: stop learning from samples whose decision is already settled.

A fundamental, assumption-free training mechanism that modifies CE so that
each sample's contribution to the loss is gated by its current confidence
margin. Once a sample is correctly classified by margin >= m, its loss is
smoothly suppressed; the model stops over-learning from it.

The key insight is the distinction from "ignore easy samples":
MarginGate does NOT downweight easy samples a priori. It uses every sample
for gradient updates *up to* the configured confidence margin, and only
then suppresses further learning from it. Bias-aligned samples whose label
is correctly predicted *by the bias alone* will reach margin quickly and
get gated out; bias-conflicting samples that need the true signal to be
predicted will keep contributing gradient. Crucially, easy samples whose
correct prediction also relies on the true signal still contribute — they
are gated only when the model is sufficiently confident in their label,
regardless of *which* feature drove that confidence.

Objective
---------
For a batch of N samples with logits z_i in R^C and labels y_i:

    margin_i = z_{i, y_i} - max_{j != y_i} z_{i, j}
    w_i      = sigmoid((m - margin_i) / T)             [stop_grad]
    L        = sum_i w_i * CE(z_i, y_i)  /  sum_i w_i

The gate weights w_i carry NO gradient with respect to model parameters
(stop_grad). This is essential: a differentiable gate would let the model
inflate logit norms to push every margin past m and silence the loss.

Hyper-parameters
----------------
  - MARGIN m:       the confidence cutoff. m=0 with hard gating recovers
                    ERM (almost; see note in trainer). Larger m means the
                    model continues learning from a sample longer.
                    Typical: 1.0 to 5.0.
  - TEMPERATURE T:  smoothness of the gate. Small T -> hard cutoff.
                    Typical: 0.3 to 1.0.
  - WARMUP_EPOCHS:  epochs of pure CE before the gate engages. The gate
                    needs reasonable margins to gate on, so a warmup of
                    at least 1 epoch is recommended.

Connection to prior work
------------------------
  - Logit-margin losses (Cao et al., LDAM, NeurIPS 2019) use a related
    margin idea but with a class-imbalance motivation and additive margin.
  - Curriculum learning / anti-curriculum: similar in spirit but operates
    via sample ordering, not loss gating.
  - GAS / saturating-gradient observations: the underlying mechanism here
    is the same gradient-saturation that motivates SD, but addressed at
    the sample level rather than the logit level.
  - LfF/JTT: superficially similar (both produce per-sample weights), but
    LfF/JTT need an *auxiliary biased model* and equate hardness with
    bias-conflicting status. MarginGate uses only the current model and
    makes no claim about which samples are bias-conflicting.

What MarginGate does NOT do
---------------------------
  - No auxiliary models.
  - No discovery phase.
  - No bias attribute labels, environments, or groups.
  - No clustering or pseudo-labels.
  - No assumption that "easy = spurious."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_trainer import BaseTrainer


# ---------------------------------------------------------------------------
# Margin gate primitive
# ---------------------------------------------------------------------------


def _margin_gate_weights(
    logits: torch.Tensor,
    targets: torch.Tensor,
    m: float,
    T: float,
) -> torch.Tensor:
    """Compute per-sample gate weights w_i in [0, 1].

    w_i = sigmoid((m - margin_i) / T), where
        margin_i = z_{i, y_i} - max_{j != y_i} z_{i, j}.

    The returned tensor is detached from the autograd graph so that no
    gradient flows back through the gate. This is critical: a
    differentiable gate would let the model game the gate by inflating
    logit norms, which is exactly the failure mode SD addresses for
    standard CE.

    Args:
        logits: shape [N, C].
        targets: shape [N], int64.
        m: margin threshold (>= 0). Larger -> samples kept longer.
        T: gate temperature (> 0). Smaller -> harder cutoff.

    Returns:
        weights: shape [N], in [0, 1], detached.
    """
    if T <= 0:
        raise ValueError(f"TEMPERATURE must be > 0, got {T}.")

    n, c = logits.shape

    # True-class logit per sample.
    true_logit = logits[torch.arange(n, device=logits.device), targets]  # [N]

    # Highest logit among non-true classes per sample. We do this by
    # masking the true class with -inf and taking the max.
    masked = logits.clone()
    masked[torch.arange(n, device=logits.device), targets] = float("-inf")
    other_max, _ = masked.max(dim=1)  # [N]

    margin = true_logit - other_max  # [N]
    weights = torch.sigmoid((m - margin) / T)
    return weights.detach()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class MarginGateTrainer(BaseTrainer):
    """
    Trainer implementing MarginGate in the vb-mitigator framework.

    Configuration (under ``cfg.MITIGATOR.MARGINGATE``):
      - ``MARGIN``:         margin cutoff m. Default 1.0.
      - ``TEMPERATURE``:    gate smoothness T. Default 0.5.
      - ``WARMUP_EPOCHS``:  pure-CE warmup before gating. Default 1.
      - ``MIN_EFFECTIVE_BATCH``: floor on effective batch size (sum of
                            weights). If the gate suppresses nearly all
                            samples in a batch, gradient becomes
                            high-variance; this floor ensures we do not
                            update on pathologically small effective
                            batches. Default 1.0 (unit sample-equivalent).
      - ``RENORMALIZE``:    divide weighted sum by sum of weights instead
                            of by batch size. Default True. When False
                            the loss magnitude shrinks as the gate engages,
                            which interacts predictably with the LR.

    Implementation notes
    --------------------
    The trainer uses the standard ERM-style loader (no per-group sampling,
    no auxiliary models). One forward, one backward, one optimizer step
    per batch, identical in structure to ERM.

    A small but important detail: the per-sample CE is computed with
    ``reduction='none'``, multiplied by the detached gate weights, and
    then averaged. The model parameters receive gradient only through the
    CE term, never through the gate.
    """

    def _setup_criterion(self):
        if self.cfg.SOLVER.CRITERION == "CE":
            self.criterion_train = nn.CrossEntropyLoss(reduction="none")
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {self.cfg.SOLVER.CRITERION}")

    def _method_specific_setups(self):
        mcfg = self.cfg.MITIGATOR.MARGINGATE
        self.margin = float(mcfg.MARGIN)
        self.temperature = float(mcfg.TEMPERATURE)
        self.warmup_epochs = int(mcfg.WARMUP_EPOCHS)
        self.min_effective_batch = float(mcfg.MIN_EFFECTIVE_BATCH)
        self.renormalize = bool(mcfg.RENORMALIZE)

        if self.margin < 0:
            raise ValueError(f"MARGIN must be >= 0, got {self.margin}.")
        if self.temperature <= 0:
            raise ValueError(f"TEMPERATURE must be > 0, got {self.temperature}.")
        if self.min_effective_batch < 0:
            raise ValueError(
                f"MIN_EFFECTIVE_BATCH must be >= 0, got {self.min_effective_batch}."
            )

    def _gate_active(self) -> bool:
        """The gate engages only after the warmup. During warmup we run
        pure CE so that the model has meaningful margins to gate on once
        the gate turns on.
        """
        ep = getattr(self, "current_epoch", 0)
        return ep > self.warmup_epochs

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs

        per_sample_ce = self.criterion_train(logits, targets)  # [N]

        if self._gate_active():
            with torch.no_grad():
                weights = _margin_gate_weights(
                    logits.detach(), targets, self.margin, self.temperature
                )
                eff_batch = weights.sum()

            # Guard against pathological batches where the gate suppresses
            # nearly everything: skip the update and log it.
            if eff_batch.item() < self.min_effective_batch:
                # Still step the scheduler so LR schedules don't drift.
                self.scheduler.step()
                return {
                    "train_cls_loss": per_sample_ce.mean().detach(),
                    "train_gate_weight": weights.mean().detach(),
                    "train_eff_batch": eff_batch.detach(),
                    "train_skipped": torch.tensor(1.0),
                }

            if self.renormalize:
                loss = (weights * per_sample_ce).sum() / eff_batch
            else:
                loss = (weights * per_sample_ce).mean()

            log_dict = {
                "train_cls_loss": loss.detach(),
                "train_gate_weight": weights.mean().detach(),
                "train_eff_batch": eff_batch.detach(),
                "train_skipped": torch.tensor(0.0),
            }
        else:
            # Warmup: pure CE, full batch.
            loss = per_sample_ce.mean()
            log_dict = {
                "train_cls_loss": loss.detach(),
                "train_gate_weight": torch.tensor(1.0),
                "train_eff_batch": torch.tensor(float(per_sample_ce.shape[0])),
                "train_skipped": torch.tensor(0.0),
            }

        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()
        return log_dict
