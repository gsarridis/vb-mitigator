"""
GERNE: Gradient Extrapolation for Debiased Representation Learning.

Faithful port of the official GERNE codebase (Asaad et al., ICCV 2025) to the
vb-mitigator framework.

Reference
---------
Asaad, I., Shadaydeh, M., Denzler, J. "Gradient Extrapolation for Debiased
Representation Learning". Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), 2025.
https://arxiv.org/abs/2503.13236
https://github.com/Ihab-Asaad/GERNE

This module reproduces the canonical, *known training attributes* training
loop from `gerne/train_utils.py:train` and the per-group sampling from
`gerne/data_utils.py:get_loaders`, adapted to fit BaseTrainer's lifecycle.

Algorithm (matches the official code line-for-line where possible)
------------------------------------------------------------------
  1. Build one DataLoader per group g = (y, j) using an EnvSampler that
     samples `num_batches` mini-batches of size
        batch_size * (2 - corr)
     with replacement, restricted to the indices of group g.

  2. Compute the per-group sample budgets, where alpha_yj is the natural
     (training-set) joint distribution P(y, j):
        b_samples[y,j]  = num_attrs * batch_size * alpha_yj[y,j]
        d_samples[y,j]  = num_attrs * batch_size / N_nonzero(y) , for j with alpha>eps
        lb_samples[y,j] = (1 - c) * b_samples[y,j] + c * d_samples[y,j]
     and round each to int.  (Identical to gerne/train_utils.py.)

  3. For one optimizer step:
        - zip per-group loaders, take one batch from each group,
        - sample `b_samples[y,j]` items from the FIRST `batch_size` window
          and `lb_samples[y,j]` items from the LAST `batch_size` window of
          each group's batch (so the overlap is controlled by `corr`),
        - concatenate across groups -> X_b / Y_b, X_lb / Y_lb,
        - forward + (1-gamma)*L_b/num_classes . backward(),
        - forward + gamma   *L_lb/num_classes . backward(),
        - optimizer.step().

  4. Use gamma = beta + 1, matching gerne/main.py:
        L_total = L_lb + beta * (L_lb - L_b).

  5. Set every BatchNorm2d to eval() mode during training (model.train()
     is still called first), exactly as the official code does.

The "iteration" structure (`num_batches` optimizer steps per outer iteration,
many outer iterations until the equivalent epoch budget is exhausted) is
mapped onto BaseTrainer's epoch loop: one BaseTrainer epoch == one outer
GERNE iteration of `num_batches` steps. Set `cfg.SOLVER.EPOCHS` to the same
value you would have passed as `--epochs` to the official code.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from tools.metrics.utils import AverageMeter
from .base_trainer import BaseTrainer


# ----------------------------------------------------------------------
# Per-group sampler (mirrors gerne/data_utils.py:EnvSampler)
# ----------------------------------------------------------------------
class _GroupBatchSampler(Sampler[List[int]]):
    """
    Yield ``num_batches`` independent mini-batches drawn (with replacement
    from a per-group rng's perspective: ``rng.sample`` is without
    replacement *within* one batch but a fresh sample for each batch) from
    a fixed ``idx_list`` of dataset indices belonging to a single group
    g = (y, j). Direct counterpart of ``gerne/data_utils.EnvSampler`` for
    the train split with positive ``num_batches``.
    """

    def __init__(
        self,
        num_batches: int,
        batch_size: int,
        idx_list: List[int],
        seed: int = 0,
    ) -> None:
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.idx_list = list(idx_list)
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            if not self.idx_list:
                # Empty group: nothing to yield. zip_longest will fill None.
                yield []
                continue
            if self.batch_size < len(self.idx_list):
                yield self.rng.sample(self.idx_list, self.batch_size)
            else:
                # Fewer items than batch_size: yield all of them, like the
                # official sampler.
                yield list(self.idx_list)

    def __len__(self) -> int:
        return self.num_batches


def _zip_longest_loaders(loaders: List[DataLoader]):
    """
    Yield tuples ``(batch_g0, batch_g1, ..., batch_gG)`` for one outer
    iteration. Mirrors the ``zip_longest(*train_loaders, fillvalue=None)``
    behaviour of the official training loop. When a sampler yielded an
    empty list (empty group), the corresponding batch is None.
    """
    iters = [iter(ld) for ld in loaders]
    while True:
        out = []
        any_alive = False
        for it in iters:
            try:
                b = next(it)
            except StopIteration:
                b = None
            else:
                any_alive = True
            # An empty-group sampler can still produce empty batches: treat
            # those as None for downstream consumers.
            if b is not None and (
                (hasattr(b, "__len__") and len(b) == 0)
                or (isinstance(b, dict) and "inputs" in b and len(b["inputs"]) == 0)
            ):
                b = None
            out.append(b)
        if not any_alive:
            return
        yield out


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------
class GERNETrainer(BaseTrainer):
    """
    Faithful port of GERNE's known-training-attributes training loop.

    Configuration (under ``cfg.MITIGATOR.GERNE``):
      - ``BETA``: extrapolation factor (paper's beta). Internally
                   ``gamma = beta + 1``. Default 1.0 (matches official code).
      - ``C``:    bias-reduction factor c in [0, 1]. Default 0.5.
      - ``CORR``: correlation coefficient in [0, 1]. Default 0.5.
      - ``NUM_BATCHES``: number of optimizer steps per epoch (== outer
                   iteration). Default 50, same as the official default.
      - ``EPS``:  numerical guard for empty groups. Default 1e-4.
      - ``BN_EVAL``: freeze BatchNorm2d statistics during training, as in
                   the official code. Default True.
      - ``DROP_LAST``: drop a step if either the biased or less-biased
                   sub-batch is empty (matches ``continue`` in the official
                   loop). Default True.

    The dataset must expose bias attributes per sample (``batch[b]`` for
    every ``b`` in ``self.biases``) AND must have its underlying training
    set indexable as a torch ``Dataset`` returning the standard
    vb-mitigator dict (``inputs``, ``targets``, ``<bias>`` per attribute).
    Both conditions hold for every BLA dataset already supported in the
    framework (waterbirds, celeba, urbancars, biased_mnist, etc.).
    """

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
    def _setup_criterion(self):
        if self.cfg.SOLVER.CRITERION == "CE":
            # Match gerne/model_utils.py: nn.CrossEntropyLoss with default
            # 'mean' reduction; the per-num_classes division is applied in
            # _train_iter, exactly like the official train loop.
            self.criterion_train = nn.CrossEntropyLoss()
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"Unsupported criterion type: {self.cfg.SOLVER.CRITERION}"
            )

    def _setup_dataset(self):
        # Standard BLA dataset setup, like GroupDRO/DI/EnD/BB/BAdd.
        from my_datasets.builder import get_dataset

        dataset = get_dataset(self.cfg)
        self.num_class = dataset["num_class"]
        self.biases = dataset["biases"]
        self.dataloaders = dataset["dataloaders"]
        self.data_root = dataset["root"]
        self.sets = dataset["sets"]
        self.target2name = dataset["target2name"]
        self.ba_groups = dataset["ba_groups"] if "ba_groups" in dataset else None
        self.num_group = int(dataset["num_groups"])
        # GERNE thinks of groups as (y, j) with num_attrs = product of
        # bias-attribute cardinalities.  We compute it the same way GroupDRO
        # does so the indexing is consistent across BLA mitigators.
        self.num_attrs = self.num_group // self.num_class

    def _method_specific_setups(self):
        gerne_cfg = self.cfg.MITIGATOR.GERNE
        self.beta: float = float(gerne_cfg.BETA)
        self.gamma: float = self.beta + 1.0  # matches `args.gamma = args.beta + 1`.
        self.c: float = float(gerne_cfg.C)
        self.corr: float = float(gerne_cfg.CORR)
        self.num_batches: int = int(gerne_cfg.NUM_BATCHES)
        self.eps: float = float(gerne_cfg.EPS)
        self.bn_eval: bool = bool(gerne_cfg.BN_EVAL)
        self.drop_last_if_empty: bool = bool(gerne_cfg.DROP_LAST)

        if not (0.0 <= self.c <= 1.0):
            raise ValueError(f"MITIGATOR.GERNE.C must be in [0, 1], got {self.c}.")
        if not (0.0 <= self.corr <= 1.0):
            raise ValueError(f"MITIGATOR.GERNE.CORR must be in [0, 1], got {self.corr}.")

        # Build per-group training loaders and the per-group sample budgets.
        self._build_group_loaders_and_budgets()

    # -----------------------------------------------------------------
    # Per-group loader construction (mirrors gerne/data_utils.get_loaders)
    # -----------------------------------------------------------------
    def _index_groups(self):
        """Walk the training Dataset once, recording per-group index lists.

        Returns
        -------
        groups : list[list[int]]
            ``groups[y * num_attrs + j]`` is the list of dataset indices
            belonging to (target=y, joint-attribute=j).
        alpha_yj : torch.Tensor of shape (num_class, num_attrs)
            Empirical joint distribution P(y, j) on the training split,
            same convention as `BiasedDataset.dist_yj()` in the official
            codebase: ``alpha[y][j] = N_yj / N_y``.
        """
        train_set = self.sets["train"]
        num_attrs = self.num_attrs
        num_class = self.num_class
        biases = self.biases

        groups: List[List[int]] = [[] for _ in range(num_class * num_attrs)]
        per_class_count = [0 for _ in range(num_class)]
        alpha_counts = torch.zeros(num_class, num_attrs)

        for i in range(len(train_set)):
            sample = train_set[i]
            y = int(sample["targets"])
            # Joint attribute index = sum_k a_k * (num_biases ** (k + 1)).
            # Single-bias datasets degenerate to j = a_0.
            if len(biases) == 1:
                j = int(sample[biases[0]])
            else:
                # Match the GroupDRO mapping: targets * num_biases is *not*
                # used here because we only need j (the joint attribute), so
                # we accumulate the bias indices using the same per-bias
                # cardinality scheme.
                num_biases_per_attr = self.num_attrs ** (1.0 / len(biases))
                # Round to int to be safe for products like 2 * 2 = 4.
                num_biases_per_attr = int(round(num_biases_per_attr))
                j = 0
                for k, b_name in enumerate(biases):
                    j += int(sample[b_name]) * (num_biases_per_attr ** k)
            j = max(0, min(num_attrs - 1, j))
            groups[y * num_attrs + j].append(i)
            per_class_count[y] += 1
            alpha_counts[y][j] += 1

        # Normalize per row -> alpha[y][j] = P(j | y) * P(y_obs) ?  In the
        # official code this is P(j | y) (rows sum to 1) when the loaders
        # zip_longest balances per-class sampling. We match that.
        alpha_yj = torch.zeros(num_class, num_attrs)
        for y in range(num_class):
            n_y = per_class_count[y]
            if n_y == 0:
                continue
            for j in range(num_attrs):
                alpha_yj[y][j] = alpha_counts[y][j] / n_y

        return groups, alpha_yj

    def _build_group_loaders_and_budgets(self):
        groups, alpha_yj = self._index_groups()
        self.alpha_yj = alpha_yj.to(self.device)

        num_class = self.num_class
        num_attrs = self.num_attrs
        seed = int(getattr(self.cfg.EXPERIMENT, "SEED", 0))

        # Per-group loader batch size -- multiplied by (2 - corr) so the
        # first/last windows can be sampled with the configured overlap.
        bs = int(self.cfg.SOLVER.BATCH_SIZE)
        scaled_bs = max(1, int(round(bs * (2.0 - self.corr))))
        self._group_loader_batch_size = scaled_bs
        self._inner_batch_size = bs  # the "window size" for first/last splits.

        train_set = self.sets["train"]
        num_workers = int(getattr(self.cfg.DATASET, "NUM_WORKERS", 0))

        self.train_loaders: List[DataLoader] = []
        for y in range(num_class):
            for j in range(num_attrs):
                idx_list = groups[y * num_attrs + j]
                # Per-group rng seed identical in spirit to the official code.
                sampler = _GroupBatchSampler(
                    num_batches=self.num_batches,
                    batch_size=scaled_bs,
                    idx_list=idx_list,
                    seed=seed + 4000 + y * num_attrs + j,
                )
                loader = DataLoader(
                    train_set,
                    batch_sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=False,
                )
                self.train_loaders.append(loader)

        # Per-group sample budgets for biased / less-biased / debiased.
        # These are the integer counts of samples to pull from each group's
        # batch, exactly as the official code computes them.
        b = num_attrs * bs * alpha_yj                                 # [Y, A]
        nz = (b > self.eps).sum(dim=1, keepdim=True).clamp(min=1)     # [Y, 1]
        d = (num_attrs * bs / nz).repeat(1, num_attrs) * (b > self.eps).float()
        lb = (1.0 - self.c) * b + self.c * d

        self.b_samples_per_group = b.round().to(torch.int).cpu()
        self.lb_samples_per_group = lb.round().to(torch.int).cpu()
        self.d_samples_per_group = d.round().to(torch.int).cpu()

        # Log distributions like the official code.
        self.logger.info(f"GERNE: alpha_yj (P(j|y)):\n{alpha_yj}")
        self.logger.info(f"GERNE: Biased samples/group:\n{self.b_samples_per_group}")
        self.logger.info(f"GERNE: Targeted samples/group:\n{self.d_samples_per_group}")
        self.logger.info(f"GERNE: Less-biased samples/group:\n{self.lb_samples_per_group}")
        self.logger.info(
            f"GERNE: beta={self.beta}, gamma={self.gamma}, c={self.c}, "
            f"corr={self.corr}, num_batches={self.num_batches}, "
            f"per-group loader BS={scaled_bs}, window={bs}"
        )

    # -----------------------------------------------------------------
    # Train loop overrides
    # -----------------------------------------------------------------
    def _set_train(self):
        """Like BaseTrainer._set_train() but freezes BN statistics.

        This is the line-for-line counterpart of the official:

            model.train()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        """
        self.model.train()
        if self.bn_eval:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def _gather_window(self, batch_dict, n_take: int, side: str):
        """Sample ``n_take`` items from a per-group batch's first or last
        ``batch_size`` window with replacement, mirroring the
        ``torch.randint`` calls in the official loop.
        """
        if n_take <= 0:
            return None, None

        inputs = batch_dict["inputs"]
        targets = batch_dict["targets"]
        cur = inputs.shape[0]
        win = min(self._inner_batch_size, cur)
        if side == "first":
            lo, hi = 0, win
        elif side == "last":
            lo, hi = cur - win, cur
        else:
            raise ValueError(side)
        if hi <= lo:
            return None, None

        # torch.randint does sampling WITH replacement, matching the
        # official code's `torch.randint(0, win, (n_take,))`.
        idx = torch.randint(lo, hi, (n_take,))
        return inputs[idx], targets[idx]

    def _train_epoch(self):
        """One outer GERNE iteration: ``num_batches`` optimizer steps."""
        self._set_train()
        self.current_lr = self.scheduler.get_last_lr()[0]
        avg_loss = None

        num_class = self.num_class
        num_attrs = self.num_attrs

        show_progress = self.cfg.EXPERIMENT.PROGRESS_BAR
        loader_iter = _zip_longest_loaders(self.train_loaders)
        if show_progress:
            loader_iter = tqdm(
                loader_iter,
                total=self.num_batches,
                desc=f"Epoch {getattr(self, 'current_epoch', '?')}/"
                     f"{self.cfg.SOLVER.EPOCHS} GERNE",
                unit="step",
            )

        steps_taken = 0
        for step_idx, batches in enumerate(loader_iter):
            if step_idx >= self.num_batches:
                break

            self.optimizer.zero_grad()

            x_b_chunks, y_b_chunks = [], []
            x_lb_chunks, y_lb_chunks = [], []

            for y in range(num_class):
                for j in range(num_attrs):
                    batch = batches[y * num_attrs + j]
                    if batch is None:
                        continue

                    n_b = int(self.b_samples_per_group[y][j].item())
                    n_lb = int(self.lb_samples_per_group[y][j].item())

                    xb, yb = self._gather_window(batch, n_b, side="first")
                    if xb is not None:
                        x_b_chunks.append(xb)
                        y_b_chunks.append(yb)

                    xlb, ylb = self._gather_window(batch, n_lb, side="last")
                    if xlb is not None:
                        x_lb_chunks.append(xlb)
                        y_lb_chunks.append(ylb)

            if (
                self.drop_last_if_empty
                and (len(x_b_chunks) == 0 or len(x_lb_chunks) == 0)
            ):
                # Match `if len(x_b) == 0 or len(x_lb) == 0: continue`.
                continue

            # ---------- Biased pass ----------
            biased_loss_val = 0.0
            if len(x_b_chunks) > 0:
                xb = torch.cat(x_b_chunks, dim=0).to(self.device)
                yb = torch.cat(y_b_chunks, dim=0).to(self.device)
                out_b = self.model(xb)
                if isinstance(out_b, tuple):
                    out_b, _ = out_b
                # Official: criterion(...) / num_classes  with mean-reduction CE.
                b_loss = self.criterion_train(out_b, yb) / num_class
                ((1.0 - self.gamma) * b_loss).backward()
                biased_loss_val = b_loss.item()

            # ---------- Less-biased pass ----------
            lessbiased_loss_val = 0.0
            if len(x_lb_chunks) > 0:
                xlb = torch.cat(x_lb_chunks, dim=0).to(self.device)
                ylb = torch.cat(y_lb_chunks, dim=0).to(self.device)
                out_lb = self.model(xlb)
                if isinstance(out_lb, tuple):
                    out_lb, _ = out_lb
                lb_loss = self.criterion_train(out_lb, ylb) / num_class
                (self.gamma * lb_loss).backward()
                lessbiased_loss_val = lb_loss.item()

            self._optimizer_step()
            self.scheduler.step()
            steps_taken += 1

            d_loss_val = (
                self.gamma * lessbiased_loss_val
                + (1.0 - self.gamma) * biased_loss_val
            )
            loss_dict = {
                "train_cls_loss": torch.tensor(d_loss_val),
                "train_biased_loss": torch.tensor(biased_loss_val),
                "train_lessbiased_loss": torch.tensor(lessbiased_loss_val),
            }
            if avg_loss is None:
                avg_loss = {k: AverageMeter() for k in loss_dict}
            # bsz: number of samples actually used in this step (b + lb).
            bsz = (
                (sum(c.shape[0] for c in x_b_chunks) if x_b_chunks else 0)
                + (sum(c.shape[0] for c in x_lb_chunks) if x_lb_chunks else 0)
            )
            for k, v in loss_dict.items():
                avg_loss[k].update(v.item(), max(bsz, 1))

        if steps_taken == 0:
            self.logger.warning(
                "GERNE: no optimizer steps were taken this epoch. "
                "All groups may have produced empty windows; consider "
                "lowering BATCH_SIZE or raising NUM_BATCHES."
            )
            return {
                "train_cls_loss": 0.0,
                "train_biased_loss": 0.0,
                "train_lessbiased_loss": 0.0,
            }

        return {k: m.avg for k, m in avg_loss.items()}