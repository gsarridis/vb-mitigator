# Sebra — "Sebra: Debiasing through Self-Guided Bias Ranking"
# (Adarsh, Patravali, Krishnan, ICLR 2025)
# https://github.com/kadarsh22/Sebra
#
# Faithful adaptation of the official `celeba_trainers/sebra.py` +
# `loss/upweighted_training_loss.py` + `loss/contrastive_loss.py` to
# the vb-mitigator framework.
#
# Pipeline (two stages):
#
#   Stage 1 — Spuriosity ranking:
#     Train a model with UpweightedTrainingLoss for `RANK_ROUNDS` rounds.
#     The loss multiplies CE by p_y^β (β = beta_inverse, default 0.8),
#     which UPweights easy samples — i.e., samples with high p_y get
#     larger gradients — encouraging the model to lock in on the most
#     spurious examples first. After each round, score every training
#     sample; any sample whose p_y exceeds `p_critical` (default 0.7) is
#     considered "learned" and dropped from the active training pool.
#     Each sample's spuriosity rank = the round at which it was dropped.
#
#   Stage 2 — Contrastive learning:
#     Train a fresh model end-to-end with:
#       loss = SupervisedContrastiveLoss(anchor, positive, negative)
#              + classifier_weight * CE(classifier_head(anchor), y)
#     where the anchor is a normal training sample, the positive is a
#     same-class sample with rank min(anchor_rank + gap, max_rank) (i.e.,
#     a LESS spurious sample of the same class), and the negative is a
#     same-class sample with the SAME rank.
#
#     The contrastive objective pulls the anchor's features toward
#     less-spurious counterparts and pushes away from same-spuriosity
#     ones, learning spurious-feature-invariant representations.

import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.builder import get_model
from tools.utils import load_checkpoint, log_msg, save_checkpoint
from .base_trainer import BaseTrainer


# ---------------------------------------------------------------------------
# Loss functions — faithful ports of loss/upweighted_training_loss.py and
# loss/contrastive_loss.py
# ---------------------------------------------------------------------------


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


class _UpweightedTrainingLoss(nn.Module):
    """CE loss multiplied by p_y^β. Encourages the model to focus on
    easy/high-confidence samples first (the most spurious ones).
    """

    def __init__(self, beta_inverse=0.8):
        super().__init__()
        self.beta_inverse = beta_inverse

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        py = torch.gather(p, 1, targets.unsqueeze(1)).squeeze(1)
        loss_weight = py.detach() ** self.beta_inverse
        loss = F.cross_entropy(logits, targets, reduction="none") * loss_weight
        return loss


class _SupervisedContrastiveLoss(nn.Module):
    """Sebra's contrastive loss: pulls anchor toward positives (less
    spurious same-class), pushes away from negatives (same-rank
    same-class).
    """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.sim = nn.CosineSimilarity(dim=1)

    def _compute_exp_sim(self, anchor, other):
        s = self.sim(anchor, other)
        return torch.exp(s / self.temperature)

    def forward(self, feature_anchor, features_pos, features_neg):
        exp_neg = self._compute_exp_sim(feature_anchor, features_neg)
        sum_exp_neg = exp_neg.sum(0, keepdim=True)
        exp_pos = self._compute_exp_sim(feature_anchor, features_pos)
        log_probs = torch.log(exp_pos) - torch.log(
            sum_exp_neg + exp_pos.sum(0, keepdim=True)
        )
        return -log_probs.mean()


# ---------------------------------------------------------------------------
# Stage 2 dataset wrapper — produces (anchor, positive, negative) triplets
# ---------------------------------------------------------------------------


class _SebraTripletDataset(torch.utils.data.Dataset):
    """Wraps a vb-mitigator dataset to produce contrastive triplets given
    spuriosity ranks. For each anchor index:
      - positive = random same-class sample with rank min(r + gap, max_rank)
        (a LESS spurious sample of the same class)
      - negative = random same-class sample with the same rank r
    Falls back to higher ranks if the requested positive rank has no
    samples (matching the official code).
    """

    def __init__(self, base_dataset, ranks, indices_by_label_rank, max_rank, gap):
        self.base = base_dataset
        self.ranks = ranks  # array of length N
        self.indices_by_label_rank = indices_by_label_rank  # {label: {rank: np.array}}
        self.max_rank = max_rank  # {label: int}
        self.gap = int(gap)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        anchor = self.base[index]
        targets = anchor["targets"]
        # vb-mitigator targets are scalars or 0-d tensors
        if torch.is_tensor(targets):
            label = (
                int(targets.item())
                if targets.ndim == 0
                else int(targets.flatten()[0].item())
            )
        else:
            label = int(targets)

        rank = int(self.ranks[index])
        max_rank = int(self.max_rank.get(label, rank))
        pos_rank = min(rank + self.gap, max_rank)

        # Negative: same class, same rank
        neg_pool = self.indices_by_label_rank[label].get(rank, np.array([]))
        if len(neg_pool) == 0:
            return None
        neg_idx = int(np.random.choice(neg_pool))

        # Positive: same class, rank = pos_rank, with fallback
        pos_pool = self.indices_by_label_rank[label].get(pos_rank, np.array([]))
        if len(pos_pool) == 0:
            # Try higher ranks until we find one
            pos_idx = -1
            for r in range(pos_rank + 1, max_rank + 1):
                fallback = self.indices_by_label_rank[label].get(r, np.array([]))
                if len(fallback) > 0:
                    pos_idx = int(np.random.choice(fallback))
                    break
            if pos_idx == -1:
                # Final fallback: use anchor itself (degenerate but avoids None)
                pos_idx = index
        else:
            pos_idx = int(np.random.choice(pos_pool))

        positive = self.base[pos_idx]
        negative = self.base[neg_idx]

        return {
            "anchor": anchor["inputs"],
            "positive": positive["inputs"],
            "negative": negative["inputs"],
            "targets": anchor["targets"],
        }


def _sebra_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    anchor = torch.stack([b["anchor"] for b in batch])
    positive = torch.stack([b["positive"] for b in batch])
    negative = torch.stack([b["negative"] for b in batch])
    targets = torch.stack(
        [
            (
                b["targets"]
                if torch.is_tensor(b["targets"])
                else torch.tensor(b["targets"])
            )
            for b in batch
        ]
    )
    return {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "targets": targets,
    }


# ---------------------------------------------------------------------------
# Sebra Trainer
# ---------------------------------------------------------------------------


class SebraTrainer(BaseTrainer):
    """
    Sebra trainer.

    Stage 1: train a model with UpweightedTrainingLoss for RANK_ROUNDS
    iterations, dropping samples whose p_y exceeds p_critical after each
    round. Records the round at which each sample is dropped as its
    spuriosity rank.

    Stage 2: trains a fresh classifier with a supervised contrastive
    loss + a CE classifier-head loss using the discovered ranks.
    """

    def _setup_criterion(self):
        # Stage 1 criterion is set up dynamically; Stage 2 uses a fresh
        # CE for the classifier head. We expose the standard `criterion`
        # so BaseTrainer's validation code (which uses `self.criterion`)
        # works.
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_train = nn.CrossEntropyLoss(reduction="none")

    def _method_specific_setups(self):
        # Stage 1 model is `self.model` (already set up by BaseTrainer).
        # Stage 2 model and classifier head will be created in train().
        pass

    # ------------------------------------------------------------------
    # Override train() — Sebra has its own two-stage flow
    # ------------------------------------------------------------------
    def train(self):
        cfg = self.cfg

        # =====================================================
        # STAGE 1: Spuriosity ranking
        # =====================================================
        print(
            log_msg(
                "Sebra Stage 1: spuriosity ranking via iterative ERM-with-dropping",
                "INFO",
                self.logger,
            )
        )
        ranks = self._stage1_rank_spuriosity()

        # =====================================================
        # Build stage 2 dataset (rank-based contrastive triplets)
        # =====================================================
        train_set = self.sets["train"]
        n_samples = len(train_set)

        # Collect class label per sample (we walk the dataset once)
        class_labels = self._collect_class_labels()

        unique_labels = np.unique(class_labels)
        indices_by_label_rank = {}
        max_rank_by_label = {}
        for label in unique_labels:
            label = int(label)
            indices_by_label_rank[label] = {}
            class_mask = class_labels == label
            class_ranks = ranks[class_mask]
            class_indices = np.where(class_mask)[0]
            unique_ranks = np.unique(class_ranks[class_ranks >= 0])
            if len(unique_ranks) == 0:
                # No samples ever got dropped — fall back to a single rank
                indices_by_label_rank[label][0] = class_indices
                max_rank_by_label[label] = 0
                continue
            for r in unique_ranks:
                idx = class_indices[class_ranks == r]
                np.random.shuffle(idx)
                indices_by_label_rank[label][int(r)] = idx
            # Samples never dropped get the max rank (treated as least spurious)
            never_dropped = class_indices[
                ~np.isin(
                    class_indices,
                    np.concatenate(list(indices_by_label_rank[label].values())),
                )
            ]
            if len(never_dropped) > 0:
                final_rank = int(unique_ranks.max()) + 1
                indices_by_label_rank[label][final_rank] = never_dropped
                ranks[never_dropped] = final_rank
            max_rank_by_label[label] = int(max(indices_by_label_rank[label].keys()))

            counts = {r: len(idx) for r, idx in indices_by_label_rank[label].items()}
            print(
                log_msg(
                    f"Sebra: class {label} rank distribution = {counts}",
                    "INFO",
                    self.logger,
                )
            )

        # =====================================================
        # STAGE 2: Contrastive training
        # =====================================================
        print(
            log_msg(
                "Sebra Stage 2: contrastive learning with rank-based pairs",
                "INFO",
                self.logger,
            )
        )

        triplet_dataset = _SebraTripletDataset(
            base_dataset=train_set,
            ranks=ranks,
            indices_by_label_rank=indices_by_label_rank,
            max_rank=max_rank_by_label,
            gap=cfg.MITIGATOR.SEBRA.GAP,
        )
        triplet_loader = torch.utils.data.DataLoader(
            triplet_dataset,
            batch_size=cfg.MITIGATOR.SEBRA.BATCH_SIZE_STAGE2,
            shuffle=True,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
            collate_fn=_sebra_collate,
            drop_last=True,
        )

        # Build a fresh stage-2 model + classifier head. We use the
        # same model architecture as stage 1, treating its forward as
        # (logits, features). The classifier head is a separate linear
        # layer on the features.
        stage2_model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
            self.cfg.MODEL.PRETRAINED,
        ).to(self.device)
        # Determine feature dim by inspecting the existing fc layer
        attr_name, cl_layer = get_classifier_attr(stage2_model)
        cl_layer = get_last_linear(cl_layer)
        feat_dim = cl_layer.in_features
        # feat_dim = stage2_model.fc.in_features
        classifier_head = nn.Linear(feat_dim, self.num_class).to(self.device)

        # Optimizer over the model + classifier head
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = torch.optim.SGD(
                list(stage2_model.parameters()) + list(classifier_head.parameters()),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            optimizer = torch.optim.Adam(
                list(stage2_model.parameters()) + list(classifier_head.parameters()),
                lr=cfg.SOLVER.LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )

        contrastive_loss = _SupervisedContrastiveLoss(
            temperature=cfg.MITIGATOR.SEBRA.TEMPERATURE
        )
        ce_loss = nn.CrossEntropyLoss()

        stage2_epochs = cfg.MITIGATOR.SEBRA.STAGE2_EPOCHS
        for epoch in range(1, stage2_epochs + 1):
            self.current_epoch = epoch
            stage2_model.train()
            classifier_head.train()
            total_loss, total_cls, total_con, n_batches = 0.0, 0.0, 0.0, 0

            for batch in triplet_loader:
                if batch is None:
                    continue
                anchor = batch["anchor"].to(self.device)
                positive = batch["positive"].to(self.device)
                negative = batch["negative"].to(self.device)
                targets = batch["targets"].to(self.device).long()

                optimizer.zero_grad()

                # Stack anchor + positive so we get features for both
                # in a single forward pass (matches the official code:
                # `image_ = torch.cat((image, image_pos))`)
                stacked = torch.cat([anchor, positive], dim=0)
                outputs = stage2_model(stacked)
                if isinstance(outputs, tuple):
                    _, features_stacked = outputs
                else:
                    features_stacked = outputs

                bs = anchor.size(0)
                feat_anchor = features_stacked[:bs]
                feat_pos = features_stacked[bs:]

                # Negatives: forward through the same model in eval mode
                # (no grad), matching the official code which freezes the
                # classifier when extracting positive/negative features.
                with torch.no_grad():
                    neg_outputs = stage2_model(negative)
                    if isinstance(neg_outputs, tuple):
                        _, feat_neg = neg_outputs
                    else:
                        feat_neg = neg_outputs

                # Contrastive loss: anchor pulled toward pos, away from neg
                con_loss = contrastive_loss(
                    feat_anchor, feat_pos.detach(), feat_neg.detach()
                )

                # Classifier head loss on the positive (less spurious)
                # branch — matches the official code which slices
                # `output[bs:]` (the second half of the stacked batch).
                logits_pos = classifier_head(feat_pos)
                cls_loss = ce_loss(logits_pos, targets)

                loss = con_loss + cfg.MITIGATOR.SEBRA.CLASSIFIER_WEIGHT * cls_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_cls += cls_loss.item()
                total_con += con_loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            avg_cls = total_cls / max(n_batches, 1)
            avg_con = total_con / max(n_batches, 1)
            print(
                f"  Sebra Stage 2 epoch {epoch}/{stage2_epochs} "
                f"loss={avg_loss:.4f} cls={avg_cls:.4f} con={avg_con:.4f}"
            )

            # =====================================================
            # Plug stage-2 model + head into self.model so the standard
            # validation pipeline works unchanged.
            # =====================================================
            # Replace the model's fc with the classifier_head
            # stage2_model.fc = classifier_head
            setattr(stage2_model, attr_name, classifier_head)
            self.model = stage2_model.to(self.device)
            log_dict = {}
            # Final evaluation + checkpoint
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
    # Stage 1 — spuriosity ranking
    # ------------------------------------------------------------------
    def _stage1_rank_spuriosity(self):
        cfg = self.cfg
        train_set = self.sets["train"]
        n_samples = len(train_set)

        # Active mask: 1 = sample is still in the active training pool
        # 0 = sample has been "learned" and dropped
        active = torch.ones(n_samples, dtype=torch.float)

        # ranks[i] = round at which sample i was dropped (-1 = never)
        ranks = np.full(n_samples, -1, dtype=np.int64)

        # Stage 1 model: a fresh ERM model
        stage1_model = self.model
        stage1_model.train()
        criterion = _UpweightedTrainingLoss(
            beta_inverse=cfg.MITIGATOR.SEBRA.BETA_INVERSE
        )

        if cfg.SOLVER.TYPE == "SGD":
            stage1_opt = torch.optim.SGD(
                stage1_model.parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            stage1_opt = torch.optim.Adam(
                stage1_model.parameters(),
                lr=cfg.SOLVER.LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )

        rank_rounds = cfg.MITIGATOR.SEBRA.RANK_ROUNDS

        for round_idx in range(1, rank_rounds + 1):
            n_active = int(active.sum().item())
            if n_active == 0:
                print(
                    log_msg(
                        f"Sebra Stage 1 round {round_idx}: no active samples left, stopping",
                        "INFO",
                        self.logger,
                    )
                )
                break

            # Build a WeightedRandomSampler from the active mask. This
            # mirrors the official code which uses a WeightedRandomSampler
            # whose weights are 0/1.
            sampler = torch.utils.data.WeightedRandomSampler(
                active, num_samples=n_active, replacement=True
            )
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                sampler=sampler,
                num_workers=cfg.DATASET.NUM_WORKERS,
                pin_memory=True,
                persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
            )

            # ----- Train one round -----
            stage1_model.train()
            total_loss, n_batches = 0.0, 0
            for batch in train_loader:
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device).long()
                stage1_opt.zero_grad()
                outputs = stage1_model(inputs)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
                loss = criterion(outputs, targets).mean()
                loss.backward()
                stage1_opt.step()
                total_loss += loss.item()
                n_batches += 1

            # ----- Score every sample, drop those with p_y > p_critical -----
            stage1_model.eval()
            score_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.DATASET.NUM_WORKERS,
                pin_memory=True,
                persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
            )
            p_critical = cfg.MITIGATOR.SEBRA.P_CRITICAL
            n_dropped_this_round = 0
            with torch.no_grad():
                for batch in score_loader:
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device).long()
                    indices = batch["index"].long()

                    outputs = stage1_model(inputs)
                    if isinstance(outputs, tuple):
                        outputs, _ = outputs
                    p = F.softmax(outputs, dim=1)
                    py = torch.gather(p, 1, targets.unsqueeze(1)).squeeze(1).cpu()

                    learned = (py > p_critical) & (active[indices] == 1)
                    drop_idx = indices[learned]
                    if drop_idx.numel() > 0:
                        active[drop_idx] = 0
                        ranks[drop_idx.numpy()] = round_idx
                        n_dropped_this_round += int(drop_idx.numel())

            print(
                f"  Sebra Stage 1 round {round_idx}/{rank_rounds}: "
                f"loss={total_loss/max(n_batches,1):.4f} "
                f"active={n_active} dropped={n_dropped_this_round} "
                f"remaining={int(active.sum().item())}"
            )

        # Any sample never dropped gets a final rank one higher than the max
        max_seen = int(ranks.max()) if (ranks >= 0).any() else 0
        never_dropped = ranks == -1
        if never_dropped.any():
            ranks[never_dropped] = max_seen + 1
            print(
                log_msg(
                    f"Sebra Stage 1: {int(never_dropped.sum())} samples never dropped, "
                    f"assigned rank {max_seen + 1}",
                    "INFO",
                    self.logger,
                )
            )

        return ranks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collect_class_labels(self):
        """Walk the training set once to collect class labels per index."""
        train_set = self.sets["train"]
        n = len(train_set)
        labels = np.zeros(n, dtype=np.int64)
        loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.DATASET.NUM_WORKERS,
            pin_memory=False,
        )
        for batch in loader:
            indices = batch["index"].long().numpy()
            targets = batch["targets"].long().numpy()
            labels[indices] = targets
        return labels

    # ------------------------------------------------------------------
    # NSF/BPA-style minimal _train_iter (in case anything calls it)
    # ------------------------------------------------------------------
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
