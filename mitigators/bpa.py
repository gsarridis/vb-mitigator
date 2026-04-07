# BPA — "Unsupervised Learning of Debiased Representations with
# Pseudo-Attributes" (Seo, Lee & Han, CVPR 2022)
# https://github.com/skynbe/pseudo-attributes
#
# Faithful adaptation of the official `bpa_trainer.py` +
# `modules/centroids.py` to the vb-mitigator framework.
#
# High-level pipeline:
#
#   Phase 0 (setup): Train (or load) a base ERM model. The official code
#   asserts that --use_base must be set, so the base model is *required*.
#
#   Phase 1 (initial clustering, once): Use the frozen base model to
#   extract features for every training sample. For each class, run
#   k-means with cosine distance to discover `per_clusters` pseudo-groups.
#   Cluster IDs are offset by `t * per_clusters` to be disjoint across
#   classes. Total pseudo-groups = num_classes * per_clusters.
#
#   Phase 2 (online training): Train the main model end-to-end. For each
#   batch, look up the per-sample cluster weight from a persistent
#   "centroid bank" and use it to scale the per-sample CE loss. After
#   the backward pass, update the bank with each sample's correctness
#   and loss. Periodically recompute per-cluster aggregate statistics
#   (mean loss, mean acc, etc.) used by the reweighting formula.
#
# The reweighting formula (from `Centroids.get_cluster_weights`):
#   1. base_weight[k] = total_count / cluster_count[k]    # inverse-size
#   2. if per-sample losses are populated, multiply by
#      losses_weight[k] = cluster_losses[k] / sum(cluster_losses)
#   3. normalize by the batch mean
#   4. AvgFixedCentroids ('expavg') applies an EMA on top of the base
#      weight: new_w = prev_w * exp(exp_step * new_w)
#
# Notes on the port:
# - We require either an existing ERM checkpoint via CKPT_PATH, or BPA
#   will train its own ERM base model first (similar to NSF).
# - We use `sklearn.cluster.KMeans(...).fit_predict` on L2-normalized
#   features as a proxy for the official `kmeans-pytorch` with
#   distance='cosine'. Cosine k-means is equivalent to Euclidean k-means
#   on L2-normalized vectors, so this is faithful.
# - We do NOT include the HeteroCentroids "multi-k" variant from the
#   official repo (which trains with multiple k values simultaneously) —
#   that's an optional extension on top of the base method.

import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from models.builder import get_model
from models.utils import get_local_model_dict
from tools.utils import load_checkpoint, log_msg, save_checkpoint
from .base_trainer import BaseTrainer


# ---------------------------------------------------------------------------
# Centroid bank — faithful port of modules/centroids.py
# ---------------------------------------------------------------------------


class _AvgFixedCentroids:
    """Persistent per-sample bank that tracks cluster assignments,
    per-sample correctness/loss, and EMA-smoothed per-sample weights.

    Faithful port of `Centroids` + `FixedCentroids` + `AvgFixedCentroids`
    from `modules/centroids.py`. We collapse the three classes into one
    here since the trainer only ever uses the AvgFixedCentroids variant.
    """

    def __init__(
        self,
        num_classes,
        per_clusters,
        n_samples,
        device,
        momentum=0.3,
        exp_step=0.05,
    ):
        self.num_classes = num_classes
        self.per_clusters = per_clusters
        self.n_samples = n_samples
        self.device = device
        self.momentum = momentum
        self.exp_step = exp_step

        # Per-cluster aggregate stats (Y x K)
        self.cluster_losses = torch.zeros((num_classes, per_clusters), device=device)
        self.cluster_accs = torch.zeros((num_classes, per_clusters), device=device)

        # Per-sample state
        self.assigns = None  # (N,) cluster index in [0, num_clusters)
        self.corrects = None  # (N,) -1 = not yet recorded, else 0/1
        self.losses = None  # (N,) -1 = not yet recorded
        self.weights = None  # (N,) per-sample EMA weight (init 1.0)

        self.initialized = False

    @property
    def num_clusters(self):
        return self.num_classes * self.per_clusters

    @property
    def cluster_counts(self):
        if self.assigns is None:
            return torch.zeros(self.num_clusters, device=self.device)
        return torch.bincount(self.assigns, minlength=self.num_clusters).float()

    def initialize(self, cluster_assigns):
        """Called once after the initial k-means clustering."""
        self.assigns = cluster_assigns.to(self.device).long()
        self.corrects = torch.zeros(self.n_samples, device=self.device).long() - 1
        self.losses = torch.zeros(self.n_samples, device=self.device) - 1
        self.weights = torch.ones(self.n_samples, device=self.device)
        self.initialized = True

    def get_cluster_weights(self, ids):
        """Compute per-sample weights for the given batch of indices.

        This implements both the base weighting from `Centroids` and the
        EMA averaging from `AvgFixedCentroids` ('expavg' variant).
        """
        if not self.initialized:
            return torch.ones(len(ids), device=self.device)

        # ---- Base weight (Centroids.get_cluster_weights) ----
        cluster_counts = self.cluster_counts
        # Avoid division by zero
        safe_counts = cluster_counts + (cluster_counts == 0).float()
        cluster_weights = cluster_counts.sum() / safe_counts  # inverse size

        assigns_id = self.assigns[ids]

        # If all per-sample losses have been populated at least once,
        # multiply by the per-cluster loss share.
        all_losses_populated = (self.losses == -1).sum() == 0
        if all_losses_populated:
            cluster_losses_flat = self.cluster_losses.view(-1)
            losses_weight = cluster_losses_flat / (cluster_losses_flat.sum() + 1e-12)
            base_w = cluster_weights[assigns_id] * losses_weight[assigns_id]
        else:
            base_w = cluster_weights[assigns_id]

        # Normalize by batch mean
        base_w = base_w / (base_w.mean() + 1e-12)

        # ---- AvgFixedCentroids 'expavg' EMA ----
        prev = self.weights[ids]
        new_w = prev * torch.exp(self.exp_step * base_w.detach())
        new_w = new_w / (new_w.mean() + 1e-12)
        self.weights[ids] = new_w
        return new_w

    def update(self, logits, target, ids):
        """Record per-sample correctness and loss (called after each batch)."""
        if not self.initialized:
            return
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            corrects = (preds == target).long()
            self.corrects[ids] = corrects
            losses = F.cross_entropy(logits, target.long(), reduction="none").detach()
            self.losses[ids] = losses

    def compute_centroids(self):
        """Refresh per-cluster aggregate stats (mean loss, mean acc).

        Called periodically during training (every `update_cluster_iter`
        iterations) and once at the end of each epoch.
        """
        if not self.initialized:
            return
        for y in range(self.num_classes):
            for k in range(self.per_clusters):
                cluster_id = y * self.per_clusters + k
                ids = (self.assigns == cluster_id).nonzero().flatten()
                if ids.numel() == 0:
                    continue

                # Mean correctness over samples that have been seen at least once
                corr = self.corrects[ids]
                corr_seen = corr[corr >= 0]
                if corr_seen.numel() > 0:
                    self.cluster_accs[y, k] = corr_seen.float().mean()

                # Mean loss over samples that have been seen at least once
                loss = self.losses[ids]
                loss_seen = loss[loss >= 0]
                if loss_seen.numel() > 0:
                    self.cluster_losses[y, k] = loss_seen.float().mean()


# ---------------------------------------------------------------------------
# BPA Trainer
# ---------------------------------------------------------------------------


class BPATrainer(BaseTrainer):
    """
    BPA (Bias Pseudo-Attribute) trainer.

    Two-model design:
      * `self.base_model`: a frozen ERM model used only for the initial
        per-class feature clustering.
      * `self.model`: the trainable model that gets the cluster-reweighted
        training.

    Phase 1 (once, before training): cluster features from `base_model`.
    Phase 2 (every step): standard CE loss reweighted per-sample by the
        centroid bank.
    """

    def _setup_criterion(self):
        if self.cfg.SOLVER.CRITERION == "CE":
            self.criterion_train = nn.CrossEntropyLoss(reduction="none")
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {self.cfg.SOLVER.CRITERION}")

    def _method_specific_setups(self):
        # Initialize the base model (frozen, used for clustering)
        self.base_model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
            self.cfg.MODEL.PRETRAINED,
        ).to(self.device)

        # Load base model checkpoint if provided, otherwise it'll be
        # trained inline in train()
        ckpt_path = self.cfg.MITIGATOR.BPA.CKPT_PATH
        self._base_model_ready = False
        if ckpt_path:
            print(
                log_msg(
                    f"BPA: loading base model from {ckpt_path}",
                    "INFO",
                    self.logger,
                )
            )
            try:
                base_dict = get_local_model_dict(ckpt_path)
                self.base_model.load_state_dict(base_dict["model"])
                self._base_model_ready = True
            except Exception as e:
                logging.warning(
                    f"BPA: failed to load base model via get_local_model_dict ({e}); "
                    f"trying torch.load directly."
                )
                state = torch.load(ckpt_path, map_location=self.device)
                if isinstance(state, dict) and "model" in state:
                    self.base_model.load_state_dict(state["model"])
                else:
                    self.base_model.load_state_dict(state)
                self._base_model_ready = True

        # Centroid bank (initialized in _initial_clustering, after we
        # know the actual training set size)
        self.centroids = None
        self._iter_in_epoch = 0

    # ------------------------------------------------------------------
    # Override train() to handle the base-model bootstrap and initial
    # clustering before falling back to the standard training loop
    # ------------------------------------------------------------------
    def train(self):
        cfg = self.cfg

        # If no base checkpoint was provided, train one inline
        if not self._base_model_ready:
            erm_epochs = cfg.MITIGATOR.BPA.BASE_EPOCHS
            print(
                log_msg(
                    f"BPA: no CKPT_PATH provided; training base ERM model "
                    f"for {erm_epochs} epochs",
                    "INFO",
                    self.logger,
                )
            )
            self._train_base_model(erm_epochs)

        # Run initial clustering once
        self._initial_clustering()

        # Now defer to the standard training loop. Each iteration uses
        # the cluster reweighting in _train_iter.
        super().train()

    def _train_base_model(self, epochs):
        """Train an ERM base model from scratch (used when CKPT_PATH is empty)."""
        cfg = self.cfg
        if cfg.SOLVER.TYPE == "SGD":
            opt = torch.optim.SGD(
                self.base_model.parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            opt = torch.optim.Adam(
                self.base_model.parameters(),
                lr=cfg.SOLVER.LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        for ep in range(epochs):
            self.base_model.train()
            total_loss, correct, total = 0.0, 0, 0
            for batch in self.dataloaders["train"]:
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)
                outputs = self.base_model(inputs)
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
                f"  BPA base epoch {ep+1}/{epochs} "
                f"loss={total_loss/max(len(self.dataloaders['train']),1):.4f} "
                f"acc={correct/max(total,1):.4f}"
            )
        self._base_model_ready = True

    # ------------------------------------------------------------------
    # Phase 1: extract base features and run per-class k-means
    # ------------------------------------------------------------------
    def _initial_clustering(self):
        cfg = self.cfg
        per_clusters = cfg.MITIGATOR.BPA.PER_CLUSTERS

        train_set = self.sets["train"]
        n_samples = len(train_set)

        # Extract features from base model for every training sample
        print(
            log_msg(
                f"BPA: extracting base-model features for {n_samples} training samples",
                "INFO",
                self.logger,
            )
        )
        self.base_model.eval()
        loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
        )

        all_feats = torch.zeros(n_samples, dtype=torch.float)
        feat_buffer = None
        all_targets = torch.zeros(n_samples, dtype=torch.long)

        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]
                indices = batch["index"]
                outputs = self.base_model(inputs)
                if isinstance(outputs, tuple):
                    _, feats = outputs
                else:
                    feats = outputs
                feats = feats.detach().cpu().float()
                if feat_buffer is None:
                    feat_buffer = torch.zeros(
                        n_samples, feats.size(-1), dtype=torch.float
                    )
                feat_buffer[indices] = feats
                all_targets[indices] = targets.long()

        all_feats = feat_buffer

        # L2-normalize features (so Euclidean k-means matches the
        # official cosine k-means)
        all_feats = F.normalize(all_feats, p=2, dim=1)

        # Per-class k-means with disjoint cluster ID offsets
        cluster_assigns = torch.zeros(n_samples, dtype=torch.long)
        for y in range(self.num_class):
            class_mask = (all_targets == y).nonzero().flatten()
            if class_mask.numel() == 0:
                continue
            class_feats = all_feats[class_mask].numpy()

            n_in_class = class_feats.shape[0]
            k = min(per_clusters, max(2, n_in_class - 1))
            if n_in_class < 2:
                cluster_assigns[class_mask] = y * per_clusters
                continue

            kmeans = KMeans(
                n_clusters=k,
                random_state=cfg.EXPERIMENT.SEED,
                n_init=10,
            )
            ids = kmeans.fit_predict(class_feats)
            cluster_assigns[class_mask] = (
                torch.from_numpy(ids).long() + y * per_clusters
            )

            print(
                f"  BPA cluster init: class {y} -> {k} clusters from {n_in_class} samples"
            )

        # Initialize the centroid bank
        self.centroids = _AvgFixedCentroids(
            num_classes=self.num_class,
            per_clusters=per_clusters,
            n_samples=n_samples,
            device=self.device,
            momentum=cfg.MITIGATOR.BPA.MOMENTUM,
            exp_step=cfg.MITIGATOR.BPA.EXP_STEP,
        )
        self.centroids.initialize(cluster_assigns)

        cluster_count_str = ", ".join(
            f"{int(c)}" for c in self.centroids.cluster_counts.tolist()
        )
        print(
            log_msg(
                f"BPA: initial cluster counts = [{cluster_count_str}]",
                "INFO",
                self.logger,
            )
        )

    # ------------------------------------------------------------------
    # Phase 2: per-iteration training with cluster reweighting
    # ------------------------------------------------------------------
    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        indices = batch["index"].to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs

        # Per-sample loss
        loss_per_sample = self.criterion_train(outputs, targets)

        # Cluster reweighting from the centroid bank
        weights = self.centroids.get_cluster_weights(indices)

        loss = (loss_per_sample * weights).mean()

        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()

        # Update centroid bank with this batch's outputs
        self.centroids.update(outputs.detach(), targets, indices)

        # Periodically refresh per-cluster aggregate stats
        self._iter_in_epoch += 1
        update_iter = self.cfg.MITIGATOR.BPA.UPDATE_CLUSTER_ITER
        if update_iter > 0 and self._iter_in_epoch % update_iter == 0:
            self.centroids.compute_centroids()

        return {"train_cls_loss": loss}

    def _train_epoch(self):
        self._iter_in_epoch = 0
        log_dict = super()._train_epoch()
        # End-of-epoch refresh of per-cluster stats
        if self.centroids is not None:
            self.centroids.compute_centroids()
        return log_dict

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
        }
        if self.centroids is not None:
            state["centroid_assigns"] = self.centroids.assigns
            state["centroid_weights"] = self.centroids.weights
            state["centroid_losses"] = self.centroids.losses
            state["centroid_corrects"] = self.centroids.corrects
        save_checkpoint(state, os.path.join(self.log_path, tag))

    def load_checkpoint(self, tag):
        checkpoint = load_checkpoint(os.path.join(self.log_path, tag))
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_performance = checkpoint["best_performance"]
        self.current_epoch = checkpoint["epoch"]
        if self.centroids is not None and "centroid_assigns" in checkpoint:
            self.centroids.assigns = checkpoint["centroid_assigns"]
            self.centroids.weights = checkpoint["centroid_weights"]
            self.centroids.losses = checkpoint["centroid_losses"]
            self.centroids.corrects = checkpoint["centroid_corrects"]
            self.centroids.initialized = True
        print(
            log_msg(
                f"Loaded checkpoint from {os.path.join(self.log_path, tag)}",
                "INFO",
                self.logger,
            )
        )


# # BPA - Unsupervised Learning of Debiased Representations with
# # Pseudo-Attributes (Seo et al., CVPR 2022)
# #
# # Key idea:
# # 1. Train an ERM model to discover pseudo bias-attributes via
# #    clustering of misclassified vs correctly classified samples.
# # 2. Use the pseudo-attributes to apply bias-balanced (BB-style)
# #    logit correction during retraining.
# #
# # The method is BLU (Bias Label Unaware) since it discovers bias
# # structure from model behavior rather than requiring explicit
# # bias annotations.

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.cluster import KMeans

# from my_datasets.builder import get_dataset
# from models.builder import get_model
# from tools.utils import load_checkpoint, log_msg, save_checkpoint
# from .base_trainer import BaseTrainer
# from .losses import GeneralizedCECriterion


# class BPATrainer(BaseTrainer):
#     """
#     BPA trainer.

#     Phase 1: Train an ERM model, identify bias-aligned (easy/correct)
#              vs bias-conflicting (hard/misclassified) samples.
#     Phase 2: Cluster embeddings within each class to discover
#              pseudo bias-attribute groups.
#     Phase 3: Retrain with logit adjustment using the pseudo-attribute
#              prior (similar to BB but with discovered pseudo-attributes).
#     """

#     def _setup_criterion(self):
#         if self.cfg.SOLVER.CRITERION == "CE":
#             self.criterion_train = nn.CrossEntropyLoss(reduction="none")
#             self.criterion = nn.CrossEntropyLoss()
#         else:
#             raise ValueError(f"Unsupported criterion type: {self.cfg.SOLVER.CRITERION}")

#     def _method_specific_setups(self):
#         self._discover_pseudo_attributes()

#     def _discover_pseudo_attributes(self):
#         """
#         Phase 1 & 2: Train ERM, extract embeddings, cluster to find
#         pseudo bias-attributes based on model confidence patterns.
#         """
#         cfg = self.cfg
#         discovery_epochs = cfg.MITIGATOR.BPA.DISCOVERY_EPOCHS
#         MODEL_SAVE_PATH = os.path.join(self.log_path, "bpa_discovery_model")
#         PSEUDO_SAVE_PATH = os.path.join(self.log_path, "bpa_pseudo_attrs.pt")

#         # Check if pseudo-attributes already computed
#         if os.path.exists(PSEUDO_SAVE_PATH):
#             print("Loading pre-computed BPA pseudo-attributes...")
#             pseudo_data = torch.load(PSEUDO_SAVE_PATH, map_location=self.device)
#             self.pseudo_attrs = pseudo_data["pseudo_attrs"]
#             self.bias_prior = pseudo_data["bias_prior"]
#             self._setup_models()
#             self._setup_optimizer()
#             return

#         # Phase 1: Train ERM model
#         if os.path.exists(MODEL_SAVE_PATH):
#             print("Loading pre-trained BPA discovery model...")
#             self.model.load_state_dict(
#                 torch.load(MODEL_SAVE_PATH, map_location=self.device)
#             )
#         else:
#             print(
#                 f"BPA: Training ERM for {discovery_epochs} epochs for pseudo-attribute discovery."
#             )
#             erm_optimizer = torch.optim.Adam(
#                 self.model.parameters(),
#                 lr=cfg.SOLVER.LR,
#                 weight_decay=cfg.SOLVER.WEIGHT_DECAY,
#             )
#             self.model.train()
#             for epoch in range(discovery_epochs):
#                 total_loss = 0.0
#                 correct = 0
#                 total = 0
#                 for batch in self.dataloaders["train"]:
#                     inputs = batch["inputs"].to(self.device)
#                     targets = batch["targets"].to(self.device)
#                     outputs = self.model(inputs)
#                     if isinstance(outputs, tuple):
#                         outputs, _ = outputs
#                     loss = self.criterion(outputs, targets)
#                     loss.backward()
#                     erm_optimizer.step()
#                     erm_optimizer.zero_grad(set_to_none=True)
#                     total_loss += loss.item()
#                     predicted = outputs.argmax(dim=1)
#                     correct += (predicted == targets).sum().item()
#                     total += targets.size(0)
#                 avg_loss = total_loss / len(self.dataloaders["train"])
#                 accuracy = correct / total * 100 if total > 0 else 0
#                 print(
#                     f"BPA Discovery Epoch [{epoch+1}/{discovery_epochs}] - "
#                     f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
#                 )
#             os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
#             torch.save(self.model.state_dict(), MODEL_SAVE_PATH)

#         # Phase 2: Extract embeddings and discover pseudo-attributes
#         print("BPA: Extracting embeddings for pseudo-attribute discovery...")
#         self.model.eval()
#         all_features = []
#         all_targets = []
#         all_losses = []

#         ordered_loader = torch.utils.data.DataLoader(
#             self.sets["train"],
#             batch_size=cfg.SOLVER.BATCH_SIZE,
#             shuffle=False,
#             num_workers=cfg.DATASET.NUM_WORKERS,
#             pin_memory=True,
#             persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
#         )

#         with torch.no_grad():
#             for batch in ordered_loader:
#                 inputs = batch["inputs"].to(self.device)
#                 targets = batch["targets"].to(self.device)
#                 outputs = self.model(inputs)
#                 if isinstance(outputs, tuple):
#                     _, feats = outputs
#                     outputs_logits = self.model(inputs)
#                     if isinstance(outputs_logits, tuple):
#                         outputs_logits, _ = outputs_logits
#                 else:
#                     feats = outputs
#                     outputs_logits = outputs

#                 per_sample_loss = self.criterion_train(outputs_logits, targets)
#                 all_features.append(feats.cpu())
#                 all_targets.append(targets.cpu())
#                 all_losses.append(per_sample_loss.cpu())

#         all_features = torch.cat(all_features, dim=0).numpy()
#         all_targets = torch.cat(all_targets, dim=0).numpy()
#         all_losses = torch.cat(all_losses, dim=0).numpy()

#         # Cluster within each class using loss + features
#         num_pseudo = cfg.MITIGATOR.BPA.NUM_PSEUDO_ATTRS
#         print(f"BPA: Discovering {num_pseudo} pseudo-attributes per class...")

#         pseudo_attrs = np.zeros(len(all_targets), dtype=np.int64)
#         for class_idx in range(self.num_class):
#             class_mask = all_targets == class_idx
#             class_features = all_features[class_mask]
#             class_losses = all_losses[class_mask].reshape(-1, 1)

#             if len(class_features) < num_pseudo:
#                 pseudo_attrs[class_mask] = 0
#                 continue

#             # Combine features with normalized loss for clustering
#             loss_weight = cfg.MITIGATOR.BPA.LOSS_WEIGHT
#             normalized_losses = (class_losses - class_losses.mean()) / (
#                 class_losses.std() + 1e-8
#             )
#             # Normalize features
#             feat_norms = np.linalg.norm(class_features, axis=1, keepdims=True)
#             normalized_features = class_features / (feat_norms + 1e-8)
#             clustering_features = np.concatenate(
#                 [normalized_features, loss_weight * normalized_losses], axis=1
#             )

#             kmeans = KMeans(
#                 n_clusters=num_pseudo,
#                 random_state=cfg.EXPERIMENT.SEED,
#                 n_init=10,
#             )
#             cluster_ids = kmeans.fit_predict(clustering_features)
#             pseudo_attrs[class_mask] = cluster_ids

#         self.pseudo_attrs = torch.from_numpy(pseudo_attrs).long()

#         # Compute bias prior: p(pseudo_attr | class)
#         # Shape: [num_pseudo_attrs, num_class]
#         bias_prior = torch.zeros(num_pseudo, self.num_class)
#         for class_idx in range(self.num_class):
#             class_mask = torch.from_numpy(all_targets) == class_idx
#             class_pseudo = self.pseudo_attrs[class_mask]
#             for attr_idx in range(num_pseudo):
#                 bias_prior[attr_idx, class_idx] = (class_pseudo == attr_idx).float().mean()

#         self.bias_prior = bias_prior

#         # Save
#         os.makedirs(os.path.dirname(PSEUDO_SAVE_PATH) if os.path.dirname(PSEUDO_SAVE_PATH) else ".", exist_ok=True)
#         torch.save(
#             {
#                 "pseudo_attrs": self.pseudo_attrs,
#                 "bias_prior": self.bias_prior,
#             },
#             PSEUDO_SAVE_PATH,
#         )
#         print(f"BPA: Discovered pseudo-attributes for {len(self.pseudo_attrs)} samples.")

#         # Reset model for retraining phase
#         self._setup_models()
#         self._setup_optimizer()

#     def _train_iter(self, batch):
#         inputs = batch["inputs"].to(self.device)
#         targets = batch["targets"].to(self.device)
#         indices = batch["index"]

#         self.optimizer.zero_grad()
#         outputs = self.model(inputs)
#         if isinstance(outputs, tuple):
#             outputs, _ = outputs

#         # Look up pseudo bias-attributes for this batch
#         pseudo_attr = self.pseudo_attrs[indices].to(self.device)

#         # Logit adjustment using pseudo-attribute prior (BB-style)
#         # For each sample, subtract log p(pseudo_attr | class) from logits
#         bias_prior_device = self.bias_prior.to(self.device)
#         # Get the prior vector for each sample's pseudo attribute
#         # bias_prior[pseudo_attr[i], :] gives the prior over classes for that pseudo-attr
#         prior_vectors = bias_prior_device[pseudo_attr]  # [batch, num_class]
#         adjusted_logits = outputs - torch.log(prior_vectors + 1e-9)

#         loss = self.criterion(adjusted_logits, targets)
#         self._loss_backward(loss)
#         self._optimizer_step()
#         self.scheduler.step()
#         return {"train_cls_loss": loss}

#     def _save_checkpoint(self, tag):
#         state = {
#             "epoch": self.current_epoch,
#             "model": self.model.state_dict(),
#             "optimizer": self.optimizer.state_dict(),
#             "best_performance": self.best_performance,
#             "scheduler": self.scheduler.state_dict(),
#             "pseudo_attrs": self.pseudo_attrs,
#             "bias_prior": self.bias_prior,
#         }
#         save_checkpoint(state, os.path.join(self.log_path, tag))

#     def load_checkpoint(self, tag):
#         checkpoint = load_checkpoint(os.path.join(self.log_path, tag))
#         self.model.load_state_dict(checkpoint["model"])
#         self.optimizer.load_state_dict(checkpoint["optimizer"])
#         self.scheduler.load_state_dict(checkpoint["scheduler"])
#         self.best_performance = checkpoint["best_performance"]
#         self.current_epoch = checkpoint["epoch"]
#         self.pseudo_attrs = checkpoint["pseudo_attrs"]
#         self.bias_prior = checkpoint["bias_prior"]
#         print(
#             log_msg(
#                 f"Loaded checkpoint from {os.path.join(self.log_path, tag)}",
#                 "INFO",
#                 self.logger,
#             )
#         )
