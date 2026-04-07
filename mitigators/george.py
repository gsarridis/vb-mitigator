# GEORGE (Sohoni et al., NeurIPS 2020)
# "No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained
#  Classification Problems"
# https://github.com/HazyResearch/hidden-stratification
#
# Faithful re-implementation of the official algorithm:
#
#   1. Train an ERM model on the superclass labels.
#   2. Extract penultimate-layer features for every training (and val)
#      sample, partitioned by superclass.
#   3. (Optional) Reduce dimensionality with UMAP.
#   4. For each superclass independently, fit an AutoK mixture model:
#      sweep k = 2 .. max_k, fit GMM (or k-means) at each k, select the k
#      with the best mean silhouette score. Each superclass may end up
#      with a different number of clusters; cluster IDs are made disjoint
#      across superclasses by adding a per-class offset.
#   5. Train a new model from scratch with GroupDRO over the discovered
#      pseudo-groups.
#
# Notes on differences from the official repo:
# - The official repo also implements an "OverclusterModel" (overcluster
#   then merge based on per-cluster loss). We do NOT include it here:
#   the default GEORGE configs across most datasets use the plain
#   AutoKMixtureModel, and overclustering would require validation losses
#   from the discovery model that complicate the integration. The
#   `cluster_method` config key still allows choosing between gmm and
#   kmeans, matching the official `cluster_config.model`.
# - The official repo uses a "HardnessAugmentedReducer" for some datasets
#   (e.g. waterbirds), which projects features along the ERM model's
#   decision boundary and concatenates a UMAP reduction of the
#   complementary subspace. This requires accessing the linear FC weights
#   of the trained model. We support 'none', 'pca', and 'umap' reducers
#   here; 'hardness' is left as future work.

import os
import logging
import warnings

import numpy as np
import torch
import torch.nn as nn

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples

from my_datasets.builder import get_dataset
from models.builder import get_model
from tools.utils import load_checkpoint, log_msg, save_checkpoint
from .base_trainer import BaseTrainer


# ---------------------------------------------------------------------------
# Cluster + reduction helpers (adapted from official GEORGE code)
# ---------------------------------------------------------------------------


def _get_cluster_sils(data, pred_labels):
    """Per-cluster mean silhouette scores."""
    unique_preds = sorted(np.unique(pred_labels))
    if len(unique_preds) < 2:
        return {int(unique_preds[0]): 0.0}, 0.0
    sil_samples = silhouette_samples(data, pred_labels)
    sils_by_cluster = {
        int(label): float(np.mean(sil_samples[pred_labels == label]))
        for label in unique_preds
    }
    sil_global = float(np.mean(sil_samples))
    return sils_by_cluster, sil_global


class _AutoKMixtureModel:
    """Auto-k clustering: sweep k = 2..max_k, pick by mean silhouette.

    Faithful translation of the official AutoKMixtureModel (with the
    silhouette-search branch). Supports 'kmeans' and 'gmm' as in the paper.
    """

    def __init__(self, cluster_method, max_k, n_init=3, seed=None, search=True):
        if cluster_method == "kmeans":
            self.cluster_cls = KMeans
            self.k_name = "n_clusters"
        elif cluster_method == "gmm":
            self.cluster_cls = GaussianMixture
            self.k_name = "n_components"
        else:
            raise ValueError(f"Unsupported cluster_method: {cluster_method}")

        self.cluster_method = cluster_method
        self.max_k = max_k
        self.n_init = n_init
        self.seed = seed
        self.search = search

        self.best_k = None
        self.n_clusters = None
        self.cluster_obj = None

    def _gen_inner(self, k):
        return self.cluster_cls(
            **{self.k_name: k}, n_init=self.n_init, random_state=self.seed
        )

    def fit(self, activ):
        # Degenerate cases: too few samples to cluster meaningfully.
        n = len(activ)
        if n < 2:
            self.n_clusters = 1
            self.best_k = 1
            self.cluster_obj = None
            return self

        eff_max_k = max(2, min(self.max_k, n - 1))

        if not self.search or eff_max_k == 2:
            cluster_obj = self._gen_inner(eff_max_k)
            cluster_obj.fit(activ)
            self.cluster_obj = cluster_obj
            self.best_k = eff_max_k
            self.n_clusters = eff_max_k
            return self

        best_score = -2.0
        best_obj = None
        best_k = 2
        for k in range(2, eff_max_k + 1):
            cluster_obj = self._gen_inner(k)
            try:
                pred_labels = cluster_obj.fit_predict(activ)
            except Exception:
                continue
            if len(np.unique(pred_labels)) < 2:
                continue
            local_sils, _ = _get_cluster_sils(activ, pred_labels)
            score = float(np.mean(list(local_sils.values())))
            if score > best_score:
                best_score = score
                best_obj = cluster_obj
                best_k = k

        if best_obj is None:
            # Fallback: single cluster
            self.n_clusters = 1
            self.best_k = 1
            self.cluster_obj = None
            return self

        self.cluster_obj = best_obj
        self.best_k = best_k
        self.n_clusters = best_k
        return self

    def predict(self, activ):
        if self.cluster_obj is None:
            return np.zeros(len(activ), dtype=np.int64)
        return self.cluster_obj.predict(activ)

    def fit_predict(self, activ):
        self.fit(activ)
        return self.predict(activ)


def _build_reducer(name, n_components, seed):
    """Return a sklearn-style reducer with .fit/.transform.

    Supported names: 'none', 'pca', 'umap'.
    """
    name = name.lower()
    if name == "none":

        class _NoOpReducer:
            def fit(self, X):
                return self

            def transform(self, X):
                return X

        return _NoOpReducer()

    if name == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, random_state=seed)

    if name == "umap":
        try:
            from umap import UMAP
        except ImportError as e:
            raise ImportError(
                "UMAP requested but `umap-learn` is not installed. "
                "Either `pip install umap-learn` or set "
                "MITIGATOR.GEORGE.REDUCTION = 'none'."
            ) from e
        # Match the defaults the official repo uses for UMAPReducer
        return UMAP(
            n_components=n_components,
            n_neighbors=10,
            min_dist=0.0,
            random_state=seed,
        )

    raise ValueError(f"Unsupported reduction model: {name}")


# ---------------------------------------------------------------------------
# GEORGE trainer
# ---------------------------------------------------------------------------


class GeorgeTrainer(BaseTrainer):
    """
    GEORGE trainer.

    Phase 1: Train an ERM model on the superclass label.
    Phase 2: Extract per-superclass penultimate features, optionally reduce
             with PCA/UMAP, then run AutoKMixtureModel per superclass to
             discover pseudo-groups (cluster IDs are offset to be disjoint).
    Phase 3: Reset the model and retrain with GroupDRO over the discovered
             pseudo-groups.
    """

    def _setup_dataset(self):
        dataset = get_dataset(self.cfg)
        self.num_class = dataset["num_class"]
        self.biases = dataset["biases"]
        self.dataloaders = dataset["dataloaders"]
        self.data_root = dataset["root"]
        self.sets = dataset["sets"]
        self.target2name = dataset["target2name"]
        self.ba_groups = dataset["ba_groups"] if "ba_groups" in dataset else None

        # num_group will be set after subgroup discovery, but we need a
        # placeholder so other parts of BaseTrainer don't break.
        self.num_group = self.num_class
        self.num_biases = 1
        return

    def _setup_criterion(self):
        if self.cfg.SOLVER.CRITERION == "CE":
            self.criterion_train = nn.CrossEntropyLoss(reduction="none")
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {self.cfg.SOLVER.CRITERION}")

    def _method_specific_setups(self):
        # Phase 1+2: discover pseudo-groups via ERM training and clustering
        self._discover_subgroups()

        # Phase 3: initialize GroupDRO state over discovered groups
        self.adv_probs = torch.ones(self.num_group, device=self.device) / self.num_group
        self.group_range = torch.arange(
            self.num_group, dtype=torch.long, device=self.device
        ).unsqueeze(1)

    # ------------------------------------------------------------------
    # Phase 1+2: ERM training, feature extraction, clustering
    # ------------------------------------------------------------------
    def _discover_subgroups(self):
        cfg = self.cfg
        discovery_epochs = cfg.MITIGATOR.GEORGE.DISCOVERY_EPOCHS
        cluster_method = cfg.MITIGATOR.GEORGE.CLUSTER_METHOD
        max_k = cfg.MITIGATOR.GEORGE.MAX_K
        search_k = cfg.MITIGATOR.GEORGE.SEARCH_K
        reduction_name = cfg.MITIGATOR.GEORGE.REDUCTION
        reduction_components = cfg.MITIGATOR.GEORGE.REDUCTION_COMPONENTS

        DISCOVERY_MODEL_PATH = os.path.join(self.log_path, "george_discovery_model")
        CLUSTERS_PATH = os.path.join(self.log_path, "george_pseudo_groups.pt")

        # Reuse if already cached
        if os.path.exists(CLUSTERS_PATH):
            print("Loading pre-computed GEORGE pseudo-groups...")
            cluster_data = torch.load(CLUSTERS_PATH, map_location="cpu")
            self.pseudo_group_labels = cluster_data["pseudo_group_labels"]
            self.num_group = int(cluster_data["num_group"])
            print(
                f"GEORGE: loaded {self.num_group} pseudo-groups for {len(self.pseudo_group_labels)} samples."
            )
            self._setup_models()
            self._setup_optimizer()
            return

        # ----- Phase 1: train ERM discovery model -----
        if os.path.exists(DISCOVERY_MODEL_PATH):
            print("Loading pre-trained GEORGE discovery model...")
            self.model.load_state_dict(
                torch.load(DISCOVERY_MODEL_PATH, map_location=self.device)
            )
        else:
            print(
                log_msg(
                    f"GEORGE: training ERM for {discovery_epochs} epochs to discover subgroups",
                    "INFO",
                    self.logger,
                )
            )
            erm_optimizer = self._make_optimizer(self.model.parameters())
            self.model.train()
            for epoch in range(discovery_epochs):
                total_loss, correct, total = 0.0, 0, 0
                for batch in self.dataloaders["train"]:
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs, _ = outputs
                    loss = self.criterion(outputs, targets)
                    erm_optimizer.zero_grad()
                    loss.backward()
                    erm_optimizer.step()
                    total_loss += loss.item()
                    pred = outputs.argmax(dim=1)
                    correct += (pred == targets).sum().item()
                    total += targets.size(0)
                avg_loss = total_loss / max(len(self.dataloaders["train"]), 1)
                acc = correct / max(total, 1)
                print(
                    f"  GEORGE discovery epoch {epoch+1}/{discovery_epochs} loss={avg_loss:.4f} acc={acc:.4f}"
                )

            os.makedirs(self.log_path, exist_ok=True)
            torch.save(self.model.state_dict(), DISCOVERY_MODEL_PATH)

        # ----- Phase 2: extract features and cluster -----
        print("GEORGE: extracting penultimate features for clustering...")
        train_features, train_targets = self._extract_features(self.sets["train"])

        # Optional: extract val features too if reduction needs them (UMAP
        # is fit on train data only, which matches the official code that
        # fits the reducer per group on train activations).

        # Allocate output array (per-sample pseudo-group ID)
        n_train = len(train_targets)
        pseudo_group_labels = np.zeros(n_train, dtype=np.int64)

        cluster_floor = 0  # ensures disjoint group IDs across superclasses
        for class_idx in range(self.num_class):
            class_mask = train_targets == class_idx
            n_in_class = int(class_mask.sum())
            if n_in_class == 0:
                continue
            class_feats = train_features[class_mask]

            # Reduce dimensionality
            reducer = _build_reducer(
                reduction_name, reduction_components, cfg.EXPERIMENT.SEED
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    reducer.fit(class_feats)
                    reduced = reducer.transform(class_feats)
            except Exception as e:
                logging.warning(
                    f"GEORGE: reduction failed for superclass {class_idx} ({e}); using raw features."
                )
                reduced = class_feats

            # Auto-k clustering
            am = _AutoKMixtureModel(
                cluster_method=cluster_method,
                max_k=max_k,
                seed=cfg.EXPERIMENT.SEED,
                search=search_k,
            )
            am.fit(reduced)
            cluster_ids = am.predict(reduced).astype(np.int64)

            # Make IDs disjoint with previous superclasses
            pseudo_group_labels[class_mask] = cluster_ids + cluster_floor
            cluster_floor += am.n_clusters
            print(
                f"  superclass {class_idx}: {n_in_class} samples -> {am.n_clusters} pseudo-groups"
            )

        self.pseudo_group_labels = torch.from_numpy(pseudo_group_labels).long()
        self.num_group = int(cluster_floor)
        print(
            log_msg(
                f"GEORGE: discovered {self.num_group} total pseudo-groups across {self.num_class} superclasses",
                "INFO",
                self.logger,
            )
        )

        torch.save(
            {
                "pseudo_group_labels": self.pseudo_group_labels,
                "num_group": self.num_group,
            },
            CLUSTERS_PATH,
        )

        # ----- Reset model + optimizer for phase 3 -----
        self._setup_models()
        self._setup_optimizer()

    def _make_optimizer(self, params):
        cfg = self.cfg
        if cfg.SOLVER.TYPE == "SGD":
            return torch.optim.SGD(
                params,
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif cfg.SOLVER.TYPE in ("Adam", "AdamW"):
            return torch.optim.Adam(
                params,
                lr=cfg.SOLVER.LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {cfg.SOLVER.TYPE}")

    def _extract_features(self, dataset):
        """Extract penultimate-layer features for every sample.

        Models in vb-mitigator return a (logits, features) tuple from
        their forward pass, where `features` is the penultimate
        representation. We use those features directly.
        """
        cfg = self.cfg
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
        )

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
                all_feats.append(feats.detach().cpu().float().numpy())
                all_targets.append(targets.numpy())

        return np.concatenate(all_feats, axis=0), np.concatenate(all_targets, axis=0)

    # ------------------------------------------------------------------
    # Phase 3: GroupDRO over discovered pseudo-groups
    # ------------------------------------------------------------------
    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        indices = batch["index"]

        # Look up pseudo-group labels for this batch
        group_index = self.pseudo_group_labels[indices.cpu()].to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs

        loss_per_sample = self.criterion_train(outputs, targets)

        # Group-wise mean loss
        group_map = (group_index.unsqueeze(0) == self.group_range).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()
        group_loss = (group_map @ loss_per_sample.flatten()) / group_denom

        # Update adversarial probabilities (multiplicative weights)
        with torch.no_grad():
            self.adv_probs = self.adv_probs * torch.exp(
                self.cfg.MITIGATOR.GEORGE.ROBUST_STEP_SIZE * group_loss.detach()
            )
            self.adv_probs = self.adv_probs / self.adv_probs.sum()

        # Reweighted robust loss
        loss = group_loss @ self.adv_probs
        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()
        return {"train_cls_loss": loss}

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
            "adv_probs": self.adv_probs,
            "pseudo_group_labels": self.pseudo_group_labels,
            "num_group": self.num_group,
        }
        save_checkpoint(state, os.path.join(self.log_path, tag))

    def load_checkpoint(self, tag):
        checkpoint = load_checkpoint(os.path.join(self.log_path, tag))
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_performance = checkpoint["best_performance"]
        self.current_epoch = checkpoint["epoch"]
        self.adv_probs = checkpoint["adv_probs"]
        self.pseudo_group_labels = checkpoint["pseudo_group_labels"]
        self.num_group = int(checkpoint["num_group"])
        # Re-create group_range to match num_group from checkpoint
        self.group_range = torch.arange(
            self.num_group, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        print(
            log_msg(
                f"Loaded checkpoint from {os.path.join(self.log_path, tag)}",
                "INFO",
                self.logger,
            )
        )


# # George (Sohoni et al., NeurIPS 2020)
# # "No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained
# #  Classification Problems"
# #
# # Key idea: Two-stage approach:
# # 1. Train an ERM model, extract embeddings, cluster them within each
# #    class to discover hidden subgroups (pseudo-groups).
# # 2. Retrain with GroupDRO-style optimization over the discovered
# #    pseudo-groups.

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.cluster import KMeans

# from my_datasets.builder import get_dataset
# from models.builder import get_model
# from tools.utils import load_checkpoint, log_msg, save_checkpoint
# from .base_trainer import BaseTrainer


# class GeorgeTrainer(BaseTrainer):
#     """
#     George trainer (GEORGE: Group-aware ERM with Oversampling and
#     Reweighting for Group-Enhanced robustness).

#     Phase 1: Train an ERM model for a set number of epochs.
#     Phase 2: Extract feature embeddings, cluster within each class
#              using k-means to discover pseudo-groups.
#     Phase 3: Retrain with GroupDRO over the discovered pseudo-groups.
#     """

#     def _setup_dataset(self):
#         dataset = get_dataset(self.cfg)
#         self.num_class = dataset["num_class"]
#         self.biases = dataset["biases"]
#         self.dataloaders = dataset["dataloaders"]
#         self.data_root = dataset["root"]
#         self.sets = dataset["sets"]
#         self.target2name = dataset["target2name"]
#         self.ba_groups = dataset["ba_groups"] if "ba_groups" in dataset else None

#         # Total pseudo-groups = num_class * num_clusters_per_class
#         self.num_clusters = self.cfg.MITIGATOR.GEORGE.NUM_CLUSTERS
#         self.num_group = self.num_class * self.num_clusters
#         self.num_biases = self.num_clusters
#         return

#     def _setup_criterion(self):
#         if self.cfg.SOLVER.CRITERION == "CE":
#             self.criterion_train = nn.CrossEntropyLoss(reduction="none")
#             self.criterion = nn.CrossEntropyLoss()
#         else:
#             raise ValueError(f"Unsupported criterion type: {self.cfg.SOLVER.CRITERION}")

#     def _method_specific_setups(self):
#         # Run the subgroup discovery pipeline
#         self._discover_subgroups()

#         # Initialize GroupDRO parameters over discovered pseudo-groups
#         self.adv_probs = torch.ones(self.num_group, device=self.device) / self.num_group
#         self.group_range = torch.arange(
#             self.num_group, dtype=torch.long, device=self.device
#         ).unsqueeze(1)

#     def _discover_subgroups(self):
#         """
#         Phase 1 & 2: Train ERM model briefly, extract embeddings,
#         cluster to discover pseudo-groups.
#         """
#         cfg = self.cfg
#         discovery_epochs = cfg.MITIGATOR.GEORGE.DISCOVERY_EPOCHS
#         MODEL_SAVE_PATH = os.path.join(self.log_path, "george_discovery_model")
#         CLUSTERS_SAVE_PATH = os.path.join(self.log_path, "george_clusters.pt")

#         # Check if clusters already computed
#         if os.path.exists(CLUSTERS_SAVE_PATH):
#             print("Loading pre-computed George clusters...")
#             cluster_data = torch.load(CLUSTERS_SAVE_PATH, map_location=self.device)
#             self.pseudo_group_labels = cluster_data["pseudo_group_labels"]
#             self._setup_models()
#             self._setup_optimizer()
#             return

#         # Phase 1: Train ERM model
#         if os.path.exists(MODEL_SAVE_PATH):
#             print("Loading pre-trained George discovery model...")
#             self.model.load_state_dict(
#                 torch.load(MODEL_SAVE_PATH, map_location=self.device)
#             )
#         else:
#             print(
#                 f"George: Training ERM for {discovery_epochs} epochs for subgroup discovery."
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
#                     f"George Discovery Epoch [{epoch+1}/{discovery_epochs}] - "
#                     f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
#                 )
#             os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
#             torch.save(self.model.state_dict(), MODEL_SAVE_PATH)

#         # Phase 2: Extract embeddings and cluster
#         print("George: Extracting embeddings for clustering...")
#         self.model.eval()
#         all_features = []
#         all_targets = []

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
#                 targets = batch["targets"]
#                 outputs = self.model(inputs)
#                 if isinstance(outputs, tuple):
#                     _, feats = outputs
#                 else:
#                     # If model doesn't return features, use penultimate layer
#                     feats = outputs
#                 all_features.append(feats.cpu())
#                 all_targets.append(targets)

#         all_features = torch.cat(all_features, dim=0).numpy()
#         all_targets = torch.cat(all_targets, dim=0).numpy()

#         # Cluster within each class
#         print(
#             f"George: Clustering with {self.num_clusters} clusters per class..."
#         )
#         pseudo_group_labels = np.zeros(len(all_targets), dtype=np.int64)
#         for class_idx in range(self.num_class):
#             class_mask = all_targets == class_idx
#             class_features = all_features[class_mask]

#             if len(class_features) < self.num_clusters:
#                 # Not enough samples; assign all to group 0
#                 pseudo_group_labels[class_mask] = class_idx * self.num_clusters
#                 continue

#             kmeans = KMeans(
#                 n_clusters=self.num_clusters,
#                 random_state=cfg.EXPERIMENT.SEED,
#                 n_init=10,
#             )
#             cluster_ids = kmeans.fit_predict(class_features)
#             pseudo_group_labels[class_mask] = (
#                 class_idx * self.num_clusters + cluster_ids
#             )

#         self.pseudo_group_labels = torch.from_numpy(pseudo_group_labels).long()

#         # Save clusters
#         os.makedirs(os.path.dirname(CLUSTERS_SAVE_PATH) if os.path.dirname(CLUSTERS_SAVE_PATH) else ".", exist_ok=True)
#         torch.save(
#             {"pseudo_group_labels": self.pseudo_group_labels}, CLUSTERS_SAVE_PATH
#         )
#         print(f"George: Discovered {self.num_group} pseudo-groups.")

#         # Reset model for phase 3
#         self._setup_models()
#         self._setup_optimizer()

#     def _train_iter(self, batch):
#         inputs = batch["inputs"].to(self.device)
#         targets = batch["targets"].to(self.device)
#         indices = batch["index"]

#         # Look up pseudo-group labels
#         group_index = self.pseudo_group_labels[indices].to(self.device)

#         self.optimizer.zero_grad()
#         outputs = self.model(inputs)
#         if isinstance(outputs, tuple):
#             outputs, _ = outputs
#         loss_per_sample = self.criterion_train(outputs, targets)

#         # GroupDRO over pseudo-groups
#         group_map = (group_index.unsqueeze(0) == self.group_range).float()
#         group_count = group_map.sum(1)
#         group_denom = group_count + (group_count == 0).float()
#         group_loss = (group_map @ loss_per_sample.flatten()) / group_denom

#         # Update adversarial probabilities
#         with torch.no_grad():
#             self.adv_probs = self.adv_probs * torch.exp(
#                 self.cfg.MITIGATOR.GEORGE.ROBUST_STEP_SIZE * group_loss.detach()
#             )
#             self.adv_probs = self.adv_probs / (self.adv_probs.sum())

#         # Reweighted robust loss
#         loss = group_loss @ self.adv_probs
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
#             "adv_probs": self.adv_probs,
#             "pseudo_group_labels": self.pseudo_group_labels,
#         }
#         save_checkpoint(state, os.path.join(self.log_path, tag))

#     def load_checkpoint(self, tag):
#         checkpoint = load_checkpoint(os.path.join(self.log_path, tag))
#         self.model.load_state_dict(checkpoint["model"])
#         self.optimizer.load_state_dict(checkpoint["optimizer"])
#         self.scheduler.load_state_dict(checkpoint["scheduler"])
#         self.best_performance = checkpoint["best_performance"]
#         self.current_epoch = checkpoint["epoch"]
#         self.adv_probs = checkpoint["adv_probs"]
#         self.pseudo_group_labels = checkpoint["pseudo_group_labels"]
#         print(
#             log_msg(
#                 f"Loaded checkpoint from {os.path.join(self.log_path, tag)}",
#                 "INFO",
#                 self.logger,
#             )
#         )
