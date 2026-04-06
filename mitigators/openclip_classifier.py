"""
OpenCLIP Classifier Trainer.

This mitigator trains a classification head on features from a frozen OpenCLIP model.
This is the vanilla/baseline approach for classification with pretrained vision encoders.

Architecture:
    Image → OpenCLIP Encoder (frozen) → Features → Classification Head (trainable) → Predictions

The classification head can be:
    - Linear: single linear layer
    - MLP: linear -> ReLU -> dropout -> linear

Config:
    MITIGATOR:
      TYPE: "openclip_classifier"
      OPENCLIP_CLASSIFIER:
        # OpenCLIP model
        ARCH: "ViT-B-32"
        PRETRAINED: "openai"

        # Classifier architecture
        CLASSIFIER_TYPE: "linear"  # "linear" or "mlp"
        HIDDEN_DIM: 512  # For MLP
        DROPOUT: 0.1

        # Training
        EPOCHS: 50
        LR: 1e-3
        WEIGHT_DECAY: 1e-4
        BATCH_SIZE: 256

        # Optional: use precomputed features
        PRECOMPUTE_FEATURES: True
"""

import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base_trainer import BaseTrainer
from models.vlm_encoders import OpenCLIPEncoder


# ============================================
# Classification Heads
# ============================================


class LinearClassifier(nn.Module):
    """Simple linear classifier."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)


class MLPClassifier(nn.Module):
    """MLP classifier with one hidden layer."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================
# OpenCLIP Classifier Model
# ============================================


class OpenCLIPClassifier(nn.Module):
    """
    OpenCLIP encoder with classification head.

    The encoder is frozen and only the classification head is trained.
    """

    def __init__(
        self,
        encoder: OpenCLIPEncoder,
        num_classes: int,
        classifier_type: str = "linear",
        hidden_dim: int = 512,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()

        self.encoder = encoder
        self.embed_dim = encoder.embed_dim

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Create classifier head
        if classifier_type == "linear":
            self.classifier = LinearClassifier(
                input_dim=self.embed_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif classifier_type == "mlp":
            self.classifier = MLPClassifier(
                input_dim=self.embed_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_layers=num_layers,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        print(f"OpenCLIPClassifier initialized:")
        print(f"  Encoder: {encoder.arch} (frozen)")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Classifier: {classifier_type}")
        print(f"  Num classes: {num_classes}")

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        with torch.no_grad():
            features = self.encoder.encode_image(images)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode images and classify."""
        features = self.encode(images)
        logits = self.classifier(features)
        return logits

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass from precomputed features."""
        return self.classifier(features)


# ============================================
# Trainer
# ============================================


class OpenCLIPClassifierTrainer(BaseTrainer):
    """
    Trainer for OpenCLIP + classification head.

    This is a vanilla baseline that:
    1. Uses a frozen OpenCLIP encoder to extract features
    2. Trains a linear or MLP classifier on top

    Configuration:
        MITIGATOR:
          TYPE: "openclip_classifier"
          OPENCLIP_CLASSIFIER:
            ARCH: "ViT-B-32"
            PRETRAINED: "openai"
            CLASSIFIER_TYPE: "linear"
            HIDDEN_DIM: 512
            DROPOUT: 0.1
            EPOCHS: 50
            LR: 1e-3
            WEIGHT_DECAY: 1e-4
            PRECOMPUTE_FEATURES: True
    """

    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """Setup OpenCLIP encoder and classifier."""

        oc_cfg = self.cfg.MITIGATOR.OPENCLIP_CLASSIFIER

        print(f"\n{'='*60}")
        print("Setting up OpenCLIP Classifier")
        print(f"{'='*60}")

        # Create OpenCLIP encoder
        print(f"\nLoading OpenCLIP: {oc_cfg.ARCH}")
        self.encoder = OpenCLIPEncoder(
            arch=oc_cfg.ARCH,
            pretrained=oc_cfg.PRETRAINED,
            device=self.device,
        )
        print(f"  Embed dim: {self.encoder.embed_dim}")
        print(f"  Image size: {self.encoder.image_size}")

        # Create classifier model
        self.model = OpenCLIPClassifier(
            encoder=self.encoder,
            num_classes=self.num_class,
            classifier_type=oc_cfg.get("CLASSIFIER_TYPE", "linear"),
            hidden_dim=oc_cfg.get("HIDDEN_DIM", 512),
            dropout=oc_cfg.get("DROPOUT", 0.1),
            num_layers=oc_cfg.get("NUM_LAYERS", 1),
        )
        self.model.to(self.device)

        # Count trainable parameters
        num_trainable = sum(
            p.numel() for p in self.model.classifier.parameters() if p.requires_grad
        )
        print(f"  Trainable parameters: {num_trainable:,}")

        # Store config
        self.oc_cfg = oc_cfg

        # Rebuild dataloaders with encoder's transform
        self._rebuild_dataloaders_with_transform()

    def _setup_dataset(self):
        super()._setup_dataset()
        self.dataloaders["train"] = self.dataloaders["val"]

    def _rebuild_dataloaders_with_transform(self):
        """Rebuild dataloaders using the encoder's preprocessing transform."""
        print(f"\nRebuilding dataloaders with encoder preprocessing...")

        # Get encoder's transform
        transform = self.encoder.get_transform()

        # Get config
        batch_size = self.oc_cfg.get("BATCH_SIZE", 256)
        num_workers = (
            self.cfg.DATALOADER.NUM_WORKERS if hasattr(self.cfg, "DATALOADER") else 4
        )

        # Get base datasets and update transform
        train_dataset = self.dataloaders["train"].dataset
        test_dataset = self.dataloaders["test"].dataset

        # Update transforms
        if hasattr(train_dataset, "transform"):
            train_dataset.transform = transform
        if hasattr(test_dataset, "transform"):
            test_dataset.transform = transform

        # Rebuild dataloaders
        self.dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Batch size: {batch_size}")

    def _setup_optimizer(self):
        """Setup optimizer for classifier only."""
        oc_cfg = self.oc_cfg

        self.optimizer = torch.optim.AdamW(
            self.model.classifier.parameters(),
            lr=oc_cfg.get("LR", 1e-3),
            weight_decay=oc_cfg.get("WEIGHT_DECAY", 1e-4),
        )

        print(f"\nOptimizer: AdamW")
        print(f"  LR: {oc_cfg.get('LR', 1e-3)}")
        print(f"  Weight decay: {oc_cfg.get('WEIGHT_DECAY', 1e-4)}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        num_epochs = self.oc_cfg.get("EPOCHS", 50)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6,
        )

    def _extract_all_features(self, dataloader, desc="Extracting features"):
        """Extract features from all images in dataloader."""

        all_features = []
        all_targets = []
        all_biases = defaultdict(list)

        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]

                features = self.model.encode(inputs)

                all_features.append(features.cpu())
                all_targets.append(targets)

                for b in self.biases:
                    if b in batch:
                        all_biases[b].append(batch[b])

        features = torch.cat(all_features, dim=0)
        targets = torch.cat(all_targets, dim=0)
        biases = {k: torch.cat(v, dim=0) for k, v in all_biases.items()}

        return features, targets, biases

    def _train_epoch_precomputed(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict:
        """Train for one epoch using precomputed features."""

        self.model.classifier.train()

        # Create dataloader for features
        dataset = TensorDataset(features, targets)
        dataloader = DataLoader(
            dataset,
            batch_size=self.oc_cfg.get("BATCH_SIZE", 256),
            shuffle=True,
            drop_last=True,
        )

        total_loss = 0
        total_correct = 0
        total_samples = 0

        criterion = nn.CrossEntropyLoss()

        for feat_batch, target_batch in dataloader:
            feat_batch = feat_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            # Forward
            logits = self.model.forward_features(feat_batch)
            loss = criterion(logits, target_batch)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Stats
            total_loss += loss.item() * len(target_batch)
            total_correct += (logits.argmax(dim=1) == target_batch).sum().item()
            total_samples += len(target_batch)

        return {
            "train_loss": total_loss / total_samples,
            "train_acc": total_correct / total_samples,
        }

    def _eval_precomputed(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        biases: Dict[str, torch.Tensor],
    ) -> Dict:
        """Evaluate using precomputed features."""

        self.model.classifier.eval()

        # Get predictions
        all_preds = []
        batch_size = 256

        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                feat_batch = features[i : i + batch_size].to(self.device)
                logits = self.model.forward_features(feat_batch)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())

        all_preds = torch.cat(all_preds)

        # Compute metrics
        correct = all_preds == targets
        accuracy = correct.float().mean().item()

        metrics = {"accuracy": accuracy}

        # Per-class accuracy
        for c in range(self.num_class):
            mask = targets == c
            if mask.sum() > 0:
                class_acc = correct[mask].float().mean().item()
                metrics[f"acc_class_{c}"] = class_acc

        # Per-subgroup accuracy
        bias_names = list(biases.keys())
        if bias_names:
            subgroup_keys = []
            for i in range(len(targets)):
                key = [f"t={targets[i].item()}"]
                for b_name in bias_names:
                    key.append(f"{b_name}={biases[b_name][i].item()}")
                subgroup_keys.append(tuple(key))

            unique_subgroups = sorted(set(subgroup_keys))
            subgroup_accs = []

            for sg in unique_subgroups:
                mask = torch.tensor([sk == sg for sk in subgroup_keys])
                if mask.sum() > 0:
                    acc = correct[mask].float().mean().item()
                    count = mask.sum().item()
                    subgroup_accs.append(acc)
                    metrics[f"acc_{sg}"] = acc

            if subgroup_accs:
                metrics["worst_group_accuracy"] = min(subgroup_accs)
                metrics["best_group_accuracy"] = max(subgroup_accs)
                metrics["accuracy_gap"] = max(subgroup_accs) - min(subgroup_accs)
                metrics["avg_group_accuracy"] = np.mean(subgroup_accs)

        return metrics

    def train(self):
        """Main training pipeline."""

        print(f"\n{'='*60}")
        print("OpenCLIP Classifier Training")
        print(f"{'='*60}")

        oc_cfg = self.oc_cfg
        num_epochs = oc_cfg.get("EPOCHS", 50)
        precompute = oc_cfg.get("PRECOMPUTE_FEATURES", True)

        if precompute:
            # Precompute features once
            print("\nStep 1: Precomputing features...")

            train_features, train_targets, train_biases = self._extract_all_features(
                self.dataloaders["train"], desc="Extracting train features"
            )
            print(f"  Train features: {train_features.shape}")

            test_features, test_targets, test_biases = self._extract_all_features(
                self.dataloaders["test"], desc="Extracting test features"
            )
            print(f"  Test features: {test_features.shape}")

            # Training loop with precomputed features
            print(f"\nStep 2: Training classifier for {num_epochs} epochs...")

            best_wg_acc = 0
            best_epoch = 0

            for epoch in range(num_epochs):
                # Train
                train_metrics = self._train_epoch_precomputed(
                    train_features, train_targets
                )

                # Evaluate
                test_metrics = self._eval_precomputed(
                    test_features, test_targets, test_biases
                )

                # Step scheduler
                if hasattr(self, "scheduler"):
                    self.scheduler.step()

                # Track best
                wg_acc = test_metrics.get(
                    "worst_group_accuracy", test_metrics["accuracy"]
                )
                if wg_acc > best_wg_acc:
                    best_wg_acc = wg_acc
                    best_epoch = epoch
                    self._save_checkpoint("best")

                # Log
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    print(
                        f"  Epoch {epoch:3d}: "
                        f"train_loss={train_metrics['train_loss']:.4f}, "
                        f"train_acc={train_metrics['train_acc']:.4f}, "
                        f"test_acc={test_metrics['accuracy']:.4f}, "
                        f"wg_acc={wg_acc:.4f}"
                    )

            # Final evaluation
            print(f"\nBest epoch: {best_epoch} (wg_acc={best_wg_acc:.4f})")

            # Load best and evaluate
            self._load_checkpoint("best")
            final_metrics = self._eval_precomputed(
                test_features, test_targets, test_biases
            )

        else:
            # Train without precomputing (slower but uses less memory)
            print(
                f"\nTraining classifier for {num_epochs} epochs (online feature extraction)..."
            )

            criterion = nn.CrossEntropyLoss()
            best_wg_acc = 0
            best_epoch = 0

            for epoch in range(num_epochs):
                # Train epoch
                self.model.classifier.train()
                total_loss = 0
                total_correct = 0
                total_samples = 0

                for batch in tqdm(self.dataloaders["train"], desc=f"Epoch {epoch}"):
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device)

                    # Forward
                    logits = self.model(inputs)
                    loss = criterion(logits, targets)

                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item() * len(targets)
                    total_correct += (logits.argmax(dim=1) == targets).sum().item()
                    total_samples += len(targets)

                train_loss = total_loss / total_samples
                train_acc = total_correct / total_samples

                # Evaluate
                test_metrics = self.eval()

                # Step scheduler
                if hasattr(self, "scheduler"):
                    self.scheduler.step()

                # Track best
                wg_acc = test_metrics.get(
                    "worst_group_accuracy", test_metrics["accuracy"]
                )
                if wg_acc > best_wg_acc:
                    best_wg_acc = wg_acc
                    best_epoch = epoch
                    self._save_checkpoint("best")

                # Log
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    print(
                        f"  Epoch {epoch:3d}: "
                        f"train_loss={train_loss:.4f}, "
                        f"train_acc={train_acc:.4f}, "
                        f"test_acc={test_metrics['accuracy']:.4f}, "
                        f"wg_acc={wg_acc:.4f}"
                    )

            # Load best and evaluate
            self._load_checkpoint("best")
            final_metrics = self.eval()

        # Print final results
        self._print_final_results(final_metrics)

        # Save results
        self._save_results(final_metrics)

        return final_metrics

    def eval(self):
        """Evaluate on test set."""

        self.model.eval()

        all_preds = []
        all_targets = []
        all_biases = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(self.dataloaders["test"], desc="Evaluating"):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]

                logits = self.model(inputs)
                preds = logits.argmax(dim=1).cpu()

                all_preds.append(preds)
                all_targets.append(targets)

                for b in self.biases:
                    if b in batch:
                        all_biases[b].append(batch[b])

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        biases = {k: torch.cat(v, dim=0) for k, v in all_biases.items()}

        # Compute metrics
        correct = all_preds == all_targets
        accuracy = correct.float().mean().item()

        metrics = {"accuracy": accuracy}

        # Per-class accuracy
        for c in range(self.num_class):
            mask = all_targets == c
            if mask.sum() > 0:
                class_acc = correct[mask].float().mean().item()
                metrics[f"acc_class_{c}"] = class_acc

        # Per-subgroup accuracy
        bias_names = list(biases.keys())
        if bias_names:
            subgroup_keys = []
            for i in range(len(all_targets)):
                key = [f"t={all_targets[i].item()}"]
                for b_name in bias_names:
                    key.append(f"{b_name}={biases[b_name][i].item()}")
                subgroup_keys.append(tuple(key))

            unique_subgroups = sorted(set(subgroup_keys))
            subgroup_accs = []

            for sg in unique_subgroups:
                mask = torch.tensor([sk == sg for sk in subgroup_keys])
                if mask.sum() > 0:
                    acc = correct[mask].float().mean().item()
                    count = mask.sum().item()
                    subgroup_accs.append(acc)
                    metrics[f"acc_{sg}"] = acc

            if subgroup_accs:
                metrics["worst_group_accuracy"] = min(subgroup_accs)
                metrics["best_group_accuracy"] = max(subgroup_accs)
                metrics["accuracy_gap"] = max(subgroup_accs) - min(subgroup_accs)
                metrics["avg_group_accuracy"] = np.mean(subgroup_accs)

        return metrics

    def _save_checkpoint(self, name: str):
        """Save checkpoint."""
        save_dir = os.path.join(self.log_path, "openclip_classifier")
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "classifier_state_dict": self.model.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        path = os.path.join(save_dir, f"checkpoint_{name}.pt")
        torch.save(checkpoint, path)

    def _load_checkpoint(self, name: str):
        """Load checkpoint."""
        save_dir = os.path.join(self.log_path, "openclip_classifier")
        path = os.path.join(save_dir, f"checkpoint_{name}.pt")

        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.classifier.load_state_dict(checkpoint["classifier_state_dict"])

    def _print_final_results(self, metrics: Dict):
        """Print final evaluation results."""

        print(f"\n{'='*60}")
        print("Final Results")
        print(f"{'='*60}")

        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")

        if "worst_group_accuracy" in metrics:
            print(f"Worst-Group Accuracy: {metrics['worst_group_accuracy']:.4f}")
            print(f"Best-Group Accuracy: {metrics['best_group_accuracy']:.4f}")
            print(f"Accuracy Gap: {metrics['accuracy_gap']:.4f}")

        # Per-class accuracy
        print("\nPer-class accuracy:")
        for c in range(self.num_class):
            key = f"acc_class_{c}"
            if key in metrics:
                print(f"  Class {c}: {metrics[key]:.4f}")

    def _save_results(self, metrics: Dict):
        """Save results to JSON."""
        save_dir = os.path.join(self.log_path, "openclip_classifier")
        os.makedirs(save_dir, exist_ok=True)

        # Config
        results = {
            "config": {
                "arch": self.oc_cfg.ARCH,
                "pretrained": self.oc_cfg.PRETRAINED,
                "classifier_type": self.oc_cfg.get("CLASSIFIER_TYPE", "linear"),
                "hidden_dim": self.oc_cfg.get("HIDDEN_DIM", 512),
                "dropout": self.oc_cfg.get("DROPOUT", 0.1),
                "epochs": self.oc_cfg.get("EPOCHS", 50),
                "lr": self.oc_cfg.get("LR", 1e-3),
            },
            "metrics": {
                k: v for k, v in metrics.items() if isinstance(v, (int, float))
            },
        }

        path = os.path.join(save_dir, "results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {path}")
