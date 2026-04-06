"""
SAE Feature Augmentation Trainer for VB-Mitigator.

This module implements a training approach that:
1. Uses OpenCLIP (or other) as a frozen pretrained backbone
2. Passes features through a pretrained SAE encoder
3. Categorizes neurons by target class (monosemantic neurons)
4. During training, augments features by randomly activating class-specific neurons
5. Trains a classification head on the augmented sparse features

The intuition is that by randomly activating neurons specific to the sample's class,
we reinforce class-relevant patterns and create feature-space data augmentation
that helps the model focus on class-relevant features rather than spurious correlations.

Based on monosemanticity analysis from SAE training.
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.builder import get_model
from .base_trainer import BaseTrainer

# Import dictionary learning components
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dictionary_learning"))

from dictionary_learning import AutoEncoder


class SAEFeatureAugmenter(nn.Module):
    """
    Module that augments SAE sparse features by activating/deactivating class-specific neurons.

    During training:
    1. Get sparse features from SAE encoder
    2. With probability `augment_prob`:
       a. Randomly DEACTIVATE some class-specific neurons (set to 0)
       b. Randomly ACTIVATE other class-specific neurons (add activation_value)

    This creates diverse augmentations by both removing and adding class-relevant signals.

    Args:
        sae: Trained Sparse Autoencoder
        neuron_to_class: Dict mapping neuron_idx -> class_idx (monosemantic neurons only)
        num_classes: Number of target classes
        augment_prob: Probability of applying augmentation (0.0 to 1.0)
        deactivate_pct: Percentage of active class neurons to deactivate (0.0 to 1.0)
        activate_pct: Percentage of inactive class neurons to activate (0.0 to 1.0)
        activation_value: Constant value to add to activated neurons
    """

    def __init__(
        self,
        sae,
        neuron_to_class,
        num_classes,
        augment_prob=0.5,
        deactivate_pct=0.2,
        activate_pct=0.3,
        activation_value=1.0,
    ):
        super().__init__()
        self.sae = sae
        self.num_classes = num_classes
        self.augment_prob = augment_prob
        self.deactivate_pct = deactivate_pct
        self.activate_pct = activate_pct
        self.activation_value = activation_value

        # Freeze SAE parameters
        for param in self.sae.parameters():
            param.requires_grad = False

        # Organize neurons by class
        self.class_to_neurons = {c: [] for c in range(num_classes)}
        for neuron_idx, class_idx in neuron_to_class.items():
            if class_idx < num_classes:
                self.class_to_neurons[class_idx].append(neuron_idx)

        # Convert to tensors for efficient indexing
        self.class_neuron_tensors = {}
        for c, neurons in self.class_to_neurons.items():
            if neurons:
                self.register_buffer(
                    f"class_{c}_neurons", torch.tensor(neurons, dtype=torch.long)
                )
            self.class_neuron_tensors[c] = neurons

        # Store neuron assignments for interpretability
        self.neuron_to_class = neuron_to_class

        print(f"SAEFeatureAugmenter initialized:")
        print(f"  Augmentation probability: {augment_prob}")
        print(f"  Deactivation percentage: {deactivate_pct}")
        print(f"  Activation percentage: {activate_pct}")
        print(f"  Activation value: {activation_value}")
        for c in range(num_classes):
            print(f"  Class {c}: {len(self.class_to_neurons[c])} monosemantic neurons")

    def get_class_neurons(self, class_idx):
        """Get tensor of neuron indices for a class."""
        buffer_name = f"class_{class_idx}_neurons"
        if hasattr(self, buffer_name):
            return getattr(self, buffer_name)
        return None

    def encode(self, features):
        """Encode features to sparse SAE latents (no augmentation)."""
        return self.sae.encode(features)

    def decode(self, latents):
        """Decode sparse latents back to feature space."""
        return self.sae.decode(latents)

    def forward(self, features, targets, training=True):
        """
        Forward pass with optional feature augmentation.

        Args:
            features: Input features from backbone (B, feature_dim)
            targets: Target class labels (B,)
            training: Whether in training mode (augmentation only applied if True)

        Returns:
            augmented_latents: SAE latents with augmentation applied (B, dict_size)
        """
        # Encode to sparse latents
        latents = self.sae.encode(features)  # (B, dict_size)

        if not training or self.augment_prob <= 0:
            return latents

        # Apply augmentation
        augmented_latents = self._apply_augmentation(latents, targets)

        return augmented_latents

    def _apply_augmentation(self, latents, targets):
        """
        Apply feature augmentation by activating/deactivating class-specific neurons.

        Args:
            latents: SAE sparse features (B, dict_size)
            targets: Target labels (B,)

        Returns:
            Augmented latents (B, dict_size)
        """
        batch_size = latents.shape[0]
        device = latents.device

        # Clone to avoid modifying original
        augmented = latents.clone()

        for i in range(batch_size):
            # Decide whether to augment this sample
            if random.random() > self.augment_prob:
                continue

            target_class = targets[i].item()
            class_neurons = self.get_class_neurons(target_class)

            if class_neurons is None or len(class_neurons) == 0:
                continue

            # Get current activations for class neurons
            class_activations = augmented[i, class_neurons]  # (num_class_neurons,)

            # Identify active and inactive neurons (among class neurons)
            active_mask = class_activations > 0
            inactive_mask = ~active_mask

            active_indices = active_mask.nonzero(as_tuple=True)[0]
            inactive_indices = inactive_mask.nonzero(as_tuple=True)[0]

            # === DEACTIVATION: Set some active neurons to 0 ===
            if len(active_indices) > 0 and self.deactivate_pct > 0:
                num_to_deactivate = max(
                    1, int(len(active_indices) * self.deactivate_pct)
                )
                perm = torch.randperm(len(active_indices), device=device)[
                    :num_to_deactivate
                ]
                neurons_to_deactivate = class_neurons[active_indices[perm]]
                augmented[i, neurons_to_deactivate] = 0.0

            # === ACTIVATION: Add value to some inactive neurons ===
            if len(inactive_indices) > 0 and self.activate_pct > 0:
                num_to_activate = max(1, int(len(inactive_indices) * self.activate_pct))
                perm = torch.randperm(len(inactive_indices), device=device)[
                    :num_to_activate
                ]
                neurons_to_activate = class_neurons[inactive_indices[perm]]
                augmented[i, neurons_to_activate] += self.activation_value

        return augmented

    def get_augmentation_stats(self):
        """Return statistics about augmentation configuration."""
        return {
            "augment_prob": self.augment_prob,
            "deactivate_pct": self.deactivate_pct,
            "activate_pct": self.activate_pct,
            "activation_value": self.activation_value,
            "neurons_per_class": {c: len(n) for c, n in self.class_to_neurons.items()},
        }


class OpenCLIPWithSAEAugmentation(nn.Module):
    """
    Full model: OpenCLIP backbone + SAE encoder with augmentation + classification head.

    Architecture:
        Input Image -> OpenCLIP (frozen) -> Features -> SAE Encoder ->
        [Augmentation] -> Sparse Latents -> Classification Head -> Logits
    """

    def __init__(
        self,
        clip_model,
        sae_augmenter,
        num_classes,
        classifier_input="sparse",  # 'sparse' or 'decoded'
    ):
        super().__init__()
        self.clip_model = clip_model
        self.sae_augmenter = sae_augmenter
        self.classifier_input = classifier_input

        # Determine classifier input dimension
        if classifier_input == "sparse":
            self.classifier_dim = sae_augmenter.sae.dict_size
        else:  # 'decoded'
            self.classifier_dim = sae_augmenter.sae.activation_dim

        # Classification head
        self.fc = nn.Linear(self.classifier_dim, num_classes)

        # Freeze CLIP backbone
        for param in self.clip_model.parameters():
            param.requires_grad = False

    @property
    def embed_size(self):
        return self.classifier_dim

    def get_clip_features(self, x):
        """Extract features from CLIP backbone."""
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
            features = F.normalize(features, dim=-1)
        return features

    def forward(self, x, targets=None, training=True):
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)
            targets: Target labels (B,) - required for augmentation during training
            training: Whether in training mode

        Returns:
            logits: Classification logits (B, num_classes)
            features: Features used for classification (B, classifier_dim)
        """
        # Get CLIP features
        clip_features = self.get_clip_features(x)

        # Get augmented SAE features
        if targets is not None and training:
            sparse_features = self.sae_augmenter(clip_features, targets, training=True)
        else:
            sparse_features = self.sae_augmenter.encode(clip_features)

        # Choose classifier input
        if self.classifier_input == "sparse":
            classifier_features = sparse_features
        else:
            classifier_features = self.sae_augmenter.decode(sparse_features)

        # Classify
        logits = self.fc(classifier_features)

        return logits, classifier_features

    def forward_no_augment(self, x):
        """Forward pass without augmentation (for evaluation)."""
        clip_features = self.get_clip_features(x)
        sparse_features = self.sae_augmenter.encode(clip_features)

        if self.classifier_input == "sparse":
            classifier_features = sparse_features
        else:
            classifier_features = self.sae_augmenter.decode(sparse_features)

        logits = self.fc(classifier_features)
        return logits, classifier_features


class SAEAugmentationTrainer(BaseTrainer):
    """
    Trainer that uses SAE feature augmentation to enhance class-specific neurons.

    This trainer:
    1. Loads a pretrained OpenCLIP model
    2. Loads a pretrained SAE and its analysis results
    3. Categorizes neurons by target class (monosemantic)
    4. During training, randomly activates class-specific neurons as augmentation
    5. Trains a classification head on the augmented features

    Configuration:
        MITIGATOR:
          TYPE: "sae_augmentation"
          SAE_AUGMENTATION:
            SAE_CHECKPOINT_PATH: "path/to/sae/ae.pt"
            SAE_ANALYSIS_PATH: "path/to/sae/analysis_results.json"
            PURITY_THRESHOLD: 1.0  # Only use neurons with 100% class purity
            AUGMENT_PROB: 0.5      # Probability of augmenting each sample
            NEURON_SELECT_PCT: 0.3 # Percentage of class neurons to activate
            ACTIVATION_VALUE: 1.0  # Value to add to activated neurons
            CLASSIFIER_INPUT: "sparse"  # 'sparse' or 'decoded'
    """

    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """Setup OpenCLIP backbone, SAE with augmentation, and classification head."""

        aug_cfg = self.cfg.MITIGATOR.SAE_AUGMENTATION

        print(f"\n{'='*60}")
        print("Setting up SAE Feature Augmentation Model")
        print(f"{'='*60}")

        # Step 1: Load OpenCLIP model
        base_model = get_model(
            self.cfg.MODEL.TYPE, self.num_class, pretrained=self.cfg.MODEL.PRETRAINED
        )

        if hasattr(base_model, "clip_model"):
            self.clip_model = base_model.clip_model
            self.embed_size = base_model.embed_size
        else:
            raise ValueError(
                "Base model must be an OpenCLIP model with clip_model attribute"
            )

        print(f"Loaded OpenCLIP backbone with embed_size={self.embed_size}")

        # Step 2: Load pretrained SAE
        sae_checkpoint_path = aug_cfg.SAE_CHECKPOINT_PATH
        if not sae_checkpoint_path or not os.path.exists(sae_checkpoint_path):
            raise ValueError(f"SAE checkpoint not found: {sae_checkpoint_path}")

        # Get SAE dimensions
        sae_config_path = os.path.join(
            os.path.dirname(sae_checkpoint_path), "config.json"
        )
        if os.path.exists(sae_config_path):
            with open(sae_config_path, "r") as f:
                sae_config = json.load(f)
            activation_dim = sae_config["activation_dim"]
            dict_size = sae_config["dict_size"]
        else:
            checkpoint = torch.load(sae_checkpoint_path, map_location="cpu")
            if "encoder.weight" in checkpoint:
                dict_size, activation_dim = checkpoint["encoder.weight"].shape
            else:
                raise ValueError("Cannot infer SAE dimensions")

        self.sae = AutoEncoder(activation_dim, dict_size)
        self.sae.load_state_dict(torch.load(sae_checkpoint_path, map_location="cpu"))
        print(f"Loaded SAE: activation_dim={activation_dim}, dict_size={dict_size}")

        # Step 3: Load analysis results and compute neuron-to-class mapping
        analysis_path = aug_cfg.SAE_ANALYSIS_PATH
        if not analysis_path or not os.path.exists(analysis_path):
            raise ValueError(f"SAE analysis not found: {analysis_path}")

        with open(analysis_path, "r") as f:
            analysis_results = json.load(f)

        print(f"Loaded analysis from {analysis_path}")

        # Step 4: Compute neuron-to-class assignment
        neuron_to_class = self._compute_neuron_class_assignment(
            analysis_results,
            dict_size=dict_size,
            purity_threshold=aug_cfg.PURITY_THRESHOLD,
        )

        print(f"Assigned {len(neuron_to_class)} neurons to classes")

        # Step 5: Create SAE augmenter
        self.sae_augmenter = SAEFeatureAugmenter(
            sae=self.sae,
            neuron_to_class=neuron_to_class,
            num_classes=self.num_class,
            augment_prob=aug_cfg.AUGMENT_PROB,
            deactivate_pct=aug_cfg.DEACTIVATE_PCT,
            activate_pct=aug_cfg.ACTIVATE_PCT,
            activation_value=aug_cfg.ACTIVATION_VALUE,
        )

        # Step 6: Create full model
        self.model = OpenCLIPWithSAEAugmentation(
            clip_model=self.clip_model,
            sae_augmenter=self.sae_augmenter,
            num_classes=self.num_class,
            classifier_input=aug_cfg.CLASSIFIER_INPUT,
        )

        self.model.to(self.device)

        print(f"\nModel ready:")
        print(f"  Classifier input: {aug_cfg.CLASSIFIER_INPUT}")
        print(f"  Classifier dimension: {self.model.classifier_dim}")

    def _compute_neuron_class_assignment(
        self, analysis_results, dict_size, purity_threshold
    ):
        """
        Compute which class each neuron belongs to based on monosemanticity.

        Only neurons with class purity >= threshold are assigned.
        Each neuron is assigned to the majority class among its top-k activating images.

        Args:
            analysis_results: Dict from SAE analysis
            dict_size: Total number of SAE neurons
            purity_threshold: Minimum purity to include neuron

        Returns:
            Dict mapping neuron_idx -> class_idx
        """
        neuron_to_class = {}

        top_k_per_latent = analysis_results.get("top_k_per_latent", {})

        class_neuron_counts = defaultdict(int)

        for latent_idx in range(dict_size):
            latent_key = str(latent_idx)

            if latent_key not in top_k_per_latent:
                continue

            latent_data = top_k_per_latent[latent_key]
            targets = latent_data.get("targets", [])

            if len(targets) == 0:
                continue

            # Compute class distribution
            target_counts = defaultdict(int)
            for t in targets:
                target_counts[t] += 1

            # Find majority class and its purity
            majority_class = max(target_counts.keys(), key=lambda k: target_counts[k])
            purity = target_counts[majority_class] / len(targets)

            # Only assign if meets purity threshold
            if purity >= purity_threshold:
                neuron_to_class[latent_idx] = majority_class
                class_neuron_counts[majority_class] += 1

        print(f"Neuron class assignment (purity >= {purity_threshold}):")
        for class_idx in sorted(class_neuron_counts.keys()):
            print(f"  Class {class_idx}: {class_neuron_counts[class_idx]} neurons")

        return neuron_to_class

    def _setup_optimizer(self):
        """Setup optimizer for classification head only."""
        # Only optimize classification head parameters
        trainable_params = list(self.model.fc.parameters())

        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

        if self.cfg.SOLVER.TYPE == "SGD":
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        elif self.cfg.SOLVER.TYPE == "Adam":
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.cfg.SOLVER.LR,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        elif self.cfg.SOLVER.TYPE == "AdamW":
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.cfg.SOLVER.LR,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.cfg.SOLVER.TYPE}")

        # Setup scheduler
        self._setup_scheduler()

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.cfg.SOLVER.SCHEDULER.TYPE

        if scheduler_type == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.cfg.SOLVER.SCHEDULER.LR_DECAY_STAGES,
                gamma=self.cfg.SOLVER.SCHEDULER.LR_DECAY_RATE,
            )
        elif scheduler_type == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.SOLVER.EPOCHS,
            )
        elif scheduler_type == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.SOLVER.SCHEDULER.LR_DECAY_STAGES[0],
                gamma=self.cfg.SOLVER.SCHEDULER.LR_DECAY_RATE,
            )
        else:
            self.scheduler = None

    def _train_iter(self, batch):
        """Training iteration with augmentation."""
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        # Forward pass with augmentation
        logits, features = self.model(inputs, targets=targets, training=True)

        # Compute loss
        loss = F.cross_entropy(logits, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss}

    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        # But keep CLIP and SAE in eval mode
        self.model.clip_model.eval()
        self.model.sae_augmenter.sae.eval()

        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.dataloaders["train"], desc="Training")
        for batch in pbar:
            loss_dict = self._train_iter(batch)
            total_loss += loss_dict["loss"].item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{total_loss/num_batches:.4f}"})

        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": total_loss / num_batches}

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("Starting SAE Augmentation Training")
        print(f"{'='*60}")

        aug_stats = self.sae_augmenter.get_augmentation_stats()
        print(f"Augmentation config:")
        print(f"  Probability: {aug_stats['augment_prob']}")
        print(f"  Activate Neuron select %: {aug_stats['activate_pct']}")
        print(f"  Activation value: {aug_stats['activation_value']}")
        print(f"  Neurons per class: {aug_stats['neurons_per_class']}")

        best_metric = 0

        for epoch in range(self.cfg.SOLVER.EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.cfg.SOLVER.EPOCHS}")

            # Train
            train_metrics = self._train_epoch()
            print(f"Train loss: {train_metrics['loss']:.4f}")

            # Evaluate (without augmentation)
            eval_metrics = self.eval()

            # Log metrics
            current_metric = eval_metrics.get(
                self.cfg.METRIC, eval_metrics.get("accuracy", 0)
            )
            print(f"Eval {self.cfg.METRIC}: {current_metric:.4f}")
            print(eval_metrics["accs"])

            # Save best model
            if current_metric > best_metric:
                best_metric = current_metric
                self._save_checkpoint(epoch, is_best=True)
                print(f"New best model! {self.cfg.METRIC}: {best_metric:.4f}")

        print(f"\n{'='*60}")
        print(f"Training complete! Best {self.cfg.METRIC}: {best_metric:.4f}")
        print(f"{'='*60}")

        # Save final results
        self._save_results(best_metric)

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        aug_cfg = self.cfg.MITIGATOR.SAE_AUGMENTATION

        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": {
                "purity_threshold": aug_cfg.PURITY_THRESHOLD,
                "augment_prob": aug_cfg.AUGMENT_PROB,
                "deactivate_pct": aug_cfg.DEACTIVATE_PCT,
                "activate_pct": aug_cfg.ACTIVATE_PCT,
                "activation_value": aug_cfg.ACTIVATION_VALUE,
                "augmentation_stats": self.sae_augmenter.get_augmentation_stats(),
            },
        }

        save_path = os.path.join(self.log_path, "last.pth")
        torch.save(checkpoint, save_path)

        if is_best:
            best_path = os.path.join(self.log_path, "best.pth")
            torch.save(checkpoint, best_path)

    def _save_results(self, best_metric):
        """Save training results."""
        aug_cfg = self.cfg.MITIGATOR.SAE_AUGMENTATION

        results = {
            "best_metric": best_metric,
            "metric_name": self.cfg.METRIC,
            "config": {
                "purity_threshold": aug_cfg.PURITY_THRESHOLD,
                "augment_prob": aug_cfg.AUGMENT_PROB,
                "deactivate_pct": aug_cfg.DEACTIVATE_PCT,
                "activate_pct": aug_cfg.ACTIVATE_PCT,
                "activation_value": aug_cfg.ACTIVATION_VALUE,
                "classifier_input": aug_cfg.CLASSIFIER_INPUT,
            },
            "augmentation_stats": self.sae_augmenter.get_augmentation_stats(),
        }

        results_path = os.path.join(self.log_path, "augmentation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

    def eval(self):
        """Evaluate the model without augmentation."""
        self.model.eval()

        all_preds = []
        all_targets = []
        all_biases = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(self.dataloaders["test"], desc="Evaluating"):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]

                # Forward without augmentation
                logits, _ = self.model.forward_no_augment(inputs)
                preds = logits.argmax(dim=1).cpu()

                all_preds.append(preds)
                all_targets.append(targets)

                # Collect bias attributes
                for bias_name in self.biases:
                    if bias_name in batch:
                        all_biases[bias_name].append(batch[bias_name])

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Compute accuracy
        accuracy = (all_preds == all_targets).float().mean().item()

        # Compute per-group metrics if bias info available
        metrics = {"accuracy": accuracy}

        if all_biases:
            bias_name = self.biases[0]
            all_bias = torch.cat(all_biases[bias_name])

            # Compute worst-group accuracy
            group_accs = []
            for t in all_targets.unique():
                for b in all_bias.unique():
                    mask = (all_targets == t) & (all_bias == b)
                    if mask.sum() > 0:
                        group_acc = (
                            (all_preds[mask] == all_targets[mask]).float().mean().item()
                        )
                        group_accs.append(group_acc)
                        metrics[f"acc_t{t.item()}_b{b.item()}"] = group_acc

            if group_accs:
                metrics["wg_ovr"] = min(group_accs)
                metrics["avg_group_acc"] = np.mean(group_accs)
                metrics["accs"] = group_accs
        return metrics
