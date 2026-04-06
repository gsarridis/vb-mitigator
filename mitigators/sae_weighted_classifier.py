"""
SAE Weighted Classifier for VB-Mitigator.

This module implements simple, robust classification methods that extend
the training-free SAE Neuron Classifier with weighting schemes:

1. **Purity Weighted Sum** (no learning):
   - Weight each neuron by its class purity score
   - score[class] = sum(activation[n] * purity[n] for n in class_neurons)
   - Neurons with 100% purity contribute more than 90% purity ones

2. **Learnable Per-Neuron Weights** (minimal learning):
   - Learn one scalar weight per monosemantic neuron
   - score[class] = sum(weight[n] * activation[n] for n in class_neurons)
   - Very few parameters, resistant to overfitting

Both methods are simple, interpretable, and robust compared to dense layers.
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


class SAEWeightedClassifier(nn.Module):
    """
    Classifier that uses weighted sums of SAE neuron activations.

    Supports two weighting modes:
    1. 'purity': Weight by neuron purity (no learning)
    2. 'learnable': Learn per-neuron weights (minimal parameters)

    Args:
        sae: Trained Sparse Autoencoder
        neuron_to_class: Dict mapping neuron_idx -> class_idx
        neuron_purity: Dict mapping neuron_idx -> purity score (0-1)
        num_classes: Number of target classes
        weight_mode: 'purity' or 'learnable'
        temperature: Temperature for final scores
        init_with_purity: If True and mode='learnable', initialize weights with purity
    """

    def __init__(
        self,
        sae,
        neuron_to_class,
        neuron_purity,
        num_classes,
        weight_mode="purity",
        temperature=1.0,
        init_with_purity=True,
    ):
        super().__init__()
        self.sae = sae
        self.num_classes = num_classes
        self.weight_mode = weight_mode
        self.temperature = temperature

        # Freeze SAE parameters
        for param in self.sae.parameters():
            param.requires_grad = False

        # Organize neurons by class and store their purities
        self.class_to_neurons = {c: [] for c in range(num_classes)}
        self.class_to_purities = {c: [] for c in range(num_classes)}

        for neuron_idx, class_idx in neuron_to_class.items():
            if class_idx < num_classes:
                self.class_to_neurons[class_idx].append(neuron_idx)
                self.class_to_purities[class_idx].append(
                    neuron_purity.get(neuron_idx, 1.0)
                )

        # Register neuron indices and purity weights as buffers
        for c in range(num_classes):
            neurons = self.class_to_neurons[c]
            purities = self.class_to_purities[c]

            if neurons:
                self.register_buffer(
                    f"class_{c}_neurons", torch.tensor(neurons, dtype=torch.long)
                )
                self.register_buffer(
                    f"class_{c}_purity_weights",
                    torch.tensor(purities, dtype=torch.float32),
                )

        # Setup weights based on mode
        if weight_mode == "learnable":
            self._setup_learnable_weights(
                neuron_to_class, neuron_purity, init_with_purity
            )

        # Learnable per-class bias (optional, helps with class imbalance)
        self.class_bias = nn.Parameter(torch.zeros(num_classes))

        # Learnable temperature (optional)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

        # Store for reference
        self.neuron_to_class = neuron_to_class
        self.neuron_purity = neuron_purity

        self._print_init_info()

    def _setup_learnable_weights(
        self, neuron_to_class, neuron_purity, init_with_purity
    ):
        """Setup learnable per-neuron weights."""
        for c in range(self.num_classes):
            neurons = self.class_to_neurons[c]
            if neurons:
                num_neurons = len(neurons)

                if init_with_purity:
                    # Initialize with purity values
                    init_weights = torch.tensor(
                        [neuron_purity.get(n, 1.0) for n in neurons],
                        dtype=torch.float32,
                    )
                else:
                    # Initialize with ones
                    init_weights = torch.ones(num_neurons)

                # Register as learnable parameter
                setattr(self, f"class_{c}_weights", nn.Parameter(init_weights))

    def _print_init_info(self):
        """Print initialization information."""
        total_neurons = sum(
            len(self.class_to_neurons[c]) for c in range(self.num_classes)
        )
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"SAEWeightedClassifier initialized:")
        print(f"  Weight mode: {self.weight_mode}")
        print(f"  Total monosemantic neurons: {total_neurons}")
        print(f"  Learnable parameters: {total_params}")

        for c in range(self.num_classes):
            neurons = self.class_to_neurons[c]
            if neurons:
                purities = self.class_to_purities[c]
                print(
                    f"  Class {c}: {len(neurons)} neurons, "
                    f"mean purity: {np.mean(purities):.3f}"
                )

    def get_class_neurons(self, class_idx):
        """Get neuron indices for a class."""
        buffer_name = f"class_{class_idx}_neurons"
        if hasattr(self, buffer_name):
            return getattr(self, buffer_name)
        return None

    def get_weights(self, class_idx):
        """Get weights for a class's neurons."""
        if self.weight_mode == "purity":
            buffer_name = f"class_{class_idx}_purity_weights"
            if hasattr(self, buffer_name):
                return getattr(self, buffer_name)
        elif self.weight_mode == "learnable":
            param_name = f"class_{class_idx}_weights"
            if hasattr(self, param_name):
                return getattr(self, param_name)
        return None

    def forward(self, sparse_features):
        """
        Compute class scores from sparse SAE features.

        Args:
            sparse_features: SAE latent activations (B, dict_size)

        Returns:
            logits: Class scores (B, num_classes)
        """
        batch_size = sparse_features.shape[0]
        device = sparse_features.device

        scores = torch.zeros(batch_size, self.num_classes, device=device)

        for class_idx in range(self.num_classes):
            neurons = self.get_class_neurons(class_idx)
            weights = self.get_weights(class_idx)
            weights = (weights - torch.min(weights)) / (
                torch.max(weights) - torch.min(weights)
            )

            if neurons is None or weights is None or len(neurons) == 0:
                continue

            # Get activations for this class's neurons
            class_activations = sparse_features[:, neurons]  # (B, num_class_neurons)

            # Weighted sum
            # weights shape: (num_class_neurons,)
            # class_activations shape: (B, num_class_neurons)
            weighted_sum = (class_activations * weights.unsqueeze(0)).sum(dim=1)  # (B,)

            scores[:, class_idx] = weighted_sum

        # Add per-class bias
        scores = scores + self.class_bias.unsqueeze(0)

        # Apply temperature
        temperature = torch.exp(self.log_temperature)
        logits = scores / temperature

        return logits

    def get_weight_stats(self):
        """Get statistics about current weights."""
        stats = {
            "mode": self.weight_mode,
            "temperature": torch.exp(self.log_temperature).item(),
            "class_bias": self.class_bias.detach().cpu().tolist(),
        }

        for c in range(self.num_classes):
            weights = self.get_weights(c)
            if weights is not None:
                w = weights.detach().cpu().numpy()
                stats[f"class_{c}_weights"] = {
                    "mean": float(np.mean(w)),
                    "std": float(np.std(w)),
                    "min": float(np.min(w)),
                    "max": float(np.max(w)),
                    "num_neurons": len(w),
                }

        return stats

    def get_top_neurons(self, class_idx, top_k=10):
        """Get top-k highest weighted neurons for a class."""
        neurons = self.get_class_neurons(class_idx)
        weights = self.get_weights(class_idx)

        if neurons is None or weights is None:
            return []

        weights_np = weights.detach().cpu().numpy()
        neurons_np = neurons.cpu().numpy()

        top_indices = np.argsort(weights_np)[-top_k:][::-1]

        return [(int(neurons_np[i]), float(weights_np[i])) for i in top_indices]


class OpenCLIPWithSAEWeightedClassifier(nn.Module):
    """
    Full model: OpenCLIP backbone + SAE encoder + Weighted classifier.
    """

    def __init__(self, clip_model, sae, weighted_classifier):
        super().__init__()
        self.clip_model = clip_model
        self.sae = sae
        self.weighted_classifier = weighted_classifier

        # Freeze CLIP and SAE
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.sae.parameters():
            param.requires_grad = False

    def encode_to_sparse(self, x):
        """Encode images to sparse SAE features."""
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(x)
            clip_features = F.normalize(clip_features, dim=-1)
            sparse_features = self.sae.encode(clip_features)
        return sparse_features

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            logits: Class scores (B, num_classes)
        """
        sparse_features = self.encode_to_sparse(x)
        logits = self.weighted_classifier(sparse_features)
        return logits

    def forward_with_features(self, x):
        """Forward pass returning intermediate features."""
        sparse_features = self.encode_to_sparse(x)
        logits = self.weighted_classifier(sparse_features)
        return logits, sparse_features


class SAEWeightedClassifierTrainer(BaseTrainer):
    """
    Trainer for SAE Weighted Classifier.

    Supports two modes:
    1. 'purity': No training, just evaluate with purity-weighted sums
    2. 'learnable': Train per-neuron weights (very few parameters)

    Configuration:
        MITIGATOR:
          TYPE: "sae_weighted_classifier"
          SAE_WEIGHTED_CLASSIFIER:
            SAE_CHECKPOINT_PATH: "path/to/ae.pt"
            SAE_ANALYSIS_PATH: "path/to/analysis_results.json"
            PURITY_THRESHOLD: 0.8  # Can be lower since we weight by purity
            WEIGHT_MODE: "learnable"  # 'purity' or 'learnable'
            INIT_WITH_PURITY: True    # Initialize learnable weights with purity
            TEMPERATURE: 1.0
            LEARN_TEMPERATURE: True
            LEARN_CLASS_BIAS: True
    """

    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """Setup OpenCLIP backbone, SAE, and weighted classifier."""

        wcfg = self.cfg.MITIGATOR.SAE_WEIGHTED_CLASSIFIER

        print(f"\n{'='*60}")
        print("Setting up SAE Weighted Classifier")
        print(f"{'='*60}")

        # Step 1: Load OpenCLIP model
        base_model = get_model(
            self.cfg.MODEL.TYPE, self.num_class, pretrained=self.cfg.MODEL.PRETRAINED
        )

        if hasattr(base_model, "clip_model"):
            self.clip_model = base_model.clip_model
            self.embed_size = base_model.embed_size
        else:
            raise ValueError("Base model must be an OpenCLIP model")

        print(f"Loaded OpenCLIP backbone with embed_size={self.embed_size}")

        # Step 2: Load pretrained SAE
        sae_checkpoint_path = wcfg.SAE_CHECKPOINT_PATH
        if not sae_checkpoint_path or not os.path.exists(sae_checkpoint_path):
            raise ValueError(f"SAE checkpoint not found: {sae_checkpoint_path}")

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

        # Step 3: Load analysis and compute neuron assignments + purities
        analysis_path = wcfg.SAE_ANALYSIS_PATH
        if not analysis_path or not os.path.exists(analysis_path):
            raise ValueError(f"SAE analysis not found: {analysis_path}")

        with open(analysis_path, "r") as f:
            analysis_results = json.load(f)

        neuron_to_class, neuron_purity = self._compute_neuron_assignments(
            analysis_results,
            dict_size=dict_size,
            purity_threshold=wcfg.PURITY_THRESHOLD,
        )

        print(
            f"Assigned {len(neuron_to_class)} neurons with purity >= {wcfg.PURITY_THRESHOLD}"
        )

        # Step 4: Create weighted classifier
        self.weighted_classifier = SAEWeightedClassifier(
            sae=self.sae,
            neuron_to_class=neuron_to_class,
            neuron_purity=neuron_purity,
            num_classes=self.num_class,
            weight_mode=wcfg.WEIGHT_MODE,
            temperature=wcfg.TEMPERATURE,
            init_with_purity=wcfg.INIT_WITH_PURITY,
        )

        # Step 5: Create full model
        self.model = OpenCLIPWithSAEWeightedClassifier(
            clip_model=self.clip_model,
            sae=self.sae,
            weighted_classifier=self.weighted_classifier,
        )

        self.model.to(self.device)

        # Store for later
        self.neuron_to_class = neuron_to_class
        self.neuron_purity = neuron_purity
        self.dict_size = dict_size

    def _compute_neuron_assignments(
        self, analysis_results, dict_size, purity_threshold
    ):
        """
        Compute neuron-to-class assignments and purity scores.

        Returns:
            neuron_to_class: Dict mapping neuron_idx -> class_idx
            neuron_purity: Dict mapping neuron_idx -> purity score (0-1)
        """
        neuron_to_class = {}
        neuron_purity = {}

        top_k_per_latent = analysis_results.get("top_k_per_latent", {})

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

            # Find majority class and purity
            majority_class = max(target_counts.keys(), key=lambda k: target_counts[k])
            purity = target_counts[majority_class] / len(targets)

            # Only include if meets threshold
            if purity >= purity_threshold:
                neuron_to_class[latent_idx] = majority_class
                neuron_purity[latent_idx] = purity

        # Print distribution
        class_counts = defaultdict(int)
        class_purities = defaultdict(list)
        for n, c in neuron_to_class.items():
            class_counts[c] += 1
            class_purities[c].append(neuron_purity[n])

        print(f"Neuron assignments (purity >= {purity_threshold}):")
        for c in sorted(class_counts.keys()):
            mean_purity = np.mean(class_purities[c])
            print(
                f"  Class {c}: {class_counts[c]} neurons, mean purity: {mean_purity:.3f}"
            )

        return neuron_to_class, neuron_purity

    def _setup_optimizer(self):
        """Setup optimizer for learnable parameters."""
        wcfg = self.cfg.MITIGATOR.SAE_WEIGHTED_CLASSIFIER

        # Collect trainable parameters
        trainable_params = []

        if wcfg.WEIGHT_MODE == "learnable":
            # Per-neuron weights
            for c in range(self.num_class):
                param_name = f"class_{c}_weights"
                if hasattr(self.weighted_classifier, param_name):
                    trainable_params.append(
                        getattr(self.weighted_classifier, param_name)
                    )

        if wcfg.LEARN_CLASS_BIAS:
            trainable_params.append(self.weighted_classifier.class_bias)

        if wcfg.LEARN_TEMPERATURE:
            trainable_params.append(self.weighted_classifier.log_temperature)

        if not trainable_params:
            print("No trainable parameters - evaluation only mode")
            self.optimizer = None
            return

        total_params = sum(p.numel() for p in trainable_params)
        print(f"Trainable parameters: {total_params}")

        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.cfg.SOLVER.LR,
            weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
        )

        # Simple scheduler
        if self.cfg.SOLVER.EPOCHS > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.SOLVER.EPOCHS,
            )
        else:
            self.scheduler = None

    def _train_iter(self, batch):
        """Training iteration."""
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss}

    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.model.clip_model.eval()
        self.model.sae.eval()

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
        """Main training/evaluation loop."""
        wcfg = self.cfg.MITIGATOR.SAE_WEIGHTED_CLASSIFIER

        print(f"\n{'='*60}")
        print(f"SAE Weighted Classifier - Mode: {wcfg.WEIGHT_MODE}")
        print(f"{'='*60}")

        # Initial evaluation
        print("\nInitial evaluation:")
        init_metrics = self.eval()
        for k, v in init_metrics.items():
            print(f"  {k}: {v:.4f}")

        # If no training needed (purity mode without learning)
        if self.optimizer is None:
            print("\nNo training - purity weighted mode")
            self._save_results(init_metrics, epoch=-1)
            self._analyze_weights()
            return init_metrics

        # Training loop
        best_metric = 0
        best_epoch = -1

        for epoch in range(self.cfg.SOLVER.EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.cfg.SOLVER.EPOCHS}")

            train_metrics = self._train_epoch()
            print(f"Train loss: {train_metrics['loss']:.4f}")

            eval_metrics = self.eval()
            current_metric = eval_metrics.get(
                self.cfg.METRIC, eval_metrics.get("accuracy", 0)
            )
            print(f"Eval {self.cfg.METRIC}: {current_metric:.4f}")

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)
                print(f"New best! {self.cfg.METRIC}: {best_metric:.4f}")

        print(f"\n{'='*60}")
        print(
            f"Training complete! Best {self.cfg.METRIC}: {best_metric:.4f} (epoch {best_epoch+1})"
        )
        print(f"{'='*60}")

        # Final evaluation and analysis
        final_metrics = self.eval()
        self._save_results(final_metrics, epoch=best_epoch)
        self._analyze_weights()

        return final_metrics

    def eval(self):
        """Evaluate the model."""
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

                for bias_name in self.biases:
                    if bias_name in batch:
                        all_biases[bias_name].append(batch[bias_name])

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        accuracy = (all_preds == all_targets).float().mean().item()
        metrics = {"accuracy": accuracy}

        # Per-class accuracy
        for c in range(self.num_class):
            mask = all_targets == c
            if mask.sum() > 0:
                metrics[f"acc_class_{c}"] = (
                    (all_preds[mask] == all_targets[mask]).float().mean().item()
                )

        # Worst-group accuracy
        if all_biases:
            bias_name = self.biases[0]
            all_bias = torch.cat(all_biases[bias_name])

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

        return metrics

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "weighted_classifier": self.weighted_classifier.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "weight_stats": self.weighted_classifier.get_weight_stats(),
        }

        save_path = os.path.join(self.log_path, "last.pth")
        torch.save(checkpoint, save_path)

        if is_best:
            best_path = os.path.join(self.log_path, "best.pth")
            torch.save(checkpoint, best_path)

    def _save_results(self, metrics, epoch):
        """Save results to file."""
        wcfg = self.cfg.MITIGATOR.SAE_WEIGHTED_CLASSIFIER

        results = {
            "metrics": metrics,
            "epoch": epoch,
            "config": {
                "weight_mode": wcfg.WEIGHT_MODE,
                "purity_threshold": wcfg.PURITY_THRESHOLD,
                "init_with_purity": wcfg.INIT_WITH_PURITY,
                "learn_temperature": wcfg.LEARN_TEMPERATURE,
                "learn_class_bias": wcfg.LEARN_CLASS_BIAS,
            },
            "weight_stats": self.weighted_classifier.get_weight_stats(),
        }

        results_path = os.path.join(self.log_path, "weighted_classifier_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

    def _analyze_weights(self):
        """Analyze and visualize learned weights."""
        print(f"\n{'='*60}")
        print("Weight Analysis")
        print(f"{'='*60}")

        stats = self.weighted_classifier.get_weight_stats()

        print(f"Temperature: {stats['temperature']:.4f}")
        print(f"Class bias: {stats['class_bias']}")

        for c in range(self.num_class):
            print(f"\nClass {c}:")
            if f"class_{c}_weights" in stats:
                ws = stats[f"class_{c}_weights"]
                print(f"  Neurons: {ws['num_neurons']}")
                print(f"  Weight mean: {ws['mean']:.4f}, std: {ws['std']:.4f}")
                print(f"  Weight range: [{ws['min']:.4f}, {ws['max']:.4f}]")

            # Top neurons
            top_neurons = self.weighted_classifier.get_top_neurons(c, top_k=5)
            if top_neurons:
                print(f"  Top 5 neurons: {top_neurons}")
