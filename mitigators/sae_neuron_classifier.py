"""
SAE Neuron Classifier for VB-Mitigator.

This module implements an interpretable classification approach that:
1. Uses OpenCLIP (or other) as a frozen pretrained backbone
2. Passes features through a pretrained SAE encoder
3. Filters to keep only neurons with activation above threshold
4. Filters to keep only monosemantic neurons (high class purity)
5. Assigns each kept neuron to a class (based on majority class in top-k images)
6. Classifies by summing activations of neurons assigned to each class

This is a training-free, interpretable classifier where:
- Each class has a set of "dedicated" neurons
- Classification score = sum of activations from that class's neurons
- No learnable parameters (aside from optional temperature scaling)

Based on the observation that monosemantic SAE neurons act as class-specific feature detectors.
"""

import os
import json
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


class SAENeuronClassifier(nn.Module):
    """
    Classifier that uses SAE neuron activations directly for classification.

    Each monosemantic neuron is assigned to a class, and classification is done
    by summing activations of neurons belonging to each class.

    Args:
        sae: Trained Sparse Autoencoder
        neuron_to_class: Dict mapping neuron_idx -> class_idx (only for kept neurons)
        num_classes: Number of output classes
        temperature: Temperature for softmax (higher = softer predictions)
        aggregation: How to aggregate neuron activations ('sum', 'mean', 'max')
    """

    def __init__(
        self, sae, neuron_to_class, num_classes, temperature=1.0, aggregation="sum"
    ):
        super().__init__()
        self.sae = sae
        self.num_classes = num_classes
        self.temperature = temperature
        self.aggregation = aggregation

        # Freeze SAE parameters
        for param in self.sae.parameters():
            param.requires_grad = False

        # Create class masks: for each class, which neurons belong to it
        dict_size = sae.dict_size
        self.class_masks = []
        self.neurons_per_class = []

        for class_idx in range(num_classes):
            mask = torch.zeros(dict_size, dtype=torch.bool)
            neurons_for_class = [
                n for n, c in neuron_to_class.items() if c == class_idx
            ]
            for neuron_idx in neurons_for_class:
                mask[neuron_idx] = True
            self.class_masks.append(mask)
            self.neurons_per_class.append(len(neurons_for_class))

        # Register as buffers so they move to correct device
        for i, mask in enumerate(self.class_masks):
            self.register_buffer(f"class_mask_{i}", mask)

        # Store neuron assignments for interpretability
        self.neuron_to_class = neuron_to_class

        # Optional learnable temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

        print(f"SAENeuronClassifier initialized:")
        print(f"  Total neurons: {dict_size}")
        print(f"  Active neurons: {len(neuron_to_class)}")
        for c in range(num_classes):
            print(f"  Class {c}: {self.neurons_per_class[c]} neurons")

    def get_class_mask(self, class_idx):
        """Get the boolean mask for neurons belonging to a class."""
        return getattr(self, f"class_mask_{class_idx}")

    def forward(self, sparse_features):
        """
        Compute class scores from sparse SAE features.

        Args:
            sparse_features: SAE latent activations of shape (B, dict_size)

        Returns:
            logits: Class scores of shape (B, num_classes)
        """
        batch_size = sparse_features.shape[0]
        scores = torch.zeros(
            batch_size, self.num_classes, device=sparse_features.device
        )

        for class_idx in range(self.num_classes):
            mask = self.get_class_mask(class_idx)
            class_activations = sparse_features[:, mask]  # (B, num_neurons_for_class)

            if self.aggregation == "sum":
                scores[:, class_idx] = class_activations.sum(dim=1)
            elif self.aggregation == "mean":
                if class_activations.shape[1] > 0:
                    scores[:, class_idx] = class_activations.mean(dim=1)
            elif self.aggregation == "max":
                if class_activations.shape[1] > 0:
                    scores[:, class_idx] = class_activations.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Apply temperature scaling
        temperature = torch.exp(self.log_temperature)
        logits = scores / temperature

        return logits

    def get_neuron_contributions(self, sparse_features):
        """
        Get per-neuron contributions to each class for interpretability.

        Returns:
            Dict with class_idx -> tensor of shape (B, num_neurons_for_class)
        """
        contributions = {}
        for class_idx in range(self.num_classes):
            mask = self.get_class_mask(class_idx)
            contributions[class_idx] = sparse_features[:, mask]
        return contributions


class OpenCLIPWithSAENeuronClassifier(nn.Module):
    """
    Full model: OpenCLIP backbone + SAE encoder + Neuron-based classifier.

    Architecture:
        Input Image -> OpenCLIP (frozen) -> Features -> SAE Encoder ->
        Sparse Activations -> Neuron Aggregation per Class -> Class Scores
    """

    def __init__(self, clip_model, sae, neuron_classifier):
        super().__init__()
        self.clip_model = clip_model
        self.sae = sae
        self.neuron_classifier = neuron_classifier

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

    def forward(self, x, return_sparse=False):
        """
        Forward pass.

        Args:
            x: Input images of shape (B, C, H, W)
            return_sparse: If True, also return sparse features

        Returns:
            logits: Class scores (B, num_classes)
            sparse_features (optional): SAE activations (B, dict_size)
        """
        sparse_features = self.encode_to_sparse(x)
        logits = self.neuron_classifier(sparse_features)

        if return_sparse:
            return logits, sparse_features
        return logits

    def predict(self, x):
        """Get predicted class."""
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def get_interpretable_prediction(self, x):
        """
        Get prediction with interpretability info.

        Returns dict with:
            - prediction: predicted class
            - scores: per-class scores
            - top_neurons: most active neurons per class
        """
        sparse_features = self.encode_to_sparse(x)
        logits = self.neuron_classifier(sparse_features)

        contributions = self.neuron_classifier.get_neuron_contributions(sparse_features)

        return {
            "prediction": logits.argmax(dim=1),
            "scores": logits,
            "sparse_features": sparse_features,
            "contributions": contributions,
        }


class SAENeuronClassifierTrainer(BaseTrainer):
    """
    Trainer for SAE Neuron-based Classifier.

    This is primarily an evaluation/analysis tool since the classifier has
    no learnable parameters (except optional temperature). It:
    1. Loads pretrained OpenCLIP and SAE
    2. Analyzes neuron class assignments from SAE analysis results
    3. Filters neurons by activation threshold AND purity threshold
    4. Builds the neuron-based classifier
    5. Evaluates on test set

    Optionally can fine-tune temperature via a small validation set.

    Configuration:
        MITIGATOR:
          TYPE: "sae_neuron_classifier"
          SAE_NEURON_CLASSIFIER:
            SAE_CHECKPOINT_PATH: "path/to/sae/ae.pt"
            SAE_ANALYSIS_PATH: "path/to/sae/analysis_results.json"
            PURITY_THRESHOLD: 1.0
            ACTIVATION_THRESHOLD: 0.1  # Only use neurons with max activation >= this
            AGGREGATION: "sum"  # 'sum', 'mean', or 'max'
            TEMPERATURE: 1.0
            LEARN_TEMPERATURE: False  # If True, optimize temperature on train set
    """

    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """Setup OpenCLIP backbone, SAE, and neuron classifier."""

        classifier_cfg = self.cfg.MITIGATOR.SAE_NEURON_CLASSIFIER

        print(f"\n{'='*60}")
        print("Setting up SAE Neuron Classifier")
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
        sae_checkpoint_path = classifier_cfg.SAE_CHECKPOINT_PATH
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
        analysis_path = classifier_cfg.SAE_ANALYSIS_PATH
        if not analysis_path or not os.path.exists(analysis_path):
            raise ValueError(f"SAE analysis not found: {analysis_path}")

        with open(analysis_path, "r") as f:
            analysis_results = json.load(f)

        print(f"Loaded analysis from {analysis_path}")

        # Step 4: Compute neuron-to-class assignment with both filters
        neuron_to_class, filter_stats = self._compute_neuron_class_assignment(
            analysis_results,
            dict_size=dict_size,
            purity_threshold=classifier_cfg.PURITY_THRESHOLD,
            activation_threshold=classifier_cfg.ACTIVATION_THRESHOLD,
        )

        print(f"Assigned {len(neuron_to_class)} neurons to classes")

        # Store filter statistics
        self.filter_stats = filter_stats

        # Step 5: Create neuron classifier
        self.neuron_classifier = SAENeuronClassifier(
            sae=self.sae,
            neuron_to_class=neuron_to_class,
            num_classes=self.num_class,
            temperature=classifier_cfg.TEMPERATURE,
            aggregation=classifier_cfg.AGGREGATION,
        )

        # Step 6: Create full model
        self.model = OpenCLIPWithSAENeuronClassifier(
            clip_model=self.clip_model,
            sae=self.sae,
            neuron_classifier=self.neuron_classifier,
        )

        self.model.to(self.device)

        # Store for analysis
        self.neuron_to_class = neuron_to_class
        self.dict_size = dict_size

    def _compute_neuron_class_assignment(
        self, analysis_results, dict_size, purity_threshold, activation_threshold
    ):
        """
        Compute which class each neuron belongs to.

        Neurons are kept only if they pass BOTH filters:
        1. Maximum activation >= activation_threshold
        2. Class purity >= purity_threshold

        Each kept neuron is assigned to the majority class among its top-k activating images.

        Args:
            analysis_results: Dict from SAE analysis
            dict_size: Total number of SAE neurons
            purity_threshold: Minimum purity to include neuron (0.0 to 1.0)
            activation_threshold: Minimum max activation to include neuron

        Returns:
            Tuple of:
            - neuron_to_class: Dict mapping neuron_idx -> class_idx
            - filter_stats: Dict with filtering statistics
        """
        neuron_to_class = {}

        top_k_per_latent = analysis_results.get("top_k_per_latent", {})

        # Statistics tracking
        stats = {
            "total_neurons": dict_size,
            "alive_neurons": len(analysis_results.get("alive_neurons", [])),
            "passed_activation_filter": 0,
            "passed_purity_filter": 0,
            "passed_both_filters": 0,
            "failed_activation_filter": 0,
            "failed_purity_filter": 0,
            "dead_neurons": 0,
        }

        class_neuron_counts = defaultdict(int)

        for latent_idx in range(dict_size):
            latent_key = str(latent_idx)

            if latent_key not in top_k_per_latent:
                stats["dead_neurons"] += 1
                continue

            latent_data = top_k_per_latent[latent_key]
            targets = latent_data.get("targets", [])
            activations = latent_data.get("activations", [])

            if len(targets) == 0 or len(activations) == 0:
                stats["dead_neurons"] += 1
                continue

            # Filter 1: Check activation threshold (using max activation)
            max_activation = max(activations)
            if max_activation < activation_threshold:
                stats["failed_activation_filter"] += 1
                continue

            stats["passed_activation_filter"] += 1

            # Filter 2: Compute class purity
            target_counts = defaultdict(int)
            for t in targets:
                target_counts[t] += 1

            # Find majority class and its purity
            majority_class = max(target_counts.keys(), key=lambda k: target_counts[k])
            purity = target_counts[majority_class] / len(targets)

            if purity < purity_threshold:
                stats["failed_purity_filter"] += 1
                continue

            stats["passed_purity_filter"] += 1
            stats["passed_both_filters"] += 1

            # Assign neuron to majority class
            neuron_to_class[latent_idx] = majority_class
            class_neuron_counts[majority_class] += 1

        # Print filtering summary
        print(f"\nNeuron Filtering Summary:")
        print(f"  Total neurons: {stats['total_neurons']}")
        print(f"  Alive neurons: {stats['alive_neurons']}")
        print(f"  Dead neurons: {stats['dead_neurons']}")
        print(f"  Activation threshold: {activation_threshold}")
        print(f"    Passed: {stats['passed_activation_filter']}")
        print(f"    Failed: {stats['failed_activation_filter']}")
        print(f"  Purity threshold: {purity_threshold}")
        print(f"    Passed (of activation-filtered): {stats['passed_both_filters']}")
        print(f"    Failed: {stats['failed_purity_filter']}")
        print(f"  Final kept neurons: {len(neuron_to_class)}")

        print(f"\nNeurons per class:")
        for class_idx in sorted(class_neuron_counts.keys()):
            print(f"  Class {class_idx}: {class_neuron_counts[class_idx]} neurons")

        return neuron_to_class, stats

    def _setup_optimizer(self):
        """Setup optimizer for temperature learning (if enabled)."""
        classifier_cfg = self.cfg.MITIGATOR.SAE_NEURON_CLASSIFIER

        if classifier_cfg.LEARN_TEMPERATURE:
            # Only optimize temperature parameter
            self.optimizer = torch.optim.Adam(
                [self.neuron_classifier.log_temperature],
                lr=0.1,
            )
        else:
            self.optimizer = None

    def train(self):
        """
        Main entry point.

        Since this is primarily a training-free method, we just:
        1. Optionally learn temperature on train set
        2. Evaluate on test set
        3. Provide interpretability analysis
        """
        print(f"\n{'='*60}")
        print("SAE Neuron Classifier Evaluation")
        print(f"{'='*60}")

        classifier_cfg = self.cfg.MITIGATOR.SAE_NEURON_CLASSIFIER

        # Optionally learn temperature
        if classifier_cfg.LEARN_TEMPERATURE:
            print("\nLearning temperature on training set...")
            self._learn_temperature()

        # Evaluate
        print("\nEvaluating on test set...")
        metrics = self.eval()

        print(f"\n{'='*60}")
        print("Results")
        print(f"{'='*60}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Save results
        self._save_results(metrics)

        # Interpretability analysis
        print("\nRunning interpretability analysis...")
        self._interpretability_analysis()

        return metrics

    def _learn_temperature(self, num_epochs=10):
        """Learn optimal temperature on training set."""
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for batch in self.dataloaders["train"]:
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)

                logits = self.model(inputs)
                loss = F.cross_entropy(logits, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            temp = torch.exp(self.neuron_classifier.log_temperature).item()
            print(
                f"  Epoch {epoch+1}/{num_epochs}: loss={total_loss/num_batches:.4f}, temp={temp:.4f}"
            )

    def eval(self):
        """Evaluate the model."""
        self.model.eval()

        all_preds = []
        all_targets = []
        all_biases = defaultdict(list)
        all_scores = []

        with torch.no_grad():
            for batch in tqdm(self.dataloaders["test"], desc="Evaluating"):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]

                logits = self.model(inputs)
                preds = logits.argmax(dim=1).cpu()

                all_preds.append(preds)
                all_targets.append(targets)
                all_scores.append(logits.cpu())

                for bias_name in self.biases:
                    if bias_name in batch:
                        all_biases[bias_name].append(batch[bias_name])

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_scores = torch.cat(all_scores)

        # Compute metrics
        accuracy = (all_preds == all_targets).float().mean().item()
        metrics = {"accuracy": accuracy}

        # Per-class accuracy
        for c in range(self.num_class):
            mask = all_targets == c
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_targets[mask]).float().mean().item()
                metrics[f"acc_class_{c}"] = class_acc

        # Worst-group accuracy if bias info available
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
                        metrics[f"acc_t{t}_b{b}"] = group_acc

            if group_accs:
                metrics["wg_ovr"] = min(group_accs)
                metrics["avg_group_acc"] = np.mean(group_accs)

        return metrics

    def _save_results(self, metrics):
        """Save results to file."""
        results = {
            "metrics": metrics,
            "config": {
                "purity_threshold": self.cfg.MITIGATOR.SAE_NEURON_CLASSIFIER.PURITY_THRESHOLD,
                "activation_threshold": self.cfg.MITIGATOR.SAE_NEURON_CLASSIFIER.ACTIVATION_THRESHOLD,
                "aggregation": self.cfg.MITIGATOR.SAE_NEURON_CLASSIFIER.AGGREGATION,
                "temperature": torch.exp(self.neuron_classifier.log_temperature).item(),
                "num_active_neurons": len(self.neuron_to_class),
                "neurons_per_class": self.neuron_classifier.neurons_per_class,
            },
            "filter_stats": self.filter_stats,
        }

        save_path = os.path.join(self.log_path, "neuron_classifier_results.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {save_path}")

    def _interpretability_analysis(self):
        """Analyze which neurons contribute most to predictions."""
        self.model.eval()

        # Get a batch of test samples
        batch = next(iter(self.dataloaders["test"]))
        inputs = batch["inputs"][:16].to(self.device)  # Take 16 samples
        targets = batch["targets"][:16]

        with torch.no_grad():
            result = self.model.get_interpretable_prediction(inputs)

        predictions = result["prediction"].cpu()
        scores = result["scores"].cpu()
        contributions = result["contributions"]

        print(f"\nSample predictions analysis:")
        print("-" * 50)

        for i in range(min(5, len(inputs))):
            pred = predictions[i].item()
            target = targets[i].item()
            correct = "✓" if pred == target else "✗"

            print(f"\nSample {i}: target={target}, pred={pred} {correct}")
            print(f"  Scores: {[f'{s:.3f}' for s in scores[i].tolist()]}")

            # Show top contributing neurons for predicted class
            class_contrib = contributions[pred][i]
            if len(class_contrib) > 0:
                top_k = min(5, len(class_contrib))
                top_vals, top_idx = class_contrib.topk(top_k)
                print(f"  Top neurons for class {pred}: ", end="")

                # Get actual neuron indices
                class_mask = self.neuron_classifier.get_class_mask(pred)
                neuron_indices = class_mask.nonzero().squeeze(-1)
                print(
                    [
                        f"n{neuron_indices[idx].item()}({top_vals[j].item():.3f})"
                        for j, idx in enumerate(top_idx)
                    ]
                )
