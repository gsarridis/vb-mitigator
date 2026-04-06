"""
SAE (Sparse Autoencoder) Trainer/Analyzer for VB-Mitigator.

This module implements a mitigator that:
1. Loads a pretrained vanilla (ERM) model from checkpoints
2. Extracts penultimate layer features from the dataset
3. Trains a Sparse Autoencoder on those features
4. Analyzes monosemantic neurons and generates visualizations

Based on the dictionary_learning library and inspired by:
- "Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models" (Pach et al., 2025)
"""

import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from models.builder import get_model
from models.utils import get_local_model_dict
from .base_trainer import BaseTrainer
from tools.utils import log_msg, save_checkpoint, load_checkpoint

# Import dictionary learning components
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dictionary_learning"))

from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import (
    StandardTrainer as SAEStandardTrainer,
    TopKTrainer,
    JumpReluTrainer,
    BatchTopKTrainer,
)
from dictionary_learning.training import trainSAE


class SAETrainer(BaseTrainer):
    """
    SAE Trainer for interpretability analysis of visual bias models.

    This trainer:
    1. Loads a pretrained ERM model
    2. Extracts features from the penultimate layer
    3. Trains a Sparse Autoencoder on those features
    4. Identifies and visualizes monosemantic neurons

    The goal is to understand what features the model has learned and potentially
    identify bias-related features in the latent space.
    """

    def __init__(self, cfg):
        """Initialize the SAE trainer with modified setup."""
        # Store config before calling parent init
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """Setup the pretrained model for feature extraction."""
        # Load the pretrained ERM model
        self.model = get_model(
            self.cfg.MODEL.TYPE, self.num_class, pretrained=self.cfg.MODEL.PRETRAINED
        )

        # Load pretrained weights if checkpoint path is provided
        if self.cfg.MITIGATOR.SAE.CHECKPOINT_PATH != "":
            checkpoint_path = self.cfg.MITIGATOR.SAE.CHECKPOINT_PATH
            print(f"Loading pretrained model from: {checkpoint_path}")

            checkpoint = load_checkpoint(checkpoint_path)
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            else:
                self.model.load_state_dict(checkpoint)
            print("Successfully loaded pretrained model weights")

        self.model.to(self.device)
        self.model.eval()  # Set to eval mode for feature extraction

        # Determine feature dimension based on model type
        self.feature_dim = self._get_feature_dim()
        print(f"Feature dimension: {self.feature_dim}")

        # Initialize SAE (will be set up during training)
        self.sae = None
        self.sae_trainer = None

    def _get_feature_dim(self):
        """Determine the feature dimension of the penultimate layer."""
        # Try common attribute names for different architectures
        try:
            if hasattr(self.model, "embed_size"):
                return self.model.embed_size
            elif hasattr(self.model, "fc"):
                return self.model.fc.in_features
            elif hasattr(self.model, "classifier"):
                if isinstance(self.model.classifier, nn.Linear):
                    return self.model.classifier.in_features
                else:
                    return self.model.classifier[-1].in_features
            else:
                # Default for ResNet-style models
                return 512
        except:
            return 512

    def _setup_optimizer(self):
        """Override optimizer setup - SAE has its own optimizer."""
        # We don't need the standard optimizer for the main model
        # SAE training uses its own optimizer internally
        self.optimizer = None

    def _setup_scheduler(self):
        """Override scheduler setup - SAE has its own scheduler."""
        self.scheduler = None

    def _method_specific_setups(self):
        """Setup SAE-specific components."""
        self.sae_cfg = self.cfg.MITIGATOR.SAE

        # Calculate dictionary size based on expansion factor
        self.dict_size = self.feature_dim * self.sae_cfg.EXPANSION_FACTOR
        print(
            f"SAE Dictionary size: {self.dict_size} (expansion factor: {self.sae_cfg.EXPANSION_FACTOR})"
        )

    def _extract_features(self, dataloader, desc="Extracting features"):
        """
        Extract penultimate layer features from the model.

        Args:
            dataloader: DataLoader to extract features from
            desc: Description for progress bar

        Returns:
            features: Tensor of shape (N, feature_dim)
            targets: Tensor of target labels
            biases: Dict of bias attributes
            indices: List of sample indices (for visualization)
        """
        all_features = []
        all_targets = []
        all_biases = {b: [] for b in self.biases}
        all_indices = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                inputs = batch["inputs"].to(self.device)

                # Forward pass to get features
                outputs = self.model(inputs)

                # Model returns (logits, features) tuple
                if isinstance(outputs, tuple):
                    _, features = outputs
                else:
                    # If model doesn't return features, we need to hook
                    features = self._extract_with_hook(inputs)

                all_features.append(features.cpu())
                all_targets.append(batch["targets"])

                for b in self.biases:
                    if b in batch:
                        all_biases[b].append(batch[b])

                if "index" in batch:
                    all_indices.extend(batch["index"].tolist())
                else:
                    all_indices.extend(range(len(inputs)))

        features = torch.cat(all_features, dim=0)
        targets = torch.cat(all_targets, dim=0)
        biases = {b: torch.cat(v, dim=0) if v else None for b, v in all_biases.items()}

        return features, targets, biases, all_indices

    def _extract_with_hook(self, inputs):
        """Extract features using a forward hook if model doesn't return them."""
        features = None

        def hook(module, input, output):
            nonlocal features
            if isinstance(output, tuple):
                features = output[0]
            else:
                features = output

        # Try to find the layer before the final classifier
        if hasattr(self.model, "extractor"):
            handle = self.model.extractor.register_forward_hook(hook)
        elif hasattr(self.model, "avgpool"):
            handle = self.model.avgpool.register_forward_hook(hook)
        else:
            # Fallback: just use the full forward pass
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                return outputs[1]
            return outputs

        _ = self.model(inputs)
        handle.remove()

        return features.view(features.size(0), -1)

    def _train_sae(self, features):
        """
        Train the Sparse Autoencoder on extracted features.

        Args:
            features: Tensor of shape (N, feature_dim)

        Returns:
            Trained SAE model
        """
        print(f"\n{'='*60}")
        print("Training Sparse Autoencoder")
        print(f"{'='*60}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Dictionary size: {self.dict_size}")
        print(f"SAE type: {self.sae_cfg.TYPE}")
        print(f"Training steps: {self.sae_cfg.STEPS}")

        # Create dataset and dataloader for SAE training
        dataset = TensorDataset(features)

        # Determine SAE trainer type
        sae_type = self.sae_cfg.TYPE.lower()

        # Build trainer configuration
        trainer_cfg = {
            "activation_dim": self.feature_dim,
            "dict_size": self.dict_size,
            "lr": self.sae_cfg.LR,
            "device": str(self.device),
            "layer": 0,  # Not applicable for vision features
            "lm_name": f"vision_{self.cfg.MODEL.TYPE}",
            "steps": self.sae_cfg.STEPS,
            "warmup_steps": self.sae_cfg.WARMUP_STEPS,
            "seed": self.cfg.EXPERIMENT.SEED,
        }

        trainer_cfg["l1_penalty"] = self.sae_cfg.L1_PENALTY
        if sae_type == "standard":
            trainer_cfg["trainer"] = SAEStandardTrainer
            trainer_cfg["dict_class"] = AutoEncoder

            trainer_cfg["resample_steps"] = (
                self.sae_cfg.RESAMPLE_STEPS if self.sae_cfg.RESAMPLE_STEPS > 0 else None
            )
        elif sae_type == "topk":
            trainer_cfg["trainer"] = TopKTrainer
            trainer_cfg["k"] = self.sae_cfg.K
            trainer_cfg["auxk_alpha"] = self.sae_cfg.AUXK_ALPHA
        elif sae_type == "batch_topk":
            trainer_cfg["trainer"] = BatchTopKTrainer
            trainer_cfg["k"] = self.sae_cfg.K
            trainer_cfg["auxk_alpha"] = self.sae_cfg.AUXK_ALPHA
        elif sae_type == "jumprelu":
            trainer_cfg["trainer"] = JumpReluTrainer
            trainer_cfg["bandwidth"] = self.sae_cfg.BANDWIDTH
            trainer_cfg["sparsity_penalty"] = self.sae_cfg.SPARSITY_PENALTY
        else:
            raise ValueError(f"Unknown SAE type: {sae_type}")

        # Create a simple data generator
        def data_generator():
            indices = torch.randperm(len(features))
            batch_size = self.sae_cfg.BATCH_SIZE
            for i in range(0, len(features), batch_size):
                batch_indices = indices[i : i + batch_size]
                yield features[batch_indices].to(self.device)

        # Create save directory
        sae_save_dir = os.path.join(self.log_path, "sae_checkpoints")
        os.makedirs(sae_save_dir, exist_ok=True)

        # Train SAE
        # Note: trainSAE expects a buffer-like object or we can manually train
        self.sae = self._manual_sae_train(features, trainer_cfg, sae_save_dir)

        return self.sae

    def _manual_sae_train(self, features, trainer_cfg, save_dir):
        """
        Manually train SAE with more control over the process.
        """
        # Initialize SAE
        if trainer_cfg["trainer"] == SAEStandardTrainer:
            sae = AutoEncoder(self.feature_dim, self.dict_size)
        else:
            # For other types, use AutoEncoder as base
            sae = AutoEncoder(self.feature_dim, self.dict_size)

        sae = sae.to(self.device)

        # Optimizer with decoder norm constraint
        optimizer = torch.optim.Adam(sae.parameters(), lr=trainer_cfg["lr"])

        # Training loop
        steps = trainer_cfg["steps"]
        batch_size = self.sae_cfg.BATCH_SIZE
        l1_penalty = trainer_cfg.get("l1_penalty", 1e-3)

        num_epochs = max(1, steps * batch_size // len(features))
        step = 0

        losses = []

        print(f"Training for {num_epochs} epochs (~{steps} steps)")

        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(len(features))
            epoch_loss = 0
            num_batches = 0

            for i in range(0, len(features), batch_size):
                if step >= steps:
                    break

                batch_indices = indices[i : i + batch_size]
                batch = features[batch_indices].to(self.device)

                # Forward pass
                reconstructed, latents = sae(batch, output_features=True)

                # Compute loss
                mse_loss = ((batch - reconstructed) ** 2).mean()
                l1_loss = latents.abs().mean()
                loss = mse_loss + l1_penalty * l1_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Normalize decoder weights
                with torch.no_grad():
                    norms = sae.decoder.weight.norm(dim=0, keepdim=True)
                    sae.decoder.weight.div_(norms.clamp(min=1e-8))

                epoch_loss += loss.item()
                num_batches += 1
                step += 1

                if step % 100 == 0:
                    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                    l0 = (latents > 0).float().sum(dim=1).mean().item()
                    print(f"Step {step}/{steps} | Loss: {avg_loss:.4f} | L0: {l0:.1f}")

            if step >= steps:
                break

            losses.append(epoch_loss / max(num_batches, 1))

        # Save final model
        save_path = os.path.join(save_dir, "ae.pt")
        torch.save(sae.state_dict(), save_path)
        print(f"Saved SAE to {save_path}")

        # Save config
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "activation_dim": self.feature_dim,
                    "dict_size": self.dict_size,
                    "sae_type": self.sae_cfg.TYPE,
                    "l1_penalty": l1_penalty,
                    "lr": trainer_cfg["lr"],
                    "steps": steps,
                },
                f,
                indent=2,
            )

        return sae

    def _analyze_monosemanticity(self, features, targets, biases, indices):
        """
        Analyze monosemantic neurons in the trained SAE.

        For each SAE latent dimension, find the top-k most activating samples
        and compute statistics about the semantic coherence.

        Args:
            features: Original features (N, feature_dim)
            targets: Target labels (N,)
            biases: Dict of bias attributes
            indices: Sample indices for visualization

        Returns:
            analysis_results: Dict containing monosemanticity metrics
        """
        print(f"\n{'='*60}")
        print("Analyzing Monosemanticity")
        print(f"{'='*60}")

        self.sae.eval()

        # Encode all features
        with torch.no_grad():
            all_latents = []
            batch_size = self.sae_cfg.BATCH_SIZE

            for i in range(0, len(features), batch_size):
                batch = features[i : i + batch_size].to(self.device)
                latents = self.sae.encode(batch)
                all_latents.append(latents.cpu())

            all_latents = torch.cat(all_latents, dim=0)  # (N, dict_size)

        # Analyze each latent dimension
        analysis_results = {
            "num_alive_neurons": 0,
            "mean_l0": all_latents.gt(0).float().sum(dim=1).mean().item(),
            "latent_activations": {},
            "top_k_per_latent": {},
            "class_distribution_per_latent": {},
            "bias_distribution_per_latent": {},
        }

        k = self.sae_cfg.TOP_K_IMAGES

        alive_neurons = []
        for latent_idx in tqdm(range(self.dict_size), desc="Analyzing latents"):
            activations = all_latents[:, latent_idx]

            # Check if neuron is alive (has any activations)
            if activations.max() > 1e-6:
                alive_neurons.append(latent_idx)

                # Get top-k activating samples
                top_k_values, top_k_indices = activations.topk(k)

                analysis_results["top_k_per_latent"][latent_idx] = {
                    "indices": [indices[i] for i in top_k_indices.tolist()],
                    "activations": top_k_values.tolist(),
                    "targets": targets[top_k_indices].tolist(),
                }

                # Class distribution among top-k
                top_k_targets = targets[top_k_indices]
                class_counts = torch.bincount(top_k_targets, minlength=self.num_class)
                analysis_results["class_distribution_per_latent"][
                    latent_idx
                ] = class_counts.tolist()

                # Bias distribution among top-k
                for b_name, b_values in biases.items():
                    if b_values is not None:
                        top_k_bias = b_values[top_k_indices]
                        if (
                            b_name
                            not in analysis_results["bias_distribution_per_latent"]
                        ):
                            analysis_results["bias_distribution_per_latent"][
                                b_name
                            ] = {}
                        analysis_results["bias_distribution_per_latent"][b_name][
                            latent_idx
                        ] = top_k_bias.tolist()

        analysis_results["num_alive_neurons"] = len(alive_neurons)
        analysis_results["alive_neurons"] = alive_neurons
        analysis_results["percent_alive"] = len(alive_neurons) / self.dict_size * 100

        print(
            f"Alive neurons: {len(alive_neurons)}/{self.dict_size} ({analysis_results['percent_alive']:.1f}%)"
        )
        print(f"Mean L0: {analysis_results['mean_l0']:.2f}")

        return analysis_results

    def _analyze_mean_activations(self, features, targets, biases, all_latents=None):
        """
        Analyze mean neuron activation values per target class, bias attribute,
        and their combinations (data groups).

        This helps understand:
        - Which neurons are target-specific vs bias-specific
        - Which neurons fire for specific demographic groups
        - Potential spurious correlations (neurons correlating more with bias than target)

        Args:
            features: Original features (N, feature_dim)
            targets: Target labels tensor (N,)
            biases: Dict of bias attribute name -> tensor (N,)
            all_latents: Pre-computed SAE latents (N, dict_size), or None to compute

        Returns:
            Dict containing mean activation statistics and saves visualizations
        """
        print(f"\n{'='*60}")
        print("Analyzing Mean Neuron Activations by Groups")
        print(f"{'='*60}")

        self.sae.eval()

        # Compute SAE latents if not provided
        if all_latents is None:
            with torch.no_grad():
                all_latents = []
                batch_size = self.sae_cfg.BATCH_SIZE

                for i in range(0, len(features), batch_size):
                    batch = features[i : i + batch_size].to(self.device)
                    latents = self.sae.encode(batch)
                    all_latents.append(latents.cpu())

                all_latents = torch.cat(all_latents, dim=0)  # (N, dict_size)

        # Create output directory
        activation_vis_dir = os.path.join(self.log_path, "activation_analysis")
        os.makedirs(activation_vis_dir, exist_ok=True)

        # Get unique target values
        unique_targets = torch.unique(targets).tolist()
        num_targets = len(unique_targets)

        # Get primary bias attribute
        bias_name = self.biases[0] if self.biases else None
        bias_values = biases.get(bias_name) if bias_name else None
        unique_biases = (
            torch.unique(bias_values).tolist() if bias_values is not None else []
        )
        num_biases = len(unique_biases)

        results = {
            "target_names": unique_targets,
            "bias_name": bias_name,
            "bias_values": unique_biases,
        }

        # ============================================
        # 1. Mean activation per TARGET CLASS
        # ============================================
        print("\n1. Computing mean activations per target class...")

        mean_per_target = {}
        std_per_target = {}

        for t in unique_targets:
            mask = targets == t
            target_latents = all_latents[mask]
            mean_per_target[t] = target_latents.mean(dim=0).numpy()
            std_per_target[t] = target_latents.std(dim=0).numpy()

        results["mean_per_target"] = {t: m.tolist() for t, m in mean_per_target.items()}

        # Visualize: Heatmap of mean activations per target
        self._plot_mean_activation_heatmap(
            mean_per_target,
            title="Mean Neuron Activation per Target Class",
            xlabel="Neuron Index",
            ylabel="Target Class",
            save_path=os.path.join(
                activation_vis_dir, "mean_activation_per_target.png"
            ),
            labels=[f"Target {t}" for t in unique_targets],
        )

        # Visualize: Top neurons most different between targets
        if num_targets == 2:
            self._plot_target_differential_neurons(
                mean_per_target,
                unique_targets,
                save_path=os.path.join(
                    activation_vis_dir, "target_differential_neurons.png"
                ),
            )

        # ============================================
        # 2. Mean activation per BIAS ATTRIBUTE
        # ============================================
        if bias_values is not None:
            print(
                f"\n2. Computing mean activations per bias attribute ({bias_name})..."
            )

            mean_per_bias = {}
            std_per_bias = {}

            for b in unique_biases:
                mask = bias_values == b
                bias_latents = all_latents[mask]
                mean_per_bias[b] = bias_latents.mean(dim=0).numpy()
                std_per_bias[b] = bias_latents.std(dim=0).numpy()

            results["mean_per_bias"] = {b: m.tolist() for b, m in mean_per_bias.items()}

            # Visualize: Heatmap of mean activations per bias
            self._plot_mean_activation_heatmap(
                mean_per_bias,
                title=f"Mean Neuron Activation per {bias_name}",
                xlabel="Neuron Index",
                ylabel=f"{bias_name} Value",
                save_path=os.path.join(
                    activation_vis_dir, f"mean_activation_per_{bias_name}.png"
                ),
                labels=[f"{bias_name}={b}" for b in unique_biases],
            )

            # Visualize: Top neurons most different between bias groups
            if num_biases == 2:
                self._plot_bias_differential_neurons(
                    mean_per_bias,
                    unique_biases,
                    bias_name,
                    save_path=os.path.join(
                        activation_vis_dir, f"{bias_name}_differential_neurons.png"
                    ),
                )

        # ============================================
        # 3. Mean activation per DATA GROUP (target × bias)
        # ============================================
        if bias_values is not None:
            print("\n3. Computing mean activations per data group (target × bias)...")

            mean_per_group = {}
            group_labels = []

            for t in unique_targets:
                for b in unique_biases:
                    mask = (targets == t) & (bias_values == b)
                    if mask.sum() > 0:
                        group_latents = all_latents[mask]
                        group_key = f"t{t}_b{b}"
                        mean_per_group[group_key] = group_latents.mean(dim=0).numpy()
                        group_labels.append(f"Target={t}, {bias_name}={b}")

            results["mean_per_group"] = {
                k: m.tolist() for k, m in mean_per_group.items()
            }

            # Visualize: Heatmap of mean activations per group
            self._plot_mean_activation_heatmap(
                mean_per_group,
                title=f"Mean Neuron Activation per Data Group (Target × {bias_name})",
                xlabel="Neuron Index",
                ylabel="Data Group",
                save_path=os.path.join(
                    activation_vis_dir, "mean_activation_per_group.png"
                ),
                labels=group_labels,
            )

            # Visualize: Grouped bar chart for selected neurons
            self._plot_group_activation_bars(
                mean_per_group,
                unique_targets,
                unique_biases,
                bias_name,
                save_path=os.path.join(
                    activation_vis_dir, "group_activation_comparison.png"
                ),
            )

        # ============================================
        # 4. Neuron classification: Target vs Bias correlation
        # ============================================
        if bias_values is not None and num_targets == 2 and num_biases == 2:
            print("\n4. Classifying neurons by target vs bias correlation...")

            neuron_classification = self._classify_neurons_by_correlation(
                mean_per_target, mean_per_bias, unique_targets, unique_biases, bias_name
            )
            results["neuron_classification"] = neuron_classification

            # Visualize: Scatter plot of target vs bias effect
            self._plot_neuron_correlation_scatter(
                mean_per_target,
                mean_per_bias,
                unique_targets,
                unique_biases,
                bias_name,
                save_path=os.path.join(
                    activation_vis_dir, "neuron_target_vs_bias_scatter.png"
                ),
            )

        # Save results
        results_path = os.path.join(activation_vis_dir, "mean_activation_analysis.json")

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    str(k): (v if isinstance(v, list) else v) for k, v in value.items()
                }
            else:
                serializable_results[key] = value

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nSaved mean activation analysis to {activation_vis_dir}")

        return results, all_latents

    def _plot_mean_activation_heatmap(
        self, mean_dict, title, xlabel, ylabel, save_path, labels
    ):
        """
        Plot heatmap of mean activations.

        Args:
            mean_dict: Dict mapping group -> mean activation array
            title: Plot title
            xlabel, ylabel: Axis labels
            save_path: Path to save figure
            labels: Y-axis tick labels
        """
        # Stack into matrix
        keys = list(mean_dict.keys())
        matrix = np.stack([mean_dict[k] for k in keys])  # (num_groups, dict_size)

        # For large dict_size, subsample or show top neurons
        if matrix.shape[1] > 200:
            # Show top 200 neurons by variance across groups
            variance = matrix.var(axis=0)
            top_indices = np.argsort(variance)[-200:]
            matrix = matrix[:, top_indices]
            xlabel = f"Top 200 High-Variance Neuron Index"

        fig, ax = plt.subplots(figsize=(14, max(3, len(keys) * 0.5)))

        im = ax.imshow(matrix, aspect="auto", cmap="viridis")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

        plt.colorbar(im, ax=ax, label="Mean Activation")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_target_differential_neurons(
        self, mean_per_target, unique_targets, save_path, top_k=50
    ):
        """
        Plot neurons that differ most between target classes.
        """
        t0, t1 = unique_targets[0], unique_targets[1]
        diff = mean_per_target[t1] - mean_per_target[t0]

        # Get top neurons favoring each class
        sorted_idx = np.argsort(diff)
        top_t0 = sorted_idx[:top_k]  # Most favor target 0
        top_t1 = sorted_idx[-top_k:][::-1]  # Most favor target 1

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Neurons favoring target 0
        ax = axes[0]
        ax.barh(range(top_k), diff[top_t0], color="steelblue")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"N{i}" for i in top_t0])
        ax.set_xlabel("Activation Difference (Target 1 - Target 0)")
        ax.set_title(f"Top {top_k} Neurons Favoring Target {t0}")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5)

        # Neurons favoring target 1
        ax = axes[1]
        ax.barh(range(top_k), diff[top_t1], color="coral")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"N{i}" for i in top_t1])
        ax.set_xlabel("Activation Difference (Target 1 - Target 0)")
        ax.set_title(f"Top {top_k} Neurons Favoring Target {t1}")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_bias_differential_neurons(
        self, mean_per_bias, unique_biases, bias_name, save_path, top_k=50
    ):
        """
        Plot neurons that differ most between bias groups.
        """
        b0, b1 = unique_biases[0], unique_biases[1]
        diff = mean_per_bias[b1] - mean_per_bias[b0]

        sorted_idx = np.argsort(diff)
        top_b0 = sorted_idx[:top_k]
        top_b1 = sorted_idx[-top_k:][::-1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.barh(range(top_k), diff[top_b0], color="forestgreen")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"N{i}" for i in top_b0])
        ax.set_xlabel(f"Activation Difference ({bias_name}={b1} - {bias_name}={b0})")
        ax.set_title(f"Top {top_k} Neurons Favoring {bias_name}={b0}")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5)

        ax = axes[1]
        ax.barh(range(top_k), diff[top_b1], color="purple")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"N{i}" for i in top_b1])
        ax.set_xlabel(f"Activation Difference ({bias_name}={b1} - {bias_name}={b0})")
        ax.set_title(f"Top {top_k} Neurons Favoring {bias_name}={b1}")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_group_activation_bars(
        self,
        mean_per_group,
        unique_targets,
        unique_biases,
        bias_name,
        save_path,
        top_k=20,
    ):
        """
        Plot grouped bar chart showing activation patterns for top differential neurons.
        """
        # Find neurons with highest variance across groups
        group_keys = list(mean_per_group.keys())
        matrix = np.stack([mean_per_group[k] for k in group_keys])
        variance = matrix.var(axis=0)
        top_neurons = np.argsort(variance)[-top_k:][::-1]

        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(top_k)
        width = 0.8 / len(group_keys)
        colors = plt.cm.Set2(np.linspace(0, 1, len(group_keys)))

        for i, (group_key, color) in enumerate(zip(group_keys, colors)):
            offset = (i - len(group_keys) / 2 + 0.5) * width
            values = mean_per_group[group_key][top_neurons]
            ax.bar(x + offset, values, width, label=group_key, color=color)

        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Mean Activation")
        ax.set_title(f"Top {top_k} High-Variance Neurons: Activation by Data Group")
        ax.set_xticks(x)
        ax.set_xticklabels([f"N{n}" for n in top_neurons], rotation=45, ha="right")
        ax.legend(title="Data Group", bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _classify_neurons_by_correlation(
        self, mean_per_target, mean_per_bias, unique_targets, unique_biases, bias_name
    ):
        """
        Classify neurons based on whether they correlate more with target or bias.

        Returns dict with:
        - target_specific: neurons that vary mainly with target
        - bias_specific: neurons that vary mainly with bias
        - mixed: neurons that vary with both
        - neutral: neurons that don't vary much with either
        """
        t0, t1 = unique_targets[0], unique_targets[1]
        b0, b1 = unique_biases[0], unique_biases[1]

        # Compute effect sizes
        target_effect = np.abs(mean_per_target[t1] - mean_per_target[t0])
        bias_effect = np.abs(mean_per_bias[b1] - mean_per_bias[b0])

        # Thresholds (can be made configurable)
        threshold = 0.1  # Minimum effect size to be considered significant

        classification = {
            "target_specific": [],
            "bias_specific": [],
            "mixed": [],
            "neutral": [],
        }

        for i in range(len(target_effect)):
            t_sig = target_effect[i] > threshold
            b_sig = bias_effect[i] > threshold

            if t_sig and b_sig:
                classification["mixed"].append(i)
            elif t_sig:
                classification["target_specific"].append(i)
            elif b_sig:
                classification["bias_specific"].append(i)
            else:
                classification["neutral"].append(i)

        # Print summary
        print(f"\nNeuron Classification Summary:")
        print(f"  Target-specific: {len(classification['target_specific'])} neurons")
        print(
            f"  Bias-specific ({bias_name}): {len(classification['bias_specific'])} neurons"
        )
        print(f"  Mixed (both): {len(classification['mixed'])} neurons")
        print(f"  Neutral: {len(classification['neutral'])} neurons")

        return classification

    def _plot_neuron_correlation_scatter(
        self,
        mean_per_target,
        mean_per_bias,
        unique_targets,
        unique_biases,
        bias_name,
        save_path,
    ):
        """
        Scatter plot showing target effect vs bias effect for each neuron.
        """
        t0, t1 = unique_targets[0], unique_targets[1]
        b0, b1 = unique_biases[0], unique_biases[1]

        target_effect = mean_per_target[t1] - mean_per_target[t0]
        bias_effect = mean_per_bias[b1] - mean_per_bias[b0]

        fig, ax = plt.subplots(figsize=(10, 10))

        # Color by which effect is stronger
        colors = []
        for te, be in zip(target_effect, bias_effect):
            if abs(te) > abs(be) * 1.5:
                colors.append("steelblue")  # Target-dominant
            elif abs(be) > abs(te) * 1.5:
                colors.append("coral")  # Bias-dominant
            else:
                colors.append("gray")  # Mixed/neutral

        ax.scatter(target_effect, bias_effect, c=colors, alpha=0.5, s=10)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5)

        # Add diagonal lines for reference
        max_val = max(np.abs(target_effect).max(), np.abs(bias_effect).max())
        ax.plot(
            [-max_val, max_val],
            [-max_val, max_val],
            "k:",
            alpha=0.3,
            label="Equal effect",
        )
        ax.plot([-max_val, max_val], [max_val, -max_val], "k:", alpha=0.3)

        ax.set_xlabel(f"Target Effect (Target {t1} - Target {t0})")
        ax.set_ylabel(f"{bias_name} Effect ({bias_name}={b1} - {bias_name}={b0})")
        ax.set_title("Neuron Activation: Target Effect vs Bias Effect")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="steelblue", label="Target-dominant"),
            Patch(facecolor="coral", label="Bias-dominant"),
            Patch(facecolor="gray", label="Mixed/Neutral"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _visualize_top_images(self, analysis_results, num_latents=50):
        """
        Generate visualization plots showing top-k images for monosemantic neurons.

        Args:
            analysis_results: Results from _analyze_monosemanticity
            num_latents: Number of latents to visualize
        """
        print(f"\n{'='*60}")
        print("Generating Visualizations")
        print(f"{'='*60}")

        vis_dir = os.path.join(self.log_path, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Get the dataset for loading images
        train_dataset = self.sets["train"]

        alive_neurons = analysis_results.get("alive_neurons", [])

        # Sort by activation magnitude to get most active neurons
        neuron_max_activations = {}
        for latent_idx in alive_neurons:
            top_k_data = analysis_results["top_k_per_latent"].get(latent_idx, {})
            if "activations" in top_k_data and top_k_data["activations"]:
                neuron_max_activations[latent_idx] = max(top_k_data["activations"])

        sorted_neurons = sorted(
            neuron_max_activations.keys(),
            key=lambda x: neuron_max_activations[x],
            reverse=True,
        )[:num_latents]

        k = self.sae_cfg.TOP_K_IMAGES

        for rank, latent_idx in enumerate(
            tqdm(sorted_neurons, desc="Creating visualizations")
        ):
            top_k_data = analysis_results["top_k_per_latent"][latent_idx]
            sample_indices = top_k_data["indices"]
            activations = top_k_data["activations"]
            targets = top_k_data["targets"]

            # Create figure
            cols = min(k, 8)
            rows = (k + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
            if rows == 1:
                axes = [axes] if cols == 1 else axes
            axes = np.array(axes).flatten()

            # Plot each top image
            for i, (idx, act, target) in enumerate(
                zip(sample_indices, activations, targets)
            ):
                if i >= len(axes):
                    break

                ax = axes[i]

                try:
                    # Load image from dataset
                    sample = train_dataset[idx]
                    # print(sample.keys())
                    if isinstance(sample, dict):
                        img = sample["inputs"]
                    elif isinstance(sample, tuple):
                        img = sample[0]
                    else:
                        img = sample

                    # Convert tensor to numpy for display
                    if isinstance(img, torch.Tensor):
                        img = img.permute(1, 2, 0).numpy()
                        # Denormalize if needed
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                    ax.imshow(img)
                    ax.set_title(f"Act: {act:.2f}\nClass: {target}", fontsize=8)
                except Exception as e:
                    print(e)
                    ax.text(
                        0.5, 0.5, f"Error:\n{str(e)[:20]}", ha="center", va="center"
                    )

                ax.axis("off")

            # Hide empty subplots
            for i in range(len(sample_indices), len(axes)):
                axes[i].axis("off")

            # Title and save
            fig.suptitle(
                f"Latent {latent_idx} (Rank {rank+1})\nMax activation: {activations[0]:.3f}",
                fontsize=12,
                fontweight="bold",
            )
            plt.tight_layout()

            save_path = os.path.join(vis_dir, f"rank_{rank+1:03d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"Saved {len(sorted_neurons)} visualizations to {vis_dir}")

        # Create summary plot
        self._create_summary_plot(analysis_results, vis_dir)

    def _perform_clustering_analysis(
        self,
        features,
        targets,
        biases,
        indices,
        feature_type="sae",
        description="SAE Sparse Features",
    ):
        """
        Perform clustering analysis on features.

        Clusters samples based on their representations and visualizes
        the clusters with data groups (target × bias combinations) as legends.

        Args:
            features: Original features (N, feature_dim)
            targets: Target labels (N,)
            biases: Dict of bias attributes
            indices: Sample indices
            feature_type: "original" for vanilla model features, "sae" for SAE sparse features
            description: Human-readable description for plots

        Returns:
            clustering_results: Dict containing cluster assignments and metrics
        """
        print(f"\n{'='*60}")
        print(f"Performing Clustering Analysis on {description}")
        print(f"{'='*60}")

        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        # Get the features to cluster
        if feature_type == "sae":
            # Encode all features to get sparse representations
            self.sae.eval()
            with torch.no_grad():
                all_latents = []
                batch_size = self.sae_cfg.BATCH_SIZE

                for i in range(0, len(features), batch_size):
                    batch = features[i : i + batch_size].to(self.device)
                    latents = self.sae.encode(batch)
                    all_latents.append(latents.cpu())

                clustering_features = torch.cat(all_latents, dim=0)  # (N, dict_size)
        else:
            # Use original features directly
            clustering_features = features

        # Convert to numpy for sklearn
        features_np = clustering_features.numpy()
        targets_np = targets.numpy()

        # Create data groups: combinations of target and bias attributes
        data_groups = self._create_data_groups(targets, biases)
        group_labels = data_groups["labels"]  # (N,) array of group indices
        group_names = data_groups["names"]  # List of group names
        num_groups = len(group_names)

        print(f"Number of samples: {len(features_np)}")
        print(f"Feature dimension: {features_np.shape[1]}")
        print(f"Number of data groups: {num_groups}")
        print(f"Data groups: {group_names}")

        # Compute statistics per group
        stats_per_group = {}
        for g_idx, g_name in enumerate(group_names):
            mask = group_labels == g_idx
            group_features = features_np[mask]

            if feature_type == "sae":
                l0 = (group_features > 0).sum(axis=1).mean()
                stats_per_group[g_name] = {
                    "count": int(mask.sum()),
                    "mean_l0": float(l0),
                    "mean_activation": float(group_features.mean()),
                }
            else:
                stats_per_group[g_name] = {
                    "count": int(mask.sum()),
                    "mean_norm": float(np.linalg.norm(group_features, axis=1).mean()),
                    "mean_activation": float(group_features.mean()),
                }

        print(f"\nStatistics per group:")
        for g_name, stats in stats_per_group.items():
            if feature_type == "sae":
                print(f"  {g_name}: n={stats['count']}, L0={stats['mean_l0']:.2f}")
            else:
                print(f"  {g_name}: n={stats['count']}, norm={stats['mean_norm']:.2f}")

        # Dimensionality reduction for visualization
        print("\nPerforming dimensionality reduction...")

        # Use PCA first to reduce dimensionality for faster t-SNE/UMAP
        n_components_pca = min(50, features_np.shape[1], len(features_np) - 1)
        pca = PCA(n_components=n_components_pca)
        features_pca = pca.fit_transform(features_np)
        print(
            f"PCA: {features_np.shape[1]} -> {n_components_pca} (explained variance: {pca.explained_variance_ratio_.sum():.2%})"
        )

        # t-SNE
        print("Running t-SNE...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(features_np) // 4),
            random_state=self.cfg.EXPERIMENT.SEED,
            # n_iter=1000,
        )
        features_tsne = tsne.fit_transform(features_pca)

        # UMAP (if available)
        features_umap = None
        try:
            import umap

            print("Running UMAP...")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                random_state=self.cfg.EXPERIMENT.SEED,
            )
            features_umap = reducer.fit_transform(features_pca)
        except Exception as e:
            print(f"UMAP failed (may not be installed): {e}")

        # Perform clustering
        print("\nPerforming clustering...")

        # K-Means with number of clusters = number of data groups
        kmeans = KMeans(
            n_clusters=num_groups, random_state=self.cfg.EXPERIMENT.SEED, n_init=10
        )
        kmeans_labels = kmeans.fit_predict(features_pca)

        # Build results dict
        clustering_results = {
            "feature_type": feature_type,
            "description": description,
            "data_groups": {
                "names": group_names,
                "labels": (
                    group_labels.tolist()
                    if isinstance(group_labels, torch.Tensor)
                    else group_labels
                ),
                "stats": stats_per_group,
            },
            "kmeans": {
                "labels": kmeans_labels.tolist(),
                "n_clusters": num_groups,
            },
            "embeddings": {
                "tsne": features_tsne,
                "umap": features_umap,
                "pca": features_pca[:, :2],
            },
            "raw_features": clustering_features,
        }

        # Compute cluster-group alignment metrics
        alignment_metrics = self._compute_cluster_alignment(
            kmeans_labels, group_labels, group_names
        )
        clustering_results["alignment_metrics"] = alignment_metrics

        # Generate visualizations
        vis_dir = os.path.join(self.log_path, f"clustering_{feature_type}")
        os.makedirs(vis_dir, exist_ok=True)

        self._plot_clustering_results(
            features_tsne,
            features_umap,
            features_pca[:, :2],
            group_labels,
            group_names,
            kmeans_labels,
            vis_dir,
            description,
        )

        # Save clustering results
        results_path = os.path.join(vis_dir, "clustering_results.json")
        # Make JSON serializable
        serializable_results = {
            "feature_type": feature_type,
            "description": description,
            "data_groups": {
                "names": group_names,
                "labels": (
                    group_labels.tolist()
                    if isinstance(group_labels, torch.Tensor)
                    else list(group_labels)
                ),
                "stats": stats_per_group,
            },
            "kmeans": clustering_results["kmeans"],
            "alignment_metrics": alignment_metrics,
        }
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        # Save embeddings separately (they're large)
        embeddings_path = os.path.join(vis_dir, "embeddings.pt")
        torch.save(
            {
                "tsne": torch.tensor(features_tsne),
                "umap": (
                    torch.tensor(features_umap) if features_umap is not None else None
                ),
                "pca": torch.tensor(features_pca),
                "features": clustering_features,
                "group_labels": (
                    torch.tensor(group_labels)
                    if not isinstance(group_labels, torch.Tensor)
                    else group_labels
                ),
                "targets": targets,
            },
            embeddings_path,
        )

        print(f"\nClustering results saved to {vis_dir}")

        return clustering_results

    def _create_data_groups(self, targets, biases):
        """
        Create data groups from combinations of target and bias attributes.

        Args:
            targets: Target labels (N,)
            biases: Dict of bias attributes {bias_name: values}

        Returns:
            dict with:
                - labels: (N,) array of group indices
                - names: List of group names
                - mapping: Dict mapping (target, bias_tuple) -> group_idx
        """
        n_samples = len(targets)

        # Get unique target values
        unique_targets = sorted(targets.unique().tolist())

        # Get the primary bias attribute (first one)
        bias_name = self.biases[0] if self.biases else None
        bias_values = biases.get(bias_name) if bias_name and biases else None

        if bias_values is None:
            # No bias attribute, just use targets
            group_names = [f"Class_{t}" for t in unique_targets]
            group_labels = targets.clone()
            mapping = {(t,): i for i, t in enumerate(unique_targets)}
        else:
            unique_biases = sorted(bias_values.unique().tolist())

            # Create all combinations
            group_names = []
            mapping = {}
            group_idx = 0

            for t in unique_targets:
                target_name = (
                    self.target2name.get(t, f"Class_{t}")
                    if hasattr(self, "target2name")
                    else f"Class_{t}"
                )
                for b in unique_biases:
                    # Create descriptive group name
                    group_name = f"{target_name}_{bias_name}={b}"
                    group_names.append(group_name)
                    mapping[(t, b)] = group_idx
                    group_idx += 1

            # Assign each sample to its group
            group_labels = torch.zeros(n_samples, dtype=torch.long)
            for i in range(n_samples):
                t = targets[i].item()
                b = bias_values[i].item()
                group_labels[i] = mapping[(t, b)]

        return {
            "labels": group_labels,
            "names": group_names,
            "mapping": mapping,
        }

    def _compute_cluster_alignment(self, cluster_labels, group_labels, group_names):
        """
        Compute metrics measuring how well clusters align with data groups.

        Args:
            cluster_labels: Cluster assignments from K-Means
            group_labels: True data group labels
            group_names: Names of data groups

        Returns:
            Dict of alignment metrics
        """
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
            homogeneity_score,
            completeness_score,
            v_measure_score,
        )

        # Convert to numpy if needed
        if isinstance(group_labels, torch.Tensor):
            group_labels = group_labels.numpy()

        metrics = {
            "adjusted_rand_index": float(
                adjusted_rand_score(group_labels, cluster_labels)
            ),
            "normalized_mutual_info": float(
                normalized_mutual_info_score(group_labels, cluster_labels)
            ),
            "homogeneity": float(homogeneity_score(group_labels, cluster_labels)),
            "completeness": float(completeness_score(group_labels, cluster_labels)),
            "v_measure": float(v_measure_score(group_labels, cluster_labels)),
        }

        # Compute confusion matrix between clusters and groups
        num_clusters = len(np.unique(cluster_labels))
        num_groups = len(group_names)
        confusion = np.zeros((num_clusters, num_groups), dtype=int)

        for c, g in zip(cluster_labels, group_labels):
            confusion[c, g] += 1

        # Find best cluster-group mapping (majority vote)
        cluster_to_group = {}
        for c in range(num_clusters):
            best_group = np.argmax(confusion[c])
            cluster_to_group[c] = {
                "best_group": group_names[best_group],
                "purity": (
                    float(confusion[c, best_group] / confusion[c].sum())
                    if confusion[c].sum() > 0
                    else 0
                ),
                "distribution": {
                    group_names[g]: int(confusion[c, g]) for g in range(num_groups)
                },
            }

        metrics["cluster_to_group_mapping"] = cluster_to_group
        metrics["confusion_matrix"] = confusion.tolist()

        # Overall clustering purity
        purity = sum(
            confusion[c, np.argmax(confusion[c])] for c in range(num_clusters)
        ) / len(cluster_labels)
        metrics["overall_purity"] = float(purity)

        print(f"\nClustering Alignment Metrics:")
        print(f"  Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
        print(f"  Normalized Mutual Info: {metrics['normalized_mutual_info']:.4f}")
        print(f"  V-Measure: {metrics['v_measure']:.4f}")
        print(f"  Overall Purity: {metrics['overall_purity']:.4f}")

        return metrics

    def _plot_clustering_results(
        self,
        tsne_emb,
        umap_emb,
        pca_emb,
        group_labels,
        group_names,
        cluster_labels,
        vis_dir,
        description="Features",
    ):
        """
        Generate clustering visualization plots.

        Creates scatter plots colored by data groups with clear legends.
        """
        # Color palette for groups
        num_groups = len(group_names)
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_groups, 20)))[:num_groups]

        # Create a mapping for consistent colors
        group_colors = {i: colors[i] for i in range(num_groups)}

        # Convert labels to numpy
        if isinstance(group_labels, torch.Tensor):
            group_labels = group_labels.numpy()

        # Figure 1: t-SNE colored by data groups
        fig, ax = plt.subplots(figsize=(12, 10))

        for g_idx, g_name in enumerate(group_names):
            mask = group_labels == g_idx
            ax.scatter(
                tsne_emb[mask, 0],
                tsne_emb[mask, 1],
                c=[group_colors[g_idx]],
                label=g_name,
                alpha=0.6,
                s=20,
                edgecolors="none",
            )

        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title(
            f"{description} - t-SNE\n(Colored by Data Groups: Target × Bias)",
            fontsize=14,
        )
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()
        plt.savefig(
            os.path.join(vis_dir, "tsne_by_data_groups.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Figure 2: UMAP colored by data groups (if available)
        if umap_emb is not None:
            fig, ax = plt.subplots(figsize=(12, 10))

            for g_idx, g_name in enumerate(group_names):
                mask = group_labels == g_idx
                ax.scatter(
                    umap_emb[mask, 0],
                    umap_emb[mask, 1],
                    c=[group_colors[g_idx]],
                    label=g_name,
                    alpha=0.6,
                    s=20,
                    edgecolors="none",
                )

            ax.set_xlabel("UMAP 1", fontsize=12)
            ax.set_ylabel("UMAP 2", fontsize=12)
            ax.set_title(
                f"{description} - UMAP\n(Colored by Data Groups: Target × Bias)",
                fontsize=14,
            )
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
            plt.tight_layout()
            plt.savefig(
                os.path.join(vis_dir, "umap_by_data_groups.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        # Figure 3: PCA colored by data groups
        fig, ax = plt.subplots(figsize=(12, 10))

        for g_idx, g_name in enumerate(group_names):
            mask = group_labels == g_idx
            ax.scatter(
                pca_emb[mask, 0],
                pca_emb[mask, 1],
                c=[group_colors[g_idx]],
                label=g_name,
                alpha=0.6,
                s=20,
                edgecolors="none",
            )

        ax.set_xlabel("PC 1", fontsize=12)
        ax.set_ylabel("PC 2", fontsize=12)
        ax.set_title(
            f"{description} - PCA\n(Colored by Data Groups: Target × Bias)", fontsize=14
        )
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()
        plt.savefig(
            os.path.join(vis_dir, "pca_by_data_groups.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Figure 4: t-SNE colored by K-Means clusters
        fig, ax = plt.subplots(figsize=(12, 10))

        num_clusters = len(np.unique(cluster_labels))
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, num_clusters))

        for c_idx in range(num_clusters):
            mask = cluster_labels == c_idx
            ax.scatter(
                tsne_emb[mask, 0],
                tsne_emb[mask, 1],
                c=[cluster_colors[c_idx]],
                label=f"Cluster {c_idx}",
                alpha=0.6,
                s=20,
                edgecolors="none",
            )

        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title(
            f"{description} - t-SNE\n(Colored by K-Means Clusters)", fontsize=14
        )
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()
        plt.savefig(
            os.path.join(vis_dir, "tsne_by_kmeans.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Figure 5: Side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Left: Data groups
        ax = axes[0]
        for g_idx, g_name in enumerate(group_names):
            mask = group_labels == g_idx
            ax.scatter(
                tsne_emb[mask, 0],
                tsne_emb[mask, 1],
                c=[group_colors[g_idx]],
                label=g_name,
                alpha=0.6,
                s=15,
                edgecolors="none",
            )
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title("Data Groups (Target × Bias)", fontsize=14)
        ax.legend(loc="upper right", fontsize=8)

        # Right: K-Means clusters
        ax = axes[1]
        for c_idx in range(num_clusters):
            mask = cluster_labels == c_idx
            ax.scatter(
                tsne_emb[mask, 0],
                tsne_emb[mask, 1],
                c=[cluster_colors[c_idx]],
                label=f"Cluster {c_idx}",
                alpha=0.6,
                s=15,
                edgecolors="none",
            )
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title("K-Means Clusters", fontsize=14)
        ax.legend(loc="upper right", fontsize=8)

        plt.suptitle(
            f"Comparison: Data Groups vs K-Means Clustering\n{description}",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(vis_dir, "comparison_groups_vs_clusters.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Figure 6: Confusion matrix heatmap
        fig, ax = plt.subplots(
            figsize=(max(10, num_groups * 0.8), max(8, num_clusters * 0.6))
        )

        confusion = np.zeros((num_clusters, num_groups), dtype=int)
        for c, g in zip(cluster_labels, group_labels):
            confusion[c, g] += 1

        # Normalize by row (cluster)
        confusion_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-8)

        im = ax.imshow(confusion_norm, cmap="Blues", aspect="auto")

        ax.set_xticks(range(num_groups))
        ax.set_xticklabels(group_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(num_clusters))
        ax.set_yticklabels([f"Cluster {i}" for i in range(num_clusters)], fontsize=10)

        ax.set_xlabel("Data Group", fontsize=12)
        ax.set_ylabel("K-Means Cluster", fontsize=12)
        ax.set_title("Cluster-Group Confusion Matrix (Row-Normalized)", fontsize=14)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Proportion", fontsize=10)

        # Add text annotations
        for i in range(num_clusters):
            for j in range(num_groups):
                text = f"{confusion[i, j]}\n({confusion_norm[i, j]:.1%})"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if confusion_norm[i, j] > 0.5 else "black",
                )

        plt.tight_layout()
        plt.savefig(
            os.path.join(vis_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

        print(f"Saved clustering visualizations to {vis_dir}")

    def _create_clustering_comparison(
        self, original_results, sae_results, targets, biases
    ):
        """
        Create side-by-side comparison plots between original and SAE features.

        Args:
            original_results: Clustering results for original features
            sae_results: Clustering results for SAE sparse features
            targets: Target labels
            biases: Bias attributes
        """
        print(f"\n{'='*60}")
        print("Creating Original vs SAE Feature Comparison")
        print(f"{'='*60}")

        vis_dir = os.path.join(self.log_path, "clustering_comparison")
        os.makedirs(vis_dir, exist_ok=True)

        # Get data
        group_names = original_results["data_groups"]["names"]
        group_labels = original_results["data_groups"]["labels"]
        if isinstance(group_labels, list):
            group_labels = np.array(group_labels)
        elif isinstance(group_labels, torch.Tensor):
            group_labels = group_labels.numpy()

        num_groups = len(group_names)

        # Color palette
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_groups, 20)))[:num_groups]
        group_colors = {i: colors[i] for i in range(num_groups)}

        # Get embeddings
        orig_tsne = original_results["embeddings"]["tsne"]
        sae_tsne = sae_results["embeddings"]["tsne"]

        orig_umap = original_results["embeddings"]["umap"]
        sae_umap = sae_results["embeddings"]["umap"]

        orig_pca = original_results["embeddings"]["pca"]
        sae_pca = sae_results["embeddings"]["pca"]

        # ============================================================
        # Figure 1: t-SNE comparison (Original vs SAE) by Data Groups
        # ============================================================
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Left: Original features
        ax = axes[0]
        for g_idx, g_name in enumerate(group_names):
            mask = group_labels == g_idx
            ax.scatter(
                orig_tsne[mask, 0],
                orig_tsne[mask, 1],
                c=[group_colors[g_idx]],
                label=g_name,
                alpha=0.6,
                s=15,
                edgecolors="none",
            )
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title("Original Vanilla Model Features", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)

        # Right: SAE features
        ax = axes[1]
        for g_idx, g_name in enumerate(group_names):
            mask = group_labels == g_idx
            ax.scatter(
                sae_tsne[mask, 0],
                sae_tsne[mask, 1],
                c=[group_colors[g_idx]],
                label=g_name,
                alpha=0.6,
                s=15,
                edgecolors="none",
            )
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title("SAE Sparse Features", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)

        plt.suptitle(
            "t-SNE Comparison: Original vs SAE Features\n(Colored by Data Groups: Target × Bias)",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(vis_dir, "tsne_comparison_by_groups.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # ============================================================
        # Figure 2: UMAP comparison (if available)
        # ============================================================
        if orig_umap is not None and sae_umap is not None:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))

            ax = axes[0]
            for g_idx, g_name in enumerate(group_names):
                mask = group_labels == g_idx
                ax.scatter(
                    orig_umap[mask, 0],
                    orig_umap[mask, 1],
                    c=[group_colors[g_idx]],
                    label=g_name,
                    alpha=0.6,
                    s=15,
                    edgecolors="none",
                )
            ax.set_xlabel("UMAP 1", fontsize=12)
            ax.set_ylabel("UMAP 2", fontsize=12)
            ax.set_title(
                "Original Vanilla Model Features", fontsize=14, fontweight="bold"
            )
            ax.legend(loc="upper right", fontsize=8)

            ax = axes[1]
            for g_idx, g_name in enumerate(group_names):
                mask = group_labels == g_idx
                ax.scatter(
                    sae_umap[mask, 0],
                    sae_umap[mask, 1],
                    c=[group_colors[g_idx]],
                    label=g_name,
                    alpha=0.6,
                    s=15,
                    edgecolors="none",
                )
            ax.set_xlabel("UMAP 1", fontsize=12)
            ax.set_ylabel("UMAP 2", fontsize=12)
            ax.set_title("SAE Sparse Features", fontsize=14, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8)

            plt.suptitle(
                "UMAP Comparison: Original vs SAE Features\n(Colored by Data Groups: Target × Bias)",
                fontsize=16,
                y=1.02,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(vis_dir, "umap_comparison_by_groups.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        # ============================================================
        # Figure 3: PCA comparison
        # ============================================================
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        ax = axes[0]
        for g_idx, g_name in enumerate(group_names):
            mask = group_labels == g_idx
            ax.scatter(
                orig_pca[mask, 0],
                orig_pca[mask, 1],
                c=[group_colors[g_idx]],
                label=g_name,
                alpha=0.6,
                s=15,
                edgecolors="none",
            )
        ax.set_xlabel("PC 1", fontsize=12)
        ax.set_ylabel("PC 2", fontsize=12)
        ax.set_title("Original Vanilla Model Features", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)

        ax = axes[1]
        for g_idx, g_name in enumerate(group_names):
            mask = group_labels == g_idx
            ax.scatter(
                sae_pca[mask, 0],
                sae_pca[mask, 1],
                c=[group_colors[g_idx]],
                label=g_name,
                alpha=0.6,
                s=15,
                edgecolors="none",
            )
        ax.set_xlabel("PC 1", fontsize=12)
        ax.set_ylabel("PC 2", fontsize=12)
        ax.set_title("SAE Sparse Features", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)

        plt.suptitle(
            "PCA Comparison: Original vs SAE Features\n(Colored by Data Groups: Target × Bias)",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(vis_dir, "pca_comparison_by_groups.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # ============================================================
        # Figure 4: K-Means clustering comparison
        # ============================================================
        orig_kmeans = np.array(original_results["kmeans"]["labels"])
        sae_kmeans = np.array(sae_results["kmeans"]["labels"])

        num_clusters = original_results["kmeans"]["n_clusters"]
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, num_clusters))

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        ax = axes[0]
        for c_idx in range(num_clusters):
            mask = orig_kmeans == c_idx
            ax.scatter(
                orig_tsne[mask, 0],
                orig_tsne[mask, 1],
                c=[cluster_colors[c_idx]],
                label=f"Cluster {c_idx}",
                alpha=0.6,
                s=15,
                edgecolors="none",
            )
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title(
            "Original Features - K-Means Clusters", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="upper right", fontsize=8)

        ax = axes[1]
        for c_idx in range(num_clusters):
            mask = sae_kmeans == c_idx
            ax.scatter(
                sae_tsne[mask, 0],
                sae_tsne[mask, 1],
                c=[cluster_colors[c_idx]],
                label=f"Cluster {c_idx}",
                alpha=0.6,
                s=15,
                edgecolors="none",
            )
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title("SAE Features - K-Means Clusters", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)

        plt.suptitle(
            "K-Means Clustering Comparison: Original vs SAE Features",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(vis_dir, "kmeans_comparison.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

        # ============================================================
        # Figure 5: Metrics comparison bar chart
        # ============================================================
        orig_metrics = original_results["alignment_metrics"]
        sae_metrics = sae_results["alignment_metrics"]

        metric_names = [
            "adjusted_rand_index",
            "normalized_mutual_info",
            "v_measure",
            "overall_purity",
        ]
        metric_labels = [
            "Adjusted Rand\nIndex",
            "Normalized\nMutual Info",
            "V-Measure",
            "Overall\nPurity",
        ]

        orig_values = [orig_metrics[m] for m in metric_names]
        sae_values = [sae_metrics[m] for m in metric_names]

        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(
            x - width / 2,
            orig_values,
            width,
            label="Original Features",
            color="steelblue",
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            sae_values,
            width,
            label="SAE Sparse Features",
            color="coral",
            alpha=0.8,
        )

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(
            "Clustering Alignment Metrics: Original vs SAE Features\n(Higher is Better)",
            fontsize=14,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.1)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(vis_dir, "metrics_comparison.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # ============================================================
        # Figure 6: Confusion matrices side by side
        # ============================================================
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        for idx, (results, title) in enumerate(
            [
                (original_results, "Original Features"),
                (sae_results, "SAE Sparse Features"),
            ]
        ):
            ax = axes[idx]

            kmeans_labels = np.array(results["kmeans"]["labels"])
            confusion = np.zeros((num_clusters, num_groups), dtype=int)
            for c, g in zip(kmeans_labels, group_labels):
                confusion[c, g] += 1

            confusion_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-8)

            im = ax.imshow(confusion_norm, cmap="Blues", aspect="auto")

            ax.set_xticks(range(num_groups))
            ax.set_xticklabels(group_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(num_clusters))
            ax.set_yticklabels(
                [f"Cluster {i}" for i in range(num_clusters)], fontsize=9
            )

            ax.set_xlabel("Data Group", fontsize=11)
            ax.set_ylabel("K-Means Cluster", fontsize=11)
            ax.set_title(
                f"{title}\nCluster-Group Confusion Matrix",
                fontsize=12,
                fontweight="bold",
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Proportion", fontsize=9)

        plt.suptitle(
            "Confusion Matrices: Original vs SAE Features", fontsize=14, y=1.02
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(vis_dir, "confusion_matrices_comparison.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # ============================================================
        # Save comparison summary
        # ============================================================
        comparison_summary = {
            "original_features": {
                "dimension": int(original_results["raw_features"].shape[1]),
                "metrics": {
                    k: float(v)
                    for k, v in orig_metrics.items()
                    if k not in ["cluster_to_group_mapping", "confusion_matrix"]
                },
            },
            "sae_features": {
                "dimension": int(sae_results["raw_features"].shape[1]),
                "metrics": {
                    k: float(v)
                    for k, v in sae_metrics.items()
                    if k not in ["cluster_to_group_mapping", "confusion_matrix"]
                },
            },
            "improvement": {
                m: float(sae_metrics[m] - orig_metrics[m]) for m in metric_names
            },
            "data_groups": group_names,
        }

        with open(os.path.join(vis_dir, "comparison_summary.json"), "w") as f:
            json.dump(comparison_summary, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("CLUSTERING COMPARISON SUMMARY")
        print("=" * 60)
        print(
            f"\nOriginal Features Dimension: {comparison_summary['original_features']['dimension']}"
        )
        print(
            f"SAE Sparse Features Dimension: {comparison_summary['sae_features']['dimension']}"
        )
        print("\nAlignment Metrics (Original → SAE):")
        for m, label in zip(metric_names, ["ARI", "NMI", "V-Measure", "Purity"]):
            orig_val = orig_metrics[m]
            sae_val = sae_metrics[m]
            diff = sae_val - orig_val
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(
                f"  {label}: {orig_val:.4f} → {sae_val:.4f} ({arrow} {abs(diff):.4f})"
            )

        print(f"\nComparison plots saved to {vis_dir}")

        return comparison_summary

    def _create_summary_plot(self, analysis_results, vis_dir):
        """Create summary statistics plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Histogram of activation magnitudes
        ax = axes[0, 0]
        all_max_activations = []
        for latent_idx in analysis_results.get("alive_neurons", []):
            top_k_data = analysis_results["top_k_per_latent"].get(latent_idx, {})
            if "activations" in top_k_data and top_k_data["activations"]:
                all_max_activations.append(max(top_k_data["activations"]))

        if all_max_activations:
            ax.hist(all_max_activations, bins=50, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Max Activation")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Max Activations per Neuron")

        # 2. Class purity for each neuron
        ax = axes[0, 1]
        purities = []
        for latent_idx in analysis_results.get("alive_neurons", []):
            class_dist = analysis_results["class_distribution_per_latent"].get(
                latent_idx, []
            )
            if class_dist and sum(class_dist) > 0:
                max_class_count = max(class_dist)
                purity = max_class_count / sum(class_dist)
                purities.append(purity)

        if purities:
            ax.hist(purities, bins=20, edgecolor="black", alpha=0.7, range=(0, 1))
            ax.set_xlabel("Class Purity")
            ax.set_ylabel("Count")
            ax.set_title(f"Class Purity Distribution\nMean: {np.mean(purities):.3f}")

        # 3. Number of alive neurons per class preference
        ax = axes[1, 0]
        class_preferences = defaultdict(int)
        for latent_idx in analysis_results.get("alive_neurons", []):
            class_dist = analysis_results["class_distribution_per_latent"].get(
                latent_idx, []
            )
            if class_dist:
                preferred_class = np.argmax(class_dist)
                class_preferences[preferred_class] += 1

        if class_preferences:
            classes = sorted(class_preferences.keys())
            counts = [class_preferences[c] for c in classes]
            ax.bar(classes, counts, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Preferred Class")
            ax.set_ylabel("Number of Neurons")
            ax.set_title("Neurons per Preferred Class")

        # 4. Summary statistics text
        ax = axes[1, 1]
        ax.axis("off")
        summary_text = f"""
SAE Analysis Summary
====================
Feature Dimension: {self.feature_dim}
Dictionary Size: {self.dict_size}
Expansion Factor: {self.sae_cfg.EXPANSION_FACTOR}

Alive Neurons: {analysis_results['num_alive_neurons']} ({analysis_results['percent_alive']:.1f}%)
Mean L0: {analysis_results['mean_l0']:.2f}

Mean Class Purity: {np.mean(purities) if purities else 'N/A':.3f}
Max Class Purity: {np.max(purities) if purities else 'N/A':.3f}
"""
        ax.text(
            0.1,
            0.5,
            summary_text,
            fontsize=12,
            family="monospace",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        plt.tight_layout()
        save_path = os.path.join(vis_dir, "summary_statistics.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved summary plot to {save_path}")

    def train(self):
        """
        Main training/analysis pipeline.

        1. Extract features from pretrained model
        2. Train SAE on features
        3. Analyze monosemanticity
        4. Generate visualizations
        """
        print(f"\n{'='*60}")
        print("SAE Analysis Pipeline")
        print(f"{'='*60}")

        # # Step 1: Extract features from training data
        # print("\nStep 1: Extracting features from training data...")
        # features, targets, biases, indices = self._extract_features(
        #     self.dataloaders["train"], desc="Extracting training features"
        # )
        # print(
        #     f"Extracted {len(features)} feature vectors of dimension {features.shape[1]}"
        # )

        # # Save features for potential reuse
        # features_dir = os.path.join(self.log_path, "features")
        # os.makedirs(features_dir, exist_ok=True)
        # torch.save(
        #     {
        #         "features": features,
        #         "targets": targets,
        #         "biases": biases,
        #         "indices": indices,
        #     },
        #     os.path.join(features_dir, "train_features.pt"),
        # )
        # print(f"Saved features to {features_dir}")



        # Step 1: Get features (extract or load precomputed)
        precomputed_path = self.sae_cfg.get("PRECOMPUTED_FEATURES_PATH", "")
        
        if precomputed_path and os.path.exists(precomputed_path):
            # Load precomputed features
            print(f"\nStep 1: Loading precomputed features from {precomputed_path}")
            saved_data = torch.load(precomputed_path, map_location='cpu')
            
            features = saved_data["features"]
            targets = saved_data["targets"]
            biases = saved_data.get("biases", {})
            indices = saved_data.get("indices", list(range(len(features))))
            
            # Handle different save formats
            if isinstance(biases, dict) and not biases:
                biases = {b: None for b in self.biases}
            
            print(f"Loaded {len(features)} feature vectors of dimension {features.shape[1]}")
            
            # Update feature_dim if needed
            if features.shape[1] != self.feature_dim:
                print(f"Updating feature_dim from {self.feature_dim} to {features.shape[1]}")
                self.feature_dim = features.shape[1]
                self.dict_size = self.feature_dim * self.sae_cfg.EXPANSION_FACTOR
                print(f"Updated dict_size to {self.dict_size}")
        else:
            # Extract features from training data
            print("\nStep 1: Extracting features from training data...")
            features, targets, biases, indices = self._extract_features(
                self.dataloaders["train"], 
                desc="Extracting training features"
            )
            print(f"Extracted {len(features)} feature vectors of dimension {features.shape[1]}")
            
            # Save features for potential reuse
            features_dir = os.path.join(self.log_path, "features")
            os.makedirs(features_dir, exist_ok=True)
            torch.save({
                "features": features,
                "targets": targets,
                "biases": biases,
                "indices": indices,
            }, os.path.join(features_dir, "train_features.pt"))
            print(f"Saved features to {features_dir}")
        

        # Step 2: Train SAE
        print("\nStep 2: Training Sparse Autoencoder...")
        sae_path = self.sae_cfg.get("PRETRAINED_SAE_PATH", "")
        
        if os.path.exists(sae_path):
            self.sae = AutoEncoder(self.feature_dim, self.dict_size)
            self.sae.load_state_dict(torch.load(sae_path))
            self.sae.to(self.device)
            print(f"Loaded SAE from {sae_path}")
        else:
            self._train_sae(features)

        # Step 3: Analyze monosemanticity
        print("\nStep 3: Analyzing monosemantic neurons...")
        analysis_results = self._analyze_monosemanticity(
            features, targets, biases, indices
        )

        # Save analysis results
        results_path = os.path.join(self.log_path, "analysis_results.json")
        # Convert non-serializable items
        serializable_results = {
            k: v for k, v in analysis_results.items() if k not in ["latent_activations"]
        }
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"Saved analysis results to {results_path}")

        # Step 3b: Analyze mean activations per target, bias, and groups
        print("\nStep 3b: Analyzing mean activations by groups...")
        mean_activation_results, all_latents = self._analyze_mean_activations(
            features, targets, biases
        )

        # Step 4: Generate visualizations
        print("\nStep 4: Generating visualizations...")
        self._visualize_top_images(
            analysis_results, num_latents=self.sae_cfg.NUM_VISUALIZE
        )

        # Step 5: Clustering analysis on ORIGINAL features (vanilla model)
        print("\nStep 5a: Performing clustering analysis on ORIGINAL features...")
        original_clustering_results = self._perform_clustering_analysis(
            features,
            targets,
            biases,
            indices,
            feature_type="original",
            description="Original Vanilla Model Features",
        )

        # Step 6: Clustering analysis on SAE sparse features
        print("\nStep 5b: Performing clustering analysis on SAE SPARSE features...")
        sae_clustering_results = self._perform_clustering_analysis(
            features,
            targets,
            biases,
            indices,
            feature_type="sae",
            description="SAE Sparse Features",
        )

        # Step 7: Create comparison plots between original and SAE features
        print("\nStep 6: Creating comparison visualizations...")
        self._create_clustering_comparison(
            original_clustering_results, sae_clustering_results, targets, biases
        )

        print(f"\n{'='*60}")
        print("SAE Analysis Complete!")
        print(f"Results saved to: {self.log_path}")
        print(f"{'='*60}")

    def _train_iter(self, batch):
        """Not used in SAE analysis - override to prevent errors."""
        return {}

    def _train_epoch(self):
        """Not used in SAE analysis - override to prevent errors."""
        return {}
    
    # def _setup_dataset(self):
    #     self.num_class = 2
    #     self.biases = [""]
    #     self.dataloaders = None
    #     self.sets = None
    #     self.data_root = None
    #     self.target2name = None
    #     self.ba_groups = None

    def eval(self):
        """
        Evaluation mode - load existing SAE and analyze on test data.
        """
        # Load SAE if exists
        sae_path = os.path.join(self.log_path, "sae_checkpoints", "ae.pt")
        if os.path.exists(sae_path):
            self.sae = AutoEncoder(self.feature_dim, self.dict_size)
            self.sae.load_state_dict(torch.load(sae_path))
            self.sae.to(self.device)
            print(f"Loaded SAE from {sae_path}")

            # Extract and analyze test features
            features, targets, biases, indices = self._extract_features(
                self.dataloaders["test"], desc="Extracting test features"
            )

            analysis_results = self._analyze_monosemanticity(
                features, targets, biases, indices
            )

            # Save test analysis
            test_results_path = os.path.join(
                self.log_path, "test_analysis_results.json"
            )
            serializable_results = {
                k: v
                for k, v in analysis_results.items()
                if k not in ["latent_activations"]
            }
            with open(test_results_path, "w") as f:
                json.dump(serializable_results, f, indent=2, default=str)
            print(f"Saved test analysis to {test_results_path}")
        else:
            print(f"No SAE found at {sae_path}. Run training first.")
