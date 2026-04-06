"""
SAE-Filtered Zero-Shot VLM Classifier for VB-Mitigator.

This module implements zero-shot classification with SAE-based feature filtering:
1. Encode images with VLM encoder (OpenCLIP, SigLIP, or Perception Encoder)
2. Pass features through pretrained SAE encoder
3. Keep only neurons corresponding to a specific target class
4. Decode filtered sparse features back to original feature space
5. Perform zero-shot classification with text-image similarity

The key idea is that by keeping only class-specific (monosemantic) neurons,
we remove spurious features and bias-related information, leading to
cleaner representations for zero-shot classification.

This combines:
- SAE neuron filtering (from sae_neuron_classifier.py)
- Zero-shot VLM classification (from zero_shot_vlm.py)
"""

import os
import json
from collections import defaultdict
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vlm_encoders import (
    create_vlm_encoder,
    get_class_names,
    DATASET_CLASS_NAMES,
    BaseVLMEncoder,
)
from .base_trainer import BaseTrainer

# Import dictionary learning components
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dictionary_learning"))

from dictionary_learning import AutoEncoder


class SAEClassFilter(nn.Module):
    """
    SAE-based class-specific feature filter.

    This module:
    1. Encodes features to sparse SAE latents
    2. Masks out neurons NOT belonging to the target class
    3. Decodes back to original feature space

    The result is a "cleaned" feature that only contains information
    from neurons that are specific to the target class.
    """

    def __init__(
        self,
        sae: AutoEncoder,
        neuron_to_class: Dict[int, int],
        target_class: int,
        keep_all_classes: bool = False,
    ):
        """
        Args:
            sae: Trained Sparse Autoencoder
            neuron_to_class: Dict mapping neuron_idx -> class_idx
            target_class: Which class's neurons to keep (-1 for all monosemantic)
            keep_all_classes: If True, keep all monosemantic neurons (ignore target_class)
        """
        super().__init__()

        self.sae = sae
        self.neuron_to_class = neuron_to_class
        self.target_class = target_class
        self.keep_all_classes = keep_all_classes

        # Freeze SAE
        for param in self.sae.parameters():
            param.requires_grad = False

        # Create neuron mask
        dict_size = sae.dict_size
        mask = torch.zeros(dict_size, dtype=torch.bool)

        if keep_all_classes:
            # Keep all monosemantic neurons regardless of class
            for neuron_idx in neuron_to_class.keys():
                mask[neuron_idx] = True
            self.num_kept = len(neuron_to_class)
        else:
            # Keep only neurons for the target class
            for neuron_idx, class_idx in neuron_to_class.items():
                if class_idx == target_class:
                    mask[neuron_idx] = True
            self.num_kept = mask.sum().item()

        self.register_buffer("neuron_mask", mask)

        print(
            f"SAEClassFilter: keeping {self.num_kept}/{dict_size} neurons "
            f"(target_class={target_class if not keep_all_classes else 'all'})"
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Filter features through SAE, keeping only target class neurons.

        Args:
            features: Input features (B, feature_dim)

        Returns:
            Filtered and decoded features (B, feature_dim)
        """
        # Encode to sparse latents
        latents = self.sae.encode(features)

        # Apply mask - zero out non-target neurons
        masked_latents = latents * self.neuron_mask.float().unsqueeze(0)

        # Decode back to feature space
        filtered_features = self.sae.decode(masked_latents)

        return filtered_features

    def get_sparse_features(self, features: torch.Tensor) -> torch.Tensor:
        """Get masked sparse features without decoding."""
        latents = self.sae.encode(features)
        return latents * self.neuron_mask.float().unsqueeze(0)


class SAEFilteredZeroShotClassifier(nn.Module):
    """
    Zero-shot classifier with SAE-based feature filtering.

    Pipeline:
        Image -> VLM Encoder -> Features -> SAE Filter -> Filtered Features ->
        Cosine Similarity with Text Embeddings -> Class Prediction
    """

    def __init__(
        self,
        encoder: BaseVLMEncoder,
        sae_filter: SAEClassFilter,
        class_names: List[str],
        temperature: float = 100.0,
        use_filtered_for_text: bool = False,  # Whether to also filter text embeddings
    ):
        super().__init__()

        self.encoder = encoder
        self.sae_filter = sae_filter
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.temperature = temperature
        self.use_filtered_for_text = use_filtered_for_text

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Pre-compute text features
        with torch.no_grad():
            text_features = encoder.encode_text(class_names)
            text_features = F.normalize(text_features, dim=-1)

        self.register_buffer("text_features", text_features)

        # Optionally filter text features too
        if use_filtered_for_text:
            with torch.no_grad():
                filtered_text = sae_filter(text_features)
                filtered_text = F.normalize(filtered_text, dim=-1)
            self.register_buffer("filtered_text_features", filtered_text)

    def forward(
        self,
        images: torch.Tensor,
        use_filtered: bool = True,
        return_both: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Input images (B, C, H, W)
            use_filtered: Whether to use SAE-filtered features
            return_both: Return both filtered and unfiltered logits

        Returns:
            logits: Similarity scores (B, num_classes)
        """
        # Encode images
        with torch.no_grad():
            image_features = self.encoder.encode_image(images)

        # Get unfiltered prediction
        unfiltered_features = F.normalize(image_features, dim=-1)
        unfiltered_logits = self.temperature * (
            unfiltered_features @ self.text_features.T
        )

        if not use_filtered:
            return unfiltered_logits

        # Get filtered prediction
        filtered_features = self.sae_filter(image_features)
        filtered_features = F.normalize(filtered_features, dim=-1)

        text_feats = (
            self.filtered_text_features
            if self.use_filtered_for_text and hasattr(self, "filtered_text_features")
            else self.text_features
        )

        filtered_logits = self.temperature * (filtered_features @ text_feats.T)

        if return_both:
            return filtered_logits, unfiltered_logits
        return filtered_logits

    def predict(self, images: torch.Tensor, use_filtered: bool = True) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(images, use_filtered=use_filtered)
        return logits.argmax(dim=1)


class SAEFilteredZeroShotTrainer(BaseTrainer):
    """
    Trainer for SAE-filtered zero-shot VLM classification.

    This combines:
    1. VLM encoder (OpenCLIP/SigLIP/PE) for image and text encoding
    2. SAE for sparse feature decomposition
    3. Class-specific neuron filtering
    4. Zero-shot classification on filtered features

    The hypothesis is that filtering to class-specific neurons removes
    spurious correlations and bias-related features.

    Configuration:
        MITIGATOR:
          TYPE: "sae_filtered_zero_shot"
          SAE_FILTERED_ZERO_SHOT:
            # VLM Encoder settings
            ENCODER_TYPE: "openclip"
            MODEL_NAME: "ViT-L-14"
            PRETRAINED: "openai"

            # SAE settings
            SAE_CHECKPOINT_PATH: "path/to/ae.pt"
            SAE_ANALYSIS_PATH: "path/to/analysis_results.json"
            PURITY_THRESHOLD: 1.0
            ACTIVATION_THRESHOLD: 0.1

            # Filtering strategy
            FILTER_MODE: "per_class"  # "per_class", "all_mono", or "target_only"
            TARGET_CLASS: -1  # For "target_only" mode

            # Zero-shot settings
            CLASS_NAME_VARIANT: "default"
            TEMPERATURE: 100.0

            # Comparison
            COMPARE_WITH_UNFILTERED: True
    """

    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """Setup VLM encoder, SAE, and filtered classifier."""

        cfg = self.cfg.MITIGATOR.SAE_FILTERED_ZERO_SHOT

        print(f"\n{'='*60}")
        print("Setting up SAE-Filtered Zero-Shot Classifier")
        print(f"{'='*60}")

        # Step 1: Create VLM encoder
        self.encoder = create_vlm_encoder(
            encoder_type=cfg.ENCODER_TYPE,
            model_name=cfg.MODEL_NAME,
            device=self.device,
            pretrained=cfg.get("PRETRAINED", "openai"),
        )

        print(f"Loaded {cfg.ENCODER_TYPE} encoder: {cfg.MODEL_NAME}")
        print(f"  Embed dim: {self.encoder.embed_dim}")

        # Step 2: Load SAE
        sae_checkpoint_path = cfg.SAE_CHECKPOINT_PATH
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
        self.sae.to(self.device)
        print(f"Loaded SAE: activation_dim={activation_dim}, dict_size={dict_size}")

        # Step 3: Load analysis and compute neuron-to-class mapping
        analysis_path = cfg.SAE_ANALYSIS_PATH
        if not analysis_path or not os.path.exists(analysis_path):
            raise ValueError(f"SAE analysis not found: {analysis_path}")

        with open(analysis_path, "r") as f:
            analysis_results = json.load(f)

        neuron_to_class = self._compute_neuron_class_assignment(
            analysis_results,
            dict_size=dict_size,
            purity_threshold=cfg.PURITY_THRESHOLD,
            activation_threshold=cfg.ACTIVATION_THRESHOLD,
        )

        self.neuron_to_class = neuron_to_class
        self.dict_size = dict_size

        # Step 4: Get class names
        dataset_name = self.cfg.DATASET.TYPE.lower()
        variant = cfg.CLASS_NAME_VARIANT

        if isinstance(variant, list):
            class_names = variant
        else:
            class_names = get_class_names(dataset_name, variant)

        self.class_names = class_names
        self.dataset_name = dataset_name

        print(f"\nClass names (variant='{variant}'):")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name[:60]}...")

        # Step 5: Create SAE filters for each class (or single filter)
        filter_mode = cfg.get("FILTER_MODE", "per_class")

        if filter_mode == "per_class":
            # Create a separate filter for each class
            self.sae_filters = nn.ModuleList()
            for class_idx in range(self.num_class):
                sae_filter = SAEClassFilter(
                    sae=self.sae,
                    neuron_to_class=neuron_to_class,
                    target_class=class_idx,
                    keep_all_classes=False,
                )
                self.sae_filters.append(sae_filter)
            self.filter_mode = "per_class"

        elif filter_mode == "all_mono":
            # Keep all monosemantic neurons regardless of class
            self.sae_filter = SAEClassFilter(
                sae=self.sae,
                neuron_to_class=neuron_to_class,
                target_class=-1,
                keep_all_classes=True,
            )
            self.filter_mode = "all_mono"

        elif filter_mode == "target_only":
            # Keep only neurons for a specific target class
            target_class = cfg.get("TARGET_CLASS", 0)
            self.sae_filter = SAEClassFilter(
                sae=self.sae,
                neuron_to_class=neuron_to_class,
                target_class=target_class,
                keep_all_classes=False,
            )
            self.filter_mode = "target_only"
            self.target_class = target_class
        else:
            raise ValueError(f"Unknown filter_mode: {filter_mode}")
        self.sae_filter.to(self.device)

        # Step 6: Create classifier (using all_mono filter for now)
        # For per_class mode, we'll handle it differently in forward
        if filter_mode != "per_class":
            self.model = SAEFilteredZeroShotClassifier(
                encoder=self.encoder,
                sae_filter=self.sae_filter,
                class_names=class_names,
                temperature=cfg.TEMPERATURE,
            )
        else:
            # Create a dummy model for compatibility
            # Actual classification done in eval()
            self.model = nn.Identity()

        self.temperature = cfg.TEMPERATURE
        self.compare_unfiltered = cfg.get("COMPARE_WITH_UNFILTERED", True)

        # Pre-compute text features
        with torch.no_grad():
            text_features = self.encoder.encode_text(class_names)
            self.text_features = F.normalize(text_features, dim=-1).to(self.device)

    def _compute_neuron_class_assignment(
        self,
        analysis_results: Dict,
        dict_size: int,
        purity_threshold: float,
        activation_threshold: float,
    ) -> Dict[int, int]:
        """Compute neuron-to-class assignment (same as in sae_neuron_classifier)."""

        neuron_to_class = {}
        top_k_per_latent = analysis_results.get("top_k_per_latent", {})

        class_neuron_counts = defaultdict(int)
        stats = {"passed": 0, "failed_activation": 0, "failed_purity": 0}

        for latent_idx in range(dict_size):
            latent_key = str(latent_idx)

            if latent_key not in top_k_per_latent:
                continue

            latent_data = top_k_per_latent[latent_key]
            targets = latent_data.get("targets", [])
            activations = latent_data.get("activations", [])

            if len(targets) == 0 or len(activations) == 0:
                continue

            # Filter 1: Activation threshold
            max_activation = max(activations)
            if max_activation < activation_threshold:
                stats["failed_activation"] += 1
                continue

            # Filter 2: Purity threshold
            target_counts = defaultdict(int)
            for t in targets:
                target_counts[t] += 1

            majority_class = max(target_counts.keys(), key=lambda k: target_counts[k])
            purity = target_counts[majority_class] / len(targets)

            if purity < purity_threshold:
                stats["failed_purity"] += 1
                continue

            stats["passed"] += 1
            neuron_to_class[latent_idx] = majority_class
            class_neuron_counts[majority_class] += 1

        print(f"\nNeuron filtering:")
        print(f"  Passed: {stats['passed']}")
        print(f"  Failed activation: {stats['failed_activation']}")
        print(f"  Failed purity: {stats['failed_purity']}")
        print(f"\nNeurons per class:")
        for class_idx in sorted(class_neuron_counts.keys()):
            print(f"  Class {class_idx}: {class_neuron_counts[class_idx]} neurons")

        return neuron_to_class

    def _setup_optimizer(self):
        """No optimizer needed for zero-shot."""
        self.optimizer = None

    def _setup_scheduler(self):
        """No scheduler needed for zero-shot."""
        self.scheduler = None

    def train(self):
        """Main entry point - evaluate different filtering strategies."""

        print(f"\n{'='*60}")
        print("SAE-Filtered Zero-Shot Classification Evaluation")
        print(f"{'='*60}")

        all_results = {}

        # 1. Evaluate unfiltered (baseline)
        if self.compare_unfiltered:
            print("\n--- Unfiltered (Baseline) ---")
            metrics = self._eval_unfiltered()
            all_results["unfiltered"] = metrics
            self._print_metrics(metrics)

        # 2. Evaluate with filtering
        if self.filter_mode == "per_class":
            print("\n--- Per-Class Filtering ---")
            metrics = self._eval_per_class_filtering()
            all_results["per_class_filtered"] = metrics
            self._print_metrics(metrics)

        elif self.filter_mode == "all_mono":
            print("\n--- All Monosemantic Neurons ---")
            metrics = self._eval_single_filter(self.sae_filter)
            all_results["all_mono_filtered"] = metrics
            self._print_metrics(metrics)

        elif self.filter_mode == "target_only":
            print(f"\n--- Target Class {self.target_class} Only ---")
            metrics = self._eval_single_filter(self.sae_filter)
            all_results[f"target_{self.target_class}_filtered"] = metrics
            self._print_metrics(metrics)

        # Save results
        self._save_results(all_results)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _eval_unfiltered(self) -> Dict:
        """Evaluate without SAE filtering (baseline)."""

        all_preds = []
        all_targets = []
        all_biases = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(self.dataloaders["test"], desc="Unfiltered"):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]

                # Encode and classify
                image_features = self.encoder.encode_image(inputs)
                image_features = F.normalize(image_features, dim=-1)
                logits = self.temperature * (image_features @ self.text_features.T)
                preds = logits.argmax(dim=1).cpu()

                all_preds.append(preds)
                all_targets.append(targets)

                for bias_name in self.biases:
                    if bias_name in batch:
                        all_biases[bias_name].append(batch[bias_name])

        return self._compute_metrics(all_preds, all_targets, all_biases)

    def _eval_single_filter(self, sae_filter: SAEClassFilter) -> Dict:
        """Evaluate with a single SAE filter."""

        all_preds = []
        all_targets = []
        all_biases = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(self.dataloaders["test"], desc="Filtered"):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]

                # Encode, filter, and classify
                image_features = self.encoder.encode_image(inputs)
                filtered_features = sae_filter(image_features)
                filtered_features = F.normalize(filtered_features, dim=-1)
                logits = self.temperature * (filtered_features @ self.text_features.T)
                preds = logits.argmax(dim=1).cpu()

                all_preds.append(preds)
                all_targets.append(targets)

                for bias_name in self.biases:
                    if bias_name in batch:
                        all_biases[bias_name].append(batch[bias_name])

        return self._compute_metrics(all_preds, all_targets, all_biases)

    def _eval_per_class_filtering(self) -> Dict:
        """
        Evaluate with per-class filtering.

        For each sample, compute similarity using each class's filtered features,
        then predict based on which class filter gives highest similarity.
        """

        all_preds = []
        all_targets = []
        all_biases = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(self.dataloaders["test"], desc="Per-class"):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]
                batch_size = inputs.shape[0]

                # Encode images
                image_features = self.encoder.encode_image(inputs)

                # For each class, filter and compute similarity
                all_similarities = []
                for class_idx, sae_filter in enumerate(self.sae_filters):
                    # Filter features using this class's neurons
                    filtered = sae_filter(image_features)
                    filtered = F.normalize(filtered, dim=-1)

                    # Compute similarity to this class's text embedding
                    class_text = self.text_features[class_idx : class_idx + 1]
                    similarity = (filtered @ class_text.T).squeeze(-1)
                    all_similarities.append(similarity)

                # Stack and get predictions
                similarities = torch.stack(all_similarities, dim=1)  # (B, num_classes)
                preds = similarities.argmax(dim=1).cpu()

                all_preds.append(preds)
                all_targets.append(targets)

                for bias_name in self.biases:
                    if bias_name in batch:
                        all_biases[bias_name].append(batch[bias_name])

        return self._compute_metrics(all_preds, all_targets, all_biases)

    def _compute_metrics(self, all_preds, all_targets, all_biases) -> Dict:
        """Compute evaluation metrics."""

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        accuracy = (all_preds == all_targets).float().mean().item()
        metrics = {"accuracy": accuracy}

        # Per-class accuracy
        for c in range(self.num_class):
            mask = all_targets == c
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_targets[mask]).float().mean().item()
                metrics[f"acc_class_{c}"] = class_acc

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

    def _print_metrics(self, metrics: Dict):
        """Print metrics."""
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if "wg_ovr" in metrics:
            print(f"  Worst-group: {metrics['wg_ovr']:.4f}")
        print(f" all: {metrics}")

    def _print_summary(self, all_results: Dict):
        """Print summary of all results."""
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"{'Method':<25} {'Accuracy':>10} {'Worst-Group':>12}")
        print("-" * 49)

        for method, metrics in all_results.items():
            acc = metrics.get("accuracy", 0)
            wg = metrics.get("wg_ovr", 0)
            print(f"{method:<25} {acc:>10.4f} {wg:>12.4f}")

    def _save_results(self, all_results: Dict):
        """Save results to file."""
        cfg = self.cfg.MITIGATOR.SAE_FILTERED_ZERO_SHOT

        results = {
            "config": {
                "encoder_type": cfg.ENCODER_TYPE,
                "model_name": cfg.MODEL_NAME,
                "filter_mode": self.filter_mode,
                "purity_threshold": cfg.PURITY_THRESHOLD,
                "activation_threshold": cfg.ACTIVATION_THRESHOLD,
                "temperature": cfg.TEMPERATURE,
                "num_monosemantic_neurons": len(self.neuron_to_class),
            },
            "results": all_results,
        }

        save_path = os.path.join(self.log_path, "sae_filtered_zero_shot_results.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    def eval(self):
        """Compatibility method."""
        return self._eval_unfiltered()
