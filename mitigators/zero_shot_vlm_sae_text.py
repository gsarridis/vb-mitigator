"""
SAE Text-Aligned Zero-Shot Classifier for VB-Mitigator.

This module implements zero-shot classification where SAE neurons are assigned
to classes based on their alignment with text embeddings (not training data).

Key difference from sae_filtered_zero_shot.py:
- Instead of using analysis_results.json to determine neuron-class assignments
- We use the SAE decoder weights and text embeddings to compute alignment
- Each SAE neuron (decoder column) is assigned to the class whose text embedding
  it is most similar to

Pipeline:
1. Load VLM encoder and SAE
2. Encode class names with VLM text encoder → text_features
3. For each SAE decoder column (neuron direction), compute similarity to each class
4. Assign each neuron to its most aligned class
5. For classification: keep only neurons for each class, decode, compute similarity

This is more aligned with zero-shot philosophy since we don't need labeled data
to determine which neurons correspond to which class.
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


def compute_neuron_class_alignment(
    sae: AutoEncoder,
    text_features: torch.Tensor,
    method: str = "decoder",
    top_k: int = 1,
) -> Tuple[Dict[int, int], torch.Tensor]:
    """
    Compute which class each SAE neuron aligns with based on text embeddings.

    Args:
        sae: Trained Sparse Autoencoder
        text_features: Text embeddings for each class (num_classes, embed_dim)
        method: How to compute alignment:
            - "decoder": Use SAE decoder columns (neuron directions in feature space)
            - "encoder": Use SAE encoder rows
        top_k: Assign neuron to class if it's in top-k most aligned neurons for that class

    Returns:
        neuron_to_class: Dict mapping neuron_idx -> class_idx
        alignment_scores: Tensor of shape (dict_size, num_classes) with similarity scores
    """
    text_features = F.normalize(text_features, dim=-1)

    if method == "decoder":
        # Decoder columns represent neuron directions in feature space
        # decoder shape: (activation_dim, dict_size) or accessed via sae.decoder.weight
        if hasattr(sae, "decoder") and hasattr(sae.decoder, "weight"):
            # decoder.weight shape: (activation_dim, dict_size)
            neuron_directions = sae.decoder.weight.T  # (dict_size, activation_dim)
        elif hasattr(sae, "W_dec"):
            # Some SAE implementations use W_dec
            neuron_directions = sae.W_dec  # Should be (dict_size, activation_dim)
        else:
            raise ValueError("Cannot find decoder weights in SAE")
    elif method == "encoder":
        # Encoder rows represent what each neuron responds to
        if hasattr(sae, "encoder") and hasattr(sae.encoder, "weight"):
            neuron_directions = sae.encoder.weight  # (dict_size, activation_dim)
        elif hasattr(sae, "W_enc"):
            neuron_directions = sae.W_enc
        else:
            raise ValueError("Cannot find encoder weights in SAE")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize neuron directions
    neuron_directions = F.normalize(neuron_directions, dim=-1)

    # Compute alignment: (dict_size, num_classes)
    alignment_scores = neuron_directions @ text_features.T

    # Assign each neuron to its most aligned class
    neuron_to_class = {}
    best_class_per_neuron = alignment_scores.argmax(dim=1)

    for neuron_idx in range(alignment_scores.shape[0]):
        neuron_to_class[neuron_idx] = best_class_per_neuron[neuron_idx].item()

    return neuron_to_class, alignment_scores


def compute_neuron_class_alignment_thresholded(
    sae: AutoEncoder,
    text_features: torch.Tensor,
    method: str = "decoder",
    threshold: float = 0.1,
    min_margin: float = 0.05,
) -> Tuple[Dict[int, int], torch.Tensor, Dict]:
    """
    Compute neuron-class alignment with thresholding for quality control.

    Only assigns a neuron to a class if:
    1. Its alignment score exceeds the threshold
    2. The margin between best and second-best class exceeds min_margin

    Args:
        sae: Trained Sparse Autoencoder
        text_features: Text embeddings for each class (num_classes, embed_dim)
        method: "decoder" or "encoder"
        threshold: Minimum alignment score to assign a neuron
        min_margin: Minimum difference between best and second-best class

    Returns:
        neuron_to_class: Dict mapping neuron_idx -> class_idx (only for aligned neurons)
        alignment_scores: Full alignment matrix
        stats: Statistics about the alignment
    """
    text_features = F.normalize(text_features, dim=-1)

    # Get neuron directions
    if method == "decoder":
        if hasattr(sae, "decoder") and hasattr(sae.decoder, "weight"):
            neuron_directions = sae.decoder.weight.T
        elif hasattr(sae, "W_dec"):
            neuron_directions = sae.W_dec
        else:
            raise ValueError("Cannot find decoder weights")
    else:
        if hasattr(sae, "encoder") and hasattr(sae.encoder, "weight"):
            neuron_directions = sae.encoder.weight
        elif hasattr(sae, "W_enc"):
            neuron_directions = sae.W_enc
        else:
            raise ValueError("Cannot find encoder weights")

    neuron_directions = F.normalize(neuron_directions, dim=-1)

    # Compute alignment
    alignment_scores = neuron_directions @ text_features.T
    dict_size, num_classes = alignment_scores.shape

    # Get top-2 classes for each neuron
    top2_scores, top2_indices = alignment_scores.topk(2, dim=1)
    best_scores = top2_scores[:, 0]
    second_scores = top2_scores[:, 1]
    best_classes = top2_indices[:, 0]
    margins = best_scores - second_scores

    # Apply thresholds
    neuron_to_class = {}
    stats = {
        "total_neurons": dict_size,
        "num_classes": num_classes,
        "passed_threshold": 0,
        "passed_margin": 0,
        "assigned": 0,
        "per_class": defaultdict(int),
        "mean_alignment": {},
        "mean_margin": {},
    }

    for neuron_idx in range(dict_size):
        best_score = best_scores[neuron_idx].item()
        margin = margins[neuron_idx].item()
        best_class = best_classes[neuron_idx].item()

        if best_score >= threshold:
            stats["passed_threshold"] += 1
            if margin >= min_margin:
                stats["passed_margin"] += 1
                neuron_to_class[neuron_idx] = best_class
                stats["per_class"][best_class] += 1
                stats["assigned"] += 1

    # Compute per-class statistics
    for class_idx in range(num_classes):
        class_neurons = [n for n, c in neuron_to_class.items() if c == class_idx]
        if class_neurons:
            class_scores = alignment_scores[class_neurons, class_idx]
            stats["mean_alignment"][class_idx] = class_scores.mean().item()
            class_margins = margins[class_neurons]
            stats["mean_margin"][class_idx] = class_margins.mean().item()

    return neuron_to_class, alignment_scores, stats


class TextAlignedSAEFilter(nn.Module):
    """
    SAE filter where neuron-class assignments come from text alignment.
    """

    def __init__(
        self,
        sae: AutoEncoder,
        neuron_to_class: Dict[int, int],
        num_classes: int,
    ):
        super().__init__()

        self.sae = sae
        self.neuron_to_class = neuron_to_class
        self.num_classes = num_classes

        # Freeze SAE
        for param in self.sae.parameters():
            param.requires_grad = False

        # Create masks for each class
        dict_size = sae.dict_size
        for class_idx in range(num_classes):
            mask = torch.zeros(dict_size, dtype=torch.bool)
            for neuron_idx, c in neuron_to_class.items():
                if c == class_idx:
                    mask[neuron_idx] = True
            self.register_buffer(f"class_mask_{class_idx}", mask)

        # Count neurons per class
        self.neurons_per_class = []
        for class_idx in range(num_classes):
            count = sum(1 for c in neuron_to_class.values() if c == class_idx)
            self.neurons_per_class.append(count)

    def get_class_mask(self, class_idx: int) -> torch.Tensor:
        return getattr(self, f"class_mask_{class_idx}")

    def filter_for_class(self, features: torch.Tensor, class_idx: int) -> torch.Tensor:
        """Filter features keeping only neurons for the specified class."""
        latents = self.sae.encode(features)
        mask = self.get_class_mask(class_idx).float().unsqueeze(0)
        masked_latents = latents * mask
        return self.sae.decode(masked_latents)

    def forward(self, features: torch.Tensor, class_idx: int) -> torch.Tensor:
        """Alias for filter_for_class."""
        return self.filter_for_class(features, class_idx)


class SAETextAlignedZeroShotClassifier(nn.Module):
    """
    Zero-shot classifier using text-aligned SAE neuron filtering.

    For each class:
    1. Filter image features using that class's neurons
    2. Compute similarity to that class's text embedding

    Predict the class with highest similarity.
    """

    def __init__(
        self,
        encoder: BaseVLMEncoder,
        sae_filter: TextAlignedSAEFilter,
        text_features: torch.Tensor,
        temperature: float = 100.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.sae_filter = sae_filter
        self.num_classes = sae_filter.num_classes
        self.temperature = temperature

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Store normalized text features
        self.register_buffer("text_features", F.normalize(text_features, dim=-1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Classify images using per-class filtered features.

        For each class c:
            filtered_c = filter(image_features, class=c)
            score_c = similarity(filtered_c, text_c)

        Returns: logits (B, num_classes)
        """
        # Encode images
        with torch.no_grad():
            image_features = self.encoder.encode_image(images)

        batch_size = image_features.shape[0]
        scores = torch.zeros(batch_size, self.num_classes, device=images.device)

        for class_idx in range(self.num_classes):
            # Filter using this class's neurons
            filtered = self.sae_filter.filter_for_class(image_features, class_idx)
            filtered = F.normalize(filtered, dim=-1)

            # Compute similarity to this class's text
            class_text = self.text_features[class_idx : class_idx + 1]
            similarity = (filtered @ class_text.T).squeeze(-1)
            scores[:, class_idx] = similarity

        return self.temperature * scores

    def forward_unfiltered(self, images: torch.Tensor) -> torch.Tensor:
        """Classify without filtering (baseline)."""
        with torch.no_grad():
            image_features = self.encoder.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        return self.temperature * (image_features @ self.text_features.T)


class SAETextAlignedZeroShotTrainer(BaseTrainer):
    """
    Trainer for text-aligned SAE zero-shot classification.

    Key feature: Uses DATASET_CLASS_NAMES text embeddings to determine
    which SAE neurons correspond to which class, rather than using
    labeled training data.

    Configuration:
        MITIGATOR:
          TYPE: "sae_text_aligned_zero_shot"
          SAE_TEXT_ALIGNED_ZERO_SHOT:
            # VLM Encoder
            ENCODER_TYPE: "openclip"
            MODEL_NAME: "ViT-L-14"
            PRETRAINED: "openai"

            # SAE (no analysis_results needed!)
            SAE_CHECKPOINT_PATH: "path/to/ae.pt"

            # Alignment settings
            ALIGNMENT_METHOD: "decoder"  # or "encoder"
            ALIGNMENT_THRESHOLD: 0.1     # Min similarity to assign neuron
            ALIGNMENT_MARGIN: 0.05       # Min margin between best and second class

            # Zero-shot settings
            CLASS_NAME_VARIANT: "default"
            TEMPERATURE: 100.0
    """

    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_scheduler(self):
        """No scheduler needed for zero-shot."""
        self.scheduler = None

    def _setup_models(self):
        """Setup VLM encoder, SAE, and text-aligned classifier."""

        cfg = self.cfg.MITIGATOR.SAE_TEXT_ALIGNED_ZERO_SHOT

        print(f"\n{'='*60}")
        print("Setting up SAE Text-Aligned Zero-Shot Classifier")
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
        self.sae.eval()
        print(f"Loaded SAE: activation_dim={activation_dim}, dict_size={dict_size}")

        # Step 3: Get class names from DATASET_CLASS_NAMES
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
            print(f"  {i}: {name}")

        # Step 4: Encode class names with VLM text encoder
        print("\nComputing text embeddings for class names...")
        with torch.no_grad():
            text_features = self.encoder.encode_text(class_names)
            text_features = text_features.to(self.device)

        print(f"Text features shape: {text_features.shape}")

        # Step 5: Compute neuron-class alignment using text embeddings
        print("\nComputing neuron-class alignment from text embeddings...")

        neuron_to_class, alignment_scores, stats = (
            compute_neuron_class_alignment_thresholded(
                sae=self.sae,
                text_features=text_features,
                method=cfg.get("ALIGNMENT_METHOD", "decoder"),
                threshold=cfg.get("ALIGNMENT_THRESHOLD", 0.1),
                min_margin=cfg.get("ALIGNMENT_MARGIN", 0.05),
            )
        )

        self.neuron_to_class = neuron_to_class
        self.alignment_scores = alignment_scores
        self.alignment_stats = stats

        print(f"\nAlignment Statistics:")
        print(f"  Total neurons: {stats['total_neurons']}")
        print(
            f"  Passed threshold ({cfg.get('ALIGNMENT_THRESHOLD', 0.1)}): {stats['passed_threshold']}"
        )
        print(
            f"  Passed margin ({cfg.get('ALIGNMENT_MARGIN', 0.05)}): {stats['passed_margin']}"
        )
        print(f"  Assigned neurons: {stats['assigned']}")
        print(f"\nNeurons per class:")
        for class_idx in range(len(class_names)):
            count = stats["per_class"].get(class_idx, 0)
            mean_align = stats["mean_alignment"].get(class_idx, 0)
            print(
                f"  Class {class_idx} ({class_names[class_idx][:30]}...): "
                f"{count} neurons, mean alignment: {mean_align:.4f}"
            )

        # Step 6: Create text-aligned SAE filter
        self.sae_filter = TextAlignedSAEFilter(
            sae=self.sae,
            neuron_to_class=neuron_to_class,
            num_classes=len(class_names),
        )
        self.sae_filter.to(self.device)

        # Step 7: Create classifier
        self.model = SAETextAlignedZeroShotClassifier(
            encoder=self.encoder,
            sae_filter=self.sae_filter,
            text_features=text_features,
            temperature=cfg.TEMPERATURE,
        )
        self.model.to(self.device)

        self.temperature = cfg.TEMPERATURE

    def _setup_optimizer(self):
        """No optimizer needed."""
        self.optimizer = None

    def train(self):
        """Evaluate the text-aligned zero-shot classifier."""

        print(f"\n{'='*60}")
        print("SAE Text-Aligned Zero-Shot Classification")
        print(f"{'='*60}")

        all_results = {}

        # 1. Evaluate unfiltered baseline
        print("\n--- Unfiltered Baseline ---")
        metrics = self._eval_unfiltered()
        all_results["unfiltered"] = metrics
        self._print_metrics(metrics)

        # 2. Evaluate with text-aligned filtering
        print("\n--- Text-Aligned Filtered ---")
        metrics = self._eval_filtered()
        all_results["text_aligned_filtered"] = metrics
        self._print_metrics(metrics)

        # Save results
        self._save_results(all_results)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _eval_unfiltered(self) -> Dict:
        """Evaluate without SAE filtering."""
        self.model.eval()

        all_preds = []
        all_targets = []
        all_biases = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(self.dataloaders["test"], desc="Unfiltered"):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]

                logits = self.model.forward_unfiltered(inputs)
                preds = logits.argmax(dim=1).cpu()

                all_preds.append(preds)
                all_targets.append(targets)

                for bias_name in self.biases:
                    if bias_name in batch:
                        all_biases[bias_name].append(batch[bias_name])

        return self._compute_metrics(all_preds, all_targets, all_biases)

    def _eval_filtered(self) -> Dict:
        """Evaluate with text-aligned filtering."""
        self.model.eval()

        all_preds = []
        all_targets = []
        all_biases = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(self.dataloaders["test"], desc="Filtered"):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]

                logits = self.model(inputs)
                preds = logits.argmax(dim=1).cpu()

                all_preds.append(preds)
                all_targets.append(targets)

                for bias_name in self.biases:
                    if bias_name in batch:
                        all_biases[bias_name].append(batch[bias_name])

        return self._compute_metrics(all_preds, all_targets, all_biases)

    def _compute_metrics(self, all_preds, all_targets, all_biases) -> Dict:
        """Compute metrics."""
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
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if "wg_ovr" in metrics:
            print(f"  Worst-group: {metrics['wg_ovr']:.4f}")
        print(f" all: {metrics}")

    def _print_summary(self, all_results: Dict):
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
        cfg = self.cfg.MITIGATOR.SAE_TEXT_ALIGNED_ZERO_SHOT

        results = {
            "config": {
                "encoder_type": cfg.ENCODER_TYPE,
                "model_name": cfg.MODEL_NAME,
                "alignment_method": cfg.get("ALIGNMENT_METHOD", "decoder"),
                "alignment_threshold": cfg.get("ALIGNMENT_THRESHOLD", 0.1),
                "alignment_margin": cfg.get("ALIGNMENT_MARGIN", 0.05),
                "temperature": cfg.TEMPERATURE,
                "class_names": self.class_names,
            },
            "alignment_stats": {
                k: v if not isinstance(v, defaultdict) else dict(v)
                for k, v in self.alignment_stats.items()
            },
            "results": all_results,
        }

        save_path = os.path.join(self.log_path, "sae_text_aligned_results.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    def eval(self):
        return self._eval_filtered()
