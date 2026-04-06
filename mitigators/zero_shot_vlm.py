"""
Zero-Shot VLM Classifier for VB-Mitigator.

This module implements zero-shot classification using VLM encoders:
1. No training required - uses text-image similarity
2. Dataset-specific class prompts for better performance
3. Supports multiple prompt variants for ablation studies
4. Can be combined with SAE analysis

The key idea is to use more specific/descriptive class names
instead of the generic binary labels used in bias benchmarks:
- UTKFace: "male"/"female" instead of 0/1
- Waterbirds: specific bird species instead of "waterbird"/"landbird"
- UrbanCars: specific car types instead of "urban"/"country"
"""

import os
import json
from collections import defaultdict
from typing import List, Optional, Dict

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
    VLMEncoderWithHead,
    BaseVLMEncoder,
)
from .base_trainer import BaseTrainer


class ZeroShotVLMClassifier(nn.Module):
    """
    Zero-shot classifier using VLM text-image similarity.

    Classification is done by:
    1. Encoding test image with vision encoder
    2. Encoding class names with text encoder
    3. Computing cosine similarity
    4. Predicting class with highest similarity
    """

    def __init__(
        self,
        encoder: BaseVLMEncoder,
        class_names: List[str],
        temperature: float = 100.0,  # CLIP-style temperature
    ):
        super().__init__()

        self.encoder = encoder
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.temperature = temperature

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Pre-compute text features
        with torch.no_grad():
            text_features = encoder.encode_text(class_names)
            text_features = F.normalize(text_features, dim=-1)

        self.register_buffer("text_features", text_features)

        print(f"ZeroShotVLMClassifier initialized:")
        print(f"  Encoder: {type(encoder).__name__}")
        print(f"  Embed dim: {encoder.embed_dim}")
        print(f"  Num classes: {self.num_classes}")
        for i, name in enumerate(class_names):
            print(f"    Class {i}: {name[:50]}...")

    def forward(self, images: torch.Tensor, return_features: bool = False):
        """
        Forward pass.

        Args:
            images: Input images (B, C, H, W)
            return_features: If True, return features

        Returns:
            logits: Similarity scores (B, num_classes)
        """
        # Encode images
        with torch.no_grad():
            image_features = self.encoder.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

        # Compute similarity
        logits = self.temperature * (image_features @ self.text_features.T)

        if return_features:
            return logits, image_features
        return logits

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(images)
        return logits.argmax(dim=1)

    def update_class_names(self, class_names: List[str]):
        """Update class names and recompute text features."""
        self.class_names = class_names
        self.num_classes = len(class_names)

        with torch.no_grad():
            text_features = self.encoder.encode_text(class_names)
            text_features = F.normalize(text_features, dim=-1)

        # Update buffer
        del self.text_features
        self.register_buffer("text_features", text_features)


class ZeroShotVLMTrainer(BaseTrainer):
    """
    Trainer for zero-shot VLM classification.

    This is primarily an evaluation tool since zero-shot requires no training.
    It evaluates different prompt variants and reports metrics.

    Configuration:
        MITIGATOR:
          TYPE: "zero_shot_vlm"
          ZERO_SHOT_VLM:
            ENCODER_TYPE: "openclip"  # or "siglip", "perception_encoder"
            MODEL_NAME: "ViT-L-14"
            PRETRAINED: "openai"
            CLASS_NAME_VARIANT: "default"  # or "species", "detailed", custom list
            TEMPERATURE: 100.0
            EVALUATE_ALL_VARIANTS: True  # Test all prompt variants
    """

    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """Setup VLM encoder and zero-shot classifier."""

        zs_cfg = self.cfg.MITIGATOR.ZERO_SHOT_VLM

        print(f"\n{'='*60}")
        print("Setting up Zero-Shot VLM Classifier")
        print(f"{'='*60}")

        # Create encoder
        self.encoder = create_vlm_encoder(
            encoder_type=zs_cfg.ENCODER_TYPE,
            model_name=zs_cfg.MODEL_NAME,
            device=self.device,
            pretrained=zs_cfg.get("PRETRAINED", "laion2b_s34b_b79k"),
        )

        print(f"Loaded {zs_cfg.ENCODER_TYPE} encoder: {zs_cfg.MODEL_NAME}")
        print(f"  Embed dim: {self.encoder.embed_dim}")
        print(f"  Image size: {self.encoder.image_size}")

        # Get class names for dataset
        dataset_name = self.cfg.DATASET.TYPE.lower()
        variant = zs_cfg.CLASS_NAME_VARIANT

        # Check if variant is a custom list
        if isinstance(variant, list):
            class_names = variant
        else:
            class_names = get_class_names(dataset_name, variant)

        print(f"\nClass names (variant='{variant}'):")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")

        # Create zero-shot classifier
        self.model = ZeroShotVLMClassifier(
            encoder=self.encoder,
            class_names=class_names,
            temperature=zs_cfg.TEMPERATURE,
        )

        self.model.to(self.device)

        # Store for later analysis
        self.class_names = class_names
        self.dataset_name = dataset_name

    def _setup_optimizer(self):
        """No optimizer needed for zero-shot."""
        self.optimizer = None

    def _setup_scheduler(self):
        """No scheduler needed for zero-shot."""
        self.scheduler = None

    def train(self):
        """
        Main entry point.

        For zero-shot, we just evaluate (no training).
        Optionally test multiple prompt variants.
        """
        print(f"\n{'='*60}")
        print("Zero-Shot VLM Classification Evaluation")
        print(f"{'='*60}")

        zs_cfg = self.cfg.MITIGATOR.ZERO_SHOT_VLM
        all_results = {}

        # Evaluate main variant
        print(f"\nEvaluating main variant: {zs_cfg.CLASS_NAME_VARIANT}")
        metrics = self.eval()
        all_results[zs_cfg.CLASS_NAME_VARIANT] = metrics

        self._print_metrics(metrics)

        # Optionally evaluate all variants
        if zs_cfg.get("EVALUATE_ALL_VARIANTS", False):
            print(f"\n{'='*60}")
            print("Evaluating all prompt variants")
            print(f"{'='*60}")

            dataset_config = DATASET_CLASS_NAMES.get(self.dataset_name, {})

            for variant_name in dataset_config.keys():
                if variant_name == zs_cfg.CLASS_NAME_VARIANT:
                    continue  # Already evaluated
                if variant_name == "template":
                    continue  # Skip template entries

                print(f"\nVariant: {variant_name}")

                # Update class names
                class_names = get_class_names(self.dataset_name, variant_name)
                self.model.update_class_names(class_names)

                for i, name in enumerate(class_names):
                    print(f"  {i}: {name[:50]}...")

                # Evaluate
                metrics = self.eval()
                all_results[variant_name] = metrics

                self._print_metrics(metrics)

        # Save all results
        self._save_results(all_results)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def eval(self):
        """Evaluate the zero-shot classifier."""
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
            group_details = {}

            for t in all_targets.unique():
                for b in all_bias.unique():
                    mask = (all_targets == t) & (all_bias == b)
                    if mask.sum() > 0:
                        group_acc = (
                            (all_preds[mask] == all_targets[mask]).float().mean().item()
                        )
                        group_count = mask.sum().item()
                        group_accs.append(group_acc)
                        group_key = f"t{t.item()}_b{b.item()}"
                        metrics[f"acc_{group_key}"] = group_acc
                        group_details[group_key] = {
                            "acc": group_acc,
                            "count": group_count,
                        }

            if group_accs:
                metrics["wg_ovr"] = min(group_accs)
                metrics["avg_group_acc"] = np.mean(group_accs)
                metrics["group_details"] = group_details

        return metrics

    def _print_metrics(self, metrics):
        """Print metrics in a nice format."""
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if "wg_ovr" in metrics:
            print(f"  Worst-group: {metrics['wg_ovr']:.4f}")
            print(f"  Avg group: {metrics['avg_group_acc']:.4f}")
            print(f"  detailed: {metrics['group_details']}")

        # Per-class accuracy
        for c in range(self.num_class):
            key = f"acc_class_{c}"
            if key in metrics:
                print(f"  Class {c}: {metrics[key]:.4f}")

    def _print_summary(self, all_results):
        """Print summary of all variants."""
        print(f"\n{'='*60}")
        print("Summary: All Variants")
        print(f"{'='*60}")
        print(f"{'Variant':<20} {'Accuracy':>10} {'Worst-Group':>12}")
        print("-" * 44)

        for variant, metrics in all_results.items():
            acc = metrics.get("accuracy", 0)
            wg = metrics.get("wg_ovr", 0)
            print(f"{variant:<20} {acc:>10.4f} {wg:>12.4f}")

    def _save_results(self, all_results):
        """Save results to file."""
        zs_cfg = self.cfg.MITIGATOR.ZERO_SHOT_VLM

        results = {
            "config": {
                "encoder_type": zs_cfg.ENCODER_TYPE,
                "model_name": zs_cfg.MODEL_NAME,
                "temperature": zs_cfg.TEMPERATURE,
                "dataset": self.dataset_name,
            },
            "variants": {},
        }

        for variant, metrics in all_results.items():
            # Remove non-serializable items
            clean_metrics = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float, str, list, dict))
            }
            results["variants"][variant] = clean_metrics

        save_path = os.path.join(self.log_path, "zero_shot_results.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")
