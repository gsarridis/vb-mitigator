"""
CLIP Knowledge Distillation Mitigator.

This module implements knowledge distillation between two OpenCLIP models:
- Teacher: Frozen OpenCLIP model (e.g., ViT-L-14)
- Student: Frozen OpenCLIP model (e.g., ViT-B-32) + learnable linear projection

The objective is to align the student's projected features with the teacher's
similarity structure using a cosine similarity-based KL divergence loss.

Pipeline:
    1. Extract teacher features (frozen)
    2. Extract student features (frozen)
    3. Project student features through learnable linear layer
    4. Minimize KL divergence between similarity matrices

Evaluation:
    - Zero-shot classification using projected student features
    - Per-subgroup accuracy analysis

Config:
    MITIGATOR:
      TYPE: "clip_distillation"
      CLIP_DISTILLATION:
        # Teacher model
        TEACHER_ARCH: "ViT-L-14"
        TEACHER_PRETRAINED: "openai"

        # Student model
        STUDENT_ARCH: "ViT-B-32"
        STUDENT_PRETRAINED: "openai"

        # Training
        STEPS: 10000
        BATCH_SIZE: 256
        LR: 1e-3
        WEIGHT_DECAY: 0.0

        # Projection layer
        PROJECTION_HIDDEN: null  # null = linear, int = MLP with hidden dim

        # Zero-shot evaluation
        CLASS_NAME_VARIANT: "default"
        TEMPERATURE: 100.0
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
from models.vlm_encoders import (
    OpenCLIPEncoder,
    get_class_names,
    DATASET_CLASS_NAMES,
)


# ============================================
# Image-Text Similarity KL Divergence Loss
# ============================================


def image_text_similarity_loss(
    student_image_features: torch.Tensor,
    teacher_image_features: torch.Tensor,
    teacher_text_features: torch.Tensor,
    eps: float = 1e-7,
    hard_sample_weight: float = 0.0,
):
    """
    Compute KL divergence between image-text similarity distributions.

    This loss encourages the student's image-text similarities to match
    the teacher's image-text similarities.

    Supports hard sample mining: samples with higher error get higher weights.

    Args:
        student_image_features: Projected student image features (B, D)
        teacher_image_features: Teacher image features (B, D)
        teacher_text_features: Teacher text features (num_texts, D) - can include bias texts
        eps: Small constant for numerical stability
        hard_sample_weight: Controls focus on hard samples (default=0.0)
            - 0.0: uniform weighting (standard mean)
            - 1.0: linear weighting by error
            - Higher values = more focus on hardest samples

    Returns:
        KL divergence loss (scalar)
    """
    # Normalize features
    student_image_features = F.normalize(student_image_features, dim=-1)
    teacher_image_features = F.normalize(teacher_image_features, dim=-1)
    teacher_text_features = F.normalize(teacher_text_features, dim=-1)

    # Compute image-text similarities (B, num_texts)
    student_similarity = torch.mm(student_image_features, teacher_text_features.T)
    teacher_similarity = torch.mm(teacher_image_features, teacher_text_features.T)

    # Scale to 0..1
    student_similarity = (student_similarity + 1.0) / 2.0
    teacher_similarity = (teacher_similarity + 1.0) / 2.0

    # Convert to probabilities (softmax over all texts)
    student_probs = student_similarity / (
        torch.sum(student_similarity, dim=1, keepdim=True) + eps
    )
    teacher_probs = teacher_similarity / (
        torch.sum(teacher_similarity, dim=1, keepdim=True) + eps
    )

    # Per-sample KL divergence (B,)
    per_sample_kl = torch.sum(
        teacher_probs * torch.log((teacher_probs + eps) / (student_probs + eps)), dim=1
    )

    if hard_sample_weight > 0:
        # Compute sample weights based on error
        with torch.no_grad():
            error_max = per_sample_kl.max()
            if error_max > eps:
                error_normalized = per_sample_kl / error_max
            else:
                error_normalized = per_sample_kl

            sample_weights = 1.0 + hard_sample_weight * error_normalized
            sample_weights = sample_weights / sample_weights.mean()

        loss = torch.mean(sample_weights * per_sample_kl)
    else:
        loss = torch.mean(per_sample_kl)

    return loss


def image_text_similarity_loss_with_rarity(
    student_image_features: torch.Tensor,
    teacher_image_features: torch.Tensor,
    teacher_text_features: torch.Tensor,
    text_group_sizes: List[int] = None,
    eps: float = 1e-7,
    rarity_weight: float = 2.0,
    return_debug_info: bool = False,
):
    """
    Compute KL divergence with rarity-based sample weighting.

    Uses extended text set (target + bias texts) and weights samples
    inversely by how common their similarity pattern is within the batch.

    Rarity is computed using continuous similarity vectors:
    - For each sample, compute its average cosine similarity to all other samples'
      similarity vectors in the batch
    - High avg similarity = common pattern = lower weight
    - Low avg similarity = rare/unique pattern = higher weight

    Args:
        student_image_features: Projected student image features (B, D)
        teacher_image_features: Teacher image features (B, D)
        teacher_text_features: Extended text features (num_texts, D)
            e.g., [target1, target2, bias1_a, bias1_b, bias2_a, bias2_b, ...]
        text_group_sizes: List of sizes for each text group [num_targets, num_bias1, num_bias2, ...]
            Used for computing subgroup assignments. If None, no subgroup info returned.
        eps: Small constant for numerical stability
        rarity_weight: How much to weight by rarity (0=uniform, higher=more focus on rare)
        return_debug_info: If True, return (loss, debug_dict) with weights and subgroups

    Returns:
        If return_debug_info=False: loss (scalar)
        If return_debug_info=True: (loss, debug_info) where debug_info contains:
            - sample_weights: (B,) tensor of weights
            - subgroup_ids: (B,) tensor of subgroup assignments
            - teacher_similarity: (B, num_texts) raw similarities
    """
    B = student_image_features.shape[0]

    # Normalize features
    student_image_features = F.normalize(student_image_features, dim=-1)
    teacher_image_features = F.normalize(teacher_image_features, dim=-1)
    teacher_text_features = F.normalize(teacher_text_features, dim=-1)

    # Compute image-text similarities (B, num_texts)
    student_similarity = torch.mm(student_image_features, teacher_text_features.T)
    teacher_similarity = torch.mm(teacher_image_features, teacher_text_features.T)

    # Scale to 0..1
    student_similarity = (student_similarity + 1.0) / 2.0
    teacher_similarity = (teacher_similarity + 1.0) / 2.0

    # Convert to probabilities (distribution over all texts)
    student_probs = student_similarity / (
        torch.sum(student_similarity, dim=1, keepdim=True) + eps
    )
    teacher_probs = teacher_similarity / (
        torch.sum(teacher_similarity, dim=1, keepdim=True) + eps
    )

    # Per-sample KL divergence (B,)
    per_sample_kl = torch.sum(
        teacher_probs * torch.log((teacher_probs + eps) / (student_probs + eps)), dim=1
    )

    # Compute rarity-based weights using continuous similarity vectors
    if rarity_weight > 0 and B > 1:

        with torch.no_grad():
            # Normalize teacher similarity vectors for cosine similarity
            teacher_sim_normalized = F.normalize(
                teacher_similarity, dim=-1
            )  # (B, num_texts)

            # Compute pairwise similarity between samples' similarity patterns
            # pattern_sim[i,j] = cosine similarity between sample i and sample j's text similarity vectors
            pattern_sim = torch.mm(
                teacher_sim_normalized, teacher_sim_normalized.T
            )  # (B, B)

            # Average similarity to OTHER samples (exclude self-similarity on diagonal)
            mask = 1.0 - torch.eye(B, device=pattern_sim.device)
            avg_similarity = (pattern_sim * mask).sum(dim=1) / (B - 1)  # (B,)
            avg_similarity = (avg_similarity - torch.min(avg_similarity)) / (
                torch.max(avg_similarity) - torch.min(avg_similarity)
            )
            # print(avg_similarity)
            # Rarity = inverse of average similarity to others
            # High avg_similarity = common pattern = low rarity
            # Low avg_similarity = unique pattern = high rarity
            rarity = 1.0 - avg_similarity  # / (avg_similarity + eps)

            # Normalize rarity to have mean=1
            # rarity = rarity / (rarity.mean() + eps)
            # print(rarity)

            # Compute sample weights
            # rarity_weight controls how much to amplify rare samples
            # When rarity_weight=0: uniform weights
            # When rarity_weight=1: weights proportional to rarity
            sample_weights = 1.0 + rarity_weight * (rarity)
            # print(sample_weights)
            # Clamp to prevent negative or extreme weights
            sample_weights = torch.clamp(sample_weights, min=0.1, max=10.0)

            # Normalize weights to mean=1 (keeps loss scale consistent)
            sample_weights = sample_weights / (sample_weights.mean() + eps)

        loss = torch.mean(sample_weights * per_sample_kl)
    else:
        sample_weights = torch.ones(B, device=student_image_features.device)
        loss = torch.mean(per_sample_kl)

    if return_debug_info:
        debug_info = {
            "sample_weights": sample_weights.detach(),
            "teacher_similarity": teacher_similarity.detach(),
        }

        # Compute subgroup IDs based on argmax within each text group
        if text_group_sizes is not None:
            with torch.no_grad():
                subgroup_ids = []
                start_idx = 0
                for group_size in text_group_sizes:
                    end_idx = start_idx + group_size
                    group_sims = teacher_similarity[
                        :, start_idx:end_idx
                    ]  # (B, group_size)
                    group_argmax = group_sims.argmax(dim=1)  # (B,)
                    subgroup_ids.append(group_argmax)
                    start_idx = end_idx

                # Combine into single subgroup ID
                # subgroup_id = target_id * (max_bias1 * max_bias2) + bias1_id * max_bias2 + bias2_id
                combined_id = torch.zeros(
                    B, dtype=torch.long, device=teacher_similarity.device
                )
                multiplier = 1
                for i in range(len(subgroup_ids) - 1, -1, -1):
                    combined_id += subgroup_ids[i] * multiplier
                    multiplier *= text_group_sizes[i]

                debug_info["subgroup_ids"] = combined_id
                debug_info["subgroup_components"] = subgroup_ids  # List of (B,) tensors

        return loss, debug_info

    return loss * 30000


def cosine_similarity_loss(output_net, target_net, eps=1e-7):
    """
    Compute KL divergence between image-image similarity distributions.
    (Kept for backward compatibility)

    Args:
        output_net: Student features (B, D)
        target_net: Teacher features (B, D')
        eps: Small constant for numerical stability

    Returns:
        KL divergence loss (scalar)
    """
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net**2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0  # Handle NaN

    target_net_norm = torch.sqrt(torch.sum(target_net**2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0  # Handle NaN

    # Calculate the cosine similarity matrices
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities (row-wise softmax-like normalization)
    model_similarity = model_similarity / torch.sum(
        model_similarity, dim=1, keepdim=True
    )
    target_similarity = target_similarity / torch.sum(
        target_similarity, dim=1, keepdim=True
    )

    # Calculate the KL-divergence
    loss = torch.mean(
        target_similarity
        * torch.log((target_similarity + eps) / (model_similarity + eps))
    )

    return loss


# ============================================
# Projection Layer
# ============================================


class ProjectionHead(nn.Module):
    """
    Learnable projection from student space to teacher space.

    Can be linear (hidden_dim=None) or MLP (hidden_dim=int).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if hidden_dim is None:
            # Linear projection
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            # MLP projection
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


# ============================================
# Distillation Model
# ============================================


class CLIPDistillationModel(nn.Module):
    """
    Model for CLIP knowledge distillation.

    Architecture:
        - Teacher encoder (frozen): images -> teacher_features
        - Student encoder (frozen): images -> student_features
        - Projection head (trainable): student_features -> projected_features

    The projection aligns student features with teacher's similarity structure.
    """

    def __init__(
        self,
        teacher_encoder: OpenCLIPEncoder,
        student_encoder: OpenCLIPEncoder,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        self.teacher_encoder = teacher_encoder
        self.student_encoder = student_encoder

        # Freeze both encoders
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.student_encoder.parameters():
            param.requires_grad = False

        # Projection head: student_dim -> teacher_dim
        self.projection = ProjectionHead(
            input_dim=student_encoder.embed_dim,
            output_dim=teacher_encoder.embed_dim,
            hidden_dim=hidden_dim,
        )

        print(f"CLIPDistillationModel initialized:")
        print(f"  Teacher: {teacher_encoder.arch} (dim={teacher_encoder.embed_dim})")
        print(f"  Student: {student_encoder.arch} (dim={student_encoder.embed_dim})")
        print(
            f"  Projection: {student_encoder.embed_dim} -> {teacher_encoder.embed_dim}"
        )
        if hidden_dim:
            print(f"  Hidden dim: {hidden_dim}")

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Input images (B, C, H, W)

        Returns:
            projected_student: Projected student features (B, teacher_dim)
            teacher_features: Teacher features (B, teacher_dim)
        """
        with torch.no_grad():
            teacher_features = self.teacher_encoder.encode_image(images)
            student_features = self.student_encoder.encode_image(images)

        projected_student = self.projection(student_features)

        return projected_student, teacher_features

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using student + projection (for inference)."""
        with torch.no_grad():
            student_features = self.student_encoder.encode_image(images)
        projected = self.projection(student_features)
        return projected

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text using teacher's text encoder."""
        return self.teacher_encoder.encode_text(texts)


# ============================================
# Zero-Shot Classifier with Distilled Model
# ============================================


class DistilledZeroShotClassifier(nn.Module):
    """
    Zero-shot classifier using the distilled model.

    Uses:
    - Student encoder + projection for image features
    - Teacher's text encoder for text features
    """

    def __init__(
        self,
        distillation_model: CLIPDistillationModel,
        class_names: List[str],
        temperature: float = 100.0,
    ):
        super().__init__()

        self.model = distillation_model
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.temperature = temperature

        # Pre-compute text features using teacher's text encoder
        with torch.no_grad():
            text_features = distillation_model.encode_text(class_names)
            text_features = F.normalize(text_features, dim=-1)

        self.register_buffer("text_features", text_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Input images (B, C, H, W)

        Returns:
            logits: Similarity scores (B, num_classes)
        """
        # Get projected student features
        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        # Compute similarity
        logits = self.temperature * (image_features @ self.text_features.T)

        return logits

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(images)
        return logits.argmax(dim=1)


# ============================================
# Trainer
# ============================================


class CLIPDistillationTrainer(BaseTrainer):
    """
    Trainer for CLIP knowledge distillation.

    Pipeline:
    1. Load teacher and student OpenCLIP models
    2. Train projection layer to align similarity structures
    3. Evaluate using zero-shot classification

    Configuration:
        MITIGATOR:
          TYPE: "clip_distillation"
          CLIP_DISTILLATION:
            TEACHER_ARCH: "ViT-L-14"
            TEACHER_PRETRAINED: "openai"
            STUDENT_ARCH: "ViT-B-32"
            STUDENT_PRETRAINED: "openai"
            STEPS: 10000
            BATCH_SIZE: 256
            LR: 1e-3
            PROJECTION_HIDDEN: null
            CLASS_NAME_VARIANT: "default"
            TEMPERATURE: 100.0
    """

    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """Setup teacher, student, and projection."""

        dist_cfg = self.cfg.MITIGATOR.CLIP_DISTILLATION

        print(f"\n{'='*60}")
        print("Setting up CLIP Distillation")
        print(f"{'='*60}")

        # Create teacher encoder
        print(f"\nLoading Teacher: {dist_cfg.TEACHER_ARCH}")
        self.teacher_encoder = OpenCLIPEncoder(
            arch=dist_cfg.TEACHER_ARCH,
            pretrained=dist_cfg.TEACHER_PRETRAINED,
            device=self.device,
        )
        print(f"  Embed dim: {self.teacher_encoder.embed_dim}")
        print(f"  Image size: {self.teacher_encoder.image_size}")

        # Create student encoder
        print(f"\nLoading Student: {dist_cfg.STUDENT_ARCH}")
        self.student_encoder = OpenCLIPEncoder(
            arch=dist_cfg.STUDENT_ARCH,
            pretrained=dist_cfg.STUDENT_PRETRAINED,
            device=self.device,
        )
        print(f"  Embed dim: {self.student_encoder.embed_dim}")
        print(f"  Image size: {self.student_encoder.image_size}")

        # Create distillation model
        hidden_dim = dist_cfg.get("PROJECTION_HIDDEN", None)
        self.model = CLIPDistillationModel(
            teacher_encoder=self.teacher_encoder,
            student_encoder=self.student_encoder,
            hidden_dim=hidden_dim,
        )
        self.model.to(self.device)

        # Store config for later
        self.dist_cfg = dist_cfg

        # Rebuild dataloaders with teacher's preprocessing
        # (teacher usually has the more demanding transform)
        self._rebuild_dataloaders_with_transform()

    def _rebuild_dataloaders_with_transform(self):
        """Create separate dataloaders for teacher and student with their own transforms."""
        print(f"\nCreating dataloaders with model-specific preprocessing...")

        # Get transforms
        teacher_transform = self.teacher_encoder.get_transform()
        student_transform = self.student_encoder.get_transform()

        # Get config
        batch_size = self.dist_cfg.get("BATCH_SIZE", 256)
        num_workers = (
            self.cfg.DATALOADER.NUM_WORKERS if hasattr(self.cfg, "DATALOADER") else 4
        )

        # Get base datasets (without transform) and create copies with different transforms
        base_train_dataset = self.dataloaders["val"].dataset
        base_test_dataset = self.dataloaders["test"].dataset

        # Create teacher dataloaders
        train_dataset_teacher = self._clone_dataset_with_transform(
            base_train_dataset, teacher_transform
        )
        test_dataset_teacher = self._clone_dataset_with_transform(
            base_test_dataset, teacher_transform
        )

        self.dataloaders["train_teacher"] = DataLoader(
            train_dataset_teacher,
            batch_size=batch_size,
            shuffle=False,  # Keep order consistent
            num_workers=num_workers,
            pin_memory=True,
        )

        self.dataloaders["test_teacher"] = DataLoader(
            test_dataset_teacher,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Create student dataloaders
        train_dataset_student = self._clone_dataset_with_transform(
            base_train_dataset, student_transform
        )
        test_dataset_student = self._clone_dataset_with_transform(
            base_test_dataset, student_transform
        )

        self.dataloaders["train_student"] = DataLoader(
            train_dataset_student,
            batch_size=batch_size,
            shuffle=False,  # Keep order consistent
            num_workers=num_workers,
            pin_memory=True,
        )

        self.dataloaders["test_student"] = DataLoader(
            test_dataset_student,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        print(f"  Train samples: {len(base_train_dataset)}")
        print(f"  Test samples: {len(base_test_dataset)}")
        print(f"  Batch size: {batch_size}")
        print(
            f"  Teacher transform: {self.teacher_encoder.arch} ({self.teacher_encoder.image_size})"
        )
        print(
            f"  Student transform: {self.student_encoder.arch} ({self.student_encoder.image_size})"
        )

    def _clone_dataset_with_transform(self, dataset, transform):
        """Create a copy of dataset with a different transform."""
        import copy

        # Try to create a shallow copy and update transform
        new_dataset = copy.copy(dataset)
        new_dataset.transform = transform

        return new_dataset

    def _setup_optimizer(self):
        """Setup optimizer for projection layer only."""
        dist_cfg = self.cfg.MITIGATOR.CLIP_DISTILLATION

        # Only optimize projection parameters
        self.optimizer = torch.optim.AdamW(
            self.model.projection.parameters(),
            lr=dist_cfg.LR,
            weight_decay=dist_cfg.get("WEIGHT_DECAY", 0.0),
        )

        print(f"\nOptimizer: AdamW")
        print(f"  LR: {dist_cfg.LR}")
        print(
            f"  Trainable params: {sum(p.numel() for p in self.model.projection.parameters())}"
        )

    def _extract_features(self, split="train", desc="Extracting"):
        """
        Extract teacher and student features using their respective transforms.

        Args:
            split: "train" or "test"
            desc: Description for progress bar
        """
        teacher_loader = self.dataloaders[f"{split}_teacher"]
        student_loader = self.dataloaders[f"{split}_student"]

        all_teacher_features = []
        all_student_features = []
        all_targets = []
        all_biases = defaultdict(list)

        self.teacher_encoder.eval()
        self.student_encoder.eval()

        # Extract teacher features
        print(f"  Extracting teacher features ({self.teacher_encoder.arch})...")
        with torch.no_grad():
            for batch in tqdm(teacher_loader, desc=f"{desc} (teacher)"):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]

                teacher_feat = self.teacher_encoder.encode_image(inputs)
                all_teacher_features.append(teacher_feat.cpu())
                all_targets.append(targets)

                for b in self.biases:
                    if b in batch:
                        all_biases[b].append(batch[b])

        # Extract student features
        print(f"  Extracting student features ({self.student_encoder.arch})...")
        with torch.no_grad():
            for batch in tqdm(student_loader, desc=f"{desc} (student)"):
                inputs = batch["inputs"].to(self.device)
                student_feat = self.student_encoder.encode_image(inputs)
                all_student_features.append(student_feat.cpu())

        teacher_features = torch.cat(all_teacher_features, dim=0)
        student_features = torch.cat(all_student_features, dim=0)
        targets = torch.cat(all_targets, dim=0)
        biases = {k: torch.cat(v, dim=0) for k, v in all_biases.items()}

        return teacher_features, student_features, targets, biases

    def _train_projection(
        self,
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
    ):
        """Train the projection layer using image-text similarity matching with rarity weighting."""

        print(f"\n{'='*60}")
        print("Training Projection Layer (Image-Text Similarity + Rarity)")
        print(f"{'='*60}")

        dist_cfg = self.dist_cfg

        steps = dist_cfg.STEPS
        batch_size = dist_cfg.BATCH_SIZE

        # Get target class names
        dataset_name = self.cfg.DATASET.TYPE.lower()
        variant = dist_cfg.get("CLASS_NAME_VARIANT", "default")
        target_texts = get_class_names(dataset_name, variant)

        # Get bias-related texts from config (user-provided)
        # Format: list of lists, where each inner list is a group of related texts
        # e.g., [["pickup truck", "sedan"], ["forest background", "city street"], ["animals", "traffic signs"]]
        bias_text_groups = dist_cfg.get("BIAS_TEXTS", [])

        # Build full text list and track group sizes
        all_texts = list(target_texts)
        text_group_sizes = [len(target_texts)]  # First group is target
        text_group_names = ["target"] + [
            f"bias{i+1}" for i in range(len(bias_text_groups))
        ]

        for group in bias_text_groups:
            all_texts.extend(group)
            text_group_sizes.append(len(group))

        print(f"\n  Texts for similarity matching:")
        print(f"    Target texts ({text_group_sizes[0]}): {target_texts}")
        for i, group in enumerate(bias_text_groups):
            print(f"    Bias group {i+1} ({text_group_sizes[i+1]}): {group}")
        print(f"  Total texts: {len(all_texts)}")

        # Build subgroup labels for logging
        # Generate all combinations of argmax indices
        from itertools import product

        subgroup_labels = {}
        all_combinations = list(product(*[range(s) for s in text_group_sizes]))
        for combo in all_combinations:
            # Compute combined ID the same way as in the loss function
            combined_id = 0
            multiplier = 1
            for i in range(len(combo) - 1, -1, -1):
                combined_id += combo[i] * multiplier
                multiplier *= text_group_sizes[i]

            # Create label
            parts = []
            idx = 0
            parts.append(f"t{combo[0]}")  # target
            for i, group in enumerate(bias_text_groups):
                parts.append(f"b{i+1}={combo[i+1]}")
            subgroup_labels[combined_id] = "_".join(parts)

        # Compute teacher text features for all texts
        with torch.no_grad():
            teacher_text_features = self.teacher_encoder.encode_text(all_texts)
            teacher_text_features = F.normalize(teacher_text_features, dim=-1)

        print(f"  Teacher text features: {teacher_text_features.shape}")

        # Get rarity weight from config
        rarity_weight = dist_cfg.get("RARITY_WEIGHT", 0.0)
        print(f"  Rarity weighting: {rarity_weight}")

        # Debug logging interval
        debug_interval = dist_cfg.get("DEBUG_INTERVAL", 10500)

        # Create dataset
        dataset = TensorDataset(student_features, teacher_features)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Training loop
        self.model.projection.train()

        step = 0
        epoch = 0
        losses = []

        pbar = tqdm(total=steps, desc="Training projection")

        while step < steps:
            epoch += 1

            for student_batch, teacher_batch in dataloader:
                if step >= steps:
                    break

                student_batch = student_batch.to(self.device)
                teacher_batch = teacher_batch.to(self.device)

                # Forward pass: project student features
                projected = self.model.projection(student_batch)

                # Check if we should log debug info this step
                should_debug = (step % debug_interval == 0) and rarity_weight > 0

                # Compute loss with rarity-based weighting (continuous similarity vectors)
                if should_debug:
                    loss, debug_info = image_text_similarity_loss_with_rarity(
                        student_image_features=projected,
                        teacher_image_features=teacher_batch,
                        teacher_text_features=teacher_text_features,
                        text_group_sizes=text_group_sizes,
                        rarity_weight=rarity_weight,
                        return_debug_info=True,
                    )

                    # Log subgroup weights
                    self._log_subgroup_weights(
                        step=step,
                        sample_weights=debug_info["sample_weights"],
                        subgroup_ids=debug_info["subgroup_ids"],
                        subgroup_labels=subgroup_labels,
                    )
                else:
                    loss = image_text_similarity_loss_with_rarity(
                        student_image_features=projected,
                        teacher_image_features=teacher_batch,
                        teacher_text_features=teacher_text_features,
                        rarity_weight=rarity_weight,
                    )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                step += 1
                pbar.update(1)

                if step % 100 == 0:
                    avg_loss = np.mean(losses[-100:])
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        pbar.close()

        print(f"\nTraining complete!")
        print(f"  Final loss: {np.mean(losses[-100:]):.4f}")

        # Save projection
        self._save_projection()

        return losses

    def _log_subgroup_weights(
        self,
        step: int,
        sample_weights: torch.Tensor,
        subgroup_ids: torch.Tensor,
        subgroup_labels: Dict[int, str],
    ):
        """Log mean weights for each subgroup in the batch."""

        unique_subgroups = subgroup_ids.unique()

        print(f"\n  [Step {step}] Subgroup weights in batch:")

        subgroup_stats = []
        for sg_id in unique_subgroups:
            sg_id_int = sg_id.item()
            mask = subgroup_ids == sg_id
            count = mask.sum().item()
            mean_weight = sample_weights[mask].mean().item()
            label = subgroup_labels.get(sg_id_int, f"sg_{sg_id_int}")
            subgroup_stats.append((label, count, mean_weight))

        # Sort by mean weight (descending) to see rare groups first
        subgroup_stats.sort(key=lambda x: x[2], reverse=True)

        for label, count, mean_weight in subgroup_stats:
            bar = "█" * int(mean_weight * 10)
            print(f"    {label:<20} n={count:>4}  weight={mean_weight:.3f}  {bar}")
        # print(f"  Final loss: {np.mean(losses[-100:]):.4f}")

        # Save projection
        self._save_projection()

        return  # losses

    def _save_projection(self):
        """Save the trained projection layer."""
        save_dir = os.path.join(self.log_path, "clip_distillation")
        os.makedirs(save_dir, exist_ok=True)

        # Save projection weights
        proj_path = os.path.join(save_dir, "projection.pt")
        torch.save(self.model.projection.state_dict(), proj_path)

        # Save config
        config = {
            "teacher_arch": self.dist_cfg.TEACHER_ARCH,
            "teacher_pretrained": self.dist_cfg.TEACHER_PRETRAINED,
            "student_arch": self.dist_cfg.STUDENT_ARCH,
            "student_pretrained": self.dist_cfg.STUDENT_PRETRAINED,
            "teacher_dim": self.teacher_encoder.embed_dim,
            "student_dim": self.student_encoder.embed_dim,
            "projection_hidden": self.dist_cfg.get("PROJECTION_HIDDEN", None),
        }

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"  Saved projection to {save_dir}")

    def _load_projection(self, path: str):
        """Load a trained projection layer."""
        config_path = os.path.join(os.path.dirname(path), "config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Recreate projection head
        self.model.projection = ProjectionHead(
            input_dim=config["student_dim"],
            output_dim=config["teacher_dim"],
            hidden_dim=config.get("projection_hidden"),
        ).to(self.device)

        self.model.projection.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        print(f"  Loaded projection from {path}")

    def _evaluate_zero_shot(
        self,
        test_features: torch.Tensor,
        test_targets: torch.Tensor,
        test_biases: Dict[str, torch.Tensor],
    ) -> dict:
        """Evaluate using zero-shot classification."""

        print(f"\n{'='*60}")
        print("Zero-Shot Evaluation")
        print(f"{'='*60}")

        dist_cfg = self.dist_cfg

        # Get class names
        dataset_name = self.cfg.DATASET.TYPE.lower()
        variant = dist_cfg.get("CLASS_NAME_VARIANT", "default")
        class_names = get_class_names(dataset_name, variant)

        print(f"Dataset: {dataset_name}")
        print(f"Class names ({variant}):")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")

        # Create zero-shot classifier
        classifier = DistilledZeroShotClassifier(
            distillation_model=self.model,
            class_names=class_names,
            temperature=dist_cfg.get("TEMPERATURE", 100.0),
        )
        classifier.to(self.device)
        classifier.eval()

        # Get predictions
        self.model.projection.eval()

        all_preds = []
        batch_size = 256

        with torch.no_grad():
            for i in range(0, len(test_features), batch_size):
                batch = test_features[i : i + batch_size].to(self.device)

                # Project student features
                projected = self.model.projection(batch)
                projected = F.normalize(projected, dim=-1)

                # Compute similarity with text features
                logits = dist_cfg.get("TEMPERATURE", 100.0) * (
                    projected @ classifier.text_features.T
                )
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())

        all_preds = torch.cat(all_preds)

        # Compute metrics
        correct = all_preds == test_targets
        accuracy = correct.float().mean().item()

        metrics = {"accuracy": accuracy}

        # Per-class accuracy
        for c in range(self.num_class):
            mask = test_targets == c
            if mask.sum() > 0:
                class_acc = correct[mask].float().mean().item()
                metrics[f"acc_class_{c}"] = class_acc

        # Build subgroups
        bias_names = list(test_biases.keys())
        if bias_names:
            # Build subgroup keys
            subgroup_keys = []
            for i in range(len(test_targets)):
                key = [f"t={test_targets[i].item()}"]
                for b_name in bias_names:
                    key.append(f"{b_name}={test_biases[b_name][i].item()}")
                subgroup_keys.append(tuple(key))

            unique_subgroups = sorted(set(subgroup_keys))

            # Per-subgroup accuracy
            subgroup_accs = []
            print(f"\nPer-subgroup accuracy:")

            for sg in unique_subgroups:
                mask = torch.tensor([sk == sg for sk in subgroup_keys])
                if mask.sum() > 0:
                    acc = correct[mask].float().mean().item()
                    count = mask.sum().item()
                    subgroup_accs.append(acc)
                    metrics[f"acc_{sg}"] = acc
                    sg_label = ", ".join([s.split("=")[1] for s in sg])
                    print(f"  {sg_label}: {acc:.4f} (n={count})")

            if subgroup_accs:
                metrics["worst_group_accuracy"] = min(subgroup_accs)
                metrics["best_group_accuracy"] = max(subgroup_accs)
                metrics["accuracy_gap"] = max(subgroup_accs) - min(subgroup_accs)
                metrics["avg_group_accuracy"] = np.mean(subgroup_accs)

                print(f"\n  Overall: {accuracy:.4f}")
                print(f"  Worst-group: {min(subgroup_accs):.4f}")
                print(f"  Best-group:  {max(subgroup_accs):.4f}")
                print(f"  Gap:         {max(subgroup_accs) - min(subgroup_accs):.4f}")

        return metrics

    def _evaluate_baseline(
        self,
        test_features_student: torch.Tensor,
        test_features_teacher: torch.Tensor,
        test_targets: torch.Tensor,
        test_biases: Dict[str, torch.Tensor],
    ) -> dict:
        """Evaluate baseline zero-shot (without distillation)."""

        print(f"\n{'='*60}")
        print("Baseline Evaluation (Teacher vs Student)")
        print(f"{'='*60}")

        dist_cfg = self.dist_cfg

        # Get class names
        dataset_name = self.cfg.DATASET.TYPE.lower()
        variant = dist_cfg.get("CLASS_NAME_VARIANT", "default")
        class_names = get_class_names(dataset_name, variant)

        results = {}

        for name, encoder, features in [
            ("teacher", self.teacher_encoder, test_features_teacher),
            ("student", self.student_encoder, test_features_student),
        ]:
            # Get text features
            with torch.no_grad():
                text_features = encoder.encode_text(class_names)
                text_features = F.normalize(text_features, dim=-1)

            # Get predictions
            all_preds = []
            batch_size = 256

            with torch.no_grad():
                for i in range(0, len(features), batch_size):
                    batch = features[i : i + batch_size].to(self.device)
                    batch = F.normalize(batch, dim=-1)

                    logits = 100.0 * (batch @ text_features.T)
                    preds = logits.argmax(dim=1)
                    all_preds.append(preds.cpu())

            all_preds = torch.cat(all_preds)

            # Compute metrics
            correct = all_preds == test_targets
            accuracy = correct.float().mean().item()

            # Per-subgroup
            bias_names = list(test_biases.keys())
            worst_group = accuracy
            gap = 0.0

            if bias_names:
                subgroup_keys = []
                for i in range(len(test_targets)):
                    key = [f"t={test_targets[i].item()}"]
                    for b_name in bias_names:
                        key.append(f"{b_name}={test_biases[b_name][i].item()}")
                    subgroup_keys.append(tuple(key))

                unique_subgroups = sorted(set(subgroup_keys))
                subgroup_accs = []

                for sg in unique_subgroups:
                    mask = torch.tensor([sk == sg for sk in subgroup_keys])
                    if mask.sum() > 0:
                        acc = correct[mask].float().mean().item()
                        subgroup_accs.append(acc)

                if subgroup_accs:
                    worst_group = min(subgroup_accs)
                    gap = max(subgroup_accs) - min(subgroup_accs)

            print(f"\n{name.upper()}:")
            print(f"  Overall: {accuracy:.4f}")
            print(f"  Worst-group: {worst_group:.4f}")
            print(f"  Gap: {gap:.4f}")

            results[name] = {
                "accuracy": accuracy,
                "worst_group_accuracy": worst_group,
                "accuracy_gap": gap,
            }

        return results

    def train(self):
        """Main training pipeline."""

        print(f"\n{'='*60}")
        print("CLIP Distillation Training Pipeline")
        print(f"{'='*60}")

        # Step 1: Extract features
        print("\nStep 1: Extracting train features...")
        teacher_features, student_features, targets, biases = self._extract_features(
            split="train", desc="Train"
        )
        print(f"  Teacher features: {teacher_features.shape}")
        print(f"  Student features: {student_features.shape}")

        # Step 2: Train projection
        print("\nStep 2: Training projection...")
        losses = self._train_projection(teacher_features, student_features)

        # Step 3: Extract test features
        print("\nStep 3: Extracting test features...")
        test_teacher, test_student, test_targets, test_biases = self._extract_features(
            split="test", desc="Test"
        )

        # Step 4: Baseline evaluation
        print("\nStep 4: Baseline evaluation...")
        baseline_results = self._evaluate_baseline(
            test_student, test_teacher, test_targets, test_biases
        )

        # Step 5: Distilled model evaluation
        print("\nStep 5: Distilled model evaluation...")
        distilled_results = self._evaluate_zero_shot(
            test_student, test_targets, test_biases
        )

        # Save all results
        all_results = {
            "baseline": baseline_results,
            "distilled": distilled_results,
            "training": {
                "final_loss": float(np.mean(losses[-100:])),
                "steps": len(losses),
            },
        }

        results_path = os.path.join(self.log_path, "clip_distillation", "results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Overall':>10} {'Worst-Group':>12} {'Gap':>10}")
        print("-" * 54)
        print(
            f"{'Teacher':<20} {baseline_results['teacher']['accuracy']:>10.4f} "
            f"{baseline_results['teacher']['worst_group_accuracy']:>12.4f} "
            f"{baseline_results['teacher']['accuracy_gap']:>10.4f}"
        )
        print(
            f"{'Student':<20} {baseline_results['student']['accuracy']:>10.4f} "
            f"{baseline_results['student']['worst_group_accuracy']:>12.4f} "
            f"{baseline_results['student']['accuracy_gap']:>10.4f}"
        )
        print(
            f"{'Distilled':<20} {distilled_results['accuracy']:>10.4f} "
            f"{distilled_results.get('worst_group_accuracy', 0):>12.4f} "
            f"{distilled_results.get('accuracy_gap', 0):>10.4f}"
        )

        print(f"\nResults saved to {results_path}")

        return all_results

    def eval(self):
        """Evaluation mode - load projection and evaluate."""

        # Try to load projection
        proj_path = os.path.join(self.log_path, "clip_distillation", "projection.pt")

        if os.path.exists(proj_path):
            self._load_projection(proj_path)
        else:
            # Check if path provided in config
            proj_path = self.dist_cfg.get("PROJECTION_PATH", "")
            if proj_path and os.path.exists(proj_path):
                self._load_projection(proj_path)
            else:
                print("No projection found. Running train() first.")
                return self.train()

        # Extract test features
        test_teacher, test_student, test_targets, test_biases = self._extract_features(
            split="test", desc="Test"
        )

        # Evaluate
        baseline_results = self._evaluate_baseline(
            test_student, test_teacher, test_targets, test_biases
        )

        distilled_results = self._evaluate_zero_shot(
            test_student, test_targets, test_biases
        )

        return {
            "baseline": baseline_results,
            "distilled": distilled_results,
        }
