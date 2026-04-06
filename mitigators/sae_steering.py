"""
SAE-Steered ERM Trainer for VB-Mitigator.

This module implements a training approach that:
1. Uses OpenCLIP (or other) as a frozen pretrained backbone
2. Passes features through a pretrained SAE
3. Suppresses/deactivates polysemantic neurons (neurons where top-k activating images
   are NOT all from the same target class)
4. Only keeps monosemantic neurons (high class purity)
5. Decodes back to feature space (or uses sparse features directly)
6. Trains a classification head

The intuition is that polysemantic neurons encode mixed/confusing features that might
include spurious correlations or bias-related information, while monosemantic neurons
encode clean, class-relevant features.

Based on steering techniques from:
- "Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models" (Pach et al., 2025)
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
from tools.utils import log_msg, save_checkpoint, load_checkpoint

# Import dictionary learning components
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dictionary_learning"))

from dictionary_learning import AutoEncoder


class SAESteeringWrapper(nn.Module):
    """
    Wrapper around SAE that applies neuron-level steering/masking.

    This module:
    1. Encodes features to sparse SAE latents
    2. Applies a mask/weight to suppress polysemantic neurons
    3. Decodes back to original feature space

    Args:
        sae: Trained Sparse Autoencoder
        neuron_weights: Tensor of shape (dict_size,) with weights for each neuron
                       - 1.0 = keep neuron as-is
                       - 0.0 = completely suppress neuron
                       - Values in between = partial suppression
                       - Values > 1.0 = amplify neuron (not typical)
        use_decode: Whether to decode back to original space or use sparse features
    """

    def __init__(self, sae, neuron_weights, use_decode=True):
        super().__init__()
        self.sae = sae
        self.register_buffer("neuron_weights", neuron_weights)
        self.use_decode = use_decode

        # Freeze SAE parameters
        for param in self.sae.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass with neuron steering.

        Args:
            x: Input features of shape (B, feature_dim)

        Returns:
            If use_decode: Reconstructed features of shape (B, feature_dim)
            Else: Masked sparse features of shape (B, dict_size)
        """
        # Encode to sparse latents
        latents = self.sae.encode(x)  # (B, dict_size)

        # Apply neuron weights (element-wise multiplication)
        masked_latents = latents * self.neuron_weights.unsqueeze(0)

        if self.use_decode:
            # Decode back to original feature space
            return self.sae.decode(masked_latents)
        else:
            return masked_latents

    def encode(self, x):
        """Encode and apply mask."""
        latents = self.sae.encode(x)
        return latents * self.neuron_weights.unsqueeze(0)

    def get_num_active_neurons(self):
        """Return number of neurons that are not fully suppressed."""
        return (self.neuron_weights > 0).sum().item()

    def get_num_suppressed_neurons(self):
        """Return number of suppressed neurons."""
        return (self.neuron_weights == 0).sum().item()


class OpenCLIPWithSAESteering(nn.Module):
    """
    OpenCLIP backbone with SAE steering applied to features.

    Architecture:
        Input Image -> OpenCLIP (frozen) -> Features -> SAE Steering -> Classification Head
    """

    def __init__(self, clip_model, sae_wrapper, num_classes, use_sae_features=False):
        super().__init__()
        self.clip_model = clip_model
        self.sae_wrapper = sae_wrapper
        self.use_sae_features = use_sae_features

        # Determine feature dimension for classifier
        if use_sae_features:
            # Use sparse features directly (larger dimension)
            self.feature_dim = sae_wrapper.sae.dict_size
        else:
            # Use decoded features (original dimension)
            self.feature_dim = sae_wrapper.sae.activation_dim

        # Classification head
        self.fc = nn.Linear(self.feature_dim, num_classes)

        # Freeze CLIP backbone
        for param in self.clip_model.parameters():
            param.requires_grad = False

    @property
    def embed_size(self):
        """For compatibility with existing code."""
        return self.feature_dim

    def forward(self, x, norm=False):
        """
        Forward pass.

        Args:
            x: Input images of shape (B, C, H, W)
            norm: Whether to normalize features

        Returns:
            logits: Classification logits (B, num_classes)
            features: Steered features (B, feature_dim)
        """
        # Extract features from CLIP (frozen)
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(x)
            clip_features = F.normalize(clip_features, dim=-1)

        # Apply SAE steering
        if self.use_sae_features:
            features = self.sae_wrapper.encode(clip_features)
        else:
            features = self.sae_wrapper(clip_features)

        if norm:
            features = F.normalize(features, dim=-1)

        # Classify
        logits = self.fc(features)

        return logits, features

    def get_clip_features(self, x):
        """Get raw CLIP features without SAE steering."""
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(x)
            return F.normalize(clip_features, dim=-1)


class SAESteeringTrainer(BaseTrainer):
    """
    Trainer that uses SAE steering to suppress polysemantic neurons.

    This trainer:
    1. Loads a pretrained OpenCLIP model
    2. Loads a pretrained SAE and its analysis results
    3. Creates a neuron mask based on class purity (monosemanticity)
    4. Trains a classification head while keeping backbone and SAE frozen

    Configuration:
        MITIGATOR:
          TYPE: "sae_steering"
          SAE_STEERING:
            SAE_CHECKPOINT_PATH: "path/to/sae/ae.pt"
            SAE_ANALYSIS_PATH: "path/to/sae/analysis_results.json"
            PURITY_THRESHOLD: 1.0  # Only keep neurons with 100% class purity
            SUPPRESSION_VALUE: 0.0  # Value to use for suppressed neurons (0 = fully suppress)
            USE_SAE_FEATURES: False  # If True, classify from sparse features; if False, decode first
            USE_DECODE: True  # Whether to decode SAE features back to original space
    """

    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """Setup OpenCLIP backbone, SAE with steering, and classification head."""

        steering_cfg = self.cfg.MITIGATOR.SAE_STEERING

        # Step 1: Load the base OpenCLIP model
        print(f"\n{'='*60}")
        print("Setting up SAE Steering Model")
        print(f"{'='*60}")

        base_model = get_model(
            self.cfg.MODEL.TYPE, self.num_class, pretrained=self.cfg.MODEL.PRETRAINED
        )

        # Extract the CLIP model from the wrapper
        if hasattr(base_model, "clip_model"):
            self.clip_model = base_model.clip_model
            self.embed_size = base_model.embed_size
        else:
            raise ValueError(
                "Base model must be an OpenCLIP model with clip_model attribute"
            )

        print(f"Loaded OpenCLIP backbone with embed_size={self.embed_size}")

        # Step 2: Load pretrained SAE
        sae_checkpoint_path = steering_cfg.SAE_CHECKPOINT_PATH
        if not sae_checkpoint_path or not os.path.exists(sae_checkpoint_path):
            raise ValueError(f"SAE checkpoint not found at: {sae_checkpoint_path}")

        # Load SAE config to get dimensions
        sae_config_path = os.path.join(
            os.path.dirname(sae_checkpoint_path), "config.json"
        )
        if os.path.exists(sae_config_path):
            with open(sae_config_path, "r") as f:
                sae_config = json.load(f)
            activation_dim = sae_config["activation_dim"]
            dict_size = sae_config["dict_size"]
        else:
            # Infer from checkpoint
            checkpoint = torch.load(sae_checkpoint_path, map_location="cpu")
            # Try to infer dimensions from encoder weight shape
            if "encoder.weight" in checkpoint:
                dict_size, activation_dim = checkpoint["encoder.weight"].shape
            else:
                raise ValueError(
                    "Cannot infer SAE dimensions. Please provide config.json"
                )

        print(f"Loading SAE: activation_dim={activation_dim}, dict_size={dict_size}")

        self.sae = AutoEncoder(activation_dim, dict_size)
        self.sae.load_state_dict(torch.load(sae_checkpoint_path, map_location="cpu"))
        print(f"Loaded SAE from {sae_checkpoint_path}")

        # Step 3: Load analysis results and compute neuron mask
        analysis_path = steering_cfg.SAE_ANALYSIS_PATH
        if not analysis_path or not os.path.exists(analysis_path):
            raise ValueError(f"SAE analysis results not found at: {analysis_path}")

        with open(analysis_path, "r") as f:
            analysis_results = json.load(f)

        print(f"Loaded analysis results from {analysis_path}")

        # Step 4: Compute neuron weights based on class purity
        neuron_weights = self._compute_neuron_weights(
            analysis_results,
            dict_size=dict_size,
            purity_threshold=steering_cfg.PURITY_THRESHOLD,
            suppression_value=steering_cfg.SUPPRESSION_VALUE,
        )

        neuron_weights = torch.tensor(neuron_weights, dtype=torch.float32)

        num_kept = (neuron_weights > 0).sum().item()
        num_suppressed = (neuron_weights == 0).sum().item()
        print(
            f"Neuron mask: {num_kept} kept, {num_suppressed} suppressed "
            f"(purity_threshold={steering_cfg.PURITY_THRESHOLD})"
        )

        # Step 5: Create SAE steering wrapper
        self.sae_wrapper = SAESteeringWrapper(
            sae=self.sae,
            neuron_weights=neuron_weights,
            use_decode=steering_cfg.USE_DECODE,
        )

        # Step 6: Create full model
        self.model = OpenCLIPWithSAESteering(
            clip_model=self.clip_model,
            sae_wrapper=self.sae_wrapper,
            num_classes=self.num_class,
            use_sae_features=steering_cfg.USE_SAE_FEATURES,
        )

        self.model.to(self.device)

        print(f"Model ready: {num_kept} active neurons out of {dict_size}")
        print(f"Classification head input dim: {self.model.feature_dim}")

    def _compute_neuron_weights(
        self, analysis_results, dict_size, purity_threshold, suppression_value
    ):
        """
        Compute neuron weights based on class purity.

        Neurons with class purity >= threshold are kept (weight=1.0).
        Neurons with class purity < threshold are suppressed (weight=suppression_value).

        Args:
            analysis_results: Dict containing SAE analysis (from _analyze_monosemanticity)
            dict_size: Total number of SAE neurons
            purity_threshold: Minimum class purity to keep neuron (0.0 to 1.0)
            suppression_value: Weight to assign to suppressed neurons (typically 0.0)

        Returns:
            numpy array of shape (dict_size,) with neuron weights
        """
        # Initialize all neurons as suppressed
        neuron_weights = np.full(dict_size, suppression_value, dtype=np.float32)

        # Get top-k analysis per latent
        top_k_per_latent = analysis_results.get("top_k_per_latent", {})

        kept_count = 0
        suppressed_count = 0
        alive_neurons = analysis_results.get("alive_neurons", [])

        for latent_idx in range(dict_size):
            latent_key = str(latent_idx)  # JSON keys are strings

            if latent_key not in top_k_per_latent:
                # Neuron not analyzed (might be dead)
                suppressed_count += 1
                continue

            latent_data = top_k_per_latent[latent_key]
            targets = latent_data.get("targets", [])

            if len(targets) == 0:
                # No activations for this neuron
                suppressed_count += 1
                continue

            # Compute class purity (fraction of top-k images from majority class)
            target_counts = {}
            for t in targets:
                target_counts[t] = target_counts.get(t, 0) + 1

            majority_count = max(target_counts.values())
            purity = majority_count / len(targets)
            if purity >= purity_threshold:
                # Monosemantic neuron - keep it
                neuron_weights[latent_idx] = 1.0
                kept_count += 1
            else:
                # Polysemantic neuron - suppress it
                suppressed_count += 1

        print(
            f"Neuron analysis: {kept_count} monosemantic (purity >= {purity_threshold}), "
            f"{suppressed_count} polysemantic/dead"
        )

        return neuron_weights

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

    # def _setup_scheduler(self):
    #     """Setup learning rate scheduler."""
    #     scheduler_type = self.cfg.SOLVER.SCHEDULER.TYPE

    #     if scheduler_type == "MultiStepLR":
    #         self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #             self.optimizer,
    #             milestones=self.cfg.SOLVER.SCHEDULER.LR_DECAY_STAGES,
    #             gamma=self.cfg.SOLVER.SCHEDULER.LR_DECAY_RATE,
    #         )
    #     elif scheduler_type == "CosineAnnealingLR":
    #         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #             self.optimizer,
    #             T_max=self.cfg.SOLVER.EPOCHS,
    #         )
    #     elif scheduler_type == "StepLR":
    #         self.scheduler = torch.optim.lr_scheduler.StepLR(
    #             self.optimizer,
    #             step_size=self.cfg.SOLVER.SCHEDULER.LR_DECAY_STAGES[0],
    #             gamma=self.cfg.SOLVER.SCHEDULER.LR_DECAY_RATE,
    #         )
    #     else:
    #         self.scheduler = None

    def _setup_resume(self):
        return

    def _train_iter(self, batch):
        """Training iteration."""
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        # Forward pass
        logits, features = self.model(inputs)

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
        self.model.sae_wrapper.eval()
        self.current_lr = self.scheduler.get_last_lr()[0]
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

    # def train(self):
    #     """Main training loop."""
    #     print(f"\n{'='*60}")
    #     print("Starting SAE Steering Training")
    #     print(f"{'='*60}")
    #     print(f"Active neurons: {self.sae_wrapper.get_num_active_neurons()}")
    #     print(f"Suppressed neurons: {self.sae_wrapper.get_num_suppressed_neurons()}")

    #     best_metric = 0

    #     for epoch in range(self.cfg.SOLVER.EPOCHS):
    #         print(f"\nEpoch {epoch+1}/{self.cfg.SOLVER.EPOCHS}")

    #         # Train
    #         train_metrics = self._train_epoch()
    #         print(f"Train loss: {train_metrics['loss']:.4f}")

    #         # Evaluate
    #         eval_metrics = self.eval()

    #         # Log metrics
    #         current_metric = eval_metrics.get(
    #             self.cfg.METRIC, eval_metrics.get("accuracy", 0)
    #         )
    #         print(f"Eval {self.cfg.METRIC}: {current_metric:.4f}")

    #         # Save best model
    #         if current_metric > best_metric:
    #             best_metric = current_metric
    #             self._save_checkpoint(epoch, is_best=True)
    #             print(f"New best model! {self.cfg.METRIC}: {best_metric:.4f}")

    #     print(f"\n{'='*60}")
    #     print(f"Training complete! Best {self.cfg.METRIC}: {best_metric:.4f}")
    #     print(f"{'='*60}")

    # def _save_checkpoint(self, epoch, is_best=False):
    #     """Save model checkpoint."""
    #     checkpoint = {
    #         "epoch": epoch,
    #         "model": self.model.state_dict(),
    #         "optimizer": self.optimizer.state_dict(),
    #         "config": {
    #             "purity_threshold": self.cfg.MITIGATOR.SAE_STEERING.PURITY_THRESHOLD,
    #             "suppression_value": self.cfg.MITIGATOR.SAE_STEERING.SUPPRESSION_VALUE,
    #             "num_active_neurons": self.sae_wrapper.get_num_active_neurons(),
    #         },
    #     }

    #     save_path = os.path.join(self.log_path, "last.pth")
    #     torch.save(checkpoint, save_path)

    #     if is_best:
    #         best_path = os.path.join(self.log_path, "best.pth")
    #         torch.save(checkpoint, best_path)

    # def eval(self):
    #     """Evaluate the model."""
    #     self.model.eval()

    #     all_preds = []
    #     all_targets = []
    #     all_biases = defaultdict(list)

    #     with torch.no_grad():
    #         for batch in tqdm(self.dataloaders["test"], desc="Evaluating"):
    #             inputs = batch["inputs"].to(self.device)
    #             targets = batch["targets"]

    #             logits, _ = self.model(inputs)
    #             preds = logits.argmax(dim=1).cpu()

    #             all_preds.append(preds)
    #             all_targets.append(targets)

    #             # Collect bias attributes
    #             for bias_name in self.biases:
    #                 if bias_name in batch:
    #                     all_biases[bias_name].append(batch[bias_name])

    #     all_preds = torch.cat(all_preds)
    #     all_targets = torch.cat(all_targets)

    #     # Compute accuracy
    #     accuracy = (all_preds == all_targets).float().mean().item()

    #     # Compute per-group metrics if bias info available
    #     metrics = {"accuracy": accuracy}

    #     if all_biases:
    #         bias_name = self.biases[0]
    #         all_bias = torch.cat(all_biases[bias_name])

    #         # Compute worst-group accuracy
    #         group_accs = []
    #         for t in all_targets.unique():
    #             for b in all_bias.unique():
    #                 mask = (all_targets == t) & (all_bias == b)
    #                 if mask.sum() > 0:
    #                     group_acc = (
    #                         (all_preds[mask] == all_targets[mask]).float().mean().item()
    #                     )
    #                     group_accs.append(group_acc)

    #         if group_accs:
    #             metrics["wg_ovr"] = min(group_accs)
    #             metrics["avg_acc"] = np.mean(group_accs)
    #     print(group_accs)
    #     return metrics
