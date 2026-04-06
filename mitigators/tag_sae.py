"""
Tag-Supervised SAE Mitigator.

This mitigator integrates the Tag-Supervised SAE with the vb-mitigator framework.
It uses the MAVIAS pipeline for tag extraction and trains an SAE where specific
neurons are supervised to activate based on tag presence.

Usage:
    1. Run MAVIAS first to generate train_tags.csv with irrelevant_tags
    2. Configure TAG_SAE mitigator with path to tags CSV
    3. Train the SAE with tag supervision
    4. Use for debiasing by zeroing out tag-associated neurons

Config:
    MITIGATOR:
      TYPE: "tag_sae"
      TAG_SAE:
        TAGS_CSV_PATH: "data/utkface/train_tags.csv"
        TAG_COLUMN: "irrelevant_tags"
        MIN_TAG_FREQUENCY: 10

        # SAE architecture
        EXPANSION_FACTOR: 8
        NUM_FREE_NEURONS: 0  # 0 = use expansion factor

        # Training
        STEPS: 20000
        BATCH_SIZE: 256
        LR: 1e-3

        # Loss weights
        LAMBDA_RECONSTRUCTION: 1.0
        LAMBDA_SPARSITY: 1e-3
        LAMBDA_TAG: 1.0

        # Tag supervision
        TAG_LOSS_TYPE: "bce"  # "bce", "hinge", "mse"
        POSITIVE_WEIGHT: 1.0
        NEGATIVE_WEIGHT: 0.5
        USE_NEGATIVE_SUPERVISION: True
        MARGIN: 0.5  # For hinge loss
        TARGET_ACTIVATION: 1.0  # For MSE loss
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base_trainer import BaseTrainer
from models.builder import get_model
from tools.utils import log_msg, save_checkpoint, load_checkpoint

# Import tag-supervised SAE components
from models.tag_supervised_sae import (
    TagSupervisedSAE,
    TagSupervisionLoss,
    TagSupervisedSAETrainer,
    load_tag_supervised_sae,
    debias_features,
)


class TagSAETrainer(BaseTrainer):
    """
    Tag-Supervised SAE Trainer for debiasing visual features.

    This trainer:
    1. Loads tags from MAVIAS pipeline (train_tags.csv)
    2. Extracts features from a pretrained model
    3. Trains an SAE with tag-supervised anchored neurons
    4. Analyzes tag neuron performance
    5. Can be used for debiasing at inference time
    """

    def __init__(self, cfg):
        """Initialize the Tag-SAE trainer."""
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """Setup the pretrained model for feature extraction."""
        # Load the pretrained model
        self.model = get_model(
            self.cfg.MODEL.TYPE, self.num_class, pretrained=self.cfg.MODEL.PRETRAINED
        )

        # Load pretrained weights if provided
        if self.cfg.MITIGATOR.TAG_SAE.CHECKPOINT_PATH != "":
            checkpoint_path = self.cfg.MITIGATOR.TAG_SAE.CHECKPOINT_PATH
            print(f"Loading pretrained model from: {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path)
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            else:
                self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Determine feature dimension
        self.feature_dim = self._get_feature_dim()
        print(f"Feature dimension: {self.feature_dim}")

        # Initialize SAE (will be set up during training)
        self.sae = None
        self.sae_trainer = None

    def _get_feature_dim(self):
        """Determine the feature dimension of the model."""
        try:
            if hasattr(self.model, "embed_size"):
                return self.model.embed_size
            elif hasattr(self.model, "embed_dim"):
                return self.model.embed_dim
            elif hasattr(self.model, "fc"):
                return self.model.fc.in_features
            elif hasattr(self.model, "classifier"):
                if isinstance(self.model.classifier, nn.Linear):
                    return self.model.classifier.in_features
                else:
                    return self.model.classifier[-1].in_features
            elif hasattr(self.model, "head"):
                if isinstance(self.model.head, nn.Linear):
                    return self.model.head.in_features
                else:
                    return self.model.head[-1].in_features
            else:
                # Default for common architectures
                return 512
        except:
            return 512

    def _setup_optimizer(self):
        """SAE has its own optimizer."""
        self.optimizer = None

    def _setup_scheduler(self):
        """SAE has its own scheduler."""
        self.scheduler = None

    def _method_specific_setups(self):
        """Setup Tag-SAE specific components."""
        self.tag_cfg = self.cfg.MITIGATOR.TAG_SAE

        # Load tags from MAVIAS pipeline
        self._load_mavias_tags()

    def _load_mavias_tags(self):
        """Load tags from MAVIAS pipeline CSV."""
        tags_csv_path = self.tag_cfg.TAGS_CSV_PATH

        # If path is relative, try to find it in data_root
        if not os.path.isabs(tags_csv_path):
            # Try data_root
            candidate = os.path.join(self.data_root, tags_csv_path)
            if os.path.exists(candidate):
                tags_csv_path = candidate
            else:
                # Try just the filename in data_root
                candidate = os.path.join(
                    self.data_root, os.path.basename(tags_csv_path)
                )
                if os.path.exists(candidate):
                    tags_csv_path = candidate

        if not os.path.exists(tags_csv_path):
            raise FileNotFoundError(
                f"Tags CSV not found: {tags_csv_path}\n"
                "Please run MAVIAS first to generate train_tags.csv with irrelevant_tags column."
            )

        print(f"\nLoading tags from: {tags_csv_path}")
        self.tags_df = pd.read_csv(tags_csv_path)

        tag_column = self.tag_cfg.TAG_COLUMN
        tag_separator = self.tag_cfg.get("TAG_SEPARATOR", " | ")
        min_freq = self.tag_cfg.MIN_TAG_FREQUENCY

        # Count tag frequencies
        tag_counts = defaultdict(int)

        for _, row in self.tags_df.iterrows():
            tags_str = row.get(tag_column, "")
            if pd.isna(tags_str) or not isinstance(tags_str, str):
                continue

            tags = [t.strip() for t in tags_str.split(tag_separator) if t.strip()]
            for tag in tags:
                tag_counts[tag] += 1

        # Filter by minimum frequency
        self.all_tags = sorted(
            [tag for tag, count in tag_counts.items() if count >= min_freq]
        )

        # Create tag to neuron mapping
        self.tag_to_neuron = {tag: i for i, tag in enumerate(self.all_tags)}

        # Build index to tags mapping
        self.index_to_tags = {}
        for _, row in self.tags_df.iterrows():
            idx = row["index"]
            tags_str = row.get(tag_column, "")
            if pd.isna(tags_str) or not isinstance(tags_str, str):
                tags = []
            else:
                tags = [t.strip() for t in tags_str.split(tag_separator) if t.strip()]
            # Filter to only known tags
            tags = [t for t in tags if t in self.tag_to_neuron]
            self.index_to_tags[idx] = tags

        print(f"  Total unique tags: {len(tag_counts)}")
        print(f"  Tags with freq >= {min_freq}: {len(self.all_tags)}")
        print(f"  Sample tags: {self.all_tags[:10]}...")

    def _extract_features(self, dataloader, desc="Extracting features"):
        """Extract features from the pretrained model."""
        all_features = []
        all_targets = []
        all_indices = []
        all_biases = {b: [] for b in self.biases}

        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                inputs = batch["inputs"].to(self.device)

                # Forward pass to get features
                outputs = self.model(inputs)

                if isinstance(outputs, tuple):
                    _, features = outputs
                else:
                    features = self._extract_with_hook(inputs)

                all_features.append(features.cpu())
                all_targets.append(batch["targets"])

                if "index" in batch:
                    all_indices.extend(batch["index"].tolist())
                else:
                    all_indices.extend(range(len(inputs)))

                for b in self.biases:
                    if b in batch:
                        all_biases[b].append(batch[b])

        features = torch.cat(all_features, dim=0)
        targets = torch.cat(all_targets, dim=0)
        indices = torch.tensor(all_indices)
        biases = {b: torch.cat(v, dim=0) if v else None for b, v in all_biases.items()}

        return features, targets, indices, biases

    def _extract_with_hook(self, inputs):
        """Extract features using a forward hook."""
        features = None

        def hook(module, input, output):
            nonlocal features
            if isinstance(output, tuple):
                features = output[0]
            else:
                features = output

        # Try to find the feature extraction layer
        if hasattr(self.model, "extractor"):
            handle = self.model.extractor.register_forward_hook(hook)
        elif hasattr(self.model, "avgpool"):
            handle = self.model.avgpool.register_forward_hook(hook)
        elif hasattr(self.model, "global_pool"):
            handle = self.model.global_pool.register_forward_hook(hook)
        else:
            # Fallback
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                return outputs[1]
            return outputs

        _ = self.model(inputs)
        handle.remove()

        return features.view(features.size(0), -1)

    def _build_tag_targets(self, indices: torch.Tensor) -> torch.Tensor:
        """Build binary tag target matrix for a batch."""
        batch_size = len(indices)
        num_anchored = len(self.all_tags)
        targets = torch.zeros(batch_size, num_anchored)

        for i, idx in enumerate(indices.tolist()):
            tags = self.index_to_tags.get(idx, [])
            for tag in tags:
                if tag in self.tag_to_neuron:
                    neuron_idx = self.tag_to_neuron[tag]
                    targets[i, neuron_idx] = 1.0

        return targets

    def _train_tag_sae(self, features, indices):
        """Train the Tag-Supervised SAE."""
        print(f"\n{'='*60}")
        print("Training Tag-Supervised SAE")
        print(f"{'='*60}")

        num_anchored = len(self.all_tags)

        # Compute dict size
        if self.tag_cfg.NUM_FREE_NEURONS > 0:
            dict_size = num_anchored + self.tag_cfg.NUM_FREE_NEURONS
        else:
            dict_size = self.feature_dim * self.tag_cfg.EXPANSION_FACTOR
            dict_size = max(dict_size, num_anchored + 100)

        print(f"  Feature dim: {self.feature_dim}")
        print(f"  Dict size: {dict_size}")
        print(f"  Anchored neurons (tags): {num_anchored}")
        print(f"  Free neurons: {dict_size - num_anchored}")

        # Compute tag statistics
        print(f"\n  Tag statistics:")
        total_samples = len(indices)
        tag_counts = torch.zeros(num_anchored)
        for idx in indices.tolist():
            tags = self.index_to_tags.get(idx, [])
            for tag in tags:
                if tag in self.tag_to_neuron:
                    tag_counts[self.tag_to_neuron[tag]] += 1

        tags_per_sample = tag_counts.sum() / total_samples
        avg_tag_freq = tag_counts.mean()
        max_tag_freq = tag_counts.max()
        min_tag_freq = tag_counts[tag_counts > 0].min() if (tag_counts > 0).any() else 0
        active_tags = (tag_counts > 0).sum()

        print(f"    Total samples: {total_samples}")
        print(f"    Active tags (with >0 samples): {active_tags}/{num_anchored}")
        print(f"    Avg tags per sample: {tags_per_sample:.2f}")
        print(
            f"    Tag frequency - mean: {avg_tag_freq:.1f}, min: {min_tag_freq:.0f}, max: {max_tag_freq:.0f}"
        )

        # Warn if tag sparsity is very high
        tag_density = tags_per_sample / num_anchored
        print(f"    Tag density (tags_per_sample / num_tags): {tag_density:.4f}")
        if tag_density < 0.01:
            print(
                f"    WARNING: Very sparse tags! Consider reducing MIN_TAG_FREQUENCY or using hinge loss."
            )

        # Create SAE
        self.sae = TagSupervisedSAE(
            input_dim=self.feature_dim,
            dict_size=dict_size,
            num_anchored=num_anchored,
            tag_to_neuron=self.tag_to_neuron,
        )
        self.sae.to(self.device)

        # Create loss functions
        tag_loss_fn = TagSupervisionLoss(
            loss_type=self.tag_cfg.TAG_LOSS_TYPE,
            positive_weight=self.tag_cfg.POSITIVE_WEIGHT,
            negative_weight=self.tag_cfg.NEGATIVE_WEIGHT,
            margin=self.tag_cfg.MARGIN,
            target_activation=self.tag_cfg.TARGET_ACTIVATION,
            use_negative_supervision=self.tag_cfg.USE_NEGATIVE_SUPERVISION,
        )

        # Create dataloader - need to recreate dataset with potentially scaled features
        dataset = TensorDataset(features, indices)
        dataloader = DataLoader(
            dataset,
            batch_size=self.tag_cfg.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )

        # Optimizer
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=self.tag_cfg.LR)

        # Training loop
        num_steps = self.tag_cfg.STEPS
        step = 0

        losses_history = {"total": [], "reconstruction": [], "sparsity": [], "tag": []}

        # Check feature statistics
        print(f"\n  Feature statistics:")
        sample_batch = features[:100].to(self.device)
        print(f"    Shape: {features.shape}")
        print(f"    Mean: {sample_batch.mean().item():.6f}")
        print(f"    Std: {sample_batch.std().item():.6f}")
        print(f"    Min: {sample_batch.min().item():.6f}")
        print(f"    Max: {sample_batch.max().item():.6f}")
        print(f"    L2 norm (mean): {sample_batch.norm(dim=1).mean().item():.4f}")

        # Test initial reconstruction
        self.sae.eval()
        with torch.no_grad():
            test_recon, test_latents = self.sae(sample_batch, return_latents=True)
            init_mse = F.mse_loss(test_recon, sample_batch).item()
            init_l0 = (test_latents > 0).float().sum(dim=1).mean().item()
            print(f"\n  Initial SAE state:")
            print(f"    Reconstruction MSE: {init_mse:.6f}")
            print(f"    L0 (active neurons): {init_l0:.1f}")
            print(f"    Latent mean: {test_latents.mean().item():.6f}")
            print(f"    Latent max: {test_latents.max().item():.6f}")

        self.sae.train()
        pbar = tqdm(total=num_steps, desc="Training Tag-SAE")

        # Get config for energy balancing
        energy_balance_weight = self.tag_cfg.get("LAMBDA_ENERGY_BALANCE", 0.0)
        orthogonality_weight = self.tag_cfg.get("LAMBDA_ORTHOGONALITY", 0.0)
        min_free_energy_ratio = self.tag_cfg.get("MIN_FREE_ENERGY_RATIO", 0.3)
        warmup_steps = self.tag_cfg.get("TAG_WARMUP_STEPS", 0)

        if energy_balance_weight > 0:
            print(f"  Energy balance weight: {energy_balance_weight}")
            print(f"  Min free energy ratio: {min_free_energy_ratio}")
        if orthogonality_weight > 0:
            print(f"  Orthogonality weight: {orthogonality_weight}")
        if warmup_steps > 0:
            print(f"  Tag supervision warmup: {warmup_steps} steps")

        while step < num_steps:
            for batch_features, batch_indices in dataloader:
                if step >= num_steps:
                    break

                batch_features = batch_features.to(self.device)

                # Forward pass
                reconstructed, latents = self.sae(batch_features, return_latents=True)

                # Split latents into anchored (tag) and free neurons
                anchored_latents = latents[:, : self.sae.num_anchored]
                free_latents = latents[:, self.sae.num_anchored :]

                # ============================================
                # 1. Reconstruction Loss
                # ============================================
                loss_recon = F.mse_loss(reconstructed, batch_features)

                # ============================================
                # 2. Sparsity Loss (L1 on all latents)
                # ============================================
                loss_sparsity = latents.abs().mean()

                # ============================================
                # 3. Tag Supervision Loss (with optional warmup)
                # ============================================
                tag_targets = self._build_tag_targets(batch_indices).to(self.device)
                loss_tag = tag_loss_fn(anchored_latents, tag_targets)

                # Apply warmup to tag loss
                if warmup_steps > 0 and step < warmup_steps:
                    tag_scale = step / warmup_steps
                else:
                    tag_scale = 1.0

                # ============================================
                # 4. Energy Balance Loss (encourage free neurons)
                # ============================================
                loss_energy_balance = torch.tensor(0.0, device=self.device)
                if energy_balance_weight > 0:
                    # Compute energy in each part
                    anchored_energy = anchored_latents.pow(2).sum(dim=1).mean()
                    free_energy = free_latents.pow(2).sum(dim=1).mean()
                    total_energy = anchored_energy + free_energy + 1e-8

                    # Current ratio of free energy
                    free_ratio = free_energy / total_energy

                    # Penalize if free_ratio < min_free_energy_ratio
                    # This encourages free neurons to have at least min_free_energy_ratio of total energy
                    loss_energy_balance = F.relu(min_free_energy_ratio - free_ratio)

                # ============================================
                # 5. Orthogonality Loss (free neurons orthogonal to tag neurons)
                # ============================================
                loss_orthogonality = torch.tensor(0.0, device=self.device)
                if orthogonality_weight > 0:
                    # Get decoder weights for anchored and free neurons
                    # decoder.weight shape: (input_dim, dict_size)
                    anchored_decoder = self.sae.decoder.weight[
                        :, : self.sae.num_anchored
                    ]  # (dim, num_anchored)
                    free_decoder = self.sae.decoder.weight[
                        :, self.sae.num_anchored :
                    ]  # (dim, num_free)

                    # Normalize
                    anchored_decoder_norm = F.normalize(anchored_decoder, dim=0)
                    free_decoder_norm = F.normalize(free_decoder, dim=0)

                    # Compute cross-correlation: should be close to 0
                    # (num_anchored, num_free) matrix of cosine similarities
                    cross_corr = torch.mm(anchored_decoder_norm.T, free_decoder_norm)

                    # Penalize high correlation
                    loss_orthogonality = cross_corr.abs().mean()

                # ============================================
                # Total Loss
                # ============================================
                loss = (
                    self.tag_cfg.LAMBDA_RECONSTRUCTION * loss_recon
                    + self.tag_cfg.LAMBDA_SPARSITY * loss_sparsity
                    + self.tag_cfg.LAMBDA_TAG * tag_scale * loss_tag
                    + energy_balance_weight * loss_energy_balance
                    + orthogonality_weight * loss_orthogonality
                )

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Normalize decoder columns to unit norm (standard SAE practice)
                with torch.no_grad():
                    norms = self.sae.decoder.weight.norm(dim=0, keepdim=True)
                    self.sae.decoder.weight.div_(norms.clamp(min=1e-6))

                # Log
                losses_history["total"].append(loss.item())
                losses_history["reconstruction"].append(loss_recon.item())
                losses_history["sparsity"].append(loss_sparsity.item())
                losses_history["tag"].append(loss_tag.item())

                step += 1
                pbar.update(1)

                if step % 1000 == 0:
                    l0 = (latents > 0).float().sum(dim=1).mean().item()
                    l0_anchored = (
                        (anchored_latents > 0).float().sum(dim=1).mean().item()
                    )
                    l0_free = (free_latents > 0).float().sum(dim=1).mean().item()

                    # Compute energy distribution
                    with torch.no_grad():
                        anch_e = anchored_latents.pow(2).sum(dim=1).mean().item()
                        free_e = free_latents.pow(2).sum(dim=1).mean().item()
                        total_e = anch_e + free_e + 1e-8
                        free_ratio = free_e / total_e

                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "recon": f"{loss_recon.item():.6f}",
                            "tag": f"{loss_tag.item():.4f}",
                            "L0": f"{l0:.0f}",
                            "free%": f"{free_ratio:.1%}",
                        }
                    )

                    # Detailed logging every 5000 steps
                    if step % 5000 == 0:
                        print(f"\n  Step {step} - Energy distribution:")
                        print(
                            f"    Anchored neurons: L0={l0_anchored:.1f}, energy={anch_e:.4f} ({1-free_ratio:.1%})"
                        )
                        print(
                            f"    Free neurons: L0={l0_free:.1f}, energy={free_e:.4f} ({free_ratio:.1%})"
                        )
                        if orthogonality_weight > 0:
                            print(
                                f"    Orthogonality loss: {loss_orthogonality.item():.4f}"
                            )

        pbar.close()

        # Save model
        self._save_tag_sae(losses_history)

        return self.sae

    def _save_tag_sae(self, losses_history):
        """Save the trained Tag-SAE."""
        save_dir = os.path.join(self.log_path, "tag_sae")
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(save_dir, "tag_sae.pt")
        torch.save(self.sae.state_dict(), model_path)
        print(f"\nSaved model to {model_path}")

        # Save config
        config = {
            "input_dim": self.feature_dim,
            "dict_size": self.sae.dict_size,
            "num_anchored": self.sae.num_anchored,
            "num_free": self.sae.num_free,
            "tag_to_neuron": self.tag_to_neuron,
            "all_tags": self.all_tags,
            "tag_loss_type": self.tag_cfg.TAG_LOSS_TYPE,
            "use_negative_supervision": self.tag_cfg.USE_NEGATIVE_SUPERVISION,
        }
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save losses
        losses_path = os.path.join(save_dir, "losses.json")
        with open(losses_path, "w") as f:
            # Save last 1000 losses
            json.dump({k: v[-1000:] for k, v in losses_history.items()}, f)

    def _analyze_tag_neurons(self, features, indices):
        """Analyze how well tag neurons have learned."""
        print(f"\n{'='*60}")
        print("Analyzing Tag Neuron Performance")
        print(f"{'='*60}")

        self.sae.eval()

        results = {
            "per_tag_stats": {},
            "overall_stats": {},
        }

        # Process all features
        with torch.no_grad():
            all_activations = []
            all_targets = []

            batch_size = self.tag_cfg.BATCH_SIZE
            for i in range(0, len(features), batch_size):
                batch_features = features[i : i + batch_size].to(self.device)
                batch_indices = indices[i : i + batch_size]

                latents = self.sae.encode(batch_features)
                anchored = self.sae.get_anchored_activations(latents)
                tag_targets = self._build_tag_targets(batch_indices).to(self.device)

                all_activations.append(anchored.cpu())
                all_targets.append(tag_targets.cpu())

            all_activations = torch.cat(all_activations, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

        # Per-tag analysis
        good_tags = []
        for tag, neuron_idx in self.tag_to_neuron.items():
            acts = all_activations[:, neuron_idx]
            targets = all_targets[:, neuron_idx]

            present_mask = targets == 1
            absent_mask = targets == 0

            act_present = (
                acts[present_mask].mean().item() if present_mask.sum() > 0 else 0
            )
            act_absent = acts[absent_mask].mean().item() if absent_mask.sum() > 0 else 0
            separation = act_present - act_absent

            results["per_tag_stats"][tag] = {
                "neuron_idx": neuron_idx,
                "num_present": present_mask.sum().item(),
                "act_when_present": act_present,
                "act_when_absent": act_absent,
                "separation": separation,
            }

            if separation > 0.1:
                good_tags.append((tag, separation))

        # Sort by separation
        good_tags.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 20 best separated tags:")
        for tag, sep in good_tags[:20]:
            stats = results["per_tag_stats"][tag]
            print(
                f"  {tag}: sep={sep:.3f}, "
                f"present={stats['act_when_present']:.3f}, "
                f"absent={stats['act_when_absent']:.3f}, "
                f"n={stats['num_present']}"
            )

        # Overall stats
        results["overall_stats"]["num_good_tags"] = len(good_tags)
        results["overall_stats"]["mean_separation"] = (
            np.mean([s for _, s in good_tags]) if good_tags else 0
        )

        # Save results
        save_dir = os.path.join(self.log_path, "tag_sae")
        results_path = os.path.join(save_dir, "tag_analysis.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved analysis to {results_path}")

        return results

    def train(self):
        """Main training pipeline."""
        print(f"\n{'='*60}")
        print("Tag-Supervised SAE Training Pipeline")
        print(f"{'='*60}")

        # Step 1: Check for precomputed features
        precomputed_path = self.tag_cfg.get("PRECOMPUTED_FEATURES_PATH", "")

        if precomputed_path and os.path.exists(precomputed_path):
            print(f"\nStep 1: Loading precomputed features from {precomputed_path}")
            saved_data = torch.load(precomputed_path, map_location="cpu")
            features = saved_data["features"]
            targets = saved_data.get("targets", None)
            indices = saved_data.get("indices", torch.arange(len(features)))
            biases = saved_data.get("biases", {})
            if isinstance(indices, list):
                indices = torch.tensor(indices)

            # Update feature_dim if needed
            if features.shape[1] != self.feature_dim:
                self.feature_dim = features.shape[1]
                print(f"  Updated feature_dim to {self.feature_dim}")
        else:
            # Extract features
            print("\nStep 1: Extracting features from training data...")
            features, targets, indices, biases = self._extract_features(
                self.dataloaders["train"], desc="Extracting features"
            )
            print(f"  Extracted {len(features)} features of dim {features.shape[1]}")

            # Save features
            features_dir = os.path.join(self.log_path, "features")
            os.makedirs(features_dir, exist_ok=True)
            torch.save(
                {
                    "features": features,
                    "targets": targets,
                    "indices": indices,
                    "biases": biases,
                },
                os.path.join(features_dir, "train_features.pt"),
            )
            print(f"  Saved features to {features_dir}")

        # Step 2: Train Tag-SAE
        print("\nStep 2: Training Tag-Supervised SAE...")
        self._train_tag_sae(features, indices)

        # Step 3: Analyze tag neurons
        print("\nStep 3: Analyzing tag neurons...")
        self._analyze_tag_neurons(features, indices)

        # Step 4: Visualize tag neurons
        print("\nStep 4: Visualizing tag neurons...")
        self._visualize_tag_neurons(features, indices)

        # Step 5: Evaluate debiasing on test set
        print("\nStep 5: Evaluating debiasing on test set...")
        self._evaluate_debiasing()

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Output directory: {self.log_path}")

    def _evaluate_debiasing(self):
        """
        Evaluate the debiasing effect on the test set.

        Compares:
        1. Original features → classifier → accuracy
        2. Debiased features (all tags removed) → classifier → accuracy
        3. Per-group accuracy to measure bias reduction
        """
        print(f"\n{'='*60}")
        print("Evaluating Debiasing Effect")
        print(f"{'='*60}")

        if "test" not in self.dataloaders:
            print("No test dataloader available, skipping evaluation.")
            return {}

        self.model.eval()
        self.sae.eval()

        # Extract test features
        print("\nExtracting test features...")
        test_features, test_targets, test_indices, test_biases = self._extract_features(
            self.dataloaders["test"], desc="Extracting test features"
        )

        # Get the classifier head from the model
        classifier = self._get_classifier_head()
        if classifier is None:
            print("Could not find classifier head, skipping classification evaluation.")
            return self._evaluate_feature_similarity(
                test_features, test_targets, test_biases
            )

        classifier.eval()

        results = {
            "original": {},
            "debiased": {},
            "per_group": {},
        }

        # Evaluate original features
        print("\nEvaluating original features...")
        with torch.no_grad():
            original_logits = classifier(test_features.to(self.device))
            original_preds = original_logits.argmax(dim=1).cpu()
            original_acc = (original_preds == test_targets).float().mean().item()

        results["original"]["accuracy"] = original_acc
        print(f"  Original accuracy: {original_acc:.4f}")

        # Evaluate debiased features (all tags removed)
        print("\nEvaluating debiased features (all tags removed)...")
        with torch.no_grad():
            # Encode through SAE
            latents = self.sae.encode(test_features.to(self.device))

            # Zero out anchored (tag) neurons
            latents_debiased = latents.clone()
            latents_debiased[:, : self.sae.num_anchored] = 0

            # Decode back to feature space
            debiased_features = self.sae.decode(latents_debiased)

            # Classify
            debiased_logits = classifier(debiased_features)
            debiased_preds = debiased_logits.argmax(dim=1).cpu()
            debiased_acc = (debiased_preds == test_targets).float().mean().item()

        results["debiased"]["accuracy"] = debiased_acc
        print(f"  Debiased accuracy: {debiased_acc:.4f}")
        print(f"  Accuracy change: {debiased_acc - original_acc:+.4f}")

        # Per-group evaluation (if bias attributes available)
        bias_name = self.biases[0] if self.biases else None
        if (
            bias_name
            and bias_name in test_biases
            and test_biases[bias_name] is not None
        ):
            print(f"\nPer-group evaluation (bias: {bias_name})...")
            bias_values = test_biases[bias_name]
            unique_targets = test_targets.unique().tolist()
            unique_biases = bias_values.unique().tolist()

            for t in unique_targets:
                for b in unique_biases:
                    mask = (test_targets == t) & (bias_values == b)
                    if mask.sum() == 0:
                        continue

                    group_orig_acc = (
                        (original_preds[mask] == test_targets[mask])
                        .float()
                        .mean()
                        .item()
                    )
                    group_debiased_acc = (
                        (debiased_preds[mask] == test_targets[mask])
                        .float()
                        .mean()
                        .item()
                    )

                    group_key = f"target_{t}_bias_{b}"
                    results["per_group"][group_key] = {
                        "original_acc": group_orig_acc,
                        "debiased_acc": group_debiased_acc,
                        "change": group_debiased_acc - group_orig_acc,
                        "n_samples": mask.sum().item(),
                    }

                    print(
                        f"  Group (target={t}, {bias_name}={b}): "
                        f"orig={group_orig_acc:.4f}, debiased={group_debiased_acc:.4f}, "
                        f"Δ={group_debiased_acc - group_orig_acc:+.4f}, n={mask.sum().item()}"
                    )

            # Compute worst-group accuracy
            group_accs_orig = [g["original_acc"] for g in results["per_group"].values()]
            group_accs_debiased = [
                g["debiased_acc"] for g in results["per_group"].values()
            ]

            results["original"]["worst_group_acc"] = min(group_accs_orig)
            results["debiased"]["worst_group_acc"] = min(group_accs_debiased)

            print(f"\n  Worst-group accuracy:")
            print(f"    Original: {results['original']['worst_group_acc']:.4f}")
            print(f"    Debiased: {results['debiased']['worst_group_acc']:.4f}")
            print(
                f"    Change: {results['debiased']['worst_group_acc'] - results['original']['worst_group_acc']:+.4f}"
            )

        # Also compute group similarity matrices
        print("\n" + "=" * 60)
        print("Computing Group Similarity Analysis")
        print("=" * 60)
        similarity_results = self._evaluate_feature_similarity(
            test_features, test_targets, test_biases
        )
        results["similarity_analysis"] = similarity_results

        # Zero-shot evaluation
        zero_shot_results = self._evaluate_zero_shot(
            test_features, test_targets, test_biases
        )
        if zero_shot_results:
            results["zero_shot"] = zero_shot_results

        # Save results
        save_dir = os.path.join(self.log_path, "tag_sae")
        results_path = os.path.join(save_dir, "debiasing_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved debiasing results to {results_path}")

        return results

    def _get_classifier_head(self):
        """Extract the classifier head from the model."""
        if hasattr(self.model, "fc"):
            return self.model.fc
        elif hasattr(self.model, "classifier"):
            return self.model.classifier
        elif hasattr(self.model, "head"):
            return self.model.head
        else:
            return None

    def _evaluate_feature_similarity(
        self, test_features, test_targets=None, test_biases=None
    ):
        """
        Evaluate how much the features change after debiasing.

        Computes:
        - Basic feature statistics
        - Group similarity matrices (confusion matrix style)
        """
        print("\nEvaluating feature changes after debiasing...")

        results = {}

        with torch.no_grad():
            test_features_gpu = test_features.to(self.device)

            # Encode through SAE
            latents = self.sae.encode(test_features_gpu)

            # Reconstruct (no debiasing)
            reconstructed = self.sae.decode(latents)

            # Debias and reconstruct
            latents_debiased = latents.clone()
            latents_debiased[:, : self.sae.num_anchored] = 0
            debiased = self.sae.decode(latents_debiased)

            # Compute basic statistics
            recon_mse = F.mse_loss(reconstructed, test_features_gpu).item()
            debias_mse = F.mse_loss(debiased, test_features_gpu).item()

            # Cosine similarity
            cos_sim_recon = (
                F.cosine_similarity(reconstructed, test_features_gpu, dim=1)
                .mean()
                .item()
            )
            cos_sim_debias = (
                F.cosine_similarity(debiased, test_features_gpu, dim=1).mean().item()
            )

            # How much of the latent is in tag neurons?
            tag_energy = (
                latents[:, : self.sae.num_anchored].pow(2).sum(dim=1).mean().item()
            )
            free_energy = (
                latents[:, self.sae.num_anchored :].pow(2).sum(dim=1).mean().item()
            )
            total_energy = latents.pow(2).sum(dim=1).mean().item()

            # Move to CPU for group analysis
            debiased_cpu = debiased.cpu()

            results["reconstruction_mse"] = recon_mse
            results["debiased_mse"] = debias_mse
            results["cosine_similarity_reconstructed"] = cos_sim_recon
            results["cosine_similarity_debiased"] = cos_sim_debias
            results["tag_neuron_energy_fraction"] = tag_energy / (total_energy + 1e-8)
            results["free_neuron_energy_fraction"] = free_energy / (total_energy + 1e-8)

        print(f"  Reconstruction MSE: {recon_mse:.6f}")
        print(f"  Debiased MSE (vs original): {debias_mse:.6f}")
        print(f"  Cosine similarity (reconstructed): {cos_sim_recon:.4f}")
        print(f"  Cosine similarity (debiased): {cos_sim_debias:.4f}")
        print(f"  Energy in tag neurons: {results['tag_neuron_energy_fraction']:.2%}")
        print(f"  Energy in free neurons: {results['free_neuron_energy_fraction']:.2%}")

        # Compute group similarity matrices if targets and biases available
        if test_targets is not None and test_biases is not None:
            bias_name = self.biases[0] if self.biases else "bias"
            if isinstance(test_biases, dict):
                bias_values = test_biases.get(bias_name)
            else:
                bias_values = test_biases

            if bias_values is not None:
                group_sim_results = self._compute_group_similarity_matrix(
                    original_features=test_features.cpu(),
                    debiased_features=debiased_cpu,
                    targets=test_targets,
                    biases=bias_values,
                    bias_name=bias_name,
                )
                results["group_similarities"] = group_sim_results

        # Save results
        save_dir = os.path.join(self.log_path, "tag_sae")
        results_path = os.path.join(save_dir, "feature_analysis.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved feature analysis to {results_path}")

        return results

    def _compute_group_similarity_matrix(
        self,
        original_features: torch.Tensor,
        debiased_features: torch.Tensor,
        targets: torch.Tensor,
        biases: torch.Tensor,
        bias_name: str = "bias",
    ) -> dict:
        """
        Compute group similarity matrices (confusion matrix style).

        Each cell (i, j) contains the cosine similarity between
        the centroid of group i and centroid of group j.

        Groups are (target, bias) combinations.
        """
        print(f"\n  Computing group similarity matrices...")

        unique_targets = sorted(targets.unique().tolist())
        unique_biases = sorted(biases.unique().tolist())

        # Build groups and compute centroids
        groups = []
        group_labels = []
        original_centroids = []
        debiased_centroids = []
        group_sizes = []

        for t in unique_targets:
            for b in unique_biases:
                mask = (targets == t) & (biases == b)
                n_samples = mask.sum().item()
                if n_samples > 0:
                    groups.append((t, b))
                    group_labels.append(f"t{t}_b{b}")
                    group_sizes.append(n_samples)

                    orig_centroid = original_features[mask].mean(dim=0)
                    debias_centroid = debiased_features[mask].mean(dim=0)

                    original_centroids.append(orig_centroid)
                    debiased_centroids.append(debias_centroid)

        num_groups = len(groups)
        original_centroids = torch.stack(original_centroids)  # (num_groups, dim)
        debiased_centroids = torch.stack(debiased_centroids)  # (num_groups, dim)

        # Normalize for cosine similarity
        original_centroids_norm = F.normalize(original_centroids, dim=1)
        debiased_centroids_norm = F.normalize(debiased_centroids, dim=1)

        # Compute similarity matrices
        original_sim_matrix = torch.mm(
            original_centroids_norm, original_centroids_norm.t()
        )
        debiased_sim_matrix = torch.mm(
            debiased_centroids_norm, debiased_centroids_norm.t()
        )
        change_matrix = debiased_sim_matrix - original_sim_matrix

        # Print group info
        print(f"\n  Groups (target, {bias_name}) with sample counts:")
        for (t, b), label, n in zip(groups, group_labels, group_sizes):
            print(f"    {label}: target={t}, {bias_name}={b}, n={n}")

        # Print matrices
        print(f"\n  Original Feature Similarity Matrix:")
        self._print_similarity_matrix(original_sim_matrix, group_labels)

        print(f"\n  Debiased Feature Similarity Matrix:")
        self._print_similarity_matrix(debiased_sim_matrix, group_labels)

        print(f"\n  Change Matrix (Debiased - Original):")
        self._print_similarity_matrix(change_matrix, group_labels, show_sign=True)

        # Compute summary statistics
        # Intra-class: same target, different bias (should increase after debiasing)
        # Inter-class: different target (should decrease or stay same)
        intra_class_orig = []
        intra_class_debias = []
        inter_class_orig = []
        inter_class_debias = []

        for i in range(num_groups):
            for j in range(i + 1, num_groups):
                t_i, b_i = groups[i]
                t_j, b_j = groups[j]

                if t_i == t_j:  # Same target, different bias = intra-class
                    intra_class_orig.append(original_sim_matrix[i, j].item())
                    intra_class_debias.append(debiased_sim_matrix[i, j].item())
                else:  # Different target = inter-class
                    inter_class_orig.append(original_sim_matrix[i, j].item())
                    inter_class_debias.append(debiased_sim_matrix[i, j].item())

        summary = {}

        if intra_class_orig:
            summary["intra_class_sim_original"] = float(np.mean(intra_class_orig))
            summary["intra_class_sim_debiased"] = float(np.mean(intra_class_debias))
            summary["intra_class_change"] = (
                summary["intra_class_sim_debiased"]
                - summary["intra_class_sim_original"]
            )

        if inter_class_orig:
            summary["inter_class_sim_original"] = float(np.mean(inter_class_orig))
            summary["inter_class_sim_debiased"] = float(np.mean(inter_class_debias))
            summary["inter_class_change"] = (
                summary["inter_class_sim_debiased"]
                - summary["inter_class_sim_original"]
            )

        print(f"\n  Summary Statistics:")
        print(f"  " + "=" * 50)

        print(f"\n  Intra-class similarity (same target, different {bias_name}):")
        if intra_class_orig:
            print(f"    Original:  {summary['intra_class_sim_original']:.4f}")
            print(f"    Debiased:  {summary['intra_class_sim_debiased']:.4f}")
            print(f"    Change:    {summary['intra_class_change']:+.4f}")
        else:
            print(f"    N/A (need multiple bias values per target)")

        print(f"\n  Inter-class similarity (different targets):")
        if inter_class_orig:
            print(f"    Original:  {summary['inter_class_sim_original']:.4f}")
            print(f"    Debiased:  {summary['inter_class_sim_debiased']:.4f}")
            print(f"    Change:    {summary['inter_class_change']:+.4f}")
        else:
            print(f"    N/A (need multiple targets)")

        # Interpretation
        if intra_class_orig and inter_class_orig:
            print(f"\n  Interpretation:")
            if summary["intra_class_change"] > 0.01:
                print(
                    f"    ✓ Intra-class similarity INCREASED by {summary['intra_class_change']:.4f}"
                )
                print(
                    f"      → Same-class samples with different bias are now MORE similar"
                )
                print(f"      → Bias information was successfully removed!")
            elif summary["intra_class_change"] < -0.01:
                print(
                    f"    ✗ Intra-class similarity DECREASED by {abs(summary['intra_class_change']):.4f}"
                )
            else:
                print(f"    ~ Intra-class similarity unchanged")

            if summary["inter_class_change"] < -0.01:
                print(
                    f"    ✓ Inter-class similarity DECREASED by {abs(summary['inter_class_change']):.4f}"
                )
                print(f"      → Different classes are now MORE distinguishable")
            elif summary["inter_class_change"] > 0.01:
                print(
                    f"    ! Inter-class similarity INCREASED by {summary['inter_class_change']:.4f}"
                )
            else:
                print(f"    ~ Inter-class similarity unchanged")

        return {
            "groups": [
                {"target": int(t), "bias": int(b), "label": l, "n_samples": n}
                for (t, b), l, n in zip(groups, group_labels, group_sizes)
            ],
            "original_similarity_matrix": original_sim_matrix.tolist(),
            "debiased_similarity_matrix": debiased_sim_matrix.tolist(),
            "change_matrix": change_matrix.tolist(),
            "summary": summary,
        }

    def _print_similarity_matrix(
        self, matrix: torch.Tensor, labels: list, show_sign: bool = False
    ):
        """Pretty print a similarity matrix."""
        n = len(labels)

        # Header
        header = "          " + "  ".join([f"{l:>8}" for l in labels])
        print(f"    {header}")
        print(f"    {'─' * (len(header) + 2)}")

        # Rows
        for i in range(n):
            row_values = []
            for j in range(n):
                val = matrix[i, j].item()
                if show_sign:
                    row_values.append(f"{val:+.4f}")
                else:
                    row_values.append(f"{val:.4f}")
            row_str = "  ".join([f"{v:>8}" for v in row_values])
            print(f"    {labels[i]:>8}  {row_str}")

    # def _visualize_tag_neurons(
    #     self,
    #     features: torch.Tensor,
    #     indices: torch.Tensor,
    #     num_tags_to_visualize: int = 50,
    #     top_k_images: int = 8,
    # ):
    #     """
    #     Visualize top-activating images for each tag neuron.

    #     Creates a grid showing:
    #     - Tag name
    #     - Top-k images that most activate this tag's neuron
    #     - Activation values

    #     This helps verify that tag neurons learned the correct concepts.
    #     """
    #     print(f"\n{'='*60}")
    #     print("Visualizing Tag Neurons")
    #     print(f"{'='*60}")

    #     try:
    #         import matplotlib.pyplot as plt
    #         from matplotlib.gridspec import GridSpec
    #         from PIL import Image
    #     except ImportError:
    #         print("matplotlib or PIL not available, skipping visualization")
    #         return

    #     self.sae.eval()

    #     # Get all activations
    #     print(f"  Computing activations for {len(features)} samples...")
    #     with torch.no_grad():
    #         all_latents = []
    #         batch_size = 256
    #         for i in range(0, len(features), batch_size):
    #             batch = features[i : i + batch_size].to(self.device)
    #             latents = self.sae.encode(batch)
    #             all_latents.append(latents.cpu())
    #         all_latents = torch.cat(all_latents, dim=0)

    #     # Get anchored activations only
    #     anchored_activations = all_latents[:, : self.sae.num_anchored]  # (N, num_tags)

    #     # Sort tags by mean activation (most active first) or by separation
    #     tag_stats = []
    #     for tag, neuron_idx in self.tag_to_neuron.items():
    #         acts = anchored_activations[:, neuron_idx]

    #         # Get indices where this tag is present
    #         present_indices = []
    #         for i, idx in enumerate(indices.tolist()):
    #             if tag in self.index_to_tags.get(idx, []):
    #                 present_indices.append(i)

    #         mean_act = acts.mean().item()
    #         max_act = acts.max().item()

    #         if present_indices:
    #             mean_when_present = acts[present_indices].mean().item()
    #         else:
    #             mean_when_present = 0

    #         tag_stats.append(
    #             {
    #                 "tag": tag,
    #                 "neuron_idx": neuron_idx,
    #                 "mean_act": mean_act,
    #                 "max_act": max_act,
    #                 "mean_when_present": mean_when_present,
    #                 "num_present": len(present_indices),
    #             }
    #         )

    #     # Sort by mean_when_present (tags with strong signal first)
    #     tag_stats.sort(key=lambda x: x["mean_when_present"], reverse=True)

    #     # Limit to top tags
    #     tags_to_viz = tag_stats[:num_tags_to_visualize]

    #     print(f"  Visualizing top {len(tags_to_viz)} tags...")

    #     # Create output directory
    #     viz_dir = os.path.join(self.log_path, "tag_sae", "tag_visualizations")
    #     os.makedirs(viz_dir, exist_ok=True)

    #     # Get image paths from dataloader
    #     image_paths = self._get_image_paths()

    #     # Create visualization for each tag
    #     for tag_info in tqdm(tags_to_viz, desc="Creating visualizations"):
    #         tag = tag_info["tag"]
    #         neuron_idx = tag_info["neuron_idx"]

    #         # Get top-k activating samples for this neuron
    #         acts = anchored_activations[:, neuron_idx]
    #         top_k_indices = acts.argsort(descending=True)[:top_k_images]
    #         top_k_acts = acts[top_k_indices]

    #         # Get original sample indices
    #         top_k_sample_indices = indices[top_k_indices].tolist()

    #         # Check which of these actually have the tag
    #         has_tag = []
    #         for sample_idx in top_k_sample_indices:
    #             tags_for_sample = self.index_to_tags.get(sample_idx, [])
    #             has_tag.append(tag in tags_for_sample)

    #         # Create figure
    #         fig = plt.figure(figsize=(16, 4))
    #         fig.suptitle(
    #             f"Tag: '{tag}' (neuron {neuron_idx})\n"
    #             f"Mean act when present: {tag_info['mean_when_present']:.3f}, "
    #             f"Samples with tag: {tag_info['num_present']}",
    #             fontsize=12,
    #         )

    #         gs = GridSpec(1, top_k_images, figure=fig)

    #         for i, (sample_idx, act_val, tag_present) in enumerate(
    #             zip(top_k_sample_indices, top_k_acts.tolist(), has_tag)
    #         ):
    #             ax = fig.add_subplot(gs[0, i])

    #             # Try to load and display image
    #             if image_paths and sample_idx in image_paths:
    #                 try:
    #                     img = Image.open(image_paths[sample_idx]).convert("RGB")
    #                     ax.imshow(img)
    #                 except Exception as e:
    #                     ax.text(0.5, 0.5, f"idx={sample_idx}", ha="center", va="center")
    #             else:
    #                 ax.text(
    #                     0.5,
    #                     0.5,
    #                     f"idx={sample_idx}",
    #                     ha="center",
    #                     va="center",
    #                     fontsize=8,
    #                 )

    #             # Set border color based on whether tag is present
    #             border_color = "green" if tag_present else "red"
    #             for spine in ax.spines.values():
    #                 spine.set_edgecolor(border_color)
    #                 spine.set_linewidth(3)

    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             ax.set_title(
    #                 f"act={act_val:.2f}\n{'✓' if tag_present else '✗'}", fontsize=9
    #             )

    #         plt.tight_layout()

    #         # Save figure
    #         safe_tag_name = tag.replace("/", "_").replace(" ", "_")[:30]
    #         save_path = os.path.join(
    #             viz_dir, f"tag_{neuron_idx:03d}_{safe_tag_name}.png"
    #         )
    #         plt.savefig(save_path, dpi=100, bbox_inches="tight")
    #         plt.close()

    #     print(f"  Saved {len(tags_to_viz)} visualizations to {viz_dir}")

    #     # Create summary HTML page
    #     self._create_tag_viz_html(tags_to_viz, viz_dir)

    #     return viz_dir

    def _get_image_paths(self) -> dict:
        """
        Get mapping from sample index to image path.
        """
        image_paths = {}

        # Try to get from dataloader dataset
        if "train" in self.dataloaders:
            dataset = self.dataloaders["train"].dataset

            # Check for common dataset structures
            if hasattr(dataset, "samples"):
                # ImageFolder style
                for idx, (path, _) in enumerate(dataset.samples):
                    image_paths[idx] = path
            if hasattr(dataset, "img_fpath_list"):
                # ImageFolder style
                for idx, path in enumerate(dataset.img_fpath_list):
                    image_paths[idx] = path
            if hasattr(dataset, "inputs"):
                # ImageFolder style
                for idx, (path, _) in enumerate(dataset.inputs):
                    image_paths[idx] = path
            elif hasattr(dataset, "imgs"):
                # Another ImageFolder style
                for idx, (path, _) in enumerate(dataset.imgs):
                    image_paths[idx] = path
            elif hasattr(dataset, "data") and hasattr(dataset, "filename"):
                # Waterbirds style
                for idx in range(len(dataset)):
                    if hasattr(dataset, "data_dir"):
                        path = os.path.join(dataset.data_dir, dataset.filename[idx])
                    else:
                        path = dataset.filename[idx]
                    image_paths[idx] = path
            elif hasattr(dataset, "image_paths"):
                # Direct image_paths attribute
                for idx, path in enumerate(dataset.image_paths):
                    image_paths[idx] = path
            elif hasattr(dataset, "df") and "img_filename" in dataset.df.columns:
                # DataFrame-based dataset
                for idx, row in dataset.df.iterrows():
                    if hasattr(dataset, "data_dir"):
                        path = os.path.join(dataset.data_dir, row["img_filename"])
                    else:
                        path = row["img_filename"]
                    image_paths[idx] = path

        if image_paths:
            print(f"  Found {len(image_paths)} image paths")
        else:
            print(f"  Warning: Could not extract image paths from dataset")

        return image_paths

    #     def _create_tag_viz_html(self, tag_stats: list, viz_dir: str):
    #         """Create an HTML page summarizing all tag visualizations."""
    #         html_content = """
    # <!DOCTYPE html>
    # <html>
    # <head>
    #     <title>Tag Neuron Visualizations</title>
    #     <style>
    #         body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
    #         h1 { color: #333; }
    #         .tag-card {
    #             background: white;
    #             border-radius: 8px;
    #             padding: 15px;
    #             margin: 10px 0;
    #             box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    #         }
    #         .tag-header {
    #             display: flex;
    #             justify-content: space-between;
    #             align-items: center;
    #             border-bottom: 1px solid #eee;
    #             padding-bottom: 10px;
    #             margin-bottom: 10px;
    #         }
    #         .tag-name { font-size: 18px; font-weight: bold; color: #2196F3; }
    #         .tag-stats { color: #666; font-size: 14px; }
    #         .tag-image { max-width: 100%; border-radius: 4px; }
    #         .legend {
    #             background: #fff3cd;
    #             padding: 10px;
    #             border-radius: 4px;
    #             margin-bottom: 20px;
    #         }
    #         .legend span { margin-right: 20px; }
    #         .green { color: green; }
    #         .red { color: red; }
    #     </style>
    # </head>
    # <body>
    #     <h1>Tag Neuron Visualizations</h1>
    #     <div class="legend">
    #         <span class="green">✓ Green border = tag IS present in image</span>
    #         <span class="red">✗ Red border = tag NOT present in image</span>
    #     </div>
    # """

    #         for tag_info in tag_stats:
    #             tag = tag_info["tag"]
    #             safe_tag_name = tag.replace("/", "_").replace(" ", "_")[:30]
    #             img_filename = f"tag_{tag_info['neuron_idx']:03d}_{safe_tag_name}.png"

    #             html_content += f"""
    #     <div class="tag-card">
    #         <div class="tag-header">
    #             <span class="tag-name">{tag}</span>
    #             <span class="tag-stats">
    #                 Neuron #{tag_info['neuron_idx']} |
    #                 Samples: {tag_info['num_present']} |
    #                 Mean activation: {tag_info['mean_when_present']:.3f}
    #             </span>
    #         </div>
    #         <img class="tag-image" src="{img_filename}" alt="{tag}">
    #     </div>
    # """

    #         html_content += """
    # </body>
    # </html>
    # """

    #         html_path = os.path.join(viz_dir, "index.html")
    #         with open(html_path, "w") as f:
    #             f.write(html_content)

    #         print(f"  Created summary HTML: {html_path}")

    def _visualize_tag_neurons(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        num_tags_to_visualize: int = 50,
        num_free_to_visualize: int = 50,
        top_k_images: int = 8,
    ):
        """
        Visualize top-activating images for tag neurons AND free neurons.

        Creates grids showing:
        - Tag neurons: Top-k images with tag presence indicated
        - Free neurons: Top-k images to discover what they learned

        This helps verify that:
        1. Tag neurons learned the correct concepts
        2. Free neurons learned something useful (not just noise)
        """
        print(f"\n{'='*60}")
        print("Visualizing SAE Neurons")
        print(f"{'='*60}")

        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            from PIL import Image
        except ImportError:
            print("matplotlib or PIL not available, skipping visualization")
            return

        self.sae.eval()

        # Get all activations
        print(f"  Computing activations for {len(features)} samples...")
        with torch.no_grad():
            all_latents = []
            batch_size = 256
            for i in range(0, len(features), batch_size):
                batch = features[i : i + batch_size].to(self.device)
                latents = self.sae.encode(batch)
                all_latents.append(latents.cpu())
            all_latents = torch.cat(all_latents, dim=0)

        # Split into anchored and free
        anchored_activations = all_latents[:, : self.sae.num_anchored]  # (N, num_tags)
        free_activations = all_latents[:, self.sae.num_anchored :]  # (N, num_free)

        print(f"  Anchored (tag) neurons: {anchored_activations.shape[1]}")
        print(f"  Free neurons: {free_activations.shape[1]}")

        # Get image paths
        image_paths = self._get_image_paths()

        # Create output directories
        viz_dir = os.path.join(self.log_path, "tag_sae", "visualizations")
        tag_viz_dir = os.path.join(viz_dir, "tag_neurons")
        free_viz_dir = os.path.join(viz_dir, "free_neurons")
        os.makedirs(tag_viz_dir, exist_ok=True)
        os.makedirs(free_viz_dir, exist_ok=True)

        # ============================================
        # 1. Visualize Tag Neurons
        # ============================================
        print(f"\n  Visualizing tag neurons...")
        tag_stats = self._visualize_neuron_set(
            activations=anchored_activations,
            indices=indices,
            image_paths=image_paths,
            output_dir=tag_viz_dir,
            neuron_type="tag",
            num_to_visualize=num_tags_to_visualize,
            top_k_images=top_k_images,
            neuron_names={idx: tag for tag, idx in self.tag_to_neuron.items()},
            check_tag_presence=True,
        )

        # ============================================
        # 2. Visualize Free Neurons
        # ============================================
        print(f"\n  Visualizing free neurons...")
        free_stats = self._visualize_neuron_set(
            activations=free_activations,
            indices=indices,
            image_paths=image_paths,
            output_dir=free_viz_dir,
            neuron_type="free",
            num_to_visualize=num_free_to_visualize,
            top_k_images=top_k_images,
            neuron_names=None,  # No predefined names
            check_tag_presence=False,
            neuron_offset=self.sae.num_anchored,  # For correct neuron numbering
        )

        # Create combined HTML summary
        self._create_combined_viz_html(tag_stats, free_stats, viz_dir)

        print(f"\n  Saved visualizations to {viz_dir}")
        return viz_dir

    def _visualize_neuron_set(
        self,
        activations: torch.Tensor,
        indices: torch.Tensor,
        image_paths: dict,
        output_dir: str,
        neuron_type: str,
        num_to_visualize: int,
        top_k_images: int,
        neuron_names: dict = None,
        check_tag_presence: bool = False,
        neuron_offset: int = 0,
    ) -> list:
        """
        Visualize a set of neurons (either tag or free).

        Returns list of neuron stats for HTML generation.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            from PIL import Image
        except ImportError:
            return []

        num_neurons = activations.shape[1]

        # Compute stats for each neuron
        neuron_stats = []
        for neuron_idx in range(num_neurons):
            acts = activations[:, neuron_idx]

            mean_act = acts.mean().item()
            max_act = acts.max().item()
            std_act = acts.std().item()
            num_active = (acts > 0).sum().item()

            # For tag neurons, compute mean when tag is present
            if check_tag_presence and neuron_names and neuron_idx in neuron_names:
                tag = neuron_names[neuron_idx]
                present_indices = []
                for i, idx in enumerate(indices.tolist()):
                    if tag in self.index_to_tags.get(idx, []):
                        present_indices.append(i)
                mean_when_present = (
                    acts[present_indices].mean().item() if present_indices else 0
                )
                num_present = len(present_indices)
            else:
                mean_when_present = mean_act
                num_present = num_active

            neuron_stats.append(
                {
                    "neuron_idx": neuron_idx,
                    "global_idx": neuron_idx + neuron_offset,
                    "name": (
                        neuron_names.get(neuron_idx, f"free_{neuron_idx}")
                        if neuron_names
                        else f"free_{neuron_idx}"
                    ),
                    "mean_act": mean_act,
                    "max_act": max_act,
                    "std_act": std_act,
                    "mean_when_present": mean_when_present,
                    "num_active": num_active,
                    "num_present": num_present,
                }
            )

        # Sort by activity (max activation or mean_when_present)
        if check_tag_presence:
            neuron_stats.sort(key=lambda x: x["mean_when_present"], reverse=True)
        else:
            # For free neurons, sort by max activation (most "interesting" neurons)
            neuron_stats.sort(key=lambda x: x["max_act"], reverse=True)

        # Limit to top neurons
        neurons_to_viz = neuron_stats[:num_to_visualize]

        print(f"    Creating {len(neurons_to_viz)} visualizations...")

        # Create visualization for each neuron
        for neuron_info in tqdm(
            neurons_to_viz, desc=f"    Visualizing {neuron_type} neurons"
        ):
            neuron_idx = neuron_info["neuron_idx"]
            global_idx = neuron_info["global_idx"]
            name = neuron_info["name"]

            # Get top-k activating samples
            acts = activations[:, neuron_idx]
            top_k_indices = acts.argsort(descending=True)[:top_k_images]
            top_k_acts = acts[top_k_indices]
            top_k_sample_indices = indices[top_k_indices].tolist()

            # Check tag presence (for tag neurons)
            if check_tag_presence and neuron_names and neuron_idx in neuron_names:
                tag = neuron_names[neuron_idx]
                has_tag = [
                    tag in self.index_to_tags.get(idx, [])
                    for idx in top_k_sample_indices
                ]
            else:
                has_tag = [None] * len(top_k_sample_indices)

            # Create figure
            fig = plt.figure(figsize=(16, 4))

            if neuron_type == "tag":
                title = (
                    f"Tag Neuron #{global_idx}: '{name}'\n"
                    f"Mean act when present: {neuron_info['mean_when_present']:.3f}, "
                    f"Samples with tag: {neuron_info['num_present']}"
                )
            else:
                title = (
                    f"Free Neuron #{global_idx}\n"
                    f"Max act: {neuron_info['max_act']:.3f}, "
                    f"Mean act: {neuron_info['mean_act']:.3f}, "
                    f"Active in: {neuron_info['num_active']} samples"
                )

            fig.suptitle(title, fontsize=11)

            gs = GridSpec(1, top_k_images, figure=fig)

            for i, (sample_idx, act_val, tag_present) in enumerate(
                zip(top_k_sample_indices, top_k_acts.tolist(), has_tag)
            ):
                ax = fig.add_subplot(gs[0, i])

                # Try to load and display image
                if image_paths and sample_idx in image_paths:
                    try:
                        img = Image.open(image_paths[sample_idx]).convert("RGB")
                        ax.imshow(img)
                    except Exception:
                        ax.text(0.5, 0.5, f"idx={sample_idx}", ha="center", va="center")
                        ax.set_facecolor("#f0f0f0")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"idx={sample_idx}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                    ax.set_facecolor("#f0f0f0")

                # Set border color
                if tag_present is True:
                    border_color = "green"
                    symbol = "✓"
                elif tag_present is False:
                    border_color = "red"
                    symbol = "✗"
                else:
                    border_color = "blue"
                    symbol = ""

                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)

                ax.set_xticks([])
                ax.set_yticks([])

                if symbol:
                    ax.set_title(f"act={act_val:.2f}\n{symbol}", fontsize=9)
                else:
                    ax.set_title(f"act={act_val:.2f}", fontsize=9)

            plt.tight_layout()

            # Save figure
            safe_name = name.replace("/", "_").replace(" ", "_")[:30]
            save_path = os.path.join(
                output_dir, f"neuron_{global_idx:04d}_{safe_name}.png"
            )
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()

        return neurons_to_viz

    def _create_combined_viz_html(
        self, tag_stats: list, free_stats: list, viz_dir: str
    ):
        """Create an HTML page with both tag and free neuron visualizations."""
        html_content = (
            """
<!DOCTYPE html>
<html>
<head>
    <title>SAE Neuron Visualizations</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1, h2 { color: #333; }
        .section { margin-bottom: 40px; }
        .section-header {
            background: #2196F3;
            color: white;
            padding: 15px;
            border-radius: 8px 8px 0 0;
            margin-bottom: 0;
        }
        .section-header.free { background: #4CAF50; }
        .neuron-card {
            background: white;
            border-radius: 0 0 8px 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .neuron-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }
        .neuron-name { font-size: 16px; font-weight: bold; color: #333; }
        .neuron-stats { color: #666; font-size: 13px; }
        .neuron-image { max-width: 100%; border-radius: 4px; }
        .legend {
            background: #fff3cd;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .legend span { margin-right: 20px; }
        .green { color: green; }
        .red { color: red; }
        .blue { color: blue; }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            background: #ddd;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }
        .tab.active { background: #2196F3; color: white; }
        .tab.free.active { background: #4CAF50; }
    </style>
</head>
<body>
    <h1>SAE Neuron Visualizations</h1>
    
    <div class="legend">
        <strong>Legend:</strong>
        <span class="green">✓ Green = tag IS present</span>
        <span class="red">✗ Red = tag NOT present</span>
        <span class="blue">Blue = free neuron (no tag label)</span>
    </div>
    
    <div class="tabs">
        <a href="#tag-neurons" class="tab active">Tag Neurons ("""
            + str(len(tag_stats))
            + """)</a>
        <a href="#free-neurons" class="tab free">Free Neurons ("""
            + str(len(free_stats))
            + """)</a>
    </div>
    
    <!-- TAG NEURONS -->
    <div class="section" id="tag-neurons">
        <h2 class="section-header">Tag Neurons (Supervised)</h2>
"""
        )

        for neuron_info in tag_stats:
            safe_name = neuron_info["name"].replace("/", "_").replace(" ", "_")[:30]
            img_filename = (
                f"tag_neurons/neuron_{neuron_info['global_idx']:04d}_{safe_name}.png"
            )

            html_content += f"""
        <div class="neuron-card">
            <div class="neuron-header">
                <span class="neuron-name">#{neuron_info['global_idx']}: {neuron_info['name']}</span>
                <span class="neuron-stats">
                    Samples: {neuron_info['num_present']} | 
                    Mean activation: {neuron_info['mean_when_present']:.3f} |
                    Max: {neuron_info['max_act']:.3f}
                </span>
            </div>
            <img class="neuron-image" src="{img_filename}" alt="{neuron_info['name']}">
        </div>
"""

        html_content += """
    </div>
    
    <!-- FREE NEURONS -->
    <div class="section" id="free-neurons">
        <h2 class="section-header free">Free Neurons (Unsupervised)</h2>
        <p style="padding: 10px; background: #e8f5e9; border-radius: 4px;">
            These neurons learned features without supervision. 
            Look for semantic patterns (objects, textures, scenes) to verify they captured useful information.
        </p>
"""

        for neuron_info in free_stats:
            safe_name = neuron_info["name"].replace("/", "_").replace(" ", "_")[:30]
            img_filename = (
                f"free_neurons/neuron_{neuron_info['global_idx']:04d}_{safe_name}.png"
            )

            html_content += f"""
        <div class="neuron-card">
            <div class="neuron-header">
                <span class="neuron-name">Free Neuron #{neuron_info['global_idx']}</span>
                <span class="neuron-stats">
                    Active in: {neuron_info['num_active']} samples | 
                    Mean: {neuron_info['mean_act']:.3f} |
                    Max: {neuron_info['max_act']:.3f} |
                    Std: {neuron_info['std_act']:.3f}
                </span>
            </div>
            <img class="neuron-image" src="{img_filename}" alt="Free neuron {neuron_info['global_idx']}">
        </div>
"""

        html_content += """
    </div>
</body>
</html>
"""

        html_path = os.path.join(viz_dir, "index.html")
        with open(html_path, "w") as f:
            f.write(html_content)

        print(f"    Created summary HTML: {html_path}")

    def _evaluate_zero_shot(self, test_features, test_targets, test_biases=None):
        """
        Evaluate using zero-shot classification.

        Compares classification accuracy with:
        1. Original features
        2. Debiased features (all tag neurons zeroed)

        Uses text embeddings of class names for classification.
        """
        print(f"\n{'='*60}")
        print("Zero-Shot Classification Evaluation")
        print(f"{'='*60}")

        # Get class names from target2name
        if not hasattr(self, "target2name") or self.target2name is None:
            print("No target2name mapping found, skipping zero-shot evaluation.")
            return {}

        # Get unique targets
        unique_targets = sorted(test_targets.unique().tolist())
        class_names = [self.target2name.get(t, f"class_{t}") for t in unique_targets]

        print(f"  Classes: {class_names}")

        # We need a text encoder - check if model has one
        text_encoder = self._get_text_encoder()
        if text_encoder is None:
            print("  No text encoder available, skipping zero-shot evaluation.")
            return {}

        # Encode class names to get text embeddings
        print("\n  Encoding class names...")
        with torch.no_grad():
            # Create prompts
            prompts = [f"a photo of a {name}" for name in class_names]
            text_features = text_encoder(prompts)
            text_features_n = F.normalize(text_features, dim=-1)

        print(f"  Text features shape: {text_features.shape}")

        results = {
            "original": {},
            "debiased": {},
            "per_group": {},
        }

        self.sae.eval()

        with torch.no_grad():
            test_features_gpu = test_features.to(self.device)

            # Normalize original features
            original_normed = F.normalize(test_features_gpu, dim=-1)

            # Encode through SAE and debias
            latents = self.sae.encode(test_features_gpu)
            latents_debiased = latents.clone()
            latents_debiased[:, : self.sae.num_anchored] = 0.0
            debiased_features = self.sae.decode(latents_debiased)
            debiased_normed = F.normalize(debiased_features, dim=-1)

            t_latents = self.sae.encode(text_features)
            t_latents_debiased = t_latents.clone()
            t_latents_debiased[:, : self.sae.num_anchored] = 0
            t_debiased_features = self.sae.decode(t_latents_debiased)
            t_debiased_normed = F.normalize(t_debiased_features, dim=-1)

            # Zero-shot classification: similarity with text embeddings
            original_sim = original_normed @ text_features_n.T  # (N, num_classes)
            debiased_sim = debiased_normed @ t_debiased_normed.T  # (N, num_classes)

            # Get predictions
            original_preds = original_sim.argmax(dim=1).cpu()
            debiased_preds = debiased_sim.argmax(dim=1).cpu()

            # Map predictions back to actual target values
            pred_to_target = {i: t for i, t in enumerate(unique_targets)}
            original_preds_mapped = torch.tensor(
                [pred_to_target[p.item()] for p in original_preds]
            )
            debiased_preds_mapped = torch.tensor(
                [pred_to_target[p.item()] for p in debiased_preds]
            )

            # Compute accuracy
            original_acc = (original_preds_mapped == test_targets).float().mean().item()
            debiased_acc = (debiased_preds_mapped == test_targets).float().mean().item()

        results["original"]["accuracy"] = original_acc
        results["debiased"]["accuracy"] = debiased_acc

        print(f"\n  Zero-Shot Classification Results:")
        print(f"    Original accuracy:  {original_acc:.4f}")
        print(f"    Debiased accuracy:  {debiased_acc:.4f}")
        print(f"    Change:             {debiased_acc - original_acc:+.4f}")

        # Per-group evaluation
        bias_name = self.biases[0] if self.biases else None
        if bias_name and test_biases is not None:
            if isinstance(test_biases, dict):
                bias_values = test_biases.get(bias_name)
            else:
                bias_values = test_biases

            if bias_values is not None:
                print(f"\n  Per-group zero-shot accuracy (bias: {bias_name}):")

                unique_biases = sorted(bias_values.unique().tolist())

                for t in unique_targets:
                    for b in unique_biases:
                        mask = (test_targets == t) & (bias_values == b)
                        if mask.sum() == 0:
                            continue

                        group_orig_acc = (
                            (original_preds_mapped[mask] == test_targets[mask])
                            .float()
                            .mean()
                            .item()
                        )
                        group_debiased_acc = (
                            (debiased_preds_mapped[mask] == test_targets[mask])
                            .float()
                            .mean()
                            .item()
                        )

                        group_key = f"target_{t}_bias_{b}"
                        results["per_group"][group_key] = {
                            "original_acc": group_orig_acc,
                            "debiased_acc": group_debiased_acc,
                            "change": group_debiased_acc - group_orig_acc,
                            "n_samples": mask.sum().item(),
                        }

                        print(
                            f"    Group (t={t}, b={b}): "
                            f"orig={group_orig_acc:.4f}, debiased={group_debiased_acc:.4f}, "
                            f"Δ={group_debiased_acc - group_orig_acc:+.4f}, n={mask.sum().item()}"
                        )

                # Worst-group accuracy
                if results["per_group"]:
                    group_accs_orig = [
                        g["original_acc"] for g in results["per_group"].values()
                    ]
                    group_accs_debiased = [
                        g["debiased_acc"] for g in results["per_group"].values()
                    ]

                    results["original"]["worst_group_acc"] = min(group_accs_orig)
                    results["debiased"]["worst_group_acc"] = min(group_accs_debiased)

                    print(f"\n    Worst-group accuracy:")
                    print(
                        f"      Original: {results['original']['worst_group_acc']:.4f}"
                    )
                    print(
                        f"      Debiased: {results['debiased']['worst_group_acc']:.4f}"
                    )
                    print(
                        f"      Change:   {results['debiased']['worst_group_acc'] - results['original']['worst_group_acc']:+.4f}"
                    )

        # Save results
        save_dir = os.path.join(self.log_path, "tag_sae")
        results_path = os.path.join(save_dir, "zero_shot_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved zero-shot results to {results_path}")

        return results

    def _get_text_encoder(self):
        """
        Get text encoder function from the model.
        Returns a callable that takes a list of strings and returns embeddings.

        The text encoder must produce embeddings of the same dimension as the image features.
        """
        target_dim = self.feature_dim
        print(f"  Looking for text encoder with dim={target_dim}...")

        # Check if model is a VLM encoder with text encoding capability
        if hasattr(self.model, "encode_text"):

            def encode_fn(texts):
                return self.model.encode_text(texts).to(self.device)

            # Verify dimension matches
            test_emb = encode_fn(["test"])
            if test_emb.shape[-1] == target_dim:
                print(f"  Using model's encode_text (dim={test_emb.shape[-1]})")
                return encode_fn
            else:
                print(
                    f"  Model's encode_text has wrong dim ({test_emb.shape[-1]} != {target_dim})"
                )

        # Try OpenCLIP with matching dimension
        try:
            import open_clip

            # Map feature dim to likely architecture
            dim_to_arch = {
                512: [
                    ("ViT-B-32", "openai"),
                    ("ViT-B-16", "openai"),
                    ("RN101", "openai"),
                ],
                768: [("ViT-L-14", "openai"), ("ViT-L-14", "laion2b_s32b_b82k")],
                1024: [("ViT-H-14", "laion2b_s32b_b79k"), ("RN50", "openai")],
                1280: [("ViT-G-14", "laion2b_s34b_b88k")],
            }

            candidates = dim_to_arch.get(target_dim, [])

            for arch, pretrained in candidates:
                try:
                    print(f"  Trying OpenCLIP {arch} ({pretrained})...")
                    model, _, _ = open_clip.create_model_and_transforms(
                        arch, pretrained=pretrained, device=self.device
                    )
                    tokenizer = open_clip.get_tokenizer(arch)
                    model.eval()

                    # Verify dimension
                    with torch.no_grad():
                        test_tokens = tokenizer(["test"]).to(self.device)
                        test_emb = model.encode_text(test_tokens)

                    if test_emb.shape[-1] == target_dim:
                        print(f"  Using OpenCLIP {arch} (dim={test_emb.shape[-1]})")

                        def encode_fn(texts):
                            tokens = tokenizer(texts).to(self.device)
                            with torch.no_grad():
                                return model.encode_text(tokens)

                        return encode_fn
                    else:
                        print(f"    Dim mismatch: {test_emb.shape[-1]} != {target_dim}")

                except Exception as e:
                    print(f"    Failed: {e}")
                    continue

        except ImportError:
            print("  OpenCLIP not available")
        except Exception as e:
            print(f"  OpenCLIP error: {e}")

        # Try HuggingFace CLIP models
        try:
            from transformers import CLIPTokenizer, CLIPModel

            # Map feature dim to likely model
            dim_to_model = {
                512: ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16"],
                768: ["openai/clip-vit-large-patch14"],
            }

            candidates = dim_to_model.get(target_dim, [])

            for model_id in candidates:
                try:
                    print(f"  Trying HuggingFace {model_id}...")
                    tokenizer = CLIPTokenizer.from_pretrained(model_id)
                    model = CLIPModel.from_pretrained(model_id).to(self.device)
                    model.eval()

                    # Verify dimension
                    with torch.no_grad():
                        inputs = tokenizer(
                            ["test"], padding=True, return_tensors="pt"
                        ).to(self.device)
                        test_emb = model.get_text_features(**inputs)

                    if test_emb.shape[-1] == target_dim:
                        print(
                            f"  Using HuggingFace {model_id} (dim={test_emb.shape[-1]})"
                        )

                        def encode_fn(texts):
                            inputs = tokenizer(
                                texts, padding=True, return_tensors="pt"
                            ).to(self.device)
                            with torch.no_grad():
                                return model.get_text_features(**inputs)

                        return encode_fn
                    else:
                        print(f"    Dim mismatch: {test_emb.shape[-1]} != {target_dim}")

                except Exception as e:
                    print(f"    Failed: {e}")
                    continue

        except ImportError:
            print("  HuggingFace transformers not available")
        except Exception as e:
            print(f"  HuggingFace error: {e}")

        print(f"  WARNING: Could not find text encoder with dim={target_dim}")
        return None

    def eval(self):
        """
        Evaluation mode: Load trained SAE and evaluate on test set.
        """
        print(f"\n{'='*60}")
        print("Tag-SAE Evaluation Mode")
        print(f"{'='*60}")

        # Load SAE checkpoint
        sae_path = self.tag_cfg.get("SAE_CHECKPOINT_PATH", "")
        if not sae_path:
            sae_path = os.path.join(self.log_path, "tag_sae", "tag_sae.pt")

        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")

        # Load config
        config_path = os.path.join(os.path.dirname(sae_path), "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Rebuild SAE
        self.sae = TagSupervisedSAE(
            input_dim=config["input_dim"],
            dict_size=config["dict_size"],
            num_anchored=config["num_anchored"],
            tag_to_neuron=config["tag_to_neuron"],
        )
        self.sae.load_state_dict(torch.load(sae_path, map_location=self.device))
        self.sae.to(self.device)
        self.sae.eval()

        self.tag_to_neuron = config["tag_to_neuron"]
        self.all_tags = config["all_tags"]

        print(f"Loaded SAE from {sae_path}")
        print(f"  Input dim: {config['input_dim']}")
        print(f"  Anchored neurons: {config['num_anchored']}")

        # Run evaluation
        return self._evaluate_debiasing()


# ============================================
# Config Defaults
# ============================================

"""
Add to configs/cfg.py:

CFG.MITIGATOR.TAG_SAE = CN()

# Path to tags CSV from MAVIAS pipeline
CFG.MITIGATOR.TAG_SAE.TAGS_CSV_PATH = "train_tags.csv"
CFG.MITIGATOR.TAG_SAE.TAG_COLUMN = "irrelevant_tags"
CFG.MITIGATOR.TAG_SAE.TAG_SEPARATOR = " | "
CFG.MITIGATOR.TAG_SAE.MIN_TAG_FREQUENCY = 10

# Model checkpoint (optional)
CFG.MITIGATOR.TAG_SAE.CHECKPOINT_PATH = ""

# Precomputed features (optional)
CFG.MITIGATOR.TAG_SAE.PRECOMPUTED_FEATURES_PATH = ""

# SAE Architecture
CFG.MITIGATOR.TAG_SAE.EXPANSION_FACTOR = 8
CFG.MITIGATOR.TAG_SAE.NUM_FREE_NEURONS = 0  # 0 = use expansion factor

# Training
CFG.MITIGATOR.TAG_SAE.STEPS = 20000
CFG.MITIGATOR.TAG_SAE.BATCH_SIZE = 256
CFG.MITIGATOR.TAG_SAE.LR = 1e-3

# Loss Weights
CFG.MITIGATOR.TAG_SAE.LAMBDA_RECONSTRUCTION = 1.0
CFG.MITIGATOR.TAG_SAE.LAMBDA_SPARSITY = 1e-3
CFG.MITIGATOR.TAG_SAE.LAMBDA_TAG = 1.0

# Tag Supervision Config
CFG.MITIGATOR.TAG_SAE.TAG_LOSS_TYPE = "bce"  # "bce", "hinge", "mse"
CFG.MITIGATOR.TAG_SAE.POSITIVE_WEIGHT = 1.0
CFG.MITIGATOR.TAG_SAE.NEGATIVE_WEIGHT = 0.5
CFG.MITIGATOR.TAG_SAE.USE_NEGATIVE_SUPERVISION = True
CFG.MITIGATOR.TAG_SAE.MARGIN = 0.5  # For hinge loss
CFG.MITIGATOR.TAG_SAE.TARGET_ACTIVATION = 1.0  # For MSE loss
"""
