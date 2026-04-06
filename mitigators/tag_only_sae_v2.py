"""
Enhanced Tag-Only SAE with Bias and Target Class Neurons.

This SAE uses ONLY supervised neurons (no free neurons).
All neurons are organized into:

IRRELEVANT (zeroed during debiasing):
  - Tag neurons from irrelevant_tags column
  - Bias label neurons (one per bias class, e.g., background=water, background=land)

RELEVANT (kept during debiasing):
  - Tag neurons from relevant tags
  - Target class neurons (one per class, e.g., class=waterbird, class=landbird)

Architecture:
    dict_size = num_irrelevant_tags + num_bias_classes + num_relevant_tags + num_target_classes

Pipeline:
    Training:
        All neurons supervised by their respective labels (tags, bias, target)

    Debiasing:
        Zero ONLY irrelevant neurons (irrelevant tags + bias labels)
        Keep relevant neurons (relevant tags + target labels)

Config:
    MITIGATOR:
      TYPE: "tag_only_sae_v2"
      TAG_ONLY_SAE_V2:
        TAGS_CSV_PATH: "train_tags.csv"
        ALL_TAGS_COLUMN: "tags"
        IRRELEVANT_TAGS_COLUMN: "irrelevant_tags"
        TAG_SEPARATOR: " | "
        MIN_TAG_FREQUENCY: 10

        # Training
        STEPS: 20000
        BATCH_SIZE: 256
        LR: 1e-3

        # Loss weights
        LAMBDA_RECONSTRUCTION: 1.0
        LAMBDA_SPARSITY: 1e-3
        LAMBDA_TAG: 1.0
        LAMBDA_BIAS: 1.0      # Weight for bias label supervision
        LAMBDA_TARGET: 1.0    # Weight for target class supervision
"""

import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base_trainer import BaseTrainer
from models.builder import get_model
from models.tag_supervised_sae import TagSupervisionLoss


class TagOnlySAEv2(nn.Module):
    """
    Enhanced SAE where ALL neurons are supervised:
    - Irrelevant tag neurons
    - Bias label neurons
    - Relevant tag neurons
    - Target class neurons

    Neuron layout:
        [0 : num_irrelevant_tags]                                    → Irrelevant tags
        [num_irrelevant_tags : num_irrelevant_tags + num_bias]       → Bias labels
        [... : ... + num_relevant_tags]                              → Relevant tags
        [... : ... + num_targets]                                    → Target classes
    """

    def __init__(
        self,
        input_dim: int,
        num_irrelevant_tags: int,
        num_bias_neurons: int,
        num_relevant_tags: int,
        num_target_neurons: int,
        irrelevant_tag_to_neuron: Dict[str, int],
        bias_to_neuron: Dict[str, int],  # e.g., {"background_0": 5, "background_1": 6}
        relevant_tag_to_neuron: Dict[str, int],
        target_to_neuron: Dict[int, int],  # e.g., {0: 10, 1: 11}
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_irrelevant_tags = num_irrelevant_tags
        self.num_bias_neurons = num_bias_neurons
        self.num_relevant_tags = num_relevant_tags
        self.num_target_neurons = num_target_neurons

        # Total neurons
        self.dict_size = (
            num_irrelevant_tags
            + num_bias_neurons
            + num_relevant_tags
            + num_target_neurons
        )

        # Neuron ranges
        self.irrelevant_tag_start = 0
        self.irrelevant_tag_end = num_irrelevant_tags

        self.bias_start = self.irrelevant_tag_end
        self.bias_end = self.bias_start + num_bias_neurons

        self.relevant_tag_start = self.bias_end
        self.relevant_tag_end = self.relevant_tag_start + num_relevant_tags

        self.target_start = self.relevant_tag_end
        self.target_end = self.target_start + num_target_neurons

        # Number of neurons to zero during debiasing
        self.num_irrelevant = num_irrelevant_tags + num_bias_neurons
        self.num_relevant = num_relevant_tags + num_target_neurons

        # Mappings
        self.irrelevant_tag_to_neuron = irrelevant_tag_to_neuron
        self.bias_to_neuron = bias_to_neuron
        self.relevant_tag_to_neuron = relevant_tag_to_neuron
        self.target_to_neuron = target_to_neuron

        # Reverse mappings
        self.neuron_to_irrelevant_tag = {
            v: k for k, v in irrelevant_tag_to_neuron.items()
        }
        self.neuron_to_bias = {v: k for k, v in bias_to_neuron.items()}
        self.neuron_to_relevant_tag = {v: k for k, v in relevant_tag_to_neuron.items()}
        self.neuron_to_target = {v: k for k, v in target_to_neuron.items()}

        # Encoder: input_dim → dict_size
        self.encoder = nn.Linear(input_dim, self.dict_size)
        self.encoder_bias = nn.Parameter(torch.zeros(input_dim))

        # Decoder: dict_size → input_dim
        self.decoder = nn.Linear(self.dict_size, input_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to neuron activations."""
        x_centered = x - self.encoder_bias
        latents = F.relu(self.encoder(x_centered))
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode neuron activations to reconstructed input."""
        return self.decoder(latents) + self.decoder_bias

    def forward(self, x: torch.Tensor, return_latents: bool = False):
        """Forward pass."""
        latents = self.encode(x)
        reconstructed = self.decode(latents)
        if return_latents:
            return reconstructed, latents
        return reconstructed

    def get_irrelevant_activations(self, latents: torch.Tensor) -> torch.Tensor:
        """Get activations for irrelevant neurons (tags + bias)."""
        return latents[:, : self.num_irrelevant]

    def get_relevant_activations(self, latents: torch.Tensor) -> torch.Tensor:
        """Get activations for relevant neurons (tags + target)."""
        return latents[:, self.num_irrelevant :]

    def get_irrelevant_tag_activations(self, latents: torch.Tensor) -> torch.Tensor:
        """Get activations for irrelevant tag neurons only."""
        return latents[:, self.irrelevant_tag_start : self.irrelevant_tag_end]

    def get_bias_activations(self, latents: torch.Tensor) -> torch.Tensor:
        """Get activations for bias label neurons."""
        return latents[:, self.bias_start : self.bias_end]

    def get_relevant_tag_activations(self, latents: torch.Tensor) -> torch.Tensor:
        """Get activations for relevant tag neurons only."""
        return latents[:, self.relevant_tag_start : self.relevant_tag_end]

    def get_target_activations(self, latents: torch.Tensor) -> torch.Tensor:
        """Get activations for target class neurons."""
        return latents[:, self.target_start : self.target_end]

    def decode_debiased(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode after zeroing all irrelevant neurons (tags + bias)."""
        latents_debiased = latents.clone()
        latents_debiased[:, : self.num_irrelevant] = 0
        return self.decode(latents_debiased)

    def decode_without_neurons(
        self, latents: torch.Tensor, neuron_indices: List[int]
    ) -> torch.Tensor:
        """Decode after zeroing specific neurons."""
        latents_modified = latents.clone()
        for idx in neuron_indices:
            latents_modified[:, idx] = 0
        return self.decode(latents_modified)


class TagOnlySAEv2Trainer(BaseTrainer):
    """
    Trainer for Enhanced Tag-Only SAE with bias and target neurons.
    """

    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """Setup the encoder model."""
        self.model = get_model(
            self.cfg.MODEL.TYPE, self.num_class, pretrained=self.cfg.MODEL.PRETRAINED
        )
        self.model.to(self.device)
        self.model.eval()

        # Freeze encoder
        for param in self.model.parameters():
            param.requires_grad = False

    def _method_specific_setups(self):
        """Setup tag-specific components."""
        self.tag_cfg = self.cfg.MITIGATOR.TAG_ONLY_SAE_V2

        # Load tags
        self._load_tags()

        # Setup bias and target info
        self._setup_bias_and_target_neurons()

        # Get feature dimension
        self._get_feature_dim()

        # Build complete neuron layout
        self._build_neuron_layout()

    def _load_tags(self):
        """Load tags from CSV."""
        csv_path = os.path.join(self.data_root, self.tag_cfg.TAGS_CSV_PATH)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Tags CSV not found: {csv_path}")

        print(f"\nLoading tags from {csv_path}")
        df = pd.read_csv(csv_path)

        all_tags_col = self.tag_cfg.ALL_TAGS_COLUMN
        irrelevant_tags_col = self.tag_cfg.IRRELEVANT_TAGS_COLUMN
        separator = self.tag_cfg.TAG_SEPARATOR
        min_freq = self.tag_cfg.MIN_TAG_FREQUENCY

        # Parse tags per sample
        self.index_to_all_tags = {}
        self.index_to_irrelevant_tags = {}
        all_tag_counts = defaultdict(int)
        irrelevant_tag_set = set()

        for _, row in df.iterrows():
            idx = int(row["index"])

            # Parse all tags
            all_tags_str = row.get(all_tags_col, "")
            if pd.isna(all_tags_str) or all_tags_str == "":
                all_tags = []
            else:
                all_tags = [
                    t.strip() for t in str(all_tags_str).split(separator) if t.strip()
                ]

            # Parse irrelevant tags
            irr_tags_str = row.get(irrelevant_tags_col, "")
            if pd.isna(irr_tags_str) or irr_tags_str == "":
                irrelevant_tags = []
            else:
                irrelevant_tags = [
                    t.strip() for t in str(irr_tags_str).split(separator) if t.strip()
                ]

            self.index_to_all_tags[idx] = all_tags
            self.index_to_irrelevant_tags[idx] = irrelevant_tags

            for tag in all_tags:
                all_tag_counts[tag] += 1
            for tag in irrelevant_tags:
                irrelevant_tag_set.add(tag)

        # Filter tags by frequency
        all_tags_filtered = set(
            tag for tag, count in all_tag_counts.items() if count >= min_freq
        )

        # Split into irrelevant and relevant
        self.irrelevant_tags = sorted(
            [tag for tag in irrelevant_tag_set if tag in all_tags_filtered]
        )
        self.relevant_tags = sorted(
            [tag for tag in all_tags_filtered if tag not in irrelevant_tag_set]
        )
        self.all_tags = self.irrelevant_tags + self.relevant_tags

        print(f"  Irrelevant tags: {len(self.irrelevant_tags)}")
        print(f"  Relevant tags: {len(self.relevant_tags)}")

    def _setup_bias_and_target_neurons(self):
        """Setup bias label and target class neurons."""
        print(f"\nSetting up bias and target neurons...")

        # Get bias information from first batch
        # We need to scan the dataset to find unique bias values
        self.bias_classes = {}  # {bias_name: [class_0, class_1, ...]}

        for bias_name in self.biases:
            unique_values = set()
            for batch in self.dataloaders["train"]:
                if bias_name in batch:
                    unique_values.update(batch[bias_name].tolist())
                if len(unique_values) > 100:  # Safety limit
                    break
            self.bias_classes[bias_name] = sorted(unique_values)
            print(
                f"  Bias '{bias_name}': {len(self.bias_classes[bias_name])} classes -> {self.bias_classes[bias_name]}"
            )

        # Get target classes
        self.target_classes = list(range(self.num_class))
        print(f"  Target classes: {len(self.target_classes)} -> {self.target_classes}")

        # Count total bias neurons
        self.num_bias_neurons = sum(
            len(classes) for classes in self.bias_classes.values()
        )
        self.num_target_neurons = len(self.target_classes)

        print(f"\n  Total bias neurons: {self.num_bias_neurons}")
        print(f"  Total target neurons: {self.num_target_neurons}")

    def _get_feature_dim(self):
        """Get feature dimension from model."""
        if hasattr(self.model, "fc"):
            self.feature_dim = self.model.fc.in_features
        elif hasattr(self.model, "classifier"):
            if hasattr(self.model.classifier, "in_features"):
                self.feature_dim = self.model.classifier.in_features
            else:
                self.feature_dim = self.model.classifier[0].in_features
        elif hasattr(self.model, "head"):
            self.feature_dim = self.model.head.in_features
        else:
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                out = self.model(dummy)
                if isinstance(out, tuple):
                    self.feature_dim = out[1].shape[-1]
                else:
                    self.feature_dim = out.shape[-1]

        print(f"  Feature dimension: {self.feature_dim}")

    def _build_neuron_layout(self):
        """Build the complete neuron layout and mappings."""
        print(f"\nBuilding neuron layout...")

        neuron_idx = 0

        # 1. Irrelevant tag neurons
        self.irrelevant_tag_to_neuron = {}
        for tag in self.irrelevant_tags:
            self.irrelevant_tag_to_neuron[tag] = neuron_idx
            neuron_idx += 1

        # 2. Bias label neurons
        self.bias_to_neuron = {}
        for bias_name in self.biases:
            for bias_class in self.bias_classes[bias_name]:
                key = f"{bias_name}_{bias_class}"
                self.bias_to_neuron[key] = neuron_idx
                neuron_idx += 1

        # 3. Relevant tag neurons
        self.relevant_tag_to_neuron = {}
        for tag in self.relevant_tags:
            self.relevant_tag_to_neuron[tag] = neuron_idx
            neuron_idx += 1

        # 4. Target class neurons
        self.target_to_neuron = {}
        for target_class in self.target_classes:
            self.target_to_neuron[target_class] = neuron_idx
            neuron_idx += 1

        self.total_neurons = neuron_idx

        print(f"\n  Neuron layout:")
        print(
            f"    [0-{len(self.irrelevant_tags)-1}]: Irrelevant tags ({len(self.irrelevant_tags)})"
        )
        print(
            f"    [{len(self.irrelevant_tags)}-{len(self.irrelevant_tags)+self.num_bias_neurons-1}]: Bias labels ({self.num_bias_neurons})"
        )
        print(
            f"    [{len(self.irrelevant_tags)+self.num_bias_neurons}-{len(self.irrelevant_tags)+self.num_bias_neurons+len(self.relevant_tags)-1}]: Relevant tags ({len(self.relevant_tags)})"
        )
        print(
            f"    [{len(self.irrelevant_tags)+self.num_bias_neurons+len(self.relevant_tags)}-{self.total_neurons-1}]: Target classes ({self.num_target_neurons})"
        )
        print(f"    Total neurons: {self.total_neurons}")

        # Neurons to zero during debiasing
        self.irrelevant_neuron_indices = list(
            self.irrelevant_tag_to_neuron.values()
        ) + list(self.bias_to_neuron.values())
        self.relevant_neuron_indices = list(
            self.relevant_tag_to_neuron.values()
        ) + list(self.target_to_neuron.values())

        print(f"\n  Debiasing will zero {len(self.irrelevant_neuron_indices)} neurons")
        print(f"  Debiasing will keep {len(self.relevant_neuron_indices)} neurons")

    def _extract_features(
        self, dataloader: DataLoader, desc: str = "Extracting features"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract features from dataloader."""
        self.model.eval()

        all_features = []
        all_targets = []
        all_indices = []
        all_biases = {b: [] for b in self.biases}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"]
                indices = batch["index"]

                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    features = outputs[1]
                else:
                    features = outputs

                all_features.append(features.cpu())
                all_targets.append(targets)
                all_indices.append(indices)

                for b in self.biases:
                    if b in batch:
                        all_biases[b].append(batch[b])

        features = torch.cat(all_features, dim=0)
        targets = torch.cat(all_targets, dim=0)
        indices = torch.cat(all_indices, dim=0)

        for b in self.biases:
            if all_biases[b]:
                all_biases[b] = torch.cat(all_biases[b], dim=0)

        return features, targets, indices, all_biases

    def _build_supervision_targets(
        self,
        indices: torch.Tensor,
        targets: torch.Tensor,
        biases: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Build complete supervision target matrix.

        Returns: (batch_size, total_neurons) binary matrix
        """
        batch_size = len(indices)
        supervision = torch.zeros(batch_size, self.total_neurons)

        # 1. Irrelevant tag targets
        for i, idx in enumerate(indices.tolist()):
            irr_tags = self.index_to_irrelevant_tags.get(idx, [])
            for tag in irr_tags:
                if tag in self.irrelevant_tag_to_neuron:
                    supervision[i, self.irrelevant_tag_to_neuron[tag]] = 1

        # 2. Bias label targets
        for bias_name in self.biases:
            if bias_name in biases and len(biases[bias_name]) > 0:
                for i, bias_val in enumerate(biases[bias_name].tolist()):
                    key = f"{bias_name}_{bias_val}"
                    if key in self.bias_to_neuron:
                        supervision[i, self.bias_to_neuron[key]] = 1

        # 3. Relevant tag targets
        for i, idx in enumerate(indices.tolist()):
            all_tags = self.index_to_all_tags.get(idx, [])
            for tag in all_tags:
                if tag in self.relevant_tag_to_neuron:
                    supervision[i, self.relevant_tag_to_neuron[tag]] = 1

        # 4. Target class targets
        for i, target_val in enumerate(targets.tolist()):
            if target_val in self.target_to_neuron:
                supervision[i, self.target_to_neuron[target_val]] = 1

        return supervision

    def _setup_optimizer(self):
        """SAE has its own optimizer."""
        self.optimizer = None

    def _setup_scheduler(self):
        """SAE has its own scheduler."""
        self.scheduler = None

    def _train_sae(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        indices: torch.Tensor,
        biases: Dict[str, torch.Tensor],
    ):
        """Train the Enhanced Tag-Only SAE."""
        print(f"\n{'='*60}")
        print("Training Enhanced Tag-Only SAE")
        print(f"{'='*60}")

        # Create SAE
        self.sae = TagOnlySAEv2(
            input_dim=self.feature_dim,
            num_irrelevant_tags=len(self.irrelevant_tags),
            num_bias_neurons=self.num_bias_neurons,
            num_relevant_tags=len(self.relevant_tags),
            num_target_neurons=self.num_target_neurons,
            irrelevant_tag_to_neuron=self.irrelevant_tag_to_neuron,
            bias_to_neuron=self.bias_to_neuron,
            relevant_tag_to_neuron=self.relevant_tag_to_neuron,
            target_to_neuron=self.target_to_neuron,
        ).to(self.device)

        print(f"\n  SAE Architecture:")
        print(f"    Input dim: {self.feature_dim}")
        print(f"    Dict size: {self.sae.dict_size}")
        print(f"    Irrelevant (tags + bias): {self.sae.num_irrelevant}")
        print(f"    Relevant (tags + target): {self.sae.num_relevant}")

        # Loss functions
        tag_loss_fn = TagSupervisionLoss(
            loss_type=self.tag_cfg.TAG_LOSS_TYPE,
            positive_weight=self.tag_cfg.POSITIVE_WEIGHT,
            negative_weight=self.tag_cfg.NEGATIVE_WEIGHT,
            margin=self.tag_cfg.MARGIN,
            target_activation=self.tag_cfg.TARGET_ACTIVATION,
            use_negative_supervision=self.tag_cfg.USE_NEGATIVE_SUPERVISION,
        )

        # Optimizer
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=self.tag_cfg.LR)

        # Create dataset with all data
        # Stack biases into tensor
        bias_tensors = []
        for b in self.biases:
            if b in biases and len(biases[b]) > 0:
                bias_tensors.append(biases[b])

        if bias_tensors:
            bias_stack = torch.stack(bias_tensors, dim=1)  # (N, num_biases)
        else:
            bias_stack = torch.zeros(len(features), 1)

        dataset = TensorDataset(features, targets, indices, bias_stack)
        dataloader = DataLoader(
            dataset, batch_size=self.tag_cfg.BATCH_SIZE, shuffle=True
        )

        # Training loop
        num_steps = self.tag_cfg.STEPS
        step = 0
        losses_history = defaultdict(list)

        # Get loss weights
        lambda_recon = self.tag_cfg.LAMBDA_RECONSTRUCTION
        lambda_sparse = self.tag_cfg.LAMBDA_SPARSITY
        lambda_tag = self.tag_cfg.LAMBDA_TAG
        lambda_bias = self.tag_cfg.get("LAMBDA_BIAS", 1.0)
        lambda_target = self.tag_cfg.get("LAMBDA_TARGET", 1.0)

        print(f"\n  Loss weights:")
        print(f"    Reconstruction: {lambda_recon}")
        print(f"    Sparsity: {lambda_sparse}")
        print(f"    Tag supervision: {lambda_tag}")
        print(f"    Bias supervision: {lambda_bias}")
        print(f"    Target supervision: {lambda_target}")

        self.sae.train()
        pbar = tqdm(total=num_steps, desc="Training SAE")

        while step < num_steps:
            for batch_data in dataloader:
                if step >= num_steps:
                    break

                batch_features = batch_data[0].to(self.device)
                batch_targets = batch_data[1]
                batch_indices = batch_data[2]
                batch_bias_stack = batch_data[3]

                # Reconstruct biases dict
                batch_biases = {}
                for i, b in enumerate(self.biases):
                    batch_biases[b] = (
                        batch_bias_stack[:, i]
                        if batch_bias_stack.shape[1] > i
                        else torch.zeros(len(batch_features))
                    )

                # Forward pass
                reconstructed, latents = self.sae(batch_features, return_latents=True)

                # Build supervision targets
                supervision = self._build_supervision_targets(
                    batch_indices, batch_targets, batch_biases
                ).to(self.device)

                # ============================================
                # Losses
                # ============================================

                # 1. Reconstruction loss
                loss_recon = F.mse_loss(reconstructed, batch_features)

                # 2. Sparsity loss
                loss_sparsity = latents.abs().mean()

                # 3. Irrelevant tag supervision
                irr_tag_acts = self.sae.get_irrelevant_tag_activations(latents)
                irr_tag_targets = supervision[
                    :, self.sae.irrelevant_tag_start : self.sae.irrelevant_tag_end
                ]
                loss_irr_tags = tag_loss_fn(irr_tag_acts, irr_tag_targets)

                # 4. Bias label supervision
                bias_acts = self.sae.get_bias_activations(latents)
                bias_targets = supervision[:, self.sae.bias_start : self.sae.bias_end]
                loss_bias = tag_loss_fn(bias_acts, bias_targets)

                # 5. Relevant tag supervision
                rel_tag_acts = self.sae.get_relevant_tag_activations(latents)
                rel_tag_targets = supervision[
                    :, self.sae.relevant_tag_start : self.sae.relevant_tag_end
                ]
                loss_rel_tags = tag_loss_fn(rel_tag_acts, rel_tag_targets)

                # 6. Target class supervision
                target_acts = self.sae.get_target_activations(latents)
                target_targets = supervision[
                    :, self.sae.target_start : self.sae.target_end
                ]
                loss_target = tag_loss_fn(target_acts, target_targets)

                # Combined tag loss
                loss_tags = loss_irr_tags + loss_rel_tags

                # Total loss
                loss = (
                    lambda_recon * loss_recon
                    + lambda_sparse * loss_sparsity
                    + lambda_tag * loss_tags
                    + lambda_bias * loss_bias
                    + lambda_target * loss_target
                )

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Normalize decoder
                with torch.no_grad():
                    norms = self.sae.decoder.weight.norm(dim=0, keepdim=True)
                    self.sae.decoder.weight.div_(norms.clamp(min=1e-6))

                # Log
                losses_history["total"].append(loss.item())
                losses_history["reconstruction"].append(loss_recon.item())
                losses_history["sparsity"].append(loss_sparsity.item())
                losses_history["irr_tags"].append(loss_irr_tags.item())
                losses_history["bias"].append(loss_bias.item())
                losses_history["rel_tags"].append(loss_rel_tags.item())
                losses_history["target"].append(loss_target.item())

                step += 1
                pbar.update(1)

                if step % 1000 == 0:
                    l0 = (latents > 0).float().sum(dim=1).mean().item()

                    # Energy distribution
                    irr_energy = (
                        self.sae.get_irrelevant_activations(latents)
                        .pow(2)
                        .sum(dim=1)
                        .mean()
                        .item()
                    )
                    rel_energy = (
                        self.sae.get_relevant_activations(latents)
                        .pow(2)
                        .sum(dim=1)
                        .mean()
                        .item()
                    )
                    total_energy = irr_energy + rel_energy + 1e-8

                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "recon": f"{loss_recon.item():.6f}",
                            "bias": f"{loss_bias.item():.4f}",
                            "tgt": f"{loss_target.item():.4f}",
                            "L0": f"{l0:.0f}",
                            "irr%": f"{100*irr_energy/total_energy:.0f}",
                        }
                    )

                if step % 5000 == 0:
                    print(f"\n  Step {step}:")
                    print(
                        f"    Irr tags: {loss_irr_tags.item():.4f}, Bias: {loss_bias.item():.4f}"
                    )
                    print(
                        f"    Rel tags: {loss_rel_tags.item():.4f}, Target: {loss_target.item():.4f}"
                    )
                    print(
                        f"    Energy - Irrelevant: {100*irr_energy/total_energy:.1f}%, Relevant: {100*rel_energy/total_energy:.1f}%"
                    )

        pbar.close()

        # Save
        self._save_sae(losses_history)

        return self.sae

    def _save_sae(self, losses_history: dict):
        """Save the trained SAE."""
        save_dir = os.path.join(self.log_path, "tag_only_sae_v2")
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(save_dir, "sae.pt")
        torch.save(self.sae.state_dict(), model_path)
        print(f"\nSaved model to {model_path}")

        # Save config
        config = {
            "input_dim": self.feature_dim,
            "dict_size": self.sae.dict_size,
            "num_irrelevant_tags": len(self.irrelevant_tags),
            "num_bias_neurons": self.num_bias_neurons,
            "num_relevant_tags": len(self.relevant_tags),
            "num_target_neurons": self.num_target_neurons,
            "num_irrelevant": self.sae.num_irrelevant,
            "num_relevant": self.sae.num_relevant,
            "irrelevant_tag_to_neuron": self.irrelevant_tag_to_neuron,
            "bias_to_neuron": self.bias_to_neuron,
            "relevant_tag_to_neuron": self.relevant_tag_to_neuron,
            "target_to_neuron": self.target_to_neuron,
            "irrelevant_tags": self.irrelevant_tags,
            "relevant_tags": self.relevant_tags,
            "bias_classes": self.bias_classes,
            "target_classes": self.target_classes,
            "biases": self.biases,
        }
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save losses
        losses_path = os.path.join(save_dir, "losses.json")
        with open(losses_path, "w") as f:
            json.dump({k: v[-1000:] for k, v in losses_history.items()}, f)

        print(f"  Saved config to {config_path}")

    def _evaluate_debiasing(
        self,
        test_features: torch.Tensor,
        test_targets: torch.Tensor,
        test_biases: Dict[str, torch.Tensor],
    ):
        """Evaluate debiasing effect."""
        print(f"\n{'='*60}")
        print("Evaluating Debiasing")
        print(f"{'='*60}")

        self.sae.eval()

        with torch.no_grad():
            test_features_gpu = test_features.to(self.device)

            # Encode
            latents = self.sae.encode(test_features_gpu)

            # Original and debiased reconstruction
            original = self.sae.decode(latents)
            debiased = self.sae.decode_debiased(latents)

            # Energy distribution
            irr_energy = (
                self.sae.get_irrelevant_activations(latents)
                .pow(2)
                .sum(dim=1)
                .mean()
                .item()
            )
            rel_energy = (
                self.sae.get_relevant_activations(latents)
                .pow(2)
                .sum(dim=1)
                .mean()
                .item()
            )
            total_energy = irr_energy + rel_energy + 1e-8

            # More detailed breakdown
            irr_tag_energy = (
                self.sae.get_irrelevant_tag_activations(latents)
                .pow(2)
                .sum(dim=1)
                .mean()
                .item()
            )
            bias_energy = (
                self.sae.get_bias_activations(latents).pow(2).sum(dim=1).mean().item()
            )
            rel_tag_energy = (
                self.sae.get_relevant_tag_activations(latents)
                .pow(2)
                .sum(dim=1)
                .mean()
                .item()
            )
            target_energy = (
                self.sae.get_target_activations(latents).pow(2).sum(dim=1).mean().item()
            )

            # Reconstruction errors
            orig_mse = F.mse_loss(original, test_features_gpu).item()
            debiased_mse = F.mse_loss(debiased, test_features_gpu).item()

            # Cosine similarities
            orig_norm = F.normalize(original, dim=-1)
            debiased_norm = F.normalize(debiased, dim=-1)
            input_norm = F.normalize(test_features_gpu, dim=-1)

            cos_orig = (orig_norm * input_norm).sum(dim=1).mean().item()
            cos_debiased = (debiased_norm * input_norm).sum(dim=1).mean().item()

        print(f"\n  Energy Distribution:")
        print(
            f"    Irrelevant tags: {irr_tag_energy:.4f} ({100*irr_tag_energy/total_energy:.1f}%)"
        )
        print(
            f"    Bias labels:     {bias_energy:.4f} ({100*bias_energy/total_energy:.1f}%)"
        )
        print(
            f"    Relevant tags:   {rel_tag_energy:.4f} ({100*rel_tag_energy/total_energy:.1f}%)"
        )
        print(
            f"    Target classes:  {target_energy:.4f} ({100*target_energy/total_energy:.1f}%)"
        )
        print(f"    ─────────────────────────────")
        print(f"    Total Irrelevant: {100*irr_energy/total_energy:.1f}%")
        print(f"    Total Relevant:   {100*rel_energy/total_energy:.1f}%")

        print(f"\n  Reconstruction:")
        print(f"    Original MSE:  {orig_mse:.6f}")
        print(f"    Debiased MSE:  {debiased_mse:.6f}")
        print(f"    Cosine (orig): {cos_orig:.4f}")
        print(f"    Cosine (deb):  {cos_debiased:.4f}")

        # Zero-shot evaluation
        results = {
            "energy": {
                "irrelevant_tags": irr_tag_energy,
                "bias_labels": bias_energy,
                "relevant_tags": rel_tag_energy,
                "target_classes": target_energy,
                "irrelevant_fraction": irr_energy / total_energy,
                "relevant_fraction": rel_energy / total_energy,
            },
            "reconstruction": {
                "original_mse": orig_mse,
                "debiased_mse": debiased_mse,
                "cosine_original": cos_orig,
                "cosine_debiased": cos_debiased,
            },
        }

        # Zero-shot evaluation
        zero_shot_results = self._evaluate_zero_shot(
            test_features, test_targets, test_biases
        )
        if zero_shot_results:
            results["zero_shot"] = zero_shot_results

        # Target neuron classification evaluation
        target_neuron_results = self._evaluate_target_neuron_classification(
            test_features, test_targets, test_biases
        )
        if target_neuron_results:
            results["target_neuron_classification"] = target_neuron_results

        # Group similarity
        similarity_results = self._evaluate_group_similarity(
            test_features, test_targets, test_biases
        )
        if similarity_results:
            results["group_similarity"] = similarity_results

        # Save results
        save_dir = os.path.join(self.log_path, "tag_only_sae_v2")
        results_path = os.path.join(save_dir, "debiasing_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {results_path}")

        return results

    def _evaluate_zero_shot(
        self,
        test_features: torch.Tensor,
        test_targets: torch.Tensor,
        test_biases: Dict[str, torch.Tensor],
    ) -> dict:
        """Zero-shot classification evaluation."""
        print(f"\n{'='*60}")
        print("Zero-Shot Classification")
        print(f"{'='*60}")

        if not hasattr(self, "target2name") or self.target2name is None:
            print("  No target2name mapping, skipping.")
            return {}

        text_encoder = self._get_text_encoder()
        if text_encoder is None:
            print("  No text encoder available, skipping.")
            return {}

        unique_targets = sorted(test_targets.unique().tolist())
        class_names = [self.target2name.get(t, f"class_{t}") for t in unique_targets]
        print(f"  Classes: {class_names}")

        with torch.no_grad():
            prompts = [f"a photo of a {name}" for name in class_names]
            text_features = text_encoder(prompts)
            text_features = F.normalize(text_features, dim=-1)

        self.sae.eval()

        with torch.no_grad():
            test_features_gpu = test_features.to(self.device)

            # Original
            original_norm = F.normalize(test_features_gpu, dim=-1)

            # Debiased
            latents = self.sae.encode(test_features_gpu)
            debiased = self.sae.decode_debiased(latents)
            debiased_norm = F.normalize(debiased, dim=-1)

            # Classification
            orig_sim = original_norm @ text_features.T
            deb_sim = debiased_norm @ text_features.T

            orig_preds = orig_sim.argmax(dim=1).cpu()
            deb_preds = deb_sim.argmax(dim=1).cpu()

            pred_to_target = {i: t for i, t in enumerate(unique_targets)}
            orig_preds_mapped = torch.tensor(
                [pred_to_target[p.item()] for p in orig_preds]
            )
            deb_preds_mapped = torch.tensor(
                [pred_to_target[p.item()] for p in deb_preds]
            )

            orig_acc = (orig_preds_mapped == test_targets).float().mean().item()
            deb_acc = (deb_preds_mapped == test_targets).float().mean().item()

        print(f"\n  Overall Accuracy:")
        print(f"    Original: {orig_acc:.4f}")
        print(f"    Debiased: {deb_acc:.4f}")
        print(f"    Change:   {deb_acc - orig_acc:+.4f}")

        results = {"original_acc": orig_acc, "debiased_acc": deb_acc}

        # Per-group accuracy
        if self.biases and test_biases:
            bias_name = self.biases[0]
            bias_values = test_biases.get(bias_name)

            if bias_values is not None:
                print(f"\n  Per-group accuracy (bias: {bias_name}):")

                unique_biases = sorted(bias_values.unique().tolist())
                group_accs_orig = []
                group_accs_deb = []

                for t in unique_targets:
                    for b in unique_biases:
                        mask = (test_targets == t) & (bias_values == b)
                        if mask.sum() == 0:
                            continue

                        o_acc = (
                            (orig_preds_mapped[mask] == test_targets[mask])
                            .float()
                            .mean()
                            .item()
                        )
                        d_acc = (
                            (deb_preds_mapped[mask] == test_targets[mask])
                            .float()
                            .mean()
                            .item()
                        )

                        group_accs_orig.append(o_acc)
                        group_accs_deb.append(d_acc)

                        results[f"group_t{t}_b{b}_orig"] = o_acc
                        results[f"group_t{t}_b{b}_deb"] = d_acc

                        print(
                            f"    (t={t}, b={b}): orig={o_acc:.4f}, deb={d_acc:.4f}, Δ={d_acc-o_acc:+.4f}, n={mask.sum().item()}"
                        )

                if group_accs_orig:
                    results["worst_group_orig"] = min(group_accs_orig)
                    results["worst_group_deb"] = min(group_accs_deb)

                    print(f"\n    Worst-group:")
                    print(f"      Original: {results['worst_group_orig']:.4f}")
                    print(f"      Debiased: {results['worst_group_deb']:.4f}")
                    print(
                        f"      Change:   {results['worst_group_deb'] - results['worst_group_orig']:+.4f}"
                    )

        return results

    def _evaluate_target_neuron_classification(
        self,
        test_features: torch.Tensor,
        test_targets: torch.Tensor,
        test_biases: Dict[str, torch.Tensor],
    ) -> dict:
        """
        Evaluate classification using target neuron activations directly.

        The target neurons are supervised to activate for their respective classes,
        so we can use argmax over target neuron activations as a classifier.

        This evaluates how well the SAE learned to separate classes.
        """
        print(f"\n{'='*60}")
        print("Target Neuron Classification")
        print(f"{'='*60}")

        self.sae.eval()

        # Get the mapping from neuron index to target class
        neuron_to_target = self.sae.neuron_to_target
        target_neuron_indices = sorted(neuron_to_target.keys())

        # Create mapping from position in target activations to actual class
        pos_to_class = {
            i: neuron_to_target[idx] for i, idx in enumerate(target_neuron_indices)
        }

        print(f"  Target neurons: {len(target_neuron_indices)}")
        print(f"  Neuron to class mapping: {neuron_to_target}")

        with torch.no_grad():
            test_features_gpu = test_features.to(self.device)

            # Encode to get latents
            latents = self.sae.encode(test_features_gpu)

            # Get target neuron activations
            target_acts = self.sae.get_target_activations(latents)  # (N, num_targets)

            # Predict class as argmax of target activations
            pred_positions = target_acts.argmax(dim=1).cpu()

            # Map positions back to actual class labels
            predictions = torch.tensor([pos_to_class[p.item()] for p in pred_positions])

            # Compute accuracy
            correct = (predictions == test_targets).float()
            overall_acc = correct.mean().item()

        results = {
            "overall_accuracy": overall_acc,
        }

        print(f"\n  Overall Accuracy: {overall_acc:.4f}")

        # Per-class accuracy
        unique_targets = sorted(test_targets.unique().tolist())
        print(f"\n  Per-class accuracy:")
        for t in unique_targets:
            mask = test_targets == t
            if mask.sum() > 0:
                class_acc = correct[mask].mean().item()
                results[f"class_{t}_accuracy"] = class_acc
                class_name = (
                    self.target2name.get(t, f"class_{t}")
                    if hasattr(self, "target2name")
                    else f"class_{t}"
                )
                print(
                    f"    {class_name} (class {t}): {class_acc:.4f} (n={mask.sum().item()})"
                )

        # Per-group accuracy (target x bias combinations)
        if self.biases and test_biases:
            bias_name = self.biases[0]
            bias_values = test_biases.get(bias_name)

            if bias_values is not None:
                print(f"\n  Per-group accuracy (bias: {bias_name}):")

                unique_biases = sorted(bias_values.unique().tolist())
                group_accs = []

                for t in unique_targets:
                    for b in unique_biases:
                        mask = (test_targets == t) & (bias_values == b)
                        if mask.sum() == 0:
                            continue

                        group_acc = correct[mask].mean().item()
                        group_accs.append(group_acc)

                        results[f"group_t{t}_b{b}_accuracy"] = group_acc

                        # Get class name
                        class_name = (
                            self.target2name.get(t, f"t{t}")
                            if hasattr(self, "target2name")
                            else f"t{t}"
                        )
                        print(
                            f"    ({class_name}, bias={b}): {group_acc:.4f} (n={mask.sum().item()})"
                        )

                if group_accs:
                    results["worst_group_accuracy"] = min(group_accs)
                    results["best_group_accuracy"] = max(group_accs)
                    results["avg_group_accuracy"] = np.mean(group_accs)
                    results["group_accuracy_gap"] = max(group_accs) - min(group_accs)

                    print(f"\n    Summary:")
                    print(f"      Worst-group: {results['worst_group_accuracy']:.4f}")
                    print(f"      Best-group:  {results['best_group_accuracy']:.4f}")
                    print(f"      Gap:         {results['group_accuracy_gap']:.4f}")

        # Analyze target neuron activations
        print(f"\n  Target Neuron Activation Statistics:")
        with torch.no_grad():
            target_acts_np = target_acts.cpu().numpy()

            for i, t in enumerate(unique_targets):
                mask = (test_targets == t).numpy()

                # Activation of correct neuron for this class
                correct_neuron_idx = i  # Position in target_acts
                correct_acts = target_acts_np[mask, correct_neuron_idx]

                # Activation of incorrect neurons for this class
                other_acts = np.delete(target_acts_np[mask], correct_neuron_idx, axis=1)

                results[f"class_{t}_correct_neuron_mean"] = float(np.mean(correct_acts))
                results[f"class_{t}_correct_neuron_std"] = float(np.std(correct_acts))
                results[f"class_{t}_other_neurons_mean"] = float(np.mean(other_acts))

                class_name = (
                    self.target2name.get(t, f"class_{t}")
                    if hasattr(self, "target2name")
                    else f"class_{t}"
                )
                print(f"    {class_name}:")
                print(
                    f"      Correct neuron: mean={np.mean(correct_acts):.4f}, std={np.std(correct_acts):.4f}"
                )
                print(f"      Other neurons:  mean={np.mean(other_acts):.4f}")
                print(
                    f"      Separation:     {np.mean(correct_acts) - np.mean(other_acts):.4f}"
                )

        return results

    def _evaluate_group_similarity(
        self,
        test_features: torch.Tensor,
        test_targets: torch.Tensor,
        test_biases: Dict[str, torch.Tensor],
    ) -> dict:
        """Compute group similarity matrices."""
        if not self.biases or not test_biases:
            return {}

        bias_name = self.biases[0]
        bias_values = test_biases.get(bias_name)
        if bias_values is None:
            return {}

        print(f"\n{'='*60}")
        print("Group Similarity Analysis")
        print(f"{'='*60}")

        self.sae.eval()

        with torch.no_grad():
            test_features_gpu = test_features.to(self.device)

            latents = self.sae.encode(test_features_gpu)
            original_norm = F.normalize(test_features_gpu, dim=-1)
            debiased = self.sae.decode_debiased(latents)
            debiased_norm = F.normalize(debiased, dim=-1)

        # Compute centroids
        unique_targets = sorted(test_targets.unique().tolist())
        unique_biases = sorted(bias_values.unique().tolist())

        groups = []
        orig_centroids = []
        deb_centroids = []

        for t in unique_targets:
            for b in unique_biases:
                mask = (test_targets == t) & (bias_values == b)
                if mask.sum() == 0:
                    continue
                groups.append((t, b))
                orig_centroids.append(original_norm[mask].mean(dim=0))
                deb_centroids.append(debiased_norm[mask].mean(dim=0))

        if len(groups) < 2:
            return {}

        orig_centroids = torch.stack(orig_centroids)
        deb_centroids = torch.stack(deb_centroids)

        orig_sim = orig_centroids @ orig_centroids.T
        deb_sim = deb_centroids @ deb_centroids.T

        # Print matrices
        labels = [f"t{t}b{b}" for t, b in groups]

        print(f"\n  Original Similarity:")
        self._print_matrix(orig_sim, labels)

        print(f"\n  Debiased Similarity:")
        self._print_matrix(deb_sim, labels)

        print(f"\n  Change (Debiased - Original):")
        self._print_matrix(deb_sim - orig_sim, labels, show_sign=True)

        # Summary metrics
        intra_orig, intra_deb = [], []
        inter_orig, inter_deb = [], []

        for i, (t1, b1) in enumerate(groups):
            for j, (t2, b2) in enumerate(groups):
                if i >= j:
                    continue
                if t1 == t2:
                    intra_orig.append(orig_sim[i, j].item())
                    intra_deb.append(deb_sim[i, j].item())
                else:
                    inter_orig.append(orig_sim[i, j].item())
                    inter_deb.append(deb_sim[i, j].item())

        results = {}

        if intra_orig:
            results["intra_class_orig"] = np.mean(intra_orig)
            results["intra_class_deb"] = np.mean(intra_deb)
            print(f"\n  Intra-class (same target, diff bias):")
            print(f"    Original: {results['intra_class_orig']:.4f}")
            print(f"    Debiased: {results['intra_class_deb']:.4f}")
            change = results["intra_class_deb"] - results["intra_class_orig"]
            print(f"    Change:   {change:+.4f} {'✓' if change > 0 else '✗'}")

        if inter_orig:
            results["inter_class_orig"] = np.mean(inter_orig)
            results["inter_class_deb"] = np.mean(inter_deb)
            print(f"\n  Inter-class (diff target):")
            print(f"    Original: {results['inter_class_orig']:.4f}")
            print(f"    Debiased: {results['inter_class_deb']:.4f}")
            change = results["inter_class_deb"] - results["inter_class_orig"]
            print(f"    Change:   {change:+.4f} {'✓' if change < 0 else '✗'}")

        return results

    def _print_matrix(self, matrix, labels, show_sign=False):
        """Print similarity matrix."""
        n = len(labels)
        header = "        " + "  ".join([f"{l:>7}" for l in labels])
        print(f"    {header}")

        for i in range(n):
            row = []
            for j in range(n):
                val = matrix[i, j].item()
                if show_sign:
                    row.append(f"{val:+.3f}")
                else:
                    row.append(f"{val:.3f}")
            row_str = "  ".join([f"{v:>7}" for v in row])
            print(f"    {labels[i]:>7}  {row_str}")

    def _get_text_encoder(self):
        """Get text encoder matching feature dimension."""
        target_dim = self.feature_dim

        try:
            import open_clip

            dim_to_arch = {
                512: [("ViT-B-32", "openai"), ("ViT-B-16", "openai")],
                768: [("ViT-L-14", "openai")],
                1024: [("ViT-H-14", "laion2b_s32b_b79k")],
            }

            for arch, pretrained in dim_to_arch.get(target_dim, []):
                try:
                    model, _, _ = open_clip.create_model_and_transforms(
                        arch, pretrained=pretrained, device=self.device
                    )
                    tokenizer = open_clip.get_tokenizer(arch)
                    model.eval()

                    with torch.no_grad():
                        test_emb = model.encode_text(
                            tokenizer(["test"]).to(self.device)
                        )

                    if test_emb.shape[-1] == target_dim:

                        def encode_fn(texts):
                            tokens = tokenizer(texts).to(self.device)
                            with torch.no_grad():
                                return model.encode_text(tokens)

                        return encode_fn
                except:
                    continue
        except ImportError:
            pass

        return None

    def _visualize_neurons(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        indices: torch.Tensor,
        biases: Dict[str, torch.Tensor],
    ):
        """Visualize all neuron types."""
        print(f"\n{'='*60}")
        print("Visualizing Neurons")
        print(f"{'='*60}")

        try:
            import matplotlib.pyplot as plt
            from PIL import Image
        except ImportError:
            print("matplotlib/PIL not available, skipping.")
            return

        self.sae.eval()

        with torch.no_grad():
            all_latents = []
            for i in range(0, len(features), 256):
                batch = features[i : i + 256].to(self.device)
                latents = self.sae.encode(batch)
                all_latents.append(latents.cpu())
            all_latents = torch.cat(all_latents, dim=0)

        image_paths = self._get_image_paths()

        viz_dir = os.path.join(self.log_path, "tag_only_sae_v2", "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        top_k = 8

        # Visualize each neuron type
        neuron_groups = [
            ("irrelevant_tags", self.irrelevant_tag_to_neuron, "Irrelevant Tags"),
            ("bias_labels", self.bias_to_neuron, "Bias Labels"),
            ("relevant_tags", self.relevant_tag_to_neuron, "Relevant Tags"),
            (
                "target_classes",
                {f"class_{k}": v for k, v in self.target_to_neuron.items()},
                "Target Classes",
            ),
        ]

        for group_name, mapping, title in neuron_groups:
            group_dir = os.path.join(viz_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)

            print(f"\n  Visualizing {title} ({len(mapping)} neurons)...")

            for name, neuron_idx in tqdm(
                list(mapping.items())[:30], desc=f"    {group_name}"
            ):
                acts = all_latents[:, neuron_idx]
                top_k_idx = acts.argsort(descending=True)[:top_k]
                top_k_acts = acts[top_k_idx]
                top_k_samples = indices[top_k_idx].tolist()

                fig, axes = plt.subplots(1, top_k, figsize=(16, 3))
                fig.suptitle(f"{title}: '{name}' (neuron {neuron_idx})", fontsize=11)

                for i, (sample_idx, act_val) in enumerate(
                    zip(top_k_samples, top_k_acts.tolist())
                ):
                    ax = axes[i]

                    if image_paths and sample_idx in image_paths:
                        try:
                            img = Image.open(image_paths[sample_idx]).convert("RGB")
                            ax.imshow(img)
                        except:
                            ax.text(
                                0.5, 0.5, f"idx={sample_idx}", ha="center", va="center"
                            )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            f"idx={sample_idx}",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"act={act_val:.2f}", fontsize=9)

                plt.tight_layout()
                safe_name = str(name).replace("/", "_").replace(" ", "_")[:25]
                plt.savefig(
                    os.path.join(group_dir, f"{neuron_idx:04d}_{safe_name}.png"), dpi=80
                )
                plt.close()

        # Create HTML
        self._create_viz_html(viz_dir, neuron_groups)
        print(f"\n  Saved visualizations to {viz_dir}")

    def _get_image_paths(self) -> dict:
        """Get image paths from dataset."""
        image_paths = {}

        if "train" in self.dataloaders:
            dataset = self.dataloaders["train"].dataset

            if hasattr(dataset, "samples"):
                for idx, (path, _) in enumerate(dataset.samples):
                    image_paths[idx] = path
            elif hasattr(dataset, "img_fpath_list"):
                # ImageFolder style
                for idx, path in enumerate(dataset.img_fpath_list):
                    image_paths[idx] = path
            elif hasattr(dataset, "imgs"):
                for idx, (path, _) in enumerate(dataset.imgs):
                    image_paths[idx] = path
            elif hasattr(dataset, "df") and "img_filename" in dataset.df.columns:
                for idx, row in dataset.df.iterrows():
                    path = row["img_filename"]
                    if hasattr(dataset, "data_dir"):
                        path = os.path.join(dataset.data_dir, path)
                    image_paths[idx] = path

        return image_paths

    def _create_viz_html(self, viz_dir, neuron_groups):
        """Create HTML summary."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Tag-Only SAE Visualizations</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        h1, h2 { color: #333; }
        .section { margin: 30px 0; padding: 20px; background: white; border-radius: 8px; }
        .irrelevant { border-left: 4px solid #f44336; }
        .relevant { border-left: 4px solid #4CAF50; }
        .grid { display: flex; flex-wrap: wrap; gap: 10px; }
        .card { background: #fafafa; padding: 10px; border-radius: 4px; }
        .card img { max-width: 100%; }
        .legend { background: #fff3cd; padding: 15px; border-radius: 4px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Enhanced Tag-Only SAE Neuron Visualizations</h1>
    <div class="legend">
        <strong>Neuron Types:</strong><br>
        <span style="color: #f44336">■ IRRELEVANT (zeroed during debiasing):</span> Irrelevant Tags + Bias Labels<br>
        <span style="color: #4CAF50">■ RELEVANT (kept during debiasing):</span> Relevant Tags + Target Classes
    </div>
"""

        for group_name, mapping, title in neuron_groups:
            is_irrelevant = group_name in ["irrelevant_tags", "bias_labels"]
            css_class = "irrelevant" if is_irrelevant else "relevant"

            html += (
                f'<div class="section {css_class}"><h2>{title}</h2><div class="grid">'
            )

            group_dir = os.path.join(viz_dir, group_name)
            if os.path.exists(group_dir):
                for img_file in sorted(os.listdir(group_dir))[:30]:
                    if img_file.endswith(".png"):
                        name = (
                            img_file.split("_", 1)[1]
                            .replace(".png", "")
                            .replace("_", " ")
                        )
                        html += f'<div class="card"><img src="{group_name}/{img_file}"><p>{name}</p></div>'

            html += "</div></div>"

        html += "</body></html>"

        with open(os.path.join(viz_dir, "index.html"), "w") as f:
            f.write(html)

    def train(self):
        """Main training pipeline."""
        print(f"\n{'='*60}")
        print("Enhanced Tag-Only SAE Training (v2)")
        print(f"{'='*60}")

        # Step 1: Extract features
        print("\nStep 1: Extracting features...")
        features, targets, indices, biases = self._extract_features(
            self.dataloaders["train"], desc="Extracting train features"
        )
        print(f"  Features: {features.shape}")
        print(f"  Targets: {targets.shape}")
        for b in self.biases:
            if b in biases:
                print(f"  Bias '{b}': {biases[b].shape}")

        # Step 2: Train SAE
        print("\nStep 2: Training SAE...")
        self._train_sae(features, targets, indices, biases)

        # Step 3: Visualize
        print("\nStep 3: Visualizing neurons...")
        self._visualize_neurons(features, targets, indices, biases)

        # Step 4: Evaluate
        print("\nStep 4: Evaluating on test set...")
        if "test" in self.dataloaders:
            test_features, test_targets, test_indices, test_biases = (
                self._extract_features(
                    self.dataloaders["test"], desc="Extracting test features"
                )
            )
            self._evaluate_debiasing(test_features, test_targets, test_biases)

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Output: {self.log_path}")

    def eval(self):
        """Evaluation mode."""
        # Load SAE
        sae_dir = os.path.join(self.log_path, "tag_only_sae_v2")
        sae_path = os.path.join(sae_dir, "sae.pt")
        config_path = os.path.join(sae_dir, "config.json")

        if not os.path.exists(sae_path):
            sae_path = self.tag_cfg.get("SAE_CHECKPOINT_PATH", "")
            config_path = os.path.join(os.path.dirname(sae_path), "config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.sae = TagOnlySAEv2(
            input_dim=config["input_dim"],
            num_irrelevant_tags=config["num_irrelevant_tags"],
            num_bias_neurons=config["num_bias_neurons"],
            num_relevant_tags=config["num_relevant_tags"],
            num_target_neurons=config["num_target_neurons"],
            irrelevant_tag_to_neuron=config["irrelevant_tag_to_neuron"],
            bias_to_neuron=config["bias_to_neuron"],
            relevant_tag_to_neuron=config["relevant_tag_to_neuron"],
            target_to_neuron={int(k): v for k, v in config["target_to_neuron"].items()},
        ).to(self.device)
        self.sae.load_state_dict(torch.load(sae_path, map_location=self.device))

        # Evaluate
        test_features, test_targets, test_indices, test_biases = self._extract_features(
            self.dataloaders["test"], desc="Extracting test features"
        )
        self._evaluate_debiasing(test_features, test_targets, test_biases)


# ============================================
# Config Defaults - Add to configs/cfg.py
# ============================================
"""
CFG.MITIGATOR.TAG_ONLY_SAE_V2 = CN()

# Tags CSV
CFG.MITIGATOR.TAG_ONLY_SAE_V2.TAGS_CSV_PATH = "train_tags.csv"
CFG.MITIGATOR.TAG_ONLY_SAE_V2.ALL_TAGS_COLUMN = "tags"
CFG.MITIGATOR.TAG_ONLY_SAE_V2.IRRELEVANT_TAGS_COLUMN = "irrelevant_tags"
CFG.MITIGATOR.TAG_ONLY_SAE_V2.TAG_SEPARATOR = " | "
CFG.MITIGATOR.TAG_ONLY_SAE_V2.MIN_TAG_FREQUENCY = 10

# Checkpoints
CFG.MITIGATOR.TAG_ONLY_SAE_V2.SAE_CHECKPOINT_PATH = ""
CFG.MITIGATOR.TAG_ONLY_SAE_V2.PRECOMPUTED_FEATURES_PATH = ""

# Training
CFG.MITIGATOR.TAG_ONLY_SAE_V2.STEPS = 20000
CFG.MITIGATOR.TAG_ONLY_SAE_V2.BATCH_SIZE = 256
CFG.MITIGATOR.TAG_ONLY_SAE_V2.LR = 1e-3

# Loss weights
CFG.MITIGATOR.TAG_ONLY_SAE_V2.LAMBDA_RECONSTRUCTION = 1.0
CFG.MITIGATOR.TAG_ONLY_SAE_V2.LAMBDA_SPARSITY = 1e-3
CFG.MITIGATOR.TAG_ONLY_SAE_V2.LAMBDA_TAG = 1.0
CFG.MITIGATOR.TAG_ONLY_SAE_V2.LAMBDA_BIAS = 1.0
CFG.MITIGATOR.TAG_ONLY_SAE_V2.LAMBDA_TARGET = 1.0

# Tag supervision
CFG.MITIGATOR.TAG_ONLY_SAE_V2.TAG_LOSS_TYPE = "bce"
CFG.MITIGATOR.TAG_ONLY_SAE_V2.POSITIVE_WEIGHT = 1.0
CFG.MITIGATOR.TAG_ONLY_SAE_V2.NEGATIVE_WEIGHT = 0.5
CFG.MITIGATOR.TAG_ONLY_SAE_V2.USE_NEGATIVE_SUPERVISION = True
CFG.MITIGATOR.TAG_ONLY_SAE_V2.MARGIN = 0.5
CFG.MITIGATOR.TAG_ONLY_SAE_V2.TARGET_ACTIVATION = 1.0
"""
