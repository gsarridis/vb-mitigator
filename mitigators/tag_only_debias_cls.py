"""
Tag-Only SAE Debiased Classifier Trainer.

This mitigator trains a classification head on selectively debiased features
from a pre-trained Tag-Only SAE. Only irrelevant (bias) tags are removed,
while relevant tags are preserved.

Pipeline:
    Image → VLM Encoder → Features → SAE Encode → Zero Irrelevant Tag Neurons → SAE Decode → Classifier → Predictions

Usage:
    1. First train a Tag-Only SAE with tag_only_sae mitigator
    2. Then train a classifier on debiased features with this mitigator

Config:
    MITIGATOR:
      TYPE: "tag_only_debias_cls"
      TAG_ONLY_DEBIAS_CLS:
        # Path to trained Tag-Only SAE
        SAE_CHECKPOINT_PATH: "outputs/tag_only_sae/tag_only_sae.pt"

        # Classifier architecture
        CLASSIFIER_TYPE: "linear"  # "linear", "mlp"
        HIDDEN_DIM: 256
        DROPOUT: 0.1

        # Training
        EPOCHS: 50
        LR: 1e-3
        WEIGHT_DECAY: 1e-4
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base_trainer import BaseTrainer
from models.builder import get_model


class TagOnlySAE(nn.Module):
    """
    SAE where ALL neurons are tag-supervised (no free neurons).
    Duplicated here for standalone loading.
    """

    def __init__(
        self,
        input_dim: int,
        num_tags: int,
        tag_to_neuron: Dict[str, int],
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_tags = num_tags
        self.dict_size = num_tags
        self.tag_to_neuron = tag_to_neuron
        self.neuron_to_tag = {v: k for k, v in tag_to_neuron.items()}

        # Encoder
        self.encoder = nn.Linear(input_dim, num_tags)
        self.encoder_bias = nn.Parameter(torch.zeros(input_dim))

        # Decoder
        self.decoder = nn.Linear(num_tags, input_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.encoder_bias
        latents = F.relu(self.encoder(x_centered))
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.decoder_bias

    def forward(self, x: torch.Tensor, return_latents: bool = False):
        latents = self.encode(x)
        reconstructed = self.decode(latents)
        if return_latents:
            return reconstructed, latents
        return reconstructed

    def decode_without_tags(
        self, latents: torch.Tensor, tags_to_remove: List[str]
    ) -> torch.Tensor:
        """Decode after zeroing specific tag neurons."""
        latents_debiased = latents.clone()
        for tag in tags_to_remove:
            if tag in self.tag_to_neuron:
                neuron_idx = self.tag_to_neuron[tag]
                latents_debiased[:, neuron_idx] = 0
        return self.decode(latents_debiased)

    def get_neuron_indices_for_tags(self, tags: List[str]) -> List[int]:
        """Get neuron indices for a list of tags."""
        return [self.tag_to_neuron[tag] for tag in tags if tag in self.tag_to_neuron]


class LinearClassifier(nn.Module):
    """Simple linear classifier."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)


class MLPClassifier(nn.Module):
    """MLP classifier with one hidden layer."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TagOnlyDebiasClassifierTrainer(BaseTrainer):
    """
    Trains a classifier on Tag-Only SAE debiased features.

    This trainer:
    1. Loads a pre-trained Tag-Only SAE
    2. Extracts features from images using a frozen VLM encoder
    3. Debiases features by zeroing ONLY irrelevant tag neurons
    4. Trains a classifier on the debiased features

    Key difference from sae_debias_cls:
    - Only removes irrelevant tags, keeps relevant tags
    """

    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)

    def _setup_models(self):
        """Setup the VLM encoder, SAE, and classifier."""
        # Load the VLM encoder (frozen)
        self.backbone = get_model(
            self.cfg.MODEL.TYPE, self.num_class, pretrained=self.cfg.MODEL.PRETRAINED
        )
        self.backbone.to(self.device)
        self.backbone.eval()

        # Freeze encoder
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Load Tag-Only SAE
        self._load_tag_only_sae()

        # Create classifier
        self._create_classifier()

    def _load_tag_only_sae(self):
        """Load the pre-trained Tag-Only SAE."""
        sae_cfg = self.cfg.MITIGATOR.TAG_ONLY_DEBIAS_CLS
        sae_path = sae_cfg.SAE_CHECKPOINT_PATH

        if not sae_path or not os.path.exists(sae_path):
            raise FileNotFoundError(
                f"Tag-Only SAE checkpoint not found: {sae_path}\n"
                "Please train a Tag-Only SAE first using the tag_only_sae mitigator."
            )

        # Load config
        config_path = os.path.join(os.path.dirname(sae_path), "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SAE config not found: {config_path}")

        with open(config_path, "r") as f:
            sae_config = json.load(f)

        print(f"\nLoading Tag-Only SAE from {sae_path}")
        print(f"  Input dim: {sae_config['input_dim']}")
        print(f"  Num tags (dict size): {sae_config['num_tags']}")
        print(f"  Irrelevant tags: {len(sae_config['irrelevant_tags'])}")
        print(f"  Relevant tags: {len(sae_config['relevant_tags'])}")

        # Create and load SAE
        self.sae = TagOnlySAE(
            input_dim=sae_config["input_dim"],
            num_tags=sae_config["num_tags"],
            tag_to_neuron=sae_config["tag_to_neuron"],
        )
        self.sae.load_state_dict(torch.load(sae_path, map_location=self.device))
        self.sae.to(self.device)
        self.sae.eval()

        # Freeze SAE
        for param in self.sae.parameters():
            param.requires_grad = False

        # Store config
        self.feature_dim = sae_config["input_dim"]
        self.tag_to_neuron = sae_config["tag_to_neuron"]
        self.all_tags = sae_config["all_tags"]
        self.irrelevant_tags = sae_config["irrelevant_tags"]
        self.relevant_tags = sae_config["relevant_tags"]

        print(f"\n  Debiasing strategy:")
        print(f"    Will ZERO {len(self.irrelevant_tags)} irrelevant tag neurons")
        print(f"    Will KEEP {len(self.relevant_tags)} relevant tag neurons")

    def _create_classifier(self):
        """Create the classification head."""
        sae_cfg = self.cfg.MITIGATOR.TAG_ONLY_DEBIAS_CLS
        classifier_type = sae_cfg.CLASSIFIER_TYPE

        if classifier_type == "linear":
            self.model = LinearClassifier(
                input_dim=self.feature_dim,
                num_classes=self.num_class,
                dropout=sae_cfg.DROPOUT,
            )
        elif classifier_type == "mlp":
            self.model = MLPClassifier(
                input_dim=self.feature_dim,
                num_classes=self.num_class,
                hidden_dim=sae_cfg.HIDDEN_DIM,
                dropout=sae_cfg.DROPOUT,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.model.to(self.device)

        print(f"\nClassifier: {classifier_type}")
        print(f"  Input dim: {self.feature_dim}")
        print(f"  Output dim: {self.num_class}")

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {num_params:,}")

    def _method_specific_setups(self):
        """Additional setup."""
        self.criterion = nn.CrossEntropyLoss()
        self.sae_cfg = self.cfg.MITIGATOR.TAG_ONLY_DEBIAS_CLS

    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract features from images using the frozen encoder."""
        self.backbone.eval()

        with torch.no_grad():
            outputs = self.backbone(inputs)

            if isinstance(outputs, tuple):
                features = outputs[1]
            else:
                features = outputs

        return features

    def _debias_features(self, features: torch.Tensor) -> torch.Tensor:
        """Debias features by zeroing only irrelevant tag neurons."""
        self.sae.eval()

        with torch.no_grad():
            # Encode
            latents = self.sae.encode(features)

            # Decode without irrelevant tags (selective debiasing)
            debiased = self.sae.decode_without_tags(latents, self.irrelevant_tags)
            # debiased = self.sae.decode(latents)

        return debiased

    def _setup_resume(self):
        return

    def _train_iter(self, batch) -> Dict:
        """Single training iteration."""
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        # Extract and debias features
        features = self._extract_features(inputs)
        debiased_features = self._debias_features(features)

        # Forward through classifier
        self.model.train()
        logits = self.model(debiased_features)

        # Compute loss
        loss = self.criterion(logits, targets)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "train_loss": loss,
        }

    def _val_iter(self, batch, stage: str = "val") -> Dict:
        """Single validation iteration."""
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        # Extract and debias features
        features = self._extract_features(inputs)
        debiased_features = self._debias_features(features)

        # Forward through classifier
        self.model.eval()
        with torch.no_grad():
            logits = self.model(debiased_features)

        loss = self.criterion(logits, targets)

        batch_dict = {
            "predictions": logits.argmax(dim=1).cpu(),
            "targets": batch["targets"],
        }

        for b in self.biases:
            if b in batch:
                batch_dict[b] = batch[b]

        return batch_dict, loss
