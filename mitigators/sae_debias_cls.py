"""
SAE-Debiased Classifier Trainer.

This mitigator trains a classification head on debiased features from a 
pre-trained Tag-Supervised SAE. The debiasing removes tag-associated 
information from features before classification.

Pipeline:
    Image → VLM Encoder → Features → SAE Encode → Zero Tag Neurons → SAE Decode → Classifier → Predictions

Usage:
    1. First train a Tag-SAE with tag_sae mitigator
    2. Then train a classifier on debiased features with this mitigator

Config:
    MITIGATOR:
      TYPE: "sae_debias_cls"
      SAE_DEBIAS_CLS:
        # Path to trained Tag-SAE
        SAE_CHECKPOINT_PATH: "outputs/tag_sae/tag_sae.pt"
        
        # Debiasing mode
        DEBIAS_MODE: "all"  # "all", "specific", "none"
        TAGS_TO_REMOVE: []  # For mode="specific"
        
        # Classifier architecture
        CLASSIFIER_TYPE: "linear"  # "linear", "mlp"
        HIDDEN_DIM: 256  # For MLP
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
from tools.utils import log_msg

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np
import torch

# Import Tag-SAE components
from models.tag_supervised_sae import TagSupervisedSAE


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
        dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class SAEDebiasClassifierTrainer(BaseTrainer):
    """
    Trains a classifier on SAE-debiased features.
    
    This trainer:
    1. Loads a pre-trained Tag-SAE
    2. Extracts features from images using a frozen VLM encoder
    3. Debiases features by zeroing tag neurons in SAE latent space
    4. Trains a classifier on the debiased features
    """
    
    def __init__(self, cfg):
        self._cfg = cfg
        super().__init__(cfg)
    
    def _setup_models(self):
        """Setup the VLM encoder, SAE, and classifier."""
        # Load the VLM encoder (frozen)
        self.backbone = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
            pretrained=self.cfg.MODEL.PRETRAINED
        )
        self.backbone.to(self.device)
        self.backbone.eval()
        
        # Freeze encoder
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Load Tag-SAE
        self._load_tag_sae()
        
        # Create classifier
        self._create_classifier()
    
    def _load_tag_sae(self):
        """Load the pre-trained Tag-SAE."""
        sae_cfg = self.cfg.MITIGATOR.SAE_DEBIAS_CLS
        sae_path = sae_cfg.SAE_CHECKPOINT_PATH
        
        if not sae_path or not os.path.exists(sae_path):
            raise FileNotFoundError(
                f"Tag-SAE checkpoint not found: {sae_path}\n"
                "Please train a Tag-SAE first using the tag_sae mitigator."
            )
        
        # Load config
        config_path = os.path.join(os.path.dirname(sae_path), "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SAE config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            sae_config = json.load(f)
        
        print(f"\nLoading Tag-SAE from {sae_path}")
        print(f"  Input dim: {sae_config['input_dim']}")
        print(f"  Dict size: {sae_config['dict_size']}")
        print(f"  Anchored neurons: {sae_config['num_anchored']}")
        print(f"  Tags: {len(sae_config['all_tags'])}")
        
        # Create and load SAE
        self.sae = TagSupervisedSAE(
            input_dim=sae_config['input_dim'],
            dict_size=sae_config['dict_size'],
            num_anchored=sae_config['num_anchored'],
            tag_to_neuron=sae_config['tag_to_neuron'],
        )
        self.sae.load_state_dict(torch.load(sae_path, map_location=self.device))
        self.sae.to(self.device)
        self.sae.eval()
        
        # Freeze SAE
        for param in self.sae.parameters():
            param.requires_grad = False
        
        self.feature_dim = sae_config['input_dim']
        self.tag_to_neuron = sae_config['tag_to_neuron']
        self.all_tags = sae_config['all_tags']
        
        # Get debiasing config
        self.debias_mode = sae_cfg.DEBIAS_MODE
        self.tags_to_remove = sae_cfg.get("TAGS_TO_REMOVE", [])
        
        print(f"  Debias mode: {self.debias_mode}")
        if self.debias_mode == "specific":
            print(f"  Tags to remove: {self.tags_to_remove}")
    
    def _create_classifier(self):
        """Create the classification head."""
        sae_cfg = self.cfg.MITIGATOR.SAE_DEBIAS_CLS
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
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {num_params:,}")
    
    

    def _method_specific_setups(self):
        """Additional setup."""
        self.criterion = nn.CrossEntropyLoss()
        self.sae_cfg = self.cfg.MITIGATOR.SAE_DEBIAS_CLS
        self.cluster_debiased_test_features()
    

    


    def cluster_debiased_test_features(self):
        """
        Collect debiased test features, perform 2-cluster KMeans,
        and print confusion matrix of clusters x (target, bias) groups.
        """
        
        self.model.eval()
        self.backbone.eval()
        self.sae.eval()

        all_features = []
        all_targets = []
        all_biases = []

        test_loader = self.dataloaders["test"]

        with torch.no_grad():
            for batch in test_loader:

                inputs = batch["inputs"].to(self.device)

                features = self._extract_features(inputs)
                debiased = self._debias_features(features)

                all_features.append(debiased.cpu())
                all_targets.append(batch["targets"])

                # assume single bias attribute
                bias_name = self.biases[0]
                all_biases.append(batch[bias_name])

        features = torch.cat(all_features).numpy()
        targets = torch.cat(all_targets).numpy()
        biases = torch.cat(all_biases).numpy()

        # ----- clustering -----
        kmeans = KMeans(n_clusters=2, random_state=0)
        clusters = kmeans.fit_predict(features)

        # ----- group encoding (target,bias) -----
        groups = list(zip(targets, biases))
        unique_groups = sorted(list(set(groups)))

        group_to_id = {g: i for i, g in enumerate(unique_groups)}
        group_ids = np.array([group_to_id[g] for g in groups])

        # ----- confusion matrix -----
        cm = confusion_matrix(clusters, group_ids)

        print("\nCluster x (target,bias) confusion matrix")
        print("-----------------------------------------")
        print("Groups:", unique_groups)
        print(cm)

    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract features from images using the frozen encoder."""
        self.backbone.eval()
        
        with torch.no_grad():
            outputs = self.backbone(inputs)
            
            if isinstance(outputs, tuple):
                # (logits, features)
                features = outputs[1]
            else:
                features = outputs
        
        return features
    
    def _debias_features(self, features: torch.Tensor) -> torch.Tensor:
        """Debias features using the SAE."""
        self.sae.eval()
        
        with torch.no_grad():
            # Encode
            latents = self.sae.encode(features)
            
            if self.debias_mode == "all":
                # Zero all tag neurons
                latents[:, :self.sae.num_anchored] = 0
                
            elif self.debias_mode == "specific":
                # Zero specific tag neurons
                for tag in self.tags_to_remove:
                    if tag in self.tag_to_neuron:
                        neuron_idx = self.tag_to_neuron[tag]
                        latents[:, neuron_idx] = 0
                        
            elif self.debias_mode == "none":
                # No debiasing (baseline)
                pass
            
            # Decode
            debiased = self.sae.decode(latents)
        
        return debiased
    
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
        # Return predictions and targets for metric computation
        batch_dict = {
            "predictions": logits.argmax(dim=1).cpu(),
            "targets": batch["targets"],
        }
        
        # Add bias if available
        for b in self.biases:
            if b in batch:
                batch_dict[b] = batch[b]
        
        return batch_dict, loss
    
  