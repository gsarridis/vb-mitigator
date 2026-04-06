"""
Debiasing with Tag-Supervised SAE.

This module provides utilities for debiasing visual features using
a trained Tag-Supervised SAE. The key idea is to encode features
through the SAE and zero out the tag-associated neurons before decoding.

Usage:
    from models.tag_sae_debias import TagSAEDebiaser
    
    debiaser = TagSAEDebiaser.from_checkpoint("outputs/tag_sae/tag_sae.pt")
    
    # Debias all tag neurons
    debiased_features = debiaser.debias(features)
    
    # Debias specific tags only
    debiased_features = debiaser.debias(features, tags=["water", "forest"])
    
    # Keep specific tags, remove all others
    debiased_features = debiaser.debias(features, keep_tags=["person", "face"])
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Union

import torch
import torch.nn as nn

from .tag_supervised_sae import TagSupervisedSAE


class TagSAEDebiaser(nn.Module):
    """
    Wrapper for using Tag-Supervised SAE for debiasing.
    
    Modes:
        - "all": Zero out all anchored (tag) neurons
        - "remove": Zero out specific tags
        - "keep": Keep only specified tags, zero out all others
    """
    
    def __init__(
        self,
        sae: TagSupervisedSAE,
        tag_to_neuron: Dict[str, int],
        all_tags: List[str],
    ):
        super().__init__()
        self.sae = sae
        self.tag_to_neuron = tag_to_neuron
        self.neuron_to_tag = {v: k for k, v in tag_to_neuron.items()}
        self.all_tags = all_tags
        self.num_anchored = len(all_tags)
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
    ) -> "TagSAEDebiaser":
        """
        Load debiaser from saved checkpoint.
        
        Args:
            checkpoint_path: Path to tag_sae.pt
            config_path: Path to config.json (auto-detected if None)
            device: Device to load on
        """
        checkpoint_path = Path(checkpoint_path)
        
        if config_path is None:
            config_path = checkpoint_path.parent / "config.json"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        sae = TagSupervisedSAE(
            input_dim=config['input_dim'],
            dict_size=config['dict_size'],
            num_anchored=config['num_anchored'],
            tag_to_neuron=config['tag_to_neuron'],
        )
        
        sae.load_state_dict(torch.load(checkpoint_path, map_location=device))
        sae.to(device)
        sae.eval()
        
        return cls(
            sae=sae,
            tag_to_neuron=config['tag_to_neuron'],
            all_tags=config['all_tags'],
        )
    
    def debias(
        self,
        features: torch.Tensor,
        mode: str = "all",
        tags: Optional[List[str]] = None,
        keep_tags: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Debias features by removing tag-associated information.
        
        Args:
            features: Input features (batch, dim)
            mode: Debiasing mode
                - "all": Remove all tag neurons
                - "remove": Remove specified tags only
                - "keep": Keep only specified tags
            tags: Tags to remove (for mode="remove")
            keep_tags: Tags to keep (for mode="keep")
            
        Returns:
            Debiased features
        """
        self.sae.eval()
        
        with torch.no_grad():
            # Encode
            latents = self.sae.encode(features)
            
            if mode == "all":
                # Zero out all anchored neurons
                latents[:, :self.num_anchored] = 0
                
            elif mode == "remove" and tags:
                # Zero out specified tags
                for tag in tags:
                    if tag in self.tag_to_neuron:
                        neuron_idx = self.tag_to_neuron[tag]
                        latents[:, neuron_idx] = 0
                        
            elif mode == "keep" and keep_tags:
                # Zero out all tags except specified ones
                keep_indices = set()
                for tag in keep_tags:
                    if tag in self.tag_to_neuron:
                        keep_indices.add(self.tag_to_neuron[tag])
                
                for i in range(self.num_anchored):
                    if i not in keep_indices:
                        latents[:, i] = 0
            
            # Decode
            return self.sae.decode(latents)
    
    def get_tag_activations(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get activation values for each tag.
        
        Returns:
            Dict mapping tag names to activation values (batch,)
        """
        self.sae.eval()
        
        with torch.no_grad():
            latents = self.sae.encode(features)
            
            return {
                tag: latents[:, neuron_idx]
                for tag, neuron_idx in self.tag_to_neuron.items()
            }
    
    def get_active_tags(
        self,
        features: torch.Tensor,
        threshold: float = 0.5,
    ) -> List[List[str]]:
        """
        Get list of active tags for each sample.
        
        Args:
            features: Input features (batch, dim)
            threshold: Activation threshold
            
        Returns:
            List of tag lists, one per sample
        """
        tag_acts = self.get_tag_activations(features)
        batch_size = features.shape[0]
        
        results = []
        for i in range(batch_size):
            active = []
            for tag, acts in tag_acts.items():
                if acts[i].item() > threshold:
                    active.append(tag)
            results.append(active)
        
        return results
    
    def forward(
        self,
        features: torch.Tensor,
        mode: str = "all",
        tags: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Forward pass (alias for debias)."""
        return self.debias(features, mode=mode, tags=tags)


class TagSAEClassifier(nn.Module):
    """
    Classifier that uses Tag-SAE for debiasing before classification.
    
    Pipeline:
        features -> SAE encode -> zero out tags -> SAE decode -> classifier -> logits
    """
    
    def __init__(
        self,
        debiaser: TagSAEDebiaser,
        classifier: nn.Module,
        debias_mode: str = "all",
        debias_tags: Optional[List[str]] = None,
    ):
        super().__init__()
        self.debiaser = debiaser
        self.classifier = classifier
        self.debias_mode = debias_mode
        self.debias_tags = debias_tags
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with debiasing.
        
        Args:
            features: Input features
            
        Returns:
            Classification logits
        """
        # Debias
        debiased = self.debiaser.debias(
            features,
            mode=self.debias_mode,
            tags=self.debias_tags,
        )
        
        # Classify
        return self.classifier(debiased)


def create_debiased_model(
    backbone: nn.Module,
    debiaser: TagSAEDebiaser,
    classifier: nn.Module,
    debias_mode: str = "all",
) -> nn.Module:
    """
    Create an end-to-end model with debiasing.
    
    Pipeline:
        image -> backbone -> features -> debias -> classifier -> logits
    """
    
    class DebiasedModel(nn.Module):
        def __init__(self, backbone, debiaser, classifier, mode):
            super().__init__()
            self.backbone = backbone
            self.debiaser = debiaser
            self.classifier = classifier
            self.mode = mode
        
        def forward(self, x):
            # Extract features
            features = self.backbone(x)
            if isinstance(features, tuple):
                features = features[1]  # (logits, features) -> features
            
            # Debias
            debiased = self.debiaser.debias(features, mode=self.mode)
            
            # Classify
            return self.classifier(debiased)
    
    return DebiasedModel(backbone, debiaser, classifier, debias_mode)