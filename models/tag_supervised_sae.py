"""
Tag-Supervised Sparse Autoencoder (TS-SAE).

This module implements an SAE where specific neurons are supervised to activate
based on the presence of semantic tags (from MAVIAS/RAM pipeline).

Key Idea:
- Anchored neurons (0 to num_tags-1): Supervised by tag presence
- Free neurons (num_tags to dict_size): Standard SAE learning

Training Loss:
    L = L_reconstruction + λ_sparsity * L_sparsity + λ_tag * L_tag_supervision

Tag Supervision Loss Options:
- BCE: Binary cross-entropy between activations and tag presence
- Hinge: Margin-based loss (activate above threshold when tag present)
- MSE: Mean squared error to target activation level

This enables debiasing by zeroing out tag-anchored neurons at inference.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class TagSupervisedSAE(nn.Module):
    """
    Sparse Autoencoder with tag-supervised anchored neurons.
    
    Architecture:
        encoder: input_dim -> dict_size (with ReLU)
        decoder: dict_size -> input_dim (linear)
    
    Neuron layout:
        [0, num_anchored): Anchored to specific tags
        [num_anchored, dict_size): Free neurons (learned)
    """
    
    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        num_anchored: int,
        tag_to_neuron: Dict[str, int],
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.num_anchored = num_anchored
        self.num_free = dict_size - num_anchored
        self.tag_to_neuron = tag_to_neuron
        self.neuron_to_tag = {v: k for k, v in tag_to_neuron.items()}
        
        # Encoder and decoder
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)
        
        # Initialize weights
        self._init_weights()
        
        print(f"TagSupervisedSAE initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  Dict size: {dict_size}")
        print(f"  Anchored neurons: {num_anchored} (tags)")
        print(f"  Free neurons: {self.num_free}")
    
    def _init_weights(self):
        """Initialize weights with appropriate scaling."""
        # Xavier initialization for encoder
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # Initialize decoder as transpose of encoder (tied initialization)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse latent representation."""
        return F.relu(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to input space."""
        return self.decoder(z)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_latents: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            return_latents: If True, also return latent activations
            
        Returns:
            reconstructed: Reconstructed input
            latents: (optional) Latent activations
        """
        latents = self.encode(x)
        reconstructed = self.decode(latents)
        
        if return_latents:
            return reconstructed, latents
        return reconstructed
    
    def get_anchored_activations(self, latents: torch.Tensor) -> torch.Tensor:
        """Get activations of anchored (tag) neurons only."""
        return latents[:, :self.num_anchored]
    
    def get_free_activations(self, latents: torch.Tensor) -> torch.Tensor:
        """Get activations of free neurons only."""
        return latents[:, self.num_anchored:]
    
    def decode_without_tags(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode and decode, but zero out tag-anchored neurons.
        This produces "debiased" features.
        """
        latents = self.encode(x)
        # Zero out anchored neurons
        latents_debiased = latents.clone()
        latents_debiased[:, :self.num_anchored] = 0
        return self.decode(latents_debiased)
    
    def decode_without_specific_tags(
        self, 
        x: torch.Tensor, 
        tags_to_remove: List[str]
    ) -> torch.Tensor:
        """
        Encode and decode, zeroing out specific tag neurons.
        """
        latents = self.encode(x)
        latents_filtered = latents.clone()
        
        for tag in tags_to_remove:
            if tag in self.tag_to_neuron:
                neuron_idx = self.tag_to_neuron[tag]
                latents_filtered[:, neuron_idx] = 0
        
        return self.decode(latents_filtered)
    
    def get_tag_activations(
        self, 
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get activation values for each tag neuron.
        
        Returns:
            Dict mapping tag names to activation values
        """
        latents = self.encode(x)
        return {
            tag: latents[:, neuron_idx]
            for tag, neuron_idx in self.tag_to_neuron.items()
        }


class TagSupervisionLoss(nn.Module):
    """
    Loss module for tag supervision.
    
    Supports multiple loss types:
    - 'bce': Binary cross-entropy (soft, differentiable)
    - 'hinge': Margin-based (enforce minimum activation gap)
    - 'mse': Mean squared error to target activation
    """
    
    def __init__(
        self,
        loss_type: str = "bce",
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        margin: float = 0.5,
        target_activation: float = 1.0,
        use_negative_supervision: bool = True,
    ):
        super().__init__()
        
        self.loss_type = loss_type.lower()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.margin = margin
        self.target_activation = target_activation
        self.use_negative_supervision = use_negative_supervision
        
        assert self.loss_type in ["bce", "hinge", "mse"], \
            f"Unknown loss type: {loss_type}"
        
        print(f"TagSupervisionLoss: {self.loss_type}")
        print(f"  Positive weight: {positive_weight}")
        print(f"  Negative weight: {negative_weight}")
        print(f"  Use negative supervision: {use_negative_supervision}")
    
    def forward(
        self,
        activations: torch.Tensor,  # (batch, num_anchored)
        tag_targets: torch.Tensor,  # (batch, num_anchored) binary
    ) -> torch.Tensor:
        """
        Compute tag supervision loss.
        
        Args:
            activations: Anchored neuron activations
            tag_targets: Binary targets (1 if tag present, 0 otherwise)
        """
        if self.loss_type == "bce":
            return self._bce_loss(activations, tag_targets)
        elif self.loss_type == "hinge":
            return self._hinge_loss(activations, tag_targets)
        elif self.loss_type == "mse":
            return self._mse_loss(activations, tag_targets)
    
    def _bce_loss(
        self, 
        activations: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Binary cross-entropy loss for tag supervision.
        
        Since activations come from ReLU (non-negative), we use
        a different formulation that works better:
        - When tag present (target=1): want high activation
        - When tag absent (target=0): want low/zero activation
        """
        # Option 1: Use BCEWithLogitsLoss-style formulation
        # Treat activations as logits (can be any positive value)
        # Scale activations to make sigmoid more sensitive
        scaled_acts = activations * 2 - 1  # Map [0, inf) to [-1, inf)
        probs = torch.sigmoid(scaled_acts)
        
        # Weighted BCE
        pos_loss = -targets * torch.log(probs + 1e-8)
        neg_loss = -(1 - targets) * torch.log(1 - probs + 1e-8)
        
        # Weight by class frequency (positive samples are rare)
        # Count actual positives and negatives
        num_pos = targets.sum()
        num_neg = (1 - targets).sum()
        
        # Adjust weights based on imbalance
        if num_pos > 0 and num_neg > 0:
            pos_weight = (num_neg / num_pos).clamp(max=10.0)  # Cap at 10x
        else:
            pos_weight = 1.0
        
        loss = self.positive_weight * pos_weight * pos_loss
        if self.use_negative_supervision:
            loss = loss + self.negative_weight * neg_loss
        
        return loss.mean()
    
    def _hinge_loss(
        self, 
        activations: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Hinge loss with margin.
        - When tag present (target=1): activation should be > margin
        - When tag absent (target=0): activation should be < margin
        """
        # Positive: penalize if activation < margin when tag present
        pos_loss = F.relu(self.margin - activations) * targets
        
        # Negative: penalize if activation > 0 when tag absent
        neg_loss = F.relu(activations) * (1 - targets)
        
        loss = self.positive_weight * pos_loss
        if self.use_negative_supervision:
            loss = loss + self.negative_weight * neg_loss
        
        return loss.mean()
    
    def _mse_loss(
        self, 
        activations: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        MSE loss to target activation level.
        - When tag present: activation should be target_activation
        - When tag absent: activation should be 0
        """
        target_values = targets * self.target_activation
        
        if self.use_negative_supervision:
            # Full MSE
            loss = F.mse_loss(activations, target_values, reduction='none')
        else:
            # Only penalize when tag is present
            loss = ((activations - target_values) ** 2) * targets
        
        # Weight positive vs negative
        weights = targets * self.positive_weight + (1 - targets) * self.negative_weight
        if not self.use_negative_supervision:
            weights = targets * self.positive_weight
        
        return (loss * weights).mean()


class TagSupervisedSAETrainer:
    """
    Trainer for Tag-Supervised SAE.
    
    Integrates with MAVIAS pipeline to load tags and train the SAE
    with tag supervision on anchored neurons.
    """
    
    def __init__(
        self,
        # Model config
        input_dim: int,
        expansion_factor: int = 8,
        num_free_neurons: int = 0,  # Additional free neurons beyond tags
        # Tag config
        tags_csv_path: str = None,
        tag_column: str = "irrelevant_tags",
        tag_separator: str = " | ",
        min_tag_frequency: int = 10,
        # Training config
        num_steps: int = 10000,
        batch_size: int = 256,
        lr: float = 1e-3,
        # Loss weights
        lambda_reconstruction: float = 1.0,
        lambda_sparsity: float = 1e-3,
        lambda_tag: float = 1.0,
        # Tag supervision config
        tag_loss_type: str = "bce",
        positive_weight: float = 1.0,
        negative_weight: float = 0.5,
        use_negative_supervision: bool = True,
        margin: float = 0.5,
        target_activation: float = 1.0,
        # Other
        device: str = "cuda",
        output_dir: str = "./outputs/tag_sae",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store config
        self.input_dim = input_dim
        self.expansion_factor = expansion_factor
        self.num_free_neurons = num_free_neurons
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.lr = lr
        
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_sparsity = lambda_sparsity
        self.lambda_tag = lambda_tag
        
        self.tag_loss_type = tag_loss_type
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.use_negative_supervision = use_negative_supervision
        self.margin = margin
        self.target_activation = target_activation
        
        # Load tags from MAVIAS pipeline
        self.tags_csv_path = tags_csv_path
        self.tag_column = tag_column
        self.tag_separator = tag_separator
        self.min_tag_frequency = min_tag_frequency
        
        self.tag_to_neuron = {}
        self.all_tags = []
        self.tags_df = None
        self.index_to_tags = {}
        
        # Will be initialized later
        self.sae = None
        self.tag_loss_fn = None
    
    def load_tags_from_mavias(self, tags_csv_path: str = None):
        """
        Load tags from MAVIAS pipeline CSV.
        
        Expected CSV format:
            index, target, tags, irrelevant_tags
            0, 0, "water | bird | sky", "water | sky"
            1, 1, "forest | tree", "forest | tree"
        """
        path = tags_csv_path or self.tags_csv_path
        
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Tags CSV not found: {path}")
        
        print(f"\nLoading tags from {path}")
        self.tags_df = pd.read_csv(path)
        
        # Count tag frequencies
        tag_counts = defaultdict(int)
        
        for _, row in self.tags_df.iterrows():
            tags_str = row.get(self.tag_column, "")
            if pd.isna(tags_str) or not isinstance(tags_str, str):
                continue
            
            tags = [t.strip() for t in tags_str.split(self.tag_separator) if t.strip()]
            for tag in tags:
                tag_counts[tag] += 1
        
        # Filter by minimum frequency
        self.all_tags = sorted([
            tag for tag, count in tag_counts.items()
            if count >= self.min_tag_frequency
        ])
        
        # Create tag to neuron mapping
        self.tag_to_neuron = {tag: i for i, tag in enumerate(self.all_tags)}
        
        # Build index to tags mapping
        self.index_to_tags = {}
        for _, row in self.tags_df.iterrows():
            idx = row["index"]
            tags_str = row.get(self.tag_column, "")
            if pd.isna(tags_str) or not isinstance(tags_str, str):
                tags = []
            else:
                tags = [t.strip() for t in tags_str.split(self.tag_separator) if t.strip()]
            # Filter to only known tags
            tags = [t for t in tags if t in self.tag_to_neuron]
            self.index_to_tags[idx] = tags
        
        print(f"  Total unique tags: {len(tag_counts)}")
        print(f"  Tags with freq >= {self.min_tag_frequency}: {len(self.all_tags)}")
        print(f"  Sample tags: {self.all_tags[:10]}...")
        
        return self.all_tags
    
    def build_tag_targets(
        self, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Build binary tag target matrix for a batch.
        
        Args:
            indices: Sample indices (batch_size,)
            
        Returns:
            targets: Binary matrix (batch_size, num_anchored)
        """
        batch_size = len(indices)
        num_anchored = len(self.all_tags)
        targets = torch.zeros(batch_size, num_anchored, device=self.device)
        
        for i, idx in enumerate(indices.tolist()):
            tags = self.index_to_tags.get(idx, [])
            for tag in tags:
                if tag in self.tag_to_neuron:
                    neuron_idx = self.tag_to_neuron[tag]
                    targets[i, neuron_idx] = 1.0
        
        return targets
    
    def initialize_sae(self):
        """Initialize the SAE model based on loaded tags."""
        if not self.all_tags:
            raise ValueError("Must load tags first with load_tags_from_mavias()")
        
        num_anchored = len(self.all_tags)
        
        # Total dict size = anchored + free + (expansion if specified)
        if self.num_free_neurons > 0:
            dict_size = num_anchored + self.num_free_neurons
        else:
            # Use expansion factor
            dict_size = self.input_dim * self.expansion_factor
            # Ensure we have at least num_anchored
            dict_size = max(dict_size, num_anchored + 100)
        
        self.sae = TagSupervisedSAE(
            input_dim=self.input_dim,
            dict_size=dict_size,
            num_anchored=num_anchored,
            tag_to_neuron=self.tag_to_neuron,
        )
        self.sae.to(self.device)
        
        # Initialize tag supervision loss
        self.tag_loss_fn = TagSupervisionLoss(
            loss_type=self.tag_loss_type,
            positive_weight=self.positive_weight,
            negative_weight=self.negative_weight,
            margin=self.margin,
            target_activation=self.target_activation,
            use_negative_supervision=self.use_negative_supervision,
        )
        
        return self.sae
    
    def train(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> TagSupervisedSAE:
        """
        Train the Tag-Supervised SAE.
        
        Args:
            features: Feature vectors (N, input_dim)
            indices: Sample indices for tag lookup (N,)
            targets: Optional class labels (N,) - not used for training
            
        Returns:
            Trained SAE model
        """
        if self.sae is None:
            self.initialize_sae()
        
        print(f"\n{'='*60}")
        print("Training Tag-Supervised SAE")
        print(f"{'='*60}")
        print(f"  Features: {features.shape}")
        print(f"  Anchored neurons: {self.sae.num_anchored}")
        print(f"  Free neurons: {self.sae.num_free}")
        print(f"  Training steps: {self.num_steps}")
        print(f"  Tag loss type: {self.tag_loss_type}")
        print(f"  Negative supervision: {self.use_negative_supervision}")
        
        # Create dataloader
        dataset = TensorDataset(features, indices)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=self.lr)
        
        # Training loop
        step = 0
        epoch = 0
        losses_history = {
            'total': [], 'reconstruction': [], 'sparsity': [], 'tag': []
        }
        
        self.sae.train()
        pbar = tqdm(total=self.num_steps, desc="Training")
        
        while step < self.num_steps:
            epoch += 1
            
            for batch_features, batch_indices in dataloader:
                if step >= self.num_steps:
                    break
                
                batch_features = batch_features.to(self.device)
                batch_indices = batch_indices.to(self.device)
                
                # Forward pass
                reconstructed, latents = self.sae(batch_features, return_latents=True)
                
                # Reconstruction loss
                loss_recon = F.mse_loss(reconstructed, batch_features)
                
                # Sparsity loss (L1 on all latents)
                loss_sparsity = latents.abs().mean()
                
                # Tag supervision loss (only on anchored neurons)
                tag_targets = self.build_tag_targets(batch_indices)
                anchored_activations = self.sae.get_anchored_activations(latents)
                loss_tag = self.tag_loss_fn(anchored_activations, tag_targets)
                
                # Total loss
                loss = (
                    self.lambda_reconstruction * loss_recon +
                    self.lambda_sparsity * loss_sparsity +
                    self.lambda_tag * loss_tag
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Normalize decoder weights (standard SAE practice)
                with torch.no_grad():
                    norms = self.sae.decoder.weight.norm(dim=0, keepdim=True)
                    self.sae.decoder.weight.div_(norms.clamp(min=1e-8))
                
                # Log
                losses_history['total'].append(loss.item())
                losses_history['reconstruction'].append(loss_recon.item())
                losses_history['sparsity'].append(loss_sparsity.item())
                losses_history['tag'].append(loss_tag.item())
                
                step += 1
                pbar.update(1)
                
                if step % 1000 == 0:
                    l0 = (latents > 0).float().sum(dim=1).mean().item()
                    tag_acc = self._compute_tag_accuracy(anchored_activations, tag_targets)
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'recon': f"{loss_recon.item():.4f}",
                        'tag': f"{loss_tag.item():.4f}",
                        'L0': f"{l0:.1f}",
                        'tag_acc': f"{tag_acc:.2%}"
                    })
        
        pbar.close()
        
        # Save model
        self._save_model(losses_history)
        
        return self.sae
    
    def _compute_tag_accuracy(
        self,
        activations: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """Compute accuracy of tag predictions."""
        with torch.no_grad():
            # Predict tag present if activation > threshold
            preds = (activations > threshold).float()
            correct = (preds == targets).float()
            
            # Only count where there's at least one tag
            mask = targets.sum(dim=1) > 0
            if mask.sum() == 0:
                return 0.0
            
            return correct[mask].mean().item()
    
    def _save_model(self, losses_history: Dict):
        """Save model and training info."""
        # Save model
        model_path = self.output_dir / "tag_sae.pt"
        torch.save(self.sae.state_dict(), model_path)
        print(f"\nSaved model to {model_path}")
        
        # Save config
        config = {
            'input_dim': self.input_dim,
            'dict_size': self.sae.dict_size,
            'num_anchored': self.sae.num_anchored,
            'num_free': self.sae.num_free,
            'tag_to_neuron': self.tag_to_neuron,
            'all_tags': self.all_tags,
            'num_steps': self.num_steps,
            'tag_loss_type': self.tag_loss_type,
            'use_negative_supervision': self.use_negative_supervision,
            'lambda_tag': self.lambda_tag,
        }
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save losses
        losses_path = self.output_dir / "losses.json"
        with open(losses_path, 'w') as f:
            json.dump({k: v[-100:] for k, v in losses_history.items()}, f)
        
        print(f"Saved config to {config_path}")
    
    def analyze_tag_neurons(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
    ) -> Dict:
        """
        Analyze how well tag neurons have learned.
        
        Returns statistics on tag neuron activations vs tag presence.
        """
        if self.sae is None:
            raise ValueError("SAE not trained yet")
        
        self.sae.eval()
        
        print(f"\n{'='*60}")
        print("Analyzing Tag Neuron Performance")
        print(f"{'='*60}")
        
        results = {
            'per_tag_stats': {},
            'overall_accuracy': 0.0,
            'tag_activation_when_present': {},
            'tag_activation_when_absent': {},
        }
        
        with torch.no_grad():
            # Process in batches
            all_activations = []
            all_targets = []
            
            for i in range(0, len(features), self.batch_size):
                batch_features = features[i:i+self.batch_size].to(self.device)
                batch_indices = indices[i:i+self.batch_size].to(self.device)
                
                latents = self.sae.encode(batch_features)
                anchored = self.sae.get_anchored_activations(latents)
                tag_targets = self.build_tag_targets(batch_indices)
                
                all_activations.append(anchored.cpu())
                all_targets.append(tag_targets.cpu())
            
            all_activations = torch.cat(all_activations, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
        
        # Per-tag analysis
        for tag, neuron_idx in self.tag_to_neuron.items():
            acts = all_activations[:, neuron_idx]
            targets = all_targets[:, neuron_idx]
            
            present_mask = targets == 1
            absent_mask = targets == 0
            
            if present_mask.sum() > 0:
                act_when_present = acts[present_mask].mean().item()
            else:
                act_when_present = 0.0
            
            if absent_mask.sum() > 0:
                act_when_absent = acts[absent_mask].mean().item()
            else:
                act_when_absent = 0.0
            
            results['tag_activation_when_present'][tag] = act_when_present
            results['tag_activation_when_absent'][tag] = act_when_absent
            
            # Compute separation
            separation = act_when_present - act_when_absent
            
            results['per_tag_stats'][tag] = {
                'neuron_idx': neuron_idx,
                'num_present': present_mask.sum().item(),
                'num_absent': absent_mask.sum().item(),
                'act_when_present': act_when_present,
                'act_when_absent': act_when_absent,
                'separation': separation,
            }
        
        # Overall accuracy
        preds = (all_activations > 0.5).float()
        results['overall_accuracy'] = (preds == all_targets).float().mean().item()
        
        # Print summary
        print(f"\nOverall tag prediction accuracy: {results['overall_accuracy']:.2%}")
        print(f"\nTop 10 best separated tags:")
        
        sorted_tags = sorted(
            results['per_tag_stats'].items(),
            key=lambda x: x[1]['separation'],
            reverse=True
        )
        
        for tag, stats in sorted_tags[:10]:
            print(f"  {tag}: separation={stats['separation']:.3f}, "
                  f"present={stats['act_when_present']:.3f}, "
                  f"absent={stats['act_when_absent']:.3f}")
        
        # Save results
        results_path = self.output_dir / "tag_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved analysis to {results_path}")
        
        return results


def load_tag_supervised_sae(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cuda",
) -> TagSupervisedSAE:
    """
    Load a trained Tag-Supervised SAE from checkpoint.
    
    Args:
        checkpoint_path: Path to model .pt file
        config_path: Path to config.json (auto-detected if None)
        device: Device to load model on
        
    Returns:
        Loaded TagSupervisedSAE model
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
    
    return sae


# ============================================
# Debiasing Functions
# ============================================

def debias_features(
    features: torch.Tensor,
    sae: TagSupervisedSAE,
    mode: str = "all_tags",
    tags_to_remove: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Debias features by removing tag-associated information.
    
    Args:
        features: Input features (batch, dim)
        sae: Trained TagSupervisedSAE
        mode: 
            - "all_tags": Remove all anchored neuron activations
            - "specific": Remove only specified tags
        tags_to_remove: List of tag names to remove (for mode="specific")
        
    Returns:
        Debiased features
    """
    sae.eval()
    
    with torch.no_grad():
        if mode == "all_tags":
            return sae.decode_without_tags(features)
        elif mode == "specific" and tags_to_remove:
            return sae.decode_without_specific_tags(features, tags_to_remove)
        else:
            # No debiasing
            return sae(features)