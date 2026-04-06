"""
SAE Trainer for Combined Dataset.

This module extends the SAE trainer to support training on the combined
dataset (CUB-200, Stanford Cars, Places365, LVIS/COCO).

The combined dataset provides diverse visual features for learning
a general-purpose SAE that can be used for bias mitigation.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.vlm_encoders import create_vlm_encoder, BaseVLMEncoder

# Import combined dataset
from datasets.sae_combined_dataset import (
    create_sae_combined_dataset,
    sae_combined_collate_fn,
    SAECombinedDataset,
)

# Import dictionary learning
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dictionary_learning'))

from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import (
    StandardTrainer as SAEStandardTrainer,
    BatchTopKSAETrainer,
)


class SAECombinedTrainer:
    """
    SAE Trainer for the combined multi-source dataset.
    
    This trainer:
    1. Loads a pretrained VLM encoder (OpenCLIP, SigLIP, or PE)
    2. Creates the combined dataset from multiple sources
    3. Extracts features from all images
    4. Trains an SAE on the combined features
    5. Analyzes which neurons respond to which sources/categories
    
    Configuration:
        SAE_COMBINED_TRAINER:
          # VLM Encoder
          ENCODER_TYPE: "openclip"
          MODEL_NAME: "ViT-L-14"
          PRETRAINED: "openai"
          
          # Dataset paths
          CUB200_ROOT: "/path/to/CUB_200_2011"
          STANFORD_CARS_ROOT: "/path/to/stanford_cars"
          PLACES365_ROOT: "/path/to/places365"
          LVIS_ROOT: "/path/to/lvis"
          COCO_IMAGES_ROOT: "/path/to/coco"
          
          # SAE settings
          DICT_SIZE: 4096
          NUM_STEPS: 50000
          BATCH_SIZE: 256
          LR: 1e-4
    """
    
    def __init__(
        self,
        # VLM Encoder settings
        encoder_type: str = "openclip",
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        # Dataset paths
        cub200_root: Optional[str] = None,
        stanford_cars_root: Optional[str] = None,
        places365_root: Optional[str] = None,
        lvis_root: Optional[str] = None,
        coco_images_root: Optional[str] = None,
        places365_categories: Optional[List[str]] = None,
        lvis_categories: Optional[List[str]] = None,
        # SAE settings
        dict_size: int = 4096,
        num_steps: int = 50000,
        batch_size: int = 256,
        lr: float = 1e-4,
        sae_type: str = "standard",
        # Other
        image_size: int = 224,
        device: str = "cuda",
        output_dir: str = "./outputs/sae_combined",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store config
        self.config = {
            "encoder_type": encoder_type,
            "model_name": model_name,
            "dict_size": dict_size,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "lr": lr,
            "sae_type": sae_type,
        }
        
        print(f"\n{'='*60}")
        print("SAE Combined Dataset Trainer")
        print(f"{'='*60}")
        
        # Step 1: Create VLM encoder
        print(f"\nLoading {encoder_type} encoder: {model_name}")
        self.encoder = create_vlm_encoder(
            encoder_type=encoder_type,
            model_name=model_name,
            device=str(self.device),
            pretrained=pretrained,
        )
        self.embed_dim = self.encoder.embed_dim
        print(f"  Embed dim: {self.embed_dim}")
        
        # Step 2: Create combined dataset
        print(f"\nCreating combined dataset...")
        self.dataset = create_sae_combined_dataset(
            cub200_root=cub200_root,
            stanford_cars_root=stanford_cars_root,
            places365_root=places365_root,
            lvis_root=lvis_root,
            coco_images_root=coco_images_root,
            places365_categories=places365_categories,
            lvis_categories=lvis_categories,
            image_size=image_size,
            normalize=True,
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=sae_combined_collate_fn,
            drop_last=True,
        )
        
        # SAE settings
        self.dict_size = dict_size
        self.num_steps = num_steps
        self.lr = lr
        self.sae_type = sae_type
        self.batch_size = batch_size
        
        # Will be set during training
        self.sae = None
        self.features = None
        self.metadata = None
    
    def extract_features(self) -> torch.Tensor:
        """Extract features from all images using the VLM encoder."""
        
        print(f"\n{'='*60}")
        print("Extracting Features")
        print(f"{'='*60}")
        
        all_features = []
        all_sources = []
        all_class_names = []
        all_paths = []
        
        self.encoder.eval() if hasattr(self.encoder, 'eval') else None
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Extracting features"):
                images = batch['inputs'].to(self.device)
                
                # Encode images
                features = self.encoder.encode_image(images)
                features = F.normalize(features, dim=-1)
                
                all_features.append(features.cpu())
                all_sources.extend(batch['sources'])
                all_class_names.extend(batch['class_names'])
                all_paths.extend(batch['paths'])
        
        self.features = torch.cat(all_features, dim=0)
        self.metadata = {
            'sources': all_sources,
            'class_names': all_class_names,
            'paths': all_paths,
        }
        
        print(f"Extracted {len(self.features)} feature vectors")
        print(f"Feature shape: {self.features.shape}")
        
        # Count per source
        source_counts = defaultdict(int)
        for s in all_sources:
            source_counts[s] += 1
        print("Features per source:")
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
        
        # Save features
        features_path = self.output_dir / "features.pt"
        torch.save({
            'features': self.features,
            'metadata': self.metadata,
        }, features_path)
        print(f"Saved features to {features_path}")
        
        return self.features
    
    def train_sae(self, features: Optional[torch.Tensor] = None) -> AutoEncoder:
        """Train the Sparse Autoencoder on extracted features."""
        
        if features is None:
            if self.features is None:
                features = self.extract_features()
            else:
                features = self.features
        
        print(f"\n{'='*60}")
        print("Training Sparse Autoencoder")
        print(f"{'='*60}")
        print(f"Feature dim: {self.embed_dim}")
        print(f"Dict size: {self.dict_size}")
        print(f"SAE type: {self.sae_type}")
        print(f"Steps: {self.num_steps}")
        
        # Create SAE
        self.sae = AutoEncoder(self.embed_dim, self.dict_size)
        self.sae.to(self.device)
        
        # Create feature dataset
        feature_dataset = TensorDataset(features)
        feature_loader = DataLoader(
            feature_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        # Create trainer
        if self.sae_type == "standard":
            trainer = SAEStandardTrainer(
                activation_dim=self.embed_dim,
                dict_size=self.dict_size,
                lr=self.lr,
                device=str(self.device),
                warmup_steps=1000,
            )
        elif self.sae_type == "batch_top_k":
            trainer = BatchTopKSAETrainer(
                activation_dim=self.embed_dim,
                dict_size=self.dict_size,
                k=32,
                lr=self.lr,
                device=str(self.device),
                warmup_steps=1000,
            )
        else:
            raise ValueError(f"Unknown SAE type: {self.sae_type}")
        
        # Training loop
        step = 0
        epoch = 0
        losses = []
        
        while step < self.num_steps:
            epoch += 1
            epoch_losses = []
            
            for batch in feature_loader:
                if step >= self.num_steps:
                    break
                
                x = batch[0].to(self.device)
                loss = trainer.loss(x, step=step)
                
                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                trainer.scheduler.step()
                
                epoch_losses.append(loss.item())
                step += 1
                
                if step % 1000 == 0:
                    avg_loss = np.mean(epoch_losses[-1000:])
                    print(f"Step {step}/{self.num_steps}, Loss: {avg_loss:.6f}")
            
            losses.extend(epoch_losses)
        
        # Get trained SAE
        self.sae = trainer.ae
        
        # Save SAE
        sae_dir = self.output_dir / "sae_checkpoints"
        sae_dir.mkdir(exist_ok=True)
        
        sae_path = sae_dir / "ae.pt"
        torch.save(self.sae.state_dict(), sae_path)
        
        config_path = sae_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'activation_dim': self.embed_dim,
                'dict_size': self.dict_size,
                'sae_type': self.sae_type,
                **self.config,
            }, f, indent=2)
        
        print(f"\nSaved SAE to {sae_path}")
        
        return self.sae
    
    def analyze_neurons(self, top_k: int = 20) -> Dict:
        """
        Analyze which neurons respond to which sources/categories.
        
        For each neuron, finds the top-k most activating images and
        records their source and class information.
        """
        
        if self.sae is None:
            raise ValueError("SAE not trained yet. Call train_sae() first.")
        
        if self.features is None or self.metadata is None:
            raise ValueError("Features not extracted. Call extract_features() first.")
        
        print(f"\n{'='*60}")
        print("Analyzing Neuron Activations")
        print(f"{'='*60}")
        
        self.sae.eval()
        
        # Encode all features with SAE
        with torch.no_grad():
            all_latents = self.sae.encode(self.features.to(self.device))
        
        all_latents = all_latents.cpu()
        num_samples, dict_size = all_latents.shape
        
        print(f"Latent shape: {all_latents.shape}")
        
        # Analyze each neuron
        analysis_results = {
            'top_k': top_k,
            'dict_size': dict_size,
            'num_samples': num_samples,
            'top_k_per_latent': {},
            'source_distribution_per_latent': {},
            'alive_neurons': [],
        }
        
        for latent_idx in tqdm(range(dict_size), desc="Analyzing neurons"):
            activations = all_latents[:, latent_idx]
            
            # Skip dead neurons
            if activations.max() < 1e-6:
                continue
            
            analysis_results['alive_neurons'].append(latent_idx)
            
            # Get top-k activating samples
            top_k_values, top_k_indices = activations.topk(min(top_k, len(activations)))
            
            # Get source and class info for top-k
            top_sources = [self.metadata['sources'][i] for i in top_k_indices.tolist()]
            top_classes = [self.metadata['class_names'][i] for i in top_k_indices.tolist()]
            
            analysis_results['top_k_per_latent'][latent_idx] = {
                'indices': top_k_indices.tolist(),
                'activations': top_k_values.tolist(),
                'sources': top_sources,
                'class_names': top_classes,
            }
            
            # Source distribution
            source_counts = defaultdict(int)
            for s in top_sources:
                source_counts[s] += 1
            analysis_results['source_distribution_per_latent'][latent_idx] = dict(source_counts)
        
        analysis_results['num_alive_neurons'] = len(analysis_results['alive_neurons'])
        analysis_results['percent_alive'] = 100 * len(analysis_results['alive_neurons']) / dict_size
        
        print(f"\nAlive neurons: {analysis_results['num_alive_neurons']}/{dict_size} "
              f"({analysis_results['percent_alive']:.1f}%)")
        
        # Compute neuron-source associations
        source_specific_neurons = defaultdict(list)
        for latent_idx, dist in analysis_results['source_distribution_per_latent'].items():
            total = sum(dist.values())
            for source, count in dist.items():
                purity = count / total
                if purity >= 0.8:  # 80% from single source
                    source_specific_neurons[source].append({
                        'neuron': latent_idx,
                        'purity': purity,
                    })
        
        analysis_results['source_specific_neurons'] = dict(source_specific_neurons)
        
        print("\nSource-specific neurons (>=80% purity):")
        for source, neurons in source_specific_neurons.items():
            print(f"  {source}: {len(neurons)} neurons")
        
        # Save results
        results_path = self.output_dir / "analysis_results.json"
        with open(results_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"\nSaved analysis to {results_path}")
        
        return analysis_results
    
    def run_full_pipeline(self):
        """Run the full training and analysis pipeline."""
        
        # Extract features
        self.extract_features()
        
        # Train SAE
        self.train_sae()
        
        # Analyze neurons
        self.analyze_neurons()
        
        print(f"\n{'='*60}")
        print("Pipeline Complete!")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        print(f"  - features.pt: Extracted features and metadata")
        print(f"  - sae_checkpoints/ae.pt: Trained SAE")
        print(f"  - analysis_results.json: Neuron analysis")


def main():
    """Example usage."""
    
    trainer = SAECombinedTrainer(
        # VLM Encoder
        encoder_type="openclip",
        model_name="ViT-L-14",
        pretrained="openai",
        
        # Dataset paths - UPDATE THESE
        cub200_root="/path/to/CUB_200_2011",
        stanford_cars_root="/path/to/stanford_cars",
        places365_root="/path/to/places365",
        lvis_root="/path/to/lvis",
        coco_images_root="/path/to/coco",
        
        # SAE settings
        dict_size=4096,
        num_steps=50000,
        batch_size=256,
        lr=1e-4,
        
        # Output
        output_dir="./outputs/sae_combined",
    )
    
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()