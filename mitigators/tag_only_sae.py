"""
Tag-Only SAE Mitigator with Selective Debiasing.

This SAE uses ONLY tag-supervised neurons (no free neurons).
All tags from the dataset are used for training, but only
irrelevant (bias) tags are zeroed during debiasing.

Key differences from tag_sae:
1. No free neurons - dict_size = num_unique_tags
2. Uses ALL tags for training (from 'tags' column)
3. Debiasing only removes irrelevant tags (from 'irrelevant_tags' column)

Pipeline:
    Training:
        Image → Encoder → Features → SAE(all tags supervised) → Reconstruction

    Debiasing:
        Features → SAE Encode → Zero ONLY irrelevant tag neurons → SAE Decode → Debiased Features

Config:
    MITIGATOR:
      TYPE: "tag_only_sae"
      TAG_ONLY_SAE:
        TAGS_CSV_PATH: "train_tags.csv"
        ALL_TAGS_COLUMN: "tags"           # Column with ALL tags (for training)
        IRRELEVANT_TAGS_COLUMN: "irrelevant_tags"  # Column with bias tags (for debiasing)
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
from models.tag_supervised_sae import TagSupervisedSAE, TagSupervisionLoss


class TagOnlySAE(nn.Module):
    """
    SAE where ALL neurons are tag-supervised (no free neurons).

    Architecture:
        encoder: input_dim → num_tags (with ReLU)
        decoder: num_tags → input_dim

    Each neuron corresponds to exactly one tag.
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
        self.dict_size = num_tags  # All neurons are tag neurons
        self.tag_to_neuron = tag_to_neuron
        self.neuron_to_tag = {v: k for k, v in tag_to_neuron.items()}

        # Encoder: input_dim → num_tags
        self.encoder = nn.Linear(input_dim, num_tags)
        self.encoder_bias = nn.Parameter(torch.zeros(input_dim))

        # Decoder: num_tags → input_dim
        self.decoder = nn.Linear(num_tags, input_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training dynamics."""
        # Initialize encoder
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        # Initialize decoder to transpose of encoder
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to tag activations."""
        x_centered = x - self.encoder_bias
        latents = F.relu(self.encoder(x_centered))
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode tag activations to reconstructed input."""
        return self.decoder(latents) + self.decoder_bias

    def forward(self, x: torch.Tensor, return_latents: bool = False):
        """Forward pass."""
        latents = self.encode(x)
        reconstructed = self.decode(latents)

        if return_latents:
            return reconstructed, latents
        return reconstructed

    def decode_without_tags(
        self, latents: torch.Tensor, tags_to_remove: List[str]
    ) -> torch.Tensor:
        """
        Decode after zeroing specific tag neurons.

        Args:
            latents: Encoded latent activations
            tags_to_remove: List of tag names to zero out

        Returns:
            Reconstructed features without specified tags
        """
        latents_debiased = latents.clone()

        for tag in tags_to_remove:
            if tag in self.tag_to_neuron:
                neuron_idx = self.tag_to_neuron[tag]
                latents_debiased[:, neuron_idx] = 0

        return self.decode(latents_debiased)

    def get_neuron_indices_for_tags(self, tags: List[str]) -> List[int]:
        """Get neuron indices for a list of tags."""
        indices = []
        for tag in tags:
            if tag in self.tag_to_neuron:
                indices.append(self.tag_to_neuron[tag])
        return indices


class TagOnlySAETrainer(BaseTrainer):
    """
    Trainer for Tag-Only SAE with selective debiasing.

    Training uses ALL tags, but debiasing only removes irrelevant tags.
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
        self.tag_cfg = self.cfg.MITIGATOR.TAG_ONLY_SAE

        # Load tags
        self._load_tags()

        # Get feature dimension
        self._get_feature_dim()

    def _load_tags(self):
        """
        Load tags from CSV.

        Loads both:
        - all_tags: ALL tags for training
        - irrelevant_tags: Bias tags for debiasing
        """
        csv_path = os.path.join(self.data_root, self.tag_cfg.TAGS_CSV_PATH)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Tags CSV not found: {csv_path}")

        print(f"\nLoading tags from {csv_path}")
        df = pd.read_csv(csv_path)

        all_tags_col = self.tag_cfg.ALL_TAGS_COLUMN
        irrelevant_tags_col = self.tag_cfg.IRRELEVANT_TAGS_COLUMN
        separator = self.tag_cfg.TAG_SEPARATOR
        min_freq = self.tag_cfg.MIN_TAG_FREQUENCY

        print(f"  All tags column: '{all_tags_col}'")
        print(f"  Irrelevant tags column: '{irrelevant_tags_col}'")

        # Parse all tags per sample
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
        self.all_tags = sorted(
            [tag for tag, count in all_tag_counts.items() if count >= min_freq]
        )

        # Irrelevant tags (subset of all_tags that appear in irrelevant_tags column)
        self.irrelevant_tags = sorted(
            [tag for tag in irrelevant_tag_set if tag in self.all_tags]
        )

        # Relevant tags = all_tags - irrelevant_tags
        self.relevant_tags = sorted(
            [tag for tag in self.all_tags if tag not in self.irrelevant_tags]
        )

        # Create tag to neuron mapping (all tags get neurons)
        self.tag_to_neuron = {tag: i for i, tag in enumerate(self.all_tags)}

        print(f"\n  Tag statistics:")
        print(f"    Total samples with tags: {len(self.index_to_all_tags)}")
        print(f"    Unique tags (freq >= {min_freq}): {len(self.all_tags)}")
        print(f"    Irrelevant (bias) tags: {len(self.irrelevant_tags)}")
        print(f"    Relevant tags: {len(self.relevant_tags)}")

        # Show some examples
        print(f"\n  Sample irrelevant tags: {self.irrelevant_tags[:10]}")
        print(f"  Sample relevant tags: {self.relevant_tags[:10]}")

        # Compute average tags per sample
        avg_all = np.mean([len(tags) for tags in self.index_to_all_tags.values()])
        avg_irr = np.mean(
            [len(tags) for tags in self.index_to_irrelevant_tags.values()]
        )
        print(f"\n  Avg all tags per sample: {avg_all:.2f}")
        print(f"  Avg irrelevant tags per sample: {avg_irr:.2f}")

    def _get_feature_dim(self):
        """Get feature dimension from model."""
        # Try to get from model
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
            # Run a dummy forward pass
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                out = self.model(dummy)
                if isinstance(out, tuple):
                    self.feature_dim = out[1].shape[-1]
                else:
                    self.feature_dim = out.shape[-1]

        print(f"  Feature dimension: {self.feature_dim}")

    def _extract_features(
        self, dataloader: DataLoader, desc: str = "Extracting features"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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

                # Extract features
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

    def _build_tag_targets(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Build binary tag target matrix for training.

        Uses ALL tags (not just irrelevant).
        """
        batch_size = len(indices)
        num_tags = len(self.all_tags)
        targets = torch.zeros(batch_size, num_tags)

        for i, idx in enumerate(indices.tolist()):
            tags = self.index_to_all_tags.get(idx, [])
            for tag in tags:
                if tag in self.tag_to_neuron:
                    targets[i, self.tag_to_neuron[tag]] = 1

        return targets

    def _train_tag_sae(self, features: torch.Tensor, indices: torch.Tensor):
        """Train the Tag-Only SAE."""
        print(f"\n{'='*60}")
        print("Training Tag-Only SAE")
        print(f"{'='*60}")

        num_tags = len(self.all_tags)

        print(f"\n  Architecture:")
        print(f"    Input dim: {self.feature_dim}")
        print(f"    Dict size: {num_tags} (= number of tags)")
        print(f"    All neurons are tag-supervised")

        # Create SAE
        self.sae = TagOnlySAE(
            input_dim=self.feature_dim,
            num_tags=num_tags,
            tag_to_neuron=self.tag_to_neuron,
        ).to(self.device)

        # Create tag supervision loss
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

        # Create dataloader
        dataset = TensorDataset(features, indices)
        dataloader = DataLoader(
            dataset, batch_size=self.tag_cfg.BATCH_SIZE, shuffle=True
        )

        # Training loop
        num_steps = self.tag_cfg.STEPS
        step = 0
        losses_history = defaultdict(list)

        self.sae.train()
        pbar = tqdm(total=num_steps, desc="Training Tag-Only SAE")

        while step < num_steps:
            for batch_features, batch_indices in dataloader:
                if step >= num_steps:
                    break

                batch_features = batch_features.to(self.device)

                # Forward pass
                reconstructed, latents = self.sae(batch_features, return_latents=True)

                # Reconstruction loss
                loss_recon = F.mse_loss(reconstructed, batch_features)

                # Sparsity loss
                loss_sparsity = latents.abs().mean()

                # Tag supervision (on ALL tags)
                tag_targets = self._build_tag_targets(batch_indices).to(self.device)
                loss_tag = tag_loss_fn(latents, tag_targets)

                # Total loss
                loss = (
                    self.tag_cfg.LAMBDA_RECONSTRUCTION * loss_recon
                    + self.tag_cfg.LAMBDA_SPARSITY * loss_sparsity
                    + self.tag_cfg.LAMBDA_TAG * loss_tag
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
                losses_history["tag"].append(loss_tag.item())

                step += 1
                pbar.update(1)

                if step % 10 == 0:
                    l0 = (latents > 0).float().sum(dim=1).mean().item()
                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "recon": f"{loss_recon.item():.6f}",
                            "tag": f"{loss_tag.item():.4f}",
                            "L0": f"{l0:.0f}",
                        }
                    )

        pbar.close()

        # Save model
        self._save_sae(losses_history)

        return self.sae

    def _setup_optimizer(self):
        return

    def _setup_scheduler(self):
        return

    def _save_sae(self, losses_history: dict):
        """Save the trained SAE."""
        save_dir = os.path.join(self.log_path, "tag_only_sae")
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(save_dir, "tag_only_sae.pt")
        torch.save(self.sae.state_dict(), model_path)
        print(f"\nSaved model to {model_path}")

        # Save config
        config = {
            "input_dim": self.feature_dim,
            "num_tags": len(self.all_tags),
            "dict_size": len(self.all_tags),
            "tag_to_neuron": self.tag_to_neuron,
            "all_tags": self.all_tags,
            "irrelevant_tags": self.irrelevant_tags,
            "relevant_tags": self.relevant_tags,
            "tag_loss_type": self.tag_cfg.TAG_LOSS_TYPE,
        }
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save losses
        losses_path = os.path.join(save_dir, "losses.json")
        with open(losses_path, "w") as f:
            json.dump({k: v[-1000:] for k, v in losses_history.items()}, f)

    def _evaluate_debiasing(self, test_features, test_targets, test_biases):
        """
        Evaluate debiasing effect.

        Compares:
        1. Original features
        2. Debiased features (irrelevant tags removed)
        """
        print(f"\n{'='*60}")
        print("Evaluating Selective Debiasing")
        print(f"{'='*60}")

        print(f"\n  Debiasing strategy:")
        print(f"    Total neurons: {len(self.all_tags)}")
        print(f"    Neurons to zero (irrelevant): {len(self.irrelevant_tags)}")
        print(f"    Neurons to keep (relevant): {len(self.relevant_tags)}")

        self.sae.eval()

        results = {
            "original": {},
            "debiased": {},
            "per_group": {},
        }

        with torch.no_grad():
            test_features_gpu = test_features.to(self.device)

            # Encode
            latents = self.sae.encode(test_features_gpu)

            # Original reconstruction
            original_recon = self.sae.decode(latents)

            # Debiased reconstruction (zero irrelevant tags)
            debiased_recon = self.sae.decode_without_tags(latents, self.irrelevant_tags)

            # Compute statistics
            # 1. Reconstruction error
            original_mse = F.mse_loss(original_recon, test_features_gpu).item()
            debiased_mse = F.mse_loss(debiased_recon, test_features_gpu).item()

            # 2. Energy in removed neurons
            removed_indices = self.sae.get_neuron_indices_for_tags(self.irrelevant_tags)
            kept_indices = self.sae.get_neuron_indices_for_tags(self.relevant_tags)

            removed_energy = latents[:, removed_indices].pow(2).sum(dim=1).mean().item()
            kept_energy = latents[:, kept_indices].pow(2).sum(dim=1).mean().item()
            total_energy = latents.pow(2).sum(dim=1).mean().item()

            # 3. Cosine similarity
            original_normed = F.normalize(original_recon, dim=-1)
            debiased_normed = F.normalize(debiased_recon, dim=-1)
            input_normed = F.normalize(test_features_gpu, dim=-1)

            cos_orig_input = (original_normed * input_normed).sum(dim=1).mean().item()
            cos_debiased_input = (
                (debiased_normed * input_normed).sum(dim=1).mean().item()
            )
            cos_orig_debiased = (
                (original_normed * debiased_normed).sum(dim=1).mean().item()
            )

        print(f"\n  Reconstruction Quality:")
        print(f"    Original MSE: {original_mse:.6f}")
        print(f"    Debiased MSE: {debiased_mse:.6f}")

        print(f"\n  Energy Distribution:")
        print(
            f"    Removed (irrelevant) neurons: {removed_energy:.4f} ({100*removed_energy/total_energy:.1f}%)"
        )
        print(
            f"    Kept (relevant) neurons: {kept_energy:.4f} ({100*kept_energy/total_energy:.1f}%)"
        )

        print(f"\n  Cosine Similarities:")
        print(f"    Original vs Input: {cos_orig_input:.4f}")
        print(f"    Debiased vs Input: {cos_debiased_input:.4f}")
        print(f"    Original vs Debiased: {cos_orig_debiased:.4f}")

        results["reconstruction"] = {
            "original_mse": original_mse,
            "debiased_mse": debiased_mse,
        }
        results["energy"] = {
            "removed_neurons": removed_energy,
            "kept_neurons": kept_energy,
            "total": total_energy,
            "removed_fraction": removed_energy / total_energy,
        }
        results["cosine_similarity"] = {
            "original_vs_input": cos_orig_input,
            "debiased_vs_input": cos_debiased_input,
            "original_vs_debiased": cos_orig_debiased,
        }

        # Zero-shot evaluation
        zero_shot_results = self._evaluate_zero_shot(
            test_features, test_targets, test_biases
        )
        if zero_shot_results:
            results["zero_shot"] = zero_shot_results

        # Group similarity analysis
        similarity_results = self._evaluate_group_similarity(
            test_features, test_targets, test_biases
        )
        results["group_similarity"] = similarity_results

        # Save results
        save_dir = os.path.join(self.log_path, "tag_only_sae")
        results_path = os.path.join(save_dir, "debiasing_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {results_path}")

        return results

    def _evaluate_zero_shot(self, test_features, test_targets, test_biases):
        """Zero-shot evaluation using text embeddings."""
        print(f"\n{'='*60}")
        print("Zero-Shot Classification Evaluation")
        print(f"{'='*60}")

        if not hasattr(self, "target2name") or self.target2name is None:
            print("  No target2name mapping, skipping zero-shot.")
            return {}

        # Get text encoder
        text_encoder = self._get_text_encoder()
        if text_encoder is None:
            print("  No text encoder available, skipping zero-shot.")
            return {}

        unique_targets = sorted(test_targets.unique().tolist())
        class_names = [self.target2name.get(t, f"class_{t}") for t in unique_targets]

        print(f"  Classes: {class_names}")

        # Encode class names
        with torch.no_grad():
            prompts = [f"a photo of a {name}" for name in class_names]
            text_features = text_encoder(prompts)
            text_features = F.normalize(text_features, dim=-1)

        self.sae.eval()

        with torch.no_grad():
            test_features_gpu = test_features.to(self.device)

            # Original features
            original_normed = F.normalize(test_features_gpu, dim=-1)

            # Debiased features
            latents = self.sae.encode(test_features_gpu)
            debiased = self.sae.decode_without_tags(latents, self.irrelevant_tags)
            debiased_normed = F.normalize(debiased, dim=-1)

            # Zero-shot classification
            original_sim = original_normed @ text_features.T
            debiased_sim = debiased_normed @ text_features.T

            original_preds = original_sim.argmax(dim=1).cpu()
            debiased_preds = debiased_sim.argmax(dim=1).cpu()

            # Map predictions
            pred_to_target = {i: t for i, t in enumerate(unique_targets)}
            original_preds_mapped = torch.tensor(
                [pred_to_target[p.item()] for p in original_preds]
            )
            debiased_preds_mapped = torch.tensor(
                [pred_to_target[p.item()] for p in debiased_preds]
            )

            # Accuracy
            original_acc = (original_preds_mapped == test_targets).float().mean().item()
            debiased_acc = (debiased_preds_mapped == test_targets).float().mean().item()

        results = {
            "original_acc": original_acc,
            "debiased_acc": debiased_acc,
        }

        print(f"\n  Zero-Shot Accuracy:")
        print(f"    Original: {original_acc:.4f}")
        print(f"    Debiased: {debiased_acc:.4f}")
        print(f"    Change:   {debiased_acc - original_acc:+.4f}")

        # Per-group accuracy
        bias_name = self.biases[0] if self.biases else None
        if bias_name and test_biases:
            bias_values = (
                test_biases.get(bias_name)
                if isinstance(test_biases, dict)
                else test_biases
            )

            if bias_values is not None:
                print(f"\n  Per-group accuracy:")

                unique_biases = sorted(bias_values.unique().tolist())
                group_accs_orig = []
                group_accs_debiased = []

                for t in unique_targets:
                    for b in unique_biases:
                        mask = (test_targets == t) & (bias_values == b)
                        if mask.sum() == 0:
                            continue

                        orig_acc = (
                            (original_preds_mapped[mask] == test_targets[mask])
                            .float()
                            .mean()
                            .item()
                        )
                        deb_acc = (
                            (debiased_preds_mapped[mask] == test_targets[mask])
                            .float()
                            .mean()
                            .item()
                        )

                        group_accs_orig.append(orig_acc)
                        group_accs_debiased.append(deb_acc)

                        print(
                            f"    (t={t}, b={b}): orig={orig_acc:.4f}, deb={deb_acc:.4f}, "
                            f"Δ={deb_acc - orig_acc:+.4f}, n={mask.sum().item()}"
                        )

                if group_accs_orig:
                    results["original_worst_group"] = min(group_accs_orig)
                    results["debiased_worst_group"] = min(group_accs_debiased)

                    print(f"\n    Worst-group accuracy:")
                    print(f"      Original: {results['original_worst_group']:.4f}")
                    print(f"      Debiased: {results['debiased_worst_group']:.4f}")
                    print(
                        f"      Change:   {results['debiased_worst_group'] - results['original_worst_group']:+.4f}"
                    )

        return results

    def _evaluate_group_similarity(self, test_features, test_targets, test_biases):
        """Compute group similarity matrices."""
        print(f"\n{'='*60}")
        print("Group Similarity Analysis")
        print(f"{'='*60}")

        bias_name = self.biases[0] if self.biases else None
        if not bias_name or not test_biases:
            print("  No bias information, skipping group similarity.")
            return {}

        bias_values = (
            test_biases.get(bias_name) if isinstance(test_biases, dict) else test_biases
        )
        if bias_values is None:
            return {}

        self.sae.eval()

        with torch.no_grad():
            test_features_gpu = test_features.to(self.device)

            # Original and debiased
            latents = self.sae.encode(test_features_gpu)
            original = test_features_gpu
            debiased = self.sae.decode_without_tags(latents, self.irrelevant_tags)

            # Normalize
            original_norm = F.normalize(original, dim=-1)
            debiased_norm = F.normalize(debiased, dim=-1)

        # Compute group centroids
        unique_targets = sorted(test_targets.unique().tolist())
        unique_biases = sorted(bias_values.unique().tolist())

        groups = []
        original_centroids = []
        debiased_centroids = []

        for t in unique_targets:
            for b in unique_biases:
                mask = (test_targets == t) & (bias_values == b)
                if mask.sum() == 0:
                    continue

                groups.append((t, b))
                original_centroids.append(original_norm[mask].mean(dim=0))
                debiased_centroids.append(debiased_norm[mask].mean(dim=0))

        if len(groups) < 2:
            return {}

        original_centroids = torch.stack(original_centroids)
        debiased_centroids = torch.stack(debiased_centroids)

        # Compute similarity matrices
        original_sim = original_centroids @ original_centroids.T
        debiased_sim = debiased_centroids @ debiased_centroids.T
        change_sim = debiased_sim - original_sim

        # Print matrices
        labels = [f"t{t}b{b}" for t, b in groups]

        print(f"\n  Original Feature Similarity:")
        self._print_matrix(original_sim, labels)

        print(f"\n  Debiased Feature Similarity:")
        self._print_matrix(debiased_sim, labels)

        print(f"\n  Change (Debiased - Original):")
        self._print_matrix(change_sim, labels, show_sign=True)

        # Compute summary metrics
        # Intra-class: same target, different bias
        # Inter-class: different target
        intra_orig = []
        intra_deb = []
        inter_orig = []
        inter_deb = []

        for i, (t1, b1) in enumerate(groups):
            for j, (t2, b2) in enumerate(groups):
                if i >= j:
                    continue
                if t1 == t2:
                    intra_orig.append(original_sim[i, j].item())
                    intra_deb.append(debiased_sim[i, j].item())
                else:
                    inter_orig.append(original_sim[i, j].item())
                    inter_deb.append(debiased_sim[i, j].item())

        results = {}
        if intra_orig:
            results["intra_class_original"] = np.mean(intra_orig)
            results["intra_class_debiased"] = np.mean(intra_deb)
            print(f"\n  Intra-class similarity (same target, diff bias):")
            print(f"    Original: {results['intra_class_original']:.4f}")
            print(f"    Debiased: {results['intra_class_debiased']:.4f}")
            print(
                f"    Change:   {results['intra_class_debiased'] - results['intra_class_original']:+.4f}"
            )

            if results["intra_class_debiased"] > results["intra_class_original"]:
                print(f"    ✓ Intra-class similarity INCREASED (bias removed)")
            else:
                print(f"    ✗ Intra-class similarity decreased")

        if inter_orig:
            results["inter_class_original"] = np.mean(inter_orig)
            results["inter_class_debiased"] = np.mean(inter_deb)
            print(f"\n  Inter-class similarity (diff target):")
            print(f"    Original: {results['inter_class_original']:.4f}")
            print(f"    Debiased: {results['inter_class_debiased']:.4f}")
            print(
                f"    Change:   {results['inter_class_debiased'] - results['inter_class_original']:+.4f}"
            )

            if results["inter_class_debiased"] < results["inter_class_original"]:
                print(f"    ✓ Inter-class similarity DECREASED (classes more distinct)")
            else:
                print(f"    ✗ Inter-class similarity increased")

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
                        test_tokens = tokenizer(["test"]).to(self.device)
                        test_emb = model.encode_text(test_tokens)

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

    def _visualize_neurons(self, features, indices):
        """Visualize tag neurons sorted by activation strength."""
        print(f"\n{'='*60}")
        print("Visualizing Tag Neurons (Train Set)")
        print(f"{'='*60}")

        try:
            import matplotlib.pyplot as plt
            from PIL import Image
        except ImportError:
            print("matplotlib or PIL not available, skipping visualization")
            return

        self.sae.eval()

        # Get activations
        with torch.no_grad():
            all_latents = []
            batch_size = 256
            for i in range(0, len(features), batch_size):
                batch = features[i : i + batch_size].to(self.device)
                latents = self.sae.encode(batch)
                all_latents.append(latents.cpu())
            all_latents = torch.cat(all_latents, dim=0)

        # Get image paths
        image_paths = self._get_image_paths()

        # Create output directory
        viz_dir = os.path.join(self.log_path, "tag_only_sae", "visualizations")
        irr_dir = os.path.join(viz_dir, "irrelevant_tags")
        rel_dir = os.path.join(viz_dir, "relevant_tags")
        os.makedirs(irr_dir, exist_ok=True)
        os.makedirs(rel_dir, exist_ok=True)

        num_to_viz = 1
        top_k_images = 16

        # Compute mean activation for each neuron to sort by
        mean_activations = all_latents.mean(dim=0)  # (num_neurons,)

        for category, tags, out_dir in [
            ("irrelevant", self.irrelevant_tags, irr_dir),
            ("relevant", self.relevant_tags, rel_dir),
        ]:
            print(f"\n  Visualizing top {num_to_viz} {category} tags by activation...")

            # Get neuron indices for this category
            neuron_indices = [self.tag_to_neuron[tag] for tag in tags]

            # Sort by mean activation (descending)
            tag_activations = [
                (tag, neuron_indices[i], mean_activations[neuron_indices[i]].item())
                for i, tag in enumerate(tags)
            ]
            tag_activations.sort(key=lambda x: x[2], reverse=True)

            # Take top-k
            top_tags = tag_activations[:num_to_viz]

            for rank, (tag, neuron_idx, mean_act) in enumerate(
                tqdm(top_tags, desc=f"    {category}")
            ):
                acts = all_latents[:, neuron_idx]

                # Get top-k images
                top_k_idx = acts.argsort(descending=True)[:top_k_images]
                top_k_acts = acts[top_k_idx]
                top_k_samples = indices[top_k_idx].tolist()

                # Check tag presence
                has_tag = [
                    tag in self.index_to_all_tags.get(idx, []) for idx in top_k_samples
                ]

                # Create figure
                fig, axes = plt.subplots(1, top_k_images, figsize=(16, 3))
                fig.suptitle(
                    f"#{rank+1} Tag: '{tag}' (neuron {neuron_idx}, mean_act={mean_act:.3f})",
                    fontsize=11,
                )

                for i, (sample_idx, act_val, present) in enumerate(
                    zip(top_k_samples, top_k_acts.tolist(), has_tag)
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

                    color = "green" if present else "red"
                    for spine in ax.spines.values():
                        spine.set_edgecolor(color)
                        spine.set_linewidth(3)

                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(
                        f"{act_val:.2f}\n{'✓' if present else '✗'}", fontsize=9
                    )

                plt.tight_layout()
                safe_name = tag.replace("/", "_").replace(" ", "_")[:25]
                plt.savefig(
                    os.path.join(
                        out_dir, f"{rank:02d}_{neuron_idx:04d}_{safe_name}.png"
                    ),
                    dpi=80,
                )
                plt.close()

        # Create HTML
        self._create_viz_html(viz_dir)

        print(f"\n  Saved visualizations to {viz_dir}")

    def _get_image_paths(self) -> dict:
        """Get image paths from dataset."""
        image_paths = {}

        if "train" in self.dataloaders:
            dataset = self.dataloaders["train"].dataset

            if hasattr(dataset, "samples"):
                for idx, (path, _) in enumerate(dataset.samples):
                    image_paths[idx] = path
            if hasattr(dataset, "img_fpath_list"):
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

    def _create_viz_html(self, viz_dir):
        """Create HTML summary."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Tag-Only SAE Visualizations</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        h1, h2 { color: #333; }
        .section { margin: 20px 0; }
        .grid { display: flex; flex-wrap: wrap; gap: 10px; }
        .card { background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card img { max-width: 100%; }
        .irrelevant { border-left: 4px solid #f44336; }
        .relevant { border-left: 4px solid #4CAF50; }
        .legend { background: #fff3cd; padding: 10px; border-radius: 4px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Tag-Only SAE Neuron Visualizations</h1>
    <div class="legend">
        <strong>Legend:</strong>
        <span style="color: green">✓ Green = tag present</span> |
        <span style="color: red">✗ Red = tag absent</span> |
        <span style="color: #f44336">■ Irrelevant tags (bias - removed during debiasing)</span> |
        <span style="color: #4CAF50">■ Relevant tags (kept during debiasing)</span>
    </div>
"""

        # Add irrelevant tags
        html += '<div class="section"><h2>Irrelevant Tags (Bias - Removed)</h2><div class="grid">'
        irr_dir = os.path.join(viz_dir, "irrelevant_tags")
        if os.path.exists(irr_dir):
            for img_file in sorted(os.listdir(irr_dir)):
                if img_file.endswith(".png"):
                    tag_name = (
                        img_file.split("_", 1)[1].replace(".png", "").replace("_", " ")
                    )
                    html += f'<div class="card irrelevant"><img src="irrelevant_tags/{img_file}"><p>{tag_name}</p></div>'
        html += "</div></div>"

        # Add relevant tags
        html += '<div class="section"><h2>Relevant Tags (Kept)</h2><div class="grid">'
        rel_dir = os.path.join(viz_dir, "relevant_tags")
        if os.path.exists(rel_dir):
            for img_file in sorted(os.listdir(rel_dir)):
                if img_file.endswith(".png"):
                    tag_name = (
                        img_file.split("_", 1)[1].replace(".png", "").replace("_", " ")
                    )
                    html += f'<div class="card relevant"><img src="relevant_tags/{img_file}"><p>{tag_name}</p></div>'
        html += "</div></div>"

        html += "</body></html>"

        with open(os.path.join(viz_dir, "index.html"), "w") as f:
            f.write(html)

    def _visualize_neurons_test(self, features, indices):
        """
        Visualize tag neurons on test set, sorted by activation strength.

        Unlike train visualization, this does NOT check if tags are present.
        Just shows neuron name and top activated test images.
        """
        print(f"\n{'='*60}")
        print("Visualizing Tag Neurons (Test Set)")
        print(f"{'='*60}")

        try:
            import matplotlib.pyplot as plt
            from PIL import Image
        except ImportError:
            print("matplotlib or PIL not available, skipping visualization")
            return

        self.sae.eval()

        # Get activations
        with torch.no_grad():
            all_latents = []
            batch_size = 256
            for i in range(0, len(features), batch_size):
                batch = features[i : i + batch_size].to(self.device)
                latents = self.sae.encode(batch)
                all_latents.append(latents.cpu())
            all_latents = torch.cat(all_latents, dim=0)

        # Get image paths from test set
        image_paths = self._get_image_paths_test()

        # Create output directory
        viz_dir = os.path.join(self.log_path, "tag_only_sae", "visualizations_test")
        irr_dir = os.path.join(viz_dir, "irrelevant_tags")
        rel_dir = os.path.join(viz_dir, "relevant_tags")
        os.makedirs(irr_dir, exist_ok=True)
        os.makedirs(rel_dir, exist_ok=True)

        num_to_viz = 1
        top_k_images = 16

        # Compute mean activation for each neuron to sort by
        mean_activations = all_latents.mean(dim=0)  # (num_neurons,)

        for category, tags, out_dir in [
            ("irrelevant", self.irrelevant_tags, irr_dir),
            ("relevant", self.relevant_tags, rel_dir),
        ]:
            print(
                f"\n  Visualizing top {num_to_viz} {category} tags by activation (test set)..."
            )

            # Get neuron indices for this category
            neuron_indices = [self.tag_to_neuron[tag] for tag in tags]

            # Sort by mean activation (descending)
            tag_activations = [
                (tag, neuron_indices[i], mean_activations[neuron_indices[i]].item())
                for i, tag in enumerate(tags)
            ]
            tag_activations.sort(key=lambda x: x[2], reverse=True)

            # Take top-k
            top_tags = tag_activations[:num_to_viz]

            for rank, (tag, neuron_idx, mean_act) in enumerate(
                tqdm(top_tags, desc=f"    {category}")
            ):
                acts = all_latents[:, neuron_idx]

                # Get top-k images
                top_k_idx = acts.argsort(descending=True)[:top_k_images]
                top_k_acts = acts[top_k_idx]
                top_k_samples = indices[top_k_idx].tolist()

                # Create figure (no tag presence check)
                fig, axes = plt.subplots(1, top_k_images, figsize=(16, 3))
                fig.suptitle(
                    f"#{rank+1} Tag: '{tag}' (neuron {neuron_idx}, mean_act={mean_act:.3f})",
                    fontsize=11,
                )

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

                    # Use blue border for test set (neutral)
                    color = "#2196F3"  # Blue
                    for spine in ax.spines.values():
                        spine.set_edgecolor(color)
                        spine.set_linewidth(3)

                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"act={act_val:.2f}", fontsize=9)

                plt.tight_layout()
                safe_name = tag.replace("/", "_").replace(" ", "_")[:25]
                plt.savefig(
                    os.path.join(
                        out_dir, f"{rank:02d}_{neuron_idx:04d}_{safe_name}.png"
                    ),
                    dpi=80,
                )
                plt.close()

        # Create HTML for test set
        self._create_viz_html_test(viz_dir)

        print(f"\n  Saved test visualizations to {viz_dir}")

    def _get_image_paths_test(self) -> dict:
        """Get image paths from test dataset."""
        image_paths = {}

        if "test" in self.dataloaders:
            dataset = self.dataloaders["test"].dataset

            if hasattr(dataset, "samples"):
                for idx, (path, _) in enumerate(dataset.samples):
                    image_paths[idx] = path
            if hasattr(dataset, "img_fpath_list"):
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

    def _create_viz_html_test(self, viz_dir):
        """Create HTML summary for test set visualizations."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Tag-Only SAE Test Set Visualizations</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        h1, h2 { color: #333; }
        .section { margin: 20px 0; }
        .grid { display: flex; flex-wrap: wrap; gap: 10px; }
        .card { background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card img { max-width: 100%; }
        .irrelevant { border-left: 4px solid #f44336; }
        .relevant { border-left: 4px solid #4CAF50; }
        .legend { background: #e3f2fd; padding: 10px; border-radius: 4px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Tag-Only SAE Neuron Visualizations (Test Set)</h1>
    <div class="legend">
        <strong>Test Set Visualization:</strong>
        Shows top activated images for each neuron on the test set.
        <br>
        <span style="color: #f44336">■ Irrelevant tags (bias - removed during debiasing)</span> |
        <span style="color: #4CAF50">■ Relevant tags (kept during debiasing)</span>
    </div>
"""

        # Add irrelevant tags
        html += '<div class="section"><h2>Irrelevant Tags (Bias - Removed)</h2><div class="grid">'
        irr_dir = os.path.join(viz_dir, "irrelevant_tags")
        if os.path.exists(irr_dir):
            for img_file in sorted(os.listdir(irr_dir)):
                if img_file.endswith(".png"):
                    tag_name = (
                        img_file.split("_", 1)[1].replace(".png", "").replace("_", " ")
                    )
                    html += f'<div class="card irrelevant"><img src="irrelevant_tags/{img_file}"><p>{tag_name}</p></div>'
        html += "</div></div>"

        # Add relevant tags
        html += '<div class="section"><h2>Relevant Tags (Kept)</h2><div class="grid">'
        rel_dir = os.path.join(viz_dir, "relevant_tags")
        if os.path.exists(rel_dir):
            for img_file in sorted(os.listdir(rel_dir)):
                if img_file.endswith(".png"):
                    tag_name = (
                        img_file.split("_", 1)[1].replace(".png", "").replace("_", " ")
                    )
                    html += f'<div class="card relevant"><img src="relevant_tags/{img_file}"><p>{tag_name}</p></div>'
        html += "</div></div>"

        html += "</body></html>"

        with open(os.path.join(viz_dir, "index.html"), "w") as f:
            f.write(html)

    def _analyze_subgroup_activations(
        self,
        train_features: torch.Tensor,
        train_targets: torch.Tensor,
        train_biases: Dict[str, torch.Tensor],
        test_features: torch.Tensor,
        test_targets: torch.Tensor,
        test_biases: Dict[str, torch.Tensor],
    ):
        """
        Analyze neuron activations across subgroups (target × bias1 × bias2 × ...).

        Uses TRAINING data for:
        - Computing discriminability scores
        - Fitting classifiers (centroids, logistic regression)

        Uses TEST data for:
        - Evaluating classification accuracy per subgroup

        Identifies:
        1. Neurons that discriminate by TARGET (good features)
        2. Neurons that discriminate by each BIAS (spurious correlations)
        3. Per-subgroup activation patterns
        """
        print(f"\n{'='*60}")
        print("Subgroup Activation Analysis")
        print(f"{'='*60}")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib/seaborn not available, skipping analysis")
            return

        self.sae.eval()

        # ============================================
        # Get activations for TRAIN and TEST
        # ============================================

        def get_latents(features):
            with torch.no_grad():
                all_latents = []
                batch_size = 256
                for i in range(0, len(features), batch_size):
                    batch = features[i : i + batch_size].to(self.device)
                    latents = self.sae.encode(batch)
                    all_latents.append(latents.cpu())
                return torch.cat(all_latents, dim=0).numpy()

        train_latents = get_latents(train_features)
        test_latents = get_latents(test_features)

        train_targets_np = train_targets.numpy()
        test_targets_np = test_targets.numpy()

        # Build subgroup keys for train and test
        bias_names = list(train_biases.keys())
        train_bias_arrays = {name: train_biases[name].numpy() for name in bias_names}
        test_bias_arrays = {name: test_biases[name].numpy() for name in bias_names}

        print(f"  Bias attributes: {bias_names}")
        print(f"  Train samples: {len(train_targets_np)}")
        print(f"  Test samples: {len(test_targets_np)}")

        def build_subgroup_keys(targets_np, bias_arrays):
            subgroup_keys = []
            for i in range(len(targets_np)):
                key = [f"t={targets_np[i]}"]
                for b_name in bias_names:
                    key.append(f"{b_name}={bias_arrays[b_name][i]}")
                subgroup_keys.append(tuple(key))
            return subgroup_keys

        train_subgroup_keys = build_subgroup_keys(train_targets_np, train_bias_arrays)
        test_subgroup_keys = build_subgroup_keys(test_targets_np, test_bias_arrays)

        # Get unique subgroups (from both train and test)
        unique_subgroups = sorted(set(train_subgroup_keys) | set(test_subgroup_keys))

        print(f"\n  Subgroups (Train / Test counts):")
        for sg in unique_subgroups:
            train_count = train_subgroup_keys.count(sg)
            test_count = test_subgroup_keys.count(sg)
            print(f"    {sg}: train={train_count}, test={test_count}")

        # ============================================
        # Compute discriminability on TRAINING data
        # ============================================

        num_neurons = train_latents.shape[1]
        num_subgroups = len(unique_subgroups)

        # Mean activation per subgroup per neuron (on train)
        test_subgroup_means = np.zeros((num_subgroups, num_neurons))
        test_subgroup_stds = np.zeros((num_subgroups, num_neurons))

        for i, sg in enumerate(unique_subgroups):
            mask = np.array([sk == sg for sk in test_subgroup_keys])
            if mask.sum() > 0:
                test_subgroup_means[i] = test_latents[mask].mean(axis=0)
                test_subgroup_stds[i] = test_latents[mask].std(axis=0)

        # Target discriminability (computed on test)
        unique_targets = sorted(set(test_targets_np))
        target_means = np.zeros((len(unique_targets), num_neurons))
        for i, t in enumerate(unique_targets):
            mask = test_targets_np == t
            if mask.sum() > 0:
                target_means[i] = test_latents[mask].mean(axis=0)

        target_variance = np.var(target_means, axis=0)
        total_variance = np.var(test_latents, axis=0) + 1e-8
        target_discriminability = target_variance / total_variance

        # Bias discriminability for each bias attribute (computed on test)
        bias_discriminability = {}
        for b_name in bias_names:
            unique_bias_vals = sorted(set(test_bias_arrays[b_name]))
            bias_means = np.zeros((len(unique_bias_vals), num_neurons))
            for i, b in enumerate(unique_bias_vals):
                mask = test_bias_arrays[b_name] == b
                if mask.sum() > 0:
                    bias_means[i] = test_latents[mask].mean(axis=0)

            bias_variance = np.var(bias_means, axis=0)
            bias_discriminability[b_name] = bias_variance / total_variance

        # ============================================
        # Identify top discriminative neurons
        # ============================================

        results = {
            "subgroups": [str(sg) for sg in unique_subgroups],
            "bias_names": bias_names,
            "neurons": {},
        }

        # Top neurons by target discriminability (GOOD)
        top_target_neurons_idx = np.argsort(target_discriminability)[::-1][:250]

        print(f"\n  Top 10 neurons discriminating by TARGET (good features):")
        for rank, neuron_idx in enumerate(top_target_neurons_idx[:10]):
            tag = self.sae.neuron_to_tag.get(neuron_idx, f"neuron_{neuron_idx}")
            score = target_discriminability[neuron_idx]
            category = "irrelevant" if tag in self.irrelevant_tags else "relevant"
            print(
                f"    #{rank+1}: '{tag}' (neuron {neuron_idx}, score={score:.4f}, {category})"
            )

            results.setdefault("target_discriminative", []).append(
                {
                    "rank": rank + 1,
                    "tag": tag,
                    "neuron_idx": int(neuron_idx),
                    "score": float(score),
                    "category": category,
                }
            )

        # Top neurons by each bias discriminability (BAD - spurious)
        for b_name in bias_names:
            top_bias_neurons_idx = np.argsort(bias_discriminability[b_name])[::-1][:30]

            print(
                f"\n  Top 10 neurons discriminating by {b_name.upper()} (potential bias):"
            )
            for rank, neuron_idx in enumerate(top_bias_neurons_idx[:10]):
                tag = self.sae.neuron_to_tag.get(neuron_idx, f"neuron_{neuron_idx}")
                score = bias_discriminability[b_name][neuron_idx]
                category = "irrelevant" if tag in self.irrelevant_tags else "relevant"
                print(
                    f"    #{rank+1}: '{tag}' (neuron {neuron_idx}, score={score:.4f}, {category})"
                )

                results.setdefault(f"bias_{b_name}_discriminative", []).append(
                    {
                        "rank": rank + 1,
                        "tag": tag,
                        "neuron_idx": int(neuron_idx),
                        "score": float(score),
                        "category": category,
                    }
                )

        # ============================================
        # Create visualizations
        # ============================================

        viz_dir = os.path.join(self.log_path, "tag_only_sae", "subgroup_analysis")
        os.makedirs(viz_dir, exist_ok=True)

        # Compute test subgroup means for visualization
        test_subgroup_means = np.zeros((num_subgroups, num_neurons))
        test_subgroup_stds = np.zeros((num_subgroups, num_neurons))

        for i, sg in enumerate(unique_subgroups):
            mask = np.array([sk == sg for sk in test_subgroup_keys])
            if mask.sum() > 0:
                test_subgroup_means[i] = test_latents[mask].mean(axis=0)
                test_subgroup_stds[i] = test_latents[mask].std(axis=0)

        # 1. Heatmap: top discriminative neurons × subgroups (test data)
        self._plot_subgroup_heatmap(
            test_subgroup_means,
            unique_subgroups,
            top_target_neurons_idx[:20],
            "Top Target-Discriminative Neurons (Test Set)",
            os.path.join(viz_dir, "heatmap_target_neurons.png"),
        )

        for b_name in bias_names:
            top_bias_idx = np.argsort(bias_discriminability[b_name])[::-1][:20]
            self._plot_subgroup_heatmap(
                test_subgroup_means,
                unique_subgroups,
                top_bias_idx,
                f"Top {b_name}-Discriminative Neurons (Test Set)",
                os.path.join(viz_dir, f"heatmap_{b_name}_neurons.png"),
            )

        # 2. Bar plot: discriminability scores comparison
        self._plot_discriminability_comparison(
            target_discriminability,
            bias_discriminability,
            os.path.join(viz_dir, "discriminability_comparison.png"),
        )

        # 3. Per-neuron subgroup activation breakdown (test data)
        self._plot_neuron_subgroup_breakdown(
            test_subgroup_means,
            test_subgroup_stds,
            unique_subgroups,
            top_target_neurons_idx[:10],
            "Target-Discriminative",
            os.path.join(viz_dir, "breakdown_target_neurons.png"),
        )

        # 4. Summary statistics
        for category_name, indices in [
            ("target", top_target_neurons_idx[:30]),
        ] + [
            (f"bias_{b}", np.argsort(bias_discriminability[b])[::-1][:30])
            for b in bias_names
        ]:

            tags_in_top = [self.sae.neuron_to_tag.get(idx, "") for idx in indices]
            num_irrelevant = sum(1 for t in tags_in_top if t in self.irrelevant_tags)
            num_relevant = sum(1 for t in tags_in_top if t in self.relevant_tags)

            results[f"{category_name}_top30_irrelevant_count"] = num_irrelevant
            results[f"{category_name}_top30_relevant_count"] = num_relevant

            print(f"\n  {category_name.upper()} top-30 discriminative neurons:")
            print(f"    Irrelevant tags: {num_irrelevant}")
            print(f"    Relevant tags: {num_relevant}")

        # ============================================
        # 5. Classification using top-k discriminative neurons
        #    Fit on TRAIN, evaluate on TEST
        # ============================================

        classification_results = self._classify_with_top_neurons(
            train_latents=train_latents,
            train_targets=train_targets_np,
            test_latents=test_latents,
            test_targets=test_targets_np,
            test_subgroup_keys=test_subgroup_keys,
            unique_subgroups=unique_subgroups,
            top_target_neurons_idx=top_target_neurons_idx,
            target_discriminability=target_discriminability,
            bias_names=bias_names,
            bias_discriminability=bias_discriminability,
            test_bias_arrays=test_bias_arrays,
            viz_dir=viz_dir,
        )
        results["classification"] = classification_results

        # Save results
        results_path = os.path.join(viz_dir, "subgroup_analysis.json")

        # Add per-neuron data
        for neuron_idx in range(num_neurons):
            tag = self.sae.neuron_to_tag.get(neuron_idx, f"neuron_{neuron_idx}")
            results["neurons"][tag] = {
                "neuron_idx": int(neuron_idx),
                "category": "irrelevant" if tag in self.irrelevant_tags else "relevant",
                "target_discriminability": float(target_discriminability[neuron_idx]),
                "bias_discriminability": {
                    b: float(bias_discriminability[b][neuron_idx]) for b in bias_names
                },
                "subgroup_means": {
                    str(sg): float(train_subgroup_means[i, neuron_idx])
                    for i, sg in enumerate(unique_subgroups)
                },
            }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n  Saved analysis to {viz_dir}")

        return results

    def _plot_subgroup_heatmap(
        self,
        subgroup_means: np.ndarray,
        subgroups: list,
        neuron_indices: np.ndarray,
        title: str,
        save_path: str,
    ):
        """Plot heatmap of neuron activations across subgroups."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get data for selected neurons
        data = subgroup_means[:, neuron_indices].T  # (neurons, subgroups)

        # Get neuron labels
        neuron_labels = [
            self.sae.neuron_to_tag.get(idx, f"n{idx}") for idx in neuron_indices
        ]
        # Truncate long labels
        neuron_labels = [l[:20] + "..." if len(l) > 20 else l for l in neuron_labels]

        # Subgroup labels (simplified)
        subgroup_labels = [", ".join([s.split("=")[1] for s in sg]) for sg in subgroups]

        # Plot
        fig, ax = plt.subplots(
            figsize=(max(10, len(subgroups) * 0.8), max(8, len(neuron_indices) * 0.4))
        )

        sns.heatmap(
            data,
            xticklabels=subgroup_labels,
            yticklabels=neuron_labels,
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "Mean Activation"},
        )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Subgroups (target, bias...)")
        ax.set_ylabel("Neurons (tags)")

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

    def _plot_discriminability_comparison(
        self, target_disc: np.ndarray, bias_disc: Dict[str, np.ndarray], save_path: str
    ):
        """Plot comparison of discriminability scores."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Distribution of discriminability scores
        ax = axes[0]
        ax.hist(target_disc, bins=50, alpha=0.7, label="Target", color="green")
        colors = plt.cm.tab10.colors
        for i, (b_name, b_disc) in enumerate(bias_disc.items()):
            ax.hist(
                b_disc, bins=50, alpha=0.5, label=f"Bias: {b_name}", color=colors[i + 1]
            )
        ax.set_xlabel("Discriminability Score")
        ax.set_ylabel("Number of Neurons")
        ax.set_title("Distribution of Discriminability Scores")
        ax.legend()

        # Right: Scatter plot - target vs bias discriminability
        ax = axes[1]
        for i, (b_name, b_disc) in enumerate(bias_disc.items()):
            # Color by category
            colors_scatter = [
                (
                    "red"
                    if self.sae.neuron_to_tag.get(j, "") in self.irrelevant_tags
                    else "green"
                )
                for j in range(len(target_disc))
            ]
            ax.scatter(
                target_disc, b_disc, c=colors_scatter, alpha=0.5, s=10, label=b_name
            )

            # Add diagonal
            max_val = max(target_disc.max(), b_disc.max())
            ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)

            ax.set_xlabel("Target Discriminability")
            ax.set_ylabel(f"Bias ({b_name}) Discriminability")
            ax.set_title(
                "Target vs Bias Discriminability\n(green=relevant, red=irrelevant)"
            )
            break  # Only first bias for scatter

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

    def _plot_neuron_subgroup_breakdown(
        self,
        subgroup_means: np.ndarray,
        subgroup_stds: np.ndarray,
        subgroups: list,
        neuron_indices: np.ndarray,
        title_prefix: str,
        save_path: str,
    ):
        """Plot per-neuron activation breakdown by subgroup."""
        import matplotlib.pyplot as plt

        n_neurons = len(neuron_indices)
        n_cols = 2
        n_rows = (n_neurons + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
        axes = axes.flatten() if n_neurons > 1 else [axes]

        subgroup_labels = [", ".join([s.split("=")[1] for s in sg]) for sg in subgroups]
        x = np.arange(len(subgroups))

        for i, neuron_idx in enumerate(neuron_indices):
            ax = axes[i]
            tag = self.sae.neuron_to_tag.get(neuron_idx, f"neuron_{neuron_idx}")

            means = subgroup_means[:, neuron_idx]
            stds = subgroup_stds[:, neuron_idx]

            # Color bars by target value
            colors = ["#4CAF50" if "t=1" in str(sg) else "#2196F3" for sg in subgroups]

            ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(subgroup_labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Mean Activation")
            ax.set_title(f"'{tag}' (neuron {neuron_idx})", fontsize=10)

        # Hide unused axes
        for i in range(len(neuron_indices), len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f"{title_prefix} Neurons - Subgroup Breakdown", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

    def _classify_with_top_neurons(
        self,
        train_latents: np.ndarray,
        train_targets: np.ndarray,
        test_latents: np.ndarray,
        test_targets: np.ndarray,
        test_subgroup_keys: list,
        unique_subgroups: list,
        top_target_neurons_idx: np.ndarray,
        target_discriminability: np.ndarray,
        bias_names: list,
        bias_discriminability: Dict[str, np.ndarray],
        test_bias_arrays: Dict[str, np.ndarray],
        viz_dir: str,
    ) -> dict:
        """
        Classify test samples using top-k target-discriminative neurons.

        IMPORTANT: Fits on TRAIN data, evaluates on TEST data.

        Methods:
        1. Nearest Centroid: Compute centroids on train, predict on test
        2. Linear Probe: Fit logistic regression on train, predict on test

        Reports accuracy per subgroup on TEST data.
        """
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        print(f"\n  {'='*50}")
        print("  Classification with Top-K Discriminative Neurons")
        print(f"  (Fit on TRAIN, Evaluate on TEST)")
        print(f"  {'='*50}")

        results = {}
        unique_targets = sorted(set(train_targets))
        k_values = [5, 10, 20, 50]

        # Filter k_values to not exceed available neurons
        max_k = len(top_target_neurons_idx)
        k_values = [k for k in k_values if k <= max_k]
        if not k_values:
            k_values = [max_k]

        for k in k_values:
            top_k_idx = top_target_neurons_idx[:k]
            train_features_k = train_latents[:, top_k_idx]
            test_features_k = test_latents[:, top_k_idx]

            # Get neuron names for reference
            neuron_names = [
                self.sae.neuron_to_tag.get(idx, f"n{idx}") for idx in top_k_idx
            ]

            print(f"\n  Using top-{k} target-discriminative neurons:")
            print(f"    Neurons: {neuron_names[:5]}{'...' if k > 5 else ''}")

            # ============================================
            # Method 1: k-NN Classifier (k=3)
            # Fit on TRAIN, predict on TEST
            # ============================================

            from sklearn.neighbors import KNeighborsClassifier

            # Use k=3 nearest neighbors (or less if not enough samples)
            n_neighbors = min(3, len(train_features_k))
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean")
            knn.fit(train_features_k, train_targets)
            predictions_knn = knn.predict(test_features_k)

            # ============================================
            # Method 2: Logistic Regression (Linear Probe)
            # Fit on TRAIN, predict on TEST
            # ============================================

            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features_k)
            test_features_scaled = scaler.transform(test_features_k)  # Use same scaler

            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(train_features_scaled, train_targets)
            predictions_lr = lr.predict(test_features_scaled)

            # ============================================
            # Compute per-subgroup accuracy on TEST
            # ============================================

            results[f"k={k}"] = {
                "neurons": neuron_names,
                "neuron_indices": [int(idx) for idx in top_k_idx],
            }

            for method_name, predictions in [
                ("knn_k3", predictions_knn),
                ("logistic_regression", predictions_lr),
            ]:
                correct = predictions == test_targets
                overall_acc = correct.mean()

                print(f"\n    {method_name}:")
                print(f"      Overall accuracy: {overall_acc:.4f}")

                method_results = {
                    "overall_accuracy": float(overall_acc),
                    "subgroup_accuracy": {},
                    "per_target_accuracy": {},
                    "per_bias_accuracy": {},
                }

                # Per-target accuracy
                for t in unique_targets:
                    mask = test_targets == t
                    acc = correct[mask].mean() if mask.sum() > 0 else 0
                    method_results["per_target_accuracy"][f"t={t}"] = {
                        "accuracy": float(acc),
                        "count": int(mask.sum()),
                    }

                # Per-bias accuracy (for each bias attribute)
                for b_name in bias_names:
                    method_results["per_bias_accuracy"][b_name] = {}
                    for b_val in sorted(set(test_bias_arrays[b_name])):
                        mask = test_bias_arrays[b_name] == b_val
                        acc = correct[mask].mean() if mask.sum() > 0 else 0
                        method_results["per_bias_accuracy"][b_name][
                            f"{b_name}={b_val}"
                        ] = {"accuracy": float(acc), "count": int(mask.sum())}

                # Per-subgroup accuracy
                subgroup_accs = []
                print(f"      Per-subgroup accuracy:")
                for sg in unique_subgroups:
                    mask = np.array([sk == sg for sk in test_subgroup_keys])
                    if mask.sum() > 0:
                        acc = correct[mask].mean()
                        subgroup_accs.append(acc)
                        method_results["subgroup_accuracy"][str(sg)] = {
                            "accuracy": float(acc),
                            "count": int(mask.sum()),
                        }
                        sg_label = ", ".join([s.split("=")[1] for s in sg])
                        print(f"        {sg_label}: {acc:.4f} (n={mask.sum()})")

                # Worst-group and gap
                if subgroup_accs:
                    method_results["worst_group_accuracy"] = float(min(subgroup_accs))
                    method_results["best_group_accuracy"] = float(max(subgroup_accs))
                    method_results["accuracy_gap"] = float(
                        max(subgroup_accs) - min(subgroup_accs)
                    )

                    print(f"      Worst-group: {min(subgroup_accs):.4f}")
                    print(f"      Best-group:  {max(subgroup_accs):.4f}")
                    print(
                        f"      Gap:         {max(subgroup_accs) - min(subgroup_accs):.4f}"
                    )

                results[f"k={k}"][method_name] = method_results

        # ============================================
        # Compare: relevant-only vs irrelevant-only vs all neurons
        # Fit on TRAIN, evaluate on TEST
        # ============================================

        print(f"\n  {'='*50}")
        print("  Comparison: Relevant vs Irrelevant Neurons")
        print(f"  (Fit on TRAIN, Evaluate on TEST)")
        print(f"  {'='*50}")

        # Get indices for relevant and irrelevant neurons
        relevant_neuron_indices = [
            self.tag_to_neuron[t] for t in self.relevant_tags if t in self.tag_to_neuron
        ]
        irrelevant_neuron_indices = [
            self.tag_to_neuron[t]
            for t in self.irrelevant_tags
            if t in self.tag_to_neuron
        ]

        for category, indices in [
            ("relevant_only", relevant_neuron_indices),
            ("irrelevant_only", irrelevant_neuron_indices),
            ("all_neurons", list(range(train_latents.shape[1]))),
        ]:
            if len(indices) == 0:
                continue

            train_features_cat = train_latents[:, indices]
            test_features_cat = test_latents[:, indices]

            # Fit scaler and classifier on TRAIN
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features_cat)
            test_features_scaled = scaler.transform(test_features_cat)

            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(train_features_scaled, train_targets)

            # Predict on TEST
            predictions = lr.predict(test_features_scaled)

            correct = predictions == test_targets
            overall_acc = correct.mean()

            # Per-subgroup on TEST
            subgroup_accs = []
            for sg in unique_subgroups:
                mask = np.array([sk == sg for sk in test_subgroup_keys])
                if mask.sum() > 0:
                    subgroup_accs.append(correct[mask].mean())

            worst_group = min(subgroup_accs) if subgroup_accs else 0
            gap = max(subgroup_accs) - min(subgroup_accs) if subgroup_accs else 0

            print(f"\n  {category} ({len(indices)} neurons):")
            print(
                f"    Overall: {overall_acc:.4f}, Worst-group: {worst_group:.4f}, Gap: {gap:.4f}"
            )

            results[category] = {
                "num_neurons": len(indices),
                "overall_accuracy": float(overall_acc),
                "worst_group_accuracy": float(worst_group),
                "accuracy_gap": float(gap),
            }

        # ============================================
        # Visualization: Accuracy vs K
        # ============================================

        self._plot_accuracy_vs_k(results, k_values, unique_subgroups, viz_dir)

        return results

    def _plot_accuracy_vs_k(
        self, results: dict, k_values: list, subgroups: list, viz_dir: str
    ):
        """Plot accuracy vs number of top-k neurons."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Overall and worst-group accuracy vs K
        ax = axes[0]

        overall_accs_nc = [
            results[f"k={k}"]["nearest_centroid"]["overall_accuracy"] for k in k_values
        ]
        worst_accs_nc = [
            results[f"k={k}"]["nearest_centroid"]["worst_group_accuracy"]
            for k in k_values
        ]
        overall_accs_lr = [
            results[f"k={k}"]["logistic_regression"]["overall_accuracy"]
            for k in k_values
        ]
        worst_accs_lr = [
            results[f"k={k}"]["logistic_regression"]["worst_group_accuracy"]
            for k in k_values
        ]

        ax.plot(k_values, overall_accs_nc, "o-", label="NC Overall", color="blue")
        ax.plot(
            k_values,
            worst_accs_nc,
            "s--",
            label="NC Worst-group",
            color="blue",
            alpha=0.6,
        )
        ax.plot(k_values, overall_accs_lr, "o-", label="LR Overall", color="green")
        ax.plot(
            k_values,
            worst_accs_lr,
            "s--",
            label="LR Worst-group",
            color="green",
            alpha=0.6,
        )

        ax.set_xlabel("Number of Top-K Neurons")
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Accuracy vs K")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: Per-subgroup accuracy for best K
        ax = axes[1]
        best_k = k_values[-1]  # Use largest K

        subgroup_labels = [", ".join([s.split("=")[1] for s in sg]) for sg in subgroups]

        lr_accs = [
            results[f"k={best_k}"]["logistic_regression"]["subgroup_accuracy"]
            .get(str(sg), {})
            .get("accuracy", 0)
            for sg in subgroups
        ]

        x = np.arange(len(subgroups))
        colors = ["#4CAF50" if "t=1" in str(sg) else "#2196F3" for sg in subgroups]

        ax.bar(x, lr_accs, color=colors, alpha=0.7)
        ax.axhline(
            y=min(lr_accs),
            color="red",
            linestyle="--",
            label=f"Worst: {min(lr_accs):.3f}",
        )
        ax.axhline(
            y=np.mean(lr_accs),
            color="gray",
            linestyle=":",
            label=f"Mean: {np.mean(lr_accs):.3f}",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(subgroup_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Per-Subgroup Accuracy (Top-{best_k} Neurons, LR)")
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(
            os.path.join(viz_dir, "classification_accuracy.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()

    def train(self):
        """Main training pipeline."""
        print(f"\n{'='*60}")
        print("Tag-Only SAE Training Pipeline")
        print(f"{'='*60}")

        # Step 1: Extract features
        print("\nStep 1: Extracting features...")
        features, targets, indices, biases = self._extract_features(
            self.dataloaders["train"], desc="Extracting train features"
        )
        print(f"  Features shape: {features.shape}")

        # Step 2: Train SAE
        print("\nStep 2: Training Tag-Only SAE...")
        self._train_tag_sae(features, indices)

        # Step 3: Visualize (train set)
        print("\nStep 3: Visualizing neurons (train set)...")
        self._visualize_neurons(features, indices)

        # Step 4: Evaluate on test set
        print("\nStep 4: Evaluating on test set...")
        if "test" in self.dataloaders:
            test_features, test_targets, test_indices, test_biases = (
                self._extract_features(
                    self.dataloaders["test"], desc="Extracting test features"
                )
            )
            self._evaluate_debiasing(test_features, test_targets, test_biases)

            # Step 5: Visualize (test set)
            print("\nStep 5: Visualizing neurons (test set)...")
            self._visualize_neurons_test(test_features, test_indices)

            # Step 6: Subgroup activation analysis
            print("\nStep 6: Analyzing subgroup activations...")
            if test_biases and any(len(v) > 0 for v in test_biases.values()):
                self._analyze_subgroup_activations(
                    train_features=features,
                    train_targets=targets,
                    train_biases=biases,
                    test_features=test_features,
                    test_targets=test_targets,
                    test_biases=test_biases,
                )
            else:
                print("  No bias attributes available, skipping subgroup analysis.")

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Output: {self.log_path}")

    def eval(self):
        """Evaluation mode."""
        # Load SAE
        sae_path = os.path.join(self.log_path, "tag_only_sae", "tag_only_sae.pt")
        config_path = os.path.join(self.log_path, "tag_only_sae", "config.json")

        if not os.path.exists(sae_path):
            # Try from config
            sae_path = self.tag_cfg.get("SAE_CHECKPOINT_PATH", "")
            config_path = os.path.join(os.path.dirname(sae_path), "config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.sae = TagOnlySAE(
            input_dim=config["input_dim"],
            num_tags=config["num_tags"],
            tag_to_neuron=config["tag_to_neuron"],
        ).to(self.device)
        self.sae.load_state_dict(torch.load(sae_path, map_location=self.device))

        self.irrelevant_tags = config["irrelevant_tags"]
        self.relevant_tags = config["relevant_tags"]
        self.all_tags = config["all_tags"]

        # Evaluate
        test_features, test_targets, test_indices, test_biases = self._extract_features(
            self.dataloaders["test"], desc="Extracting test features"
        )
        self._evaluate_debiasing(test_features, test_targets, test_biases)


# Config defaults to add to cfg.py:
"""
CFG.MITIGATOR.TAG_ONLY_SAE = CN()

# Tags CSV
CFG.MITIGATOR.TAG_ONLY_SAE.TAGS_CSV_PATH = "train_tags.csv"
CFG.MITIGATOR.TAG_ONLY_SAE.ALL_TAGS_COLUMN = "tags"
CFG.MITIGATOR.TAG_ONLY_SAE.IRRELEVANT_TAGS_COLUMN = "irrelevant_tags"
CFG.MITIGATOR.TAG_ONLY_SAE.TAG_SEPARATOR = " | "
CFG.MITIGATOR.TAG_ONLY_SAE.MIN_TAG_FREQUENCY = 10

# Checkpoint
CFG.MITIGATOR.TAG_ONLY_SAE.SAE_CHECKPOINT_PATH = ""
CFG.MITIGATOR.TAG_ONLY_SAE.PRECOMPUTED_FEATURES_PATH = ""

# Training
CFG.MITIGATOR.TAG_ONLY_SAE.STEPS = 20000
CFG.MITIGATOR.TAG_ONLY_SAE.BATCH_SIZE = 256
CFG.MITIGATOR.TAG_ONLY_SAE.LR = 1e-3

# Loss weights
CFG.MITIGATOR.TAG_ONLY_SAE.LAMBDA_RECONSTRUCTION = 1.0
CFG.MITIGATOR.TAG_ONLY_SAE.LAMBDA_SPARSITY = 1e-3
CFG.MITIGATOR.TAG_ONLY_SAE.LAMBDA_TAG = 1.0

# Tag supervision
CFG.MITIGATOR.TAG_ONLY_SAE.TAG_LOSS_TYPE = "bce"
CFG.MITIGATOR.TAG_ONLY_SAE.POSITIVE_WEIGHT = 1.0
CFG.MITIGATOR.TAG_ONLY_SAE.NEGATIVE_WEIGHT = 0.5
CFG.MITIGATOR.TAG_ONLY_SAE.USE_NEGATIVE_SUPERVISION = True
CFG.MITIGATOR.TAG_ONLY_SAE.MARGIN = 0.5
CFG.MITIGATOR.TAG_ONLY_SAE.TARGET_ACTIVATION = 1.0
"""
