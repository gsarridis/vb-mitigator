#!/usr/bin/env python
"""
Standalone SAE Training Tool for VB-Mitigator

This tool can be used to:
1. Extract features from a pretrained model
2. Train a Sparse Autoencoder on the features
3. Analyze monosemantic neurons
4. Generate visualizations

Usage:
    python tools/train_sae.py --cfg configs/utkface/sae/sae.yaml \
                              --checkpoint outputs/utkface_baselines/erm/best.pth

    # Or extract features first, then train SAE separately:
    python tools/train_sae.py --cfg configs/utkface/sae/sae.yaml \
                              --extract-only \
                              --checkpoint outputs/utkface_baselines/erm/best.pth

    python tools/train_sae.py --cfg configs/utkface/sae/sae.yaml \
                              --features-path outputs/utkface_sae/features/train_features.pt \
                              --train-sae-only
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.cfg import CFG as cfg
from my_datasets.builder import get_dataset
from models.builder import get_model
from tools.utils import load_checkpoint, seed_everything


def get_args_parser():
    parser = argparse.ArgumentParser("SAE Training Tool", add_help=True)

    # Config and checkpoints
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="Path to pretrained model checkpoint"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Mode selection
    parser.add_argument(
        "--extract-only", action="store_true", help="Only extract features"
    )
    parser.add_argument(
        "--train-sae-only", action="store_true", help="Only train SAE from features"
    )
    parser.add_argument(
        "--analyze-only", action="store_true", help="Only analyze existing SAE"
    )

    # Feature extraction
    parser.add_argument(
        "--features-path", type=str, default="", help="Path to pre-extracted features"
    )

    # SAE parameters (override config)
    parser.add_argument(
        "--sae-type", type=str, default=None, choices=["standard", "topk", "jumprelu"]
    )
    parser.add_argument("--expansion-factor", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--l1-penalty", type=float, default=None)
    parser.add_argument("--k", type=int, default=None, help="K for TopK SAE")

    # Output
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser


def load_config(args):
    """Load and merge configuration."""
    cfg.merge_from_file(args.cfg)

    # Override with command line arguments
    if args.checkpoint:
        cfg.MITIGATOR.SAE.CHECKPOINT_PATH = args.checkpoint
    if args.sae_type:
        cfg.MITIGATOR.SAE.TYPE = args.sae_type
    if args.expansion_factor:
        cfg.MITIGATOR.SAE.EXPANSION_FACTOR = args.expansion_factor
    if args.steps:
        cfg.MITIGATOR.SAE.STEPS = args.steps
    if args.l1_penalty:
        cfg.MITIGATOR.SAE.L1_PENALTY = args.l1_penalty
    if args.k:
        cfg.MITIGATOR.SAE.K = args.k

    cfg.EXPERIMENT.SEED = args.seed
    cfg.EXPERIMENT.GPU = args.device

    cfg.freeze()
    return cfg


def extract_features(model, dataloader, device, desc="Extracting"):
    """Extract features from model."""
    all_features = []
    all_targets = []
    all_indices = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            inputs = batch["inputs"].to(device)
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                _, features = outputs
            else:
                features = outputs

            all_features.append(features.cpu())
            all_targets.append(batch["targets"])

            if "index" in batch:
                all_indices.extend(batch["index"].tolist())

    features = torch.cat(all_features, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return features, targets, all_indices


def train_sae(features, cfg, device, save_dir):
    """Train Sparse Autoencoder."""
    from dictionary_learning import AutoEncoder

    feature_dim = features.shape[1]
    dict_size = feature_dim * cfg.MITIGATOR.SAE.EXPANSION_FACTOR

    print(f"Feature dim: {feature_dim}, Dict size: {dict_size}")

    # Initialize SAE
    sae = AutoEncoder(feature_dim, dict_size).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.MITIGATOR.SAE.LR)

    # Training
    steps = cfg.MITIGATOR.SAE.STEPS
    batch_size = cfg.MITIGATOR.SAE.BATCH_SIZE
    l1_penalty = cfg.MITIGATOR.SAE.L1_PENALTY

    step = 0
    pbar = tqdm(total=steps, desc="Training SAE")

    while step < steps:
        indices = torch.randperm(len(features))

        for i in range(0, len(features), batch_size):
            if step >= steps:
                break

            batch_idx = indices[i : i + batch_size]
            batch = features[batch_idx].to(device)

            # Forward
            recon, latents = sae(batch, output_features=True)

            # Loss
            mse = ((batch - recon) ** 2).mean()
            l1 = latents.abs().mean()
            loss = mse + l1_penalty * l1

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder
            with torch.no_grad():
                norms = sae.decoder.weight.norm(dim=0, keepdim=True)
                sae.decoder.weight.div_(norms.clamp(min=1e-8))

            step += 1
            pbar.update(1)

            if step % 500 == 0:
                l0 = (latents > 0).float().sum(dim=1).mean().item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "L0": f"{l0:.1f}"})

    pbar.close()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    torch.save(sae.state_dict(), os.path.join(save_dir, "ae.pt"))

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(
            {
                "feature_dim": feature_dim,
                "dict_size": dict_size,
                "steps": steps,
                "l1_penalty": l1_penalty,
            },
            f,
            indent=2,
        )

    return sae


def analyze_sae(sae, features, targets, indices, cfg, device, save_dir):
    """Analyze SAE and generate visualizations."""
    from dictionary_learning import AutoEncoder

    sae.eval()

    # Encode all features
    with torch.no_grad():
        latents = []
        batch_size = 256
        for i in range(0, len(features), batch_size):
            batch = features[i : i + batch_size].to(device)
            lat = sae.encode(batch)
            latents.append(lat.cpu())
        latents = torch.cat(latents, dim=0)

    dict_size = latents.shape[1]
    k = cfg.MITIGATOR.SAE.TOP_K_IMAGES

    # Find alive neurons and top-k images
    results = {
        "alive_neurons": [],
        "top_k_per_latent": {},
    }

    for i in tqdm(range(dict_size), desc="Analyzing neurons"):
        acts = latents[:, i]
        if acts.max() > 1e-6:
            results["alive_neurons"].append(i)
            top_vals, top_idx = acts.topk(min(k, len(acts)))
            results["top_k_per_latent"][i] = {
                "indices": [indices[j] for j in top_idx.tolist()],
                "activations": top_vals.tolist(),
                "targets": targets[top_idx].tolist(),
            }

    print(f"Alive neurons: {len(results['alive_neurons'])}/{dict_size}")

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "analysis.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    args = get_args_parser().parse_args()
    cfg = load_config(args)

    seed_everything(cfg.EXPERIMENT.SEED)
    device = torch.device(cfg.EXPERIMENT.GPU if torch.cuda.is_available() else "cpu")

    # Setup output directory
    output_dir = args.output_dir or os.path.join(
        cfg.LOG.PREFIX, cfg.EXPERIMENT.PROJECT, cfg.EXPERIMENT.NAME
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load features or extract them
    if args.features_path and os.path.exists(args.features_path):
        print(f"Loading features from {args.features_path}")
        data = torch.load(args.features_path)
        features = data["features"]
        targets = data["targets"]
        indices = data.get("indices", list(range(len(features))))
    else:
        # Load model and dataset
        print("Setting up model and dataset...")
        dataset = get_dataset(cfg)

        model = get_model(cfg.MODEL.TYPE, dataset["num_class"], cfg.MODEL.PRETRAINED)

        if cfg.MITIGATOR.SAE.CHECKPOINT_PATH:
            ckpt = load_checkpoint(cfg.MITIGATOR.SAE.CHECKPOINT_PATH)
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            print(f"Loaded checkpoint: {cfg.MITIGATOR.SAE.CHECKPOINT_PATH}")

        model.to(device)

        # Extract features
        features, targets, indices = extract_features(
            model, dataset["dataloaders"]["train"], device
        )
        print(f"Extracted features: {features.shape}")

        # Save features
        features_dir = os.path.join(output_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        torch.save(
            {
                "features": features,
                "targets": targets,
                "indices": indices,
            },
            os.path.join(features_dir, "train_features.pt"),
        )

        if args.extract_only:
            print("Feature extraction complete.")
            return

    # Train SAE
    if not args.analyze_only:
        sae_dir = os.path.join(output_dir, "sae_checkpoints")
        sae = train_sae(features, cfg, device, sae_dir)
    else:
        from dictionary_learning import AutoEncoder

        sae_path = os.path.join(output_dir, "sae_checkpoints", "ae.pt")
        with open(os.path.join(output_dir, "sae_checkpoints", "config.json")) as f:
            sae_cfg = json.load(f)
        sae = AutoEncoder(sae_cfg["feature_dim"], sae_cfg["dict_size"])
        sae.load_state_dict(torch.load(sae_path))
        sae.to(device)

    if args.train_sae_only:
        print("SAE training complete.")
        return

    # Analyze
    analysis_dir = os.path.join(output_dir, "analysis")
    results = analyze_sae(sae, features, targets, indices, cfg, device, analysis_dir)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
