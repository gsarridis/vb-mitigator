#!/usr/bin/env python3
"""
Run UrbanCars zero-shot classification analysis across ALL pretrained
OpenCLIP models. For each model:
  1. Load model, encode texts + images, compute per-subgroup accuracy
  2. Save per-model plots
  3. Delete model from GPU/CPU and clear cache
  4. Append results to a master table

Outputs:
  - results/<model_name>/  — per-model plots
  - results/master_results.csv — one row per model with all metrics
  - results/master_results.json — same data in JSON
  - results/summary_plots/  — cross-model comparison plots

Usage:
    python run_all_models.py
    python run_all_models.py --text1 "an urban car" --text2 "a country car"
    python run_all_models.py --filter "ViT-B"        # only models matching this
    python run_all_models.py --exclude "RN"           # skip ResNet models
    python run_all_models.py --resume                 # skip already completed models

Requirements:
    pip install torch open-clip-torch Pillow matplotlib numpy pandas tqdm
"""

import argparse
import gc
import glob
import json
import os
import re
import shutil
import sys
import time
import traceback
from itertools import product as iterproduct

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip


# ── Defaults ─────────────────────────────────────────────────────
DEFAULT_DATA_ROOT = "../../data/urbancars/bg-0.5_co_occur_obj-0.5/test"
DEFAULT_TEXT1 = "a photograph of a compact, sports, sedan car"
DEFAULT_TEXT2 = "a photograph of a truck, jeep, pickup car"
DEFAULT_BATCH_SIZE = 16
DEFAULT_OUTPUT_DIR = "./results"

ATTRIBUTES = ["urban", "country"]


# "a photograph of a compact car"
# "a photograph of a sports car"
# "a photograph of a sedan car"

# "a photograph of a truck car"
# "a photograph of a jeep car"
# "a photograph of a pickup car"


# "texts1": [
#             "a photo of a person with blonde hair",
#             "a photo of a person with light blonde hair",
#             "a photo of a person with golden hair",
#             "a photo of a person with platinum blonde hair",
#         ],
#         "texts2": [
#             "a photo of a person with dark hair",
#             "a photo of a person with black hair",
#             "a photo of a person with brown hair",
#             "a photo of a brunette person",
#             "a photo of a person with red hair",
#             "a photo of a person with grey hair",
#             "a photo of a bald person",
#             "a photo of a person with auburn hair",
#         ],

# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════


def get_subgroup_dirs(data_root: str) -> dict[str, str]:
    """Enumerate all 8 subgroup directories."""
    subgroups = {}
    for obj, bg, co in iterproduct(ATTRIBUTES, repeat=3):
        name = f"obj-{obj}_bg-{bg}_co_occur_obj-{co}"
        path = os.path.join(data_root, name)
        if os.path.isdir(path):
            subgroups[name] = path
    return subgroups


def parse_subgroup_label(name: str) -> str:
    m = re.match(r"obj-(\w+)_bg-(\w+)_co_occur_obj-(\w+)", name)
    if m:
        return f"obj={m.group(1)}, bg={m.group(2)}, co={m.group(3)}"
    return name


def get_ground_truth_label(name: str) -> int:
    """0 if obj=urban (text1), 1 if obj=country (text2)."""
    m = re.match(r"obj-(\w+)_bg-(\w+)_co_occur_obj-(\w+)", name)
    if m:
        return 0 if m.group(1) == "urban" else 1
    return -1


def zero_shot_classify(sim_text1: np.ndarray, sim_text2: np.ndarray) -> dict:
    """Softmax over two text similarities → predictions + probabilities."""
    logits = np.stack([sim_text1, sim_text2], axis=-1)
    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    predictions = np.argmax(probs, axis=-1)
    return {
        "predictions": predictions,
        "probs_text1": probs[:, 0],
        "probs_text2": probs[:, 1],
    }


def load_and_encode_images(
    model, preprocess, image_paths: list[str], device: str, batch_size: int
) -> torch.Tensor:
    """Load images in batches, encode, return normalized features."""
    all_features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            images.append(preprocess(img))
        batch = torch.stack(images).to(device)
        with torch.no_grad(), torch.amp.autocast(device):
            features = model.encode_image(batch)
        all_features.append(features.float().cpu())
        del batch, features
    return F.normalize(torch.cat(all_features, dim=0), dim=-1)


def encode_texts(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    """Encode texts, return normalized features on CPU."""
    tokens = tokenizer(texts).to(device)
    with torch.no_grad(), torch.amp.autocast(device):
        features = model.encode_text(tokens)
    return F.normalize(features.float().cpu(), dim=-1)


def cleanup_model(model, device: str) -> None:
    """Aggressively free model memory."""
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def delete_cached_weights(model_name: str, pretrained_tag: str) -> None:
    """
    Delete downloaded model weights from all known cache directories:
      - ~/.cache/huggingface/hub/
      - ~/.cache/clip/
      - ~/.cache/open_clip/
    """
    home = os.path.expanduser("~")
    deleted = []

    # ── Hugging Face hub cache ───────────────────────────────────
    hf_cache = os.path.join(home, ".cache", "huggingface", "hub")
    if os.path.isdir(hf_cache):
        # HF stores models in dirs like models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K
        for entry in os.listdir(hf_cache):
            entry_path = os.path.join(hf_cache, entry)
            if not os.path.isdir(entry_path):
                continue
            entry_lower = entry.lower()
            # Match by model name fragments
            model_parts = model_name.lower().replace("-", "").replace("_", "")
            tag_parts = pretrained_tag.lower().replace("-", "").replace("_", "")
            entry_clean = entry_lower.replace("-", "").replace("_", "")
            if model_parts in entry_clean and (
                tag_parts in entry_clean or "clip" in entry_lower
            ):
                try:
                    shutil.rmtree(entry_path)
                    deleted.append(entry_path)
                except Exception as e:
                    print(f"    [warn] Could not delete {entry_path}: {e}")

    # ── OpenCLIP / CLIP cache ────────────────────────────────────
    for cache_name in [".cache/clip", ".cache/open_clip"]:
        cache_dir = os.path.join(home, cache_name)
        if os.path.isdir(cache_dir):
            for entry in os.listdir(cache_dir):
                entry_path = os.path.join(cache_dir, entry)
                if os.path.isfile(entry_path) and entry.endswith(
                    (".pt", ".pth", ".bin", ".safetensors")
                ):
                    try:
                        os.remove(entry_path)
                        deleted.append(entry_path)
                    except Exception as e:
                        print(f"    [warn] Could not delete {entry_path}: {e}")

    # ── Torch hub cache ──────────────────────────────────────────
    torch_cache = os.path.join(home, ".cache", "torch", "hub", "checkpoints")
    if os.path.isdir(torch_cache):
        for entry in os.listdir(torch_cache):
            entry_path = os.path.join(torch_cache, entry)
            entry_lower = entry.lower()
            model_parts = model_name.lower().replace("-", "_")
            if model_parts in entry_lower or pretrained_tag.lower() in entry_lower:
                try:
                    os.remove(entry_path)
                    deleted.append(entry_path)
                except Exception as e:
                    print(f"    [warn] Could not delete {entry_path}: {e}")

    if deleted:
        total_freed = 0
        for p in deleted:
            # Size already gone, just report path
            print(f"    [deleted] {p}")
        print(f"    [cleanup] Removed {len(deleted)} cached file(s)/dir(s) from disk.")
    else:
        print(f"    [cleanup] No cached weight files found to delete.")


def parse_model_info(model_name: str, pretrained_tag: str) -> dict:
    """Extract architecture details from the model/pretrained names."""
    info = {
        "model_name": model_name,
        "pretrained_tag": pretrained_tag,
        "model_id": f"{model_name}_{pretrained_tag}",
    }

    # Architecture family
    if model_name.startswith("RN"):
        info["arch_family"] = "ResNet"
        m = re.match(r"RN(\d+)(x\d+)?", model_name)
        if m:
            info["resnet_depth"] = int(m.group(1))
            info["resnet_multiplier"] = m.group(2) or "x1"
        info["patch_size"] = None
    elif "ViT" in model_name or "vit" in model_name.lower():
        info["arch_family"] = "ViT"
        # Extract size letter (B, L, H, g, G, e, E)
        size_m = re.search(r"ViT-([a-zA-Z]+)-(\d+)", model_name)
        if size_m:
            info["vit_size"] = size_m.group(1)
            info["patch_size"] = int(size_m.group(2))
        else:
            info["vit_size"] = None
            info["patch_size"] = None
    elif "coca" in model_name.lower():
        info["arch_family"] = "CoCa"
        size_m = re.search(r"ViT-([a-zA-Z]+)-(\d+)", model_name)
        if size_m:
            info["vit_size"] = size_m.group(1)
            info["patch_size"] = int(size_m.group(2))
    elif "convnext" in model_name.lower():
        info["arch_family"] = "ConvNeXt"
        info["patch_size"] = None
    elif "EVA" in model_name:
        info["arch_family"] = "EVA"
        size_m = re.search(r"-(\d+)", model_name)
        if size_m:
            info["patch_size"] = int(size_m.group(1))
    elif "MobileCLIP" in model_name:
        info["arch_family"] = "MobileCLIP"
        info["patch_size"] = None
    elif "SigLIP" in model_name or "siglip" in model_name.lower():
        info["arch_family"] = "SigLIP"
        info["patch_size"] = None
    else:
        info["arch_family"] = "Other"
        info["patch_size"] = None

    # Image size from model name (e.g., ViT-B-16-plus-240, or -384)
    img_m = re.search(r"(\d{3,4})$", model_name.replace("-", ""))
    if img_m and int(img_m.group(1)) >= 200:
        info["image_size"] = int(img_m.group(1))
    else:
        info["image_size"] = None  # will be filled from model config

    # Training dataset from pretrained tag
    tag = pretrained_tag.lower()
    if "openai" in tag:
        info["training_data"] = "OpenAI WIT (400M)"
    elif "laion2b" in tag:
        info["training_data"] = "LAION-2B"
    elif "laion400m" in tag:
        info["training_data"] = "LAION-400M"
    elif "datacomp" in tag:
        info["training_data"] = "DataComp"
    elif "metaclip" in tag:
        info["training_data"] = "MetaCLIP"
    elif "dfn" in tag:
        info["training_data"] = "DFN"
    elif "commonpool" in tag:
        info["training_data"] = "CommonPool"
    elif "yfcc" in tag:
        info["training_data"] = "YFCC-15M"
    elif "cc12m" in tag:
        info["training_data"] = "CC-12M"
    elif "webli" in tag:
        info["training_data"] = "WebLI"
    elif "merged" in tag:
        info["training_data"] = "Merged"
    elif "mscoco" in tag:
        info["training_data"] = "MS-COCO (finetuned)"
    else:
        info["training_data"] = pretrained_tag

    # quickgelu
    info["quickgelu"] = "quickgelu" in model_name.lower()

    return info


# ═══════════════════════════════════════════════════════════════════
# Per-model plotting (lightweight version — just combined + accuracy)
# ═══════════════════════════════════════════════════════════════════


def plot_model_results(
    subgroup_results: dict[str, dict],
    text1: str,
    text2: str,
    model_id: str,
    output_dir: str,
) -> None:
    """Save combined histogram + accuracy bar chart for one model."""
    os.makedirs(output_dir, exist_ok=True)
    n_sub = len(subgroup_results)
    labels_short = [parse_subgroup_label(n) for n in subgroup_results]

    # ── Combined histograms ──────────────────────────────────────
    cols = 4
    rows = (n_sub + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for idx, (name, data) in enumerate(subgroup_results.items()):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        label = parse_subgroup_label(name)
        ax.hist(
            data["sim_text1"],
            bins=30,
            alpha=0.6,
            label=f'"{text1}"',
            color="steelblue",
            density=True,
        )
        ax.hist(
            data["sim_text2"],
            bins=30,
            alpha=0.6,
            label=f'"{text2}"',
            color="coral",
            density=True,
        )
        acc = data.get("accuracy", 0) * 100
        ax.set_title(f"{label}\nAcc={acc:.1f}%", fontsize=9)
        ax.set_xlabel("Cosine Sim", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    for idx in range(n_sub, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)
    fig.suptitle(f"{model_id} — Similarity Distributions", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, "combined_histograms.png"), dpi=120)
    plt.close(fig)

    # ── Accuracy bar chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    names_sorted = list(subgroup_results.keys())
    accs = [subgroup_results[n].get("accuracy", 0.0) * 100 for n in names_sorted]
    x_pos = np.arange(len(names_sorted))
    colors = [
        "steelblue" if subgroup_results[n].get("gt_label", 0) == 0 else "coral"
        for n in names_sorted
    ]
    bars = ax.bar(
        x_pos, accs, color=colors, alpha=0.75, edgecolor="black", linewidth=0.5
    )
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="Chance")
    ax.set_title(f"{model_id} — Zero-Shot Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "zeroshot_accuracy.png"), dpi=120)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Cross-model summary plots
# ═══════════════════════════════════════════════════════════════════


def plot_summary(df: pd.DataFrame, output_dir: str) -> None:
    """Generate cross-model comparison plots from the master DataFrame."""
    os.makedirs(output_dir, exist_ok=True)

    if df.empty:
        return

    df_sorted = df.sort_values("avg_accuracy", ascending=True).copy()

    # ── 1. Average accuracy bar chart ────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted) * 0.35)))
    y_pos = np.arange(len(df_sorted))
    colors = plt.cm.RdYlGn(df_sorted["avg_accuracy"].values)
    ax.barh(
        y_pos,
        df_sorted["avg_accuracy"] * 100,
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["model_id"], fontsize=6)
    ax.set_xlabel("Average Accuracy (%)")
    ax.set_title("Zero-Shot Average Accuracy — All OpenCLIP Models")
    ax.axvline(50, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "all_models_avg_accuracy.png"), dpi=150)
    plt.close(fig)

    # ── 2. Worst-group accuracy bar chart ────────────────────────
    df_sorted_wg = df.sort_values("worst_group_accuracy", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted_wg) * 0.35)))
    y_pos = np.arange(len(df_sorted_wg))
    colors = plt.cm.RdYlGn(df_sorted_wg["worst_group_accuracy"].values)
    ax.barh(
        y_pos,
        df_sorted_wg["worst_group_accuracy"] * 100,
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted_wg["model_id"], fontsize=6)
    ax.set_xlabel("Worst-Group Accuracy (%)")
    ax.set_title("Zero-Shot Worst-Group Accuracy — All OpenCLIP Models")
    ax.axvline(50, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "all_models_worst_group_accuracy.png"), dpi=150
    )
    plt.close(fig)

    # ── 3. Avg vs Worst-group scatter ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        df["avg_accuracy"] * 100,
        df["worst_group_accuracy"] * 100,
        c="steelblue",
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        s=40,
    )
    # Label a subset (top/bottom)
    for _, row in df.iterrows():
        ax.annotate(
            row["model_id"],
            (row["avg_accuracy"] * 100, row["worst_group_accuracy"] * 100),
            fontsize=4,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="avg = worst")
    ax.set_xlabel("Average Accuracy (%)")
    ax.set_ylabel("Worst-Group Accuracy (%)")
    ax.set_title("Average vs Worst-Group Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "avg_vs_worst_group.png"), dpi=150)
    plt.close(fig)

    # ── 4. Accuracy gap (avg - worst) ────────────────────────────
    df_sorted_gap = df.copy()
    df_sorted_gap["gap"] = (
        df_sorted_gap["avg_accuracy"] - df_sorted_gap["worst_group_accuracy"]
    )
    df_sorted_gap = df_sorted_gap.sort_values("gap", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted_gap) * 0.35)))
    y_pos = np.arange(len(df_sorted_gap))
    ax.barh(
        y_pos,
        df_sorted_gap["gap"] * 100,
        color="mediumpurple",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted_gap["model_id"], fontsize=6)
    ax.set_xlabel("Accuracy Gap: Avg − Worst Group (%)")
    ax.set_title("Fairness Gap — All OpenCLIP Models")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "all_models_fairness_gap.png"), dpi=150)
    plt.close(fig)

    # ── 5. Grouped by architecture family ────────────────────────
    if "arch_family" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, col, title in zip(
            axes,
            ["avg_accuracy", "worst_group_accuracy"],
            ["Average Accuracy", "Worst-Group Accuracy"],
        ):
            grouped = df.groupby("arch_family")[col].agg(["mean", "std", "count"])
            grouped = grouped.sort_values("mean", ascending=True)
            ax.barh(
                grouped.index,
                grouped["mean"] * 100,
                xerr=grouped["std"] * 100,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_xlabel(f"{title} (%)")
            ax.set_title(f"{title} by Architecture")
            ax.grid(True, alpha=0.3, axis="x")
            # Annotate count
            for i, (idx, row) in enumerate(grouped.iterrows()):
                ax.text(
                    row["mean"] * 100 + 1,
                    i,
                    f"n={int(row['count'])}",
                    va="center",
                    fontsize=8,
                )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "accuracy_by_architecture.png"), dpi=150)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Single-model analysis
# ═══════════════════════════════════════════════════════════════════


def analyze_single_model(
    model_name: str,
    pretrained_tag: str,
    subgroups: dict[str, str],
    text1: str,
    text2: str,
    device: str,
    batch_size: int,
    output_dir: str,
) -> dict | None:
    """
    Run the full analysis for one model. Returns a results dict or None on failure.
    The model is deleted and GPU memory freed before returning.
    """
    model_id = f"{model_name}_{pretrained_tag}"
    model_dir = os.path.join(output_dir, model_id)
    info = parse_model_info(model_name, pretrained_tag)

    print(f"\n{'═' * 70}")
    print(f"  MODEL: {model_name} / {pretrained_tag}")
    print(f"{'═' * 70}")

    model = None
    try:
        # ── Load model ───────────────────────────────────────────
        t0 = time.time()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_tag, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        load_time = time.time() - t0

        # Get actual model config details
        if hasattr(model, "visual"):
            visual = model.visual
            if hasattr(visual, "image_size"):
                img_size = visual.image_size
                if isinstance(img_size, (list, tuple)):
                    info["image_size"] = img_size[0]
                else:
                    info["image_size"] = img_size
            if hasattr(visual, "output_dim"):
                info["embed_dim"] = visual.output_dim
            elif hasattr(model, "embed_dim"):
                info["embed_dim"] = model.embed_dim

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        info["total_params_M"] = round(total_params / 1e6, 1)

        print(
            f"  Loaded in {load_time:.1f}s | {info['total_params_M']}M params | "
            f"img_size={info.get('image_size', '?')} | embed_dim={info.get('embed_dim', '?')}"
        )

        # ── Encode texts ─────────────────────────────────────────
        text_features = encode_texts(model, tokenizer, [text1, text2], device)
        feat_text1 = text_features[0]
        feat_text2 = text_features[1]

        # ── Process subgroups ────────────────────────────────────
        subgroup_results = {}
        t1 = time.time()

        for name, path in sorted(subgroups.items()):
            label = parse_subgroup_label(name)
            image_paths = sorted(glob.glob(os.path.join(path, "*.jpg")))
            if not image_paths:
                continue

            img_features = load_and_encode_images(
                model, preprocess, image_paths, device, batch_size
            )

            sim1 = (img_features @ feat_text1).numpy()
            sim2 = (img_features @ feat_text2).numpy()

            zs = zero_shot_classify(sim1, sim2)
            gt_label = get_ground_truth_label(name)
            accuracy = float((zs["predictions"] == gt_label).mean())

            subgroup_results[name] = {
                "sim_text1": sim1.tolist(),
                "sim_text2": sim2.tolist(),
                "count": len(image_paths),
                "mean_text1": float(sim1.mean()),
                "mean_text2": float(sim2.mean()),
                "std_text1": float(sim1.std()),
                "std_text2": float(sim2.std()),
                "gt_label": gt_label,
                "accuracy": accuracy,
                "probs_text1": zs["probs_text1"].tolist(),
                "probs_text2": zs["probs_text2"].tolist(),
            }

            del img_features, sim1, sim2
            print(f"    {label}: acc={accuracy:.1%}")

        encode_time = time.time() - t1

        # ── Aggregate metrics ────────────────────────────────────
        if not subgroup_results:
            print("  WARNING: No subgroup results, skipping.")
            return None

        accuracies = {n: d["accuracy"] for n, d in subgroup_results.items()}
        counts = {n: d["count"] for n, d in subgroup_results.items()}
        total_count = sum(counts.values())
        total_correct = sum(int(accuracies[n] * counts[n]) for n in accuracies)

        avg_acc = total_correct / total_count if total_count > 0 else 0
        worst_group_name = min(accuracies, key=accuracies.get)
        worst_group_acc = accuracies[worst_group_name]
        best_group_name = max(accuracies, key=accuracies.get)
        best_group_acc = accuracies[best_group_name]

        # Per GT-class accuracy
        for gt_val, gt_key in [(0, "gt_text1"), (1, "gt_text2")]:
            sub = {n: d for n, d in subgroup_results.items() if d["gt_label"] == gt_val}
            if sub:
                c = sum(int(d["accuracy"] * d["count"]) for d in sub.values())
                t = sum(d["count"] for d in sub.values())
                info[f"acc_{gt_key}"] = c / t if t > 0 else 0
            else:
                info[f"acc_{gt_key}"] = None

        # Build result row
        result = {
            **info,
            "avg_accuracy": avg_acc,
            "worst_group_accuracy": worst_group_acc,
            "worst_group": parse_subgroup_label(worst_group_name),
            "best_group_accuracy": best_group_acc,
            "best_group": parse_subgroup_label(best_group_name),
            "accuracy_gap": avg_acc - worst_group_acc,
            "total_images": total_count,
            "load_time_s": round(load_time, 1),
            "encode_time_s": round(encode_time, 1),
        }

        # Add per-subgroup accuracy columns
        for name, acc in sorted(accuracies.items()):
            col_name = f"acc_{parse_subgroup_label(name)}"
            result[col_name] = acc

        # ── Save per-model plots ─────────────────────────────────
        plot_model_results(subgroup_results, text1, text2, model_id, model_dir)

        # ── Save per-model JSON ──────────────────────────────────
        with open(os.path.join(model_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"\n  ✓ avg_acc={avg_acc:.1%}  worst_group={worst_group_acc:.1%} "
            f"({parse_subgroup_label(worst_group_name)})  "
            f"gap={avg_acc - worst_group_acc:.1%}"
        )

        return result

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return None

    finally:
        # ── CLEANUP: delete model from GPU/CPU memory ────────────
        if model is not None:
            cleanup_model(model, device)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  [cleanup] Model memory freed.")

        # ── CLEANUP: delete cached weights from disk ─────────────
        delete_cached_weights(model_name, pretrained_tag)
        print(f"  [cleanup] Done.")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Run UrbanCars ZS analysis across all OpenCLIP pretrained models."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--text1", default=DEFAULT_TEXT1)
    parser.add_argument("--text2", default=DEFAULT_TEXT2)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run models whose name contains this string (e.g. 'ViT-B').",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Skip models whose name contains this string (e.g. 'RN').",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models that already have results.json in the output dir.",
    )
    parser.add_argument(
        "--max-params",
        type=float,
        default=None,
        help="Skip models with more than this many million parameters.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Discover data ────────────────────────────────────────────
    subgroups = get_subgroup_dirs(args.data_root)
    if not subgroups:
        print(f"ERROR: No subgroup directories found in {args.data_root}")
        sys.exit(1)
    print(f"Found {len(subgroups)} subgroups in {args.data_root}")
    total_images = sum(
        len(glob.glob(os.path.join(p, "*.jpg"))) for p in subgroups.values()
    )
    print(f"Total images: {total_images}")

    # ── Enumerate all pretrained models ──────────────────────────
    all_pretrained = open_clip.list_pretrained()
    print(
        f"\nOpenCLIP has {len(all_pretrained)} pretrained model/weight combinations.\n"
    )

    # Apply filters
    filtered = []
    for model_name, pretrained_tag in all_pretrained:
        model_id = f"{model_name}_{pretrained_tag}"

        if args.filter and args.filter.lower() not in model_id.lower():
            continue
        if args.exclude and args.exclude.lower() in model_id.lower():
            continue
        if args.resume:
            result_path = os.path.join(args.output_dir, model_id, "results.json")
            if os.path.exists(result_path):
                print(f"  [skip] {model_id} — already completed.")
                continue
        filtered.append((model_name, pretrained_tag))

    print(f"Models to run: {len(filtered)}")
    if not filtered:
        print("Nothing to do.")
        # Still generate summary from existing results if resuming
        if args.resume:
            _generate_summary_from_existing(args.output_dir)
        return

    # ── Run analysis for each model ──────────────────────────────
    all_results = []

    # Load existing results if resuming
    if args.resume:
        all_results = _load_existing_results(args.output_dir)
        print(f"Loaded {len(all_results)} existing results.\n")

    os.makedirs(args.output_dir, exist_ok=True)

    for idx, (model_name, pretrained_tag) in enumerate(filtered):
        print(f"\n[{idx + 1}/{len(filtered)}] ", end="")

        result = analyze_single_model(
            model_name=model_name,
            pretrained_tag=pretrained_tag,
            subgroups=subgroups,
            text1=args.text1,
            text2=args.text2,
            device=device,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

        if result is not None:
            all_results.append(result)

            # ── Incrementally save master table ──────────────────
            df = pd.DataFrame(all_results)
            df = df.sort_values("avg_accuracy", ascending=False)
            df.to_csv(os.path.join(args.output_dir, "master_results.csv"), index=False)
            with open(os.path.join(args.output_dir, "master_results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

    # ── Final summary ────────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values("avg_accuracy", ascending=False)
        df.to_csv(os.path.join(args.output_dir, "master_results.csv"), index=False)
        with open(os.path.join(args.output_dir, "master_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        summary_dir = os.path.join(args.output_dir, "summary_plots")
        plot_summary(df, summary_dir)

        # Print top-level summary
        print(f"\n\n{'═' * 80}")
        print("FINAL SUMMARY")
        print(f"{'═' * 80}")
        print(f"Models analyzed: {len(df)}")
        print(
            f"Failed models: {len(filtered) - len([r for r in all_results if r is not None])}"
        )
        print(f"\nTop 10 by Average Accuracy:")
        print(
            df[
                [
                    "model_id",
                    "arch_family",
                    "total_params_M",
                    "image_size",
                    "training_data",
                    "avg_accuracy",
                    "worst_group_accuracy",
                    "accuracy_gap",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
        print(f"\nTop 10 by Worst-Group Accuracy:")
        print(
            df.sort_values("worst_group_accuracy", ascending=False)[
                [
                    "model_id",
                    "arch_family",
                    "total_params_M",
                    "image_size",
                    "training_data",
                    "avg_accuracy",
                    "worst_group_accuracy",
                    "accuracy_gap",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
        print(f"\nResults saved to:")
        print(f"  {args.output_dir}/master_results.csv")
        print(f"  {args.output_dir}/master_results.json")
        print(f"  {summary_dir}/ (comparison plots)")


def _load_existing_results(output_dir: str) -> list[dict]:
    """Load all existing results.json files from model subdirectories."""
    results = []
    if not os.path.isdir(output_dir):
        return results
    for entry in os.listdir(output_dir):
        rpath = os.path.join(output_dir, entry, "results.json")
        if os.path.isfile(rpath):
            try:
                with open(rpath) as f:
                    results.append(json.load(f))
            except Exception:
                pass
    return results


def _generate_summary_from_existing(output_dir: str) -> None:
    """Regenerate summary from existing per-model results."""
    results = _load_existing_results(output_dir)
    if not results:
        print("No existing results found.")
        return
    df = pd.DataFrame(results)
    df = df.sort_values("avg_accuracy", ascending=False)
    df.to_csv(os.path.join(output_dir, "master_results.csv"), index=False)
    with open(os.path.join(output_dir, "master_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    summary_dir = os.path.join(output_dir, "summary_plots")
    plot_summary(df, summary_dir)
    print(f"Regenerated summary from {len(results)} existing results.")


if __name__ == "__main__":
    main()
