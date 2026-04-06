#!/usr/bin/env python3
"""
Run CelebA zero-shot classification analysis across ALL pretrained
OpenCLIP models. For each model:
  1. Load model, encode texts + images, compute per-subgroup accuracy
  2. Save per-model plots
  3. Delete model from GPU/CPU and clear cache
  4. Append results to a master table

Dataset: BiasedCelebASplit (CelebA with target=blonde, bias=gender)
Subgroups: (target, bias) = (blonde/non-blonde) × (male/female)

Outputs:
  - results_celeba/<model_name>/  — per-model plots
  - results_celeba/master_results.csv
  - results_celeba/master_results.json
  - results_celeba/summary_plots/

Usage:
    python run_all_models_celeba.py
    python run_all_models_celeba.py --target-attr blonde
    python run_all_models_celeba.py --target-attr makeup
    python run_all_models_celeba.py --filter "ViT-B" --resume

Requirements:
    pip install torch open-clip-torch torchvision Pillow matplotlib numpy pandas
"""

import argparse
import gc
import json
import os
import pickle
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CelebA

import open_clip


# ── Defaults ─────────────────────────────────────────────────────
DEFAULT_DATA_ROOT = "../../data/celeba"
DEFAULT_TARGET_ATTR = "blonde"
DEFAULT_BATCH_SIZE = 32
DEFAULT_OUTPUT_DIR = "./results_celeba"

# Text prompts for zero-shot classification per target attribute.
# Each class uses MULTIPLE text descriptions — their embeddings are
# averaged and re-normalized to form a single robust class prototype.
# This avoids weak concepts like "non-blonde" and instead uses concrete
# descriptions that CLIP has actually seen during training.
TARGET_TEXTS = {
    "blonde": {
        "texts1": [  # blonde / light hair
            "a photo of a person with blonde hair",
            "a photo of a person with light blonde hair",
            "a photo of a person with golden hair",
            "a photo of a person with platinum blonde hair",
        ],
        "texts2": [  # non-blonde (concrete alternatives)
            "a photo of a person with dark hair",
            "a photo of a person with black hair",
            "a photo of a person with brown hair",
            "a photo of a brunette person",
            "a photo of a person with red hair",
            "a photo of a person with grey hair",
            "a photo of a bald person",
            "a photo of a person with auburn hair",
        ],
        "target_names": ["non-blonde", "blonde"],
    },
    "makeup": {
        "texts1": [  # heavy makeup
            "a photograph of a person wearing heavy makeup",
            "a photograph of a person with visible makeup",
            "a photo of someone with lipstick and eyeshadow",
            "a face with heavy cosmetics and makeup",
            "a photograph of a glamorous face with makeup",
        ],
        "texts2": [  # no makeup
            "a photograph of a person without makeup",
            "a photograph of a person with a natural face",
            "a photo of someone with a bare face",
            "a photograph of a person with no cosmetics",
            "a photograph of a clean face without makeup",
        ],
        "target_names": ["makeup", "no-makeup"],
    },
}

# Bias attribute index 20 = "Male" in CelebA
BIAS_IDX = 20
BIAS_NAMES = {0: "female", 1: "male"}

# CelebA attr name mappings
TARGET_ATTR_IDX = {
    "blonde": 9,  # Blond_Hair
    "makeup": 18,  # Heavy_Makeup
}


# ═══════════════════════════════════════════════════════════════════
# Dataset: BiasedCelebASplit (adapted from provided code)
# ═══════════════════════════════════════════════════════════════════


def get_confusion_matrix(num_classes, targets, biases):
    """Compute confusion matrix between targets and biases."""
    confusion_matrix_org = torch.zeros(num_classes, num_classes, dtype=torch.long)
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.float)
    for t in range(num_classes):
        for b in range(num_classes):
            confusion_matrix_org[t, b] = ((targets == t) & (biases == b)).sum()
    total = confusion_matrix_org.sum()
    confusion_matrix = (
        confusion_matrix_org.float() / total
        if total > 0
        else confusion_matrix_org.float()
    )
    confusion_matrix_by = confusion_matrix_org.float() / confusion_matrix_org.sum(
        dim=1, keepdim=True
    ).clamp(min=1)
    return confusion_matrix_org, confusion_matrix, confusion_matrix_by


def download_celeba_zip(root):
    """Placeholder — assumes data already exists or uses torchvision download."""
    pass


def download_celeba_anno(root):
    """Placeholder — assumes annotations already exist."""
    pass


class BiasedCelebASplit:
    def __init__(self, root, split, transform, target_attr, **kwargs):
        self.transform = transform
        self.target_attr = target_attr

        if os.path.basename(os.path.normpath(root)) == "celeba":
            root = os.path.dirname(root)

        self.celeba = CelebA(
            root=root,
            split="train" if split == "train_valid" else split,
            target_type="attr",
            transform=transform,
        )
        self.bias_idx = BIAS_IDX

        if target_attr == "blonde":
            self.target_idx = 9
            if split in ["train", "train_valid"]:
                save_path = Path(root) / "pickles" / "blonde"
                if save_path.is_dir():
                    self.indices = pickle.load(open(save_path / "indices.pkl", "rb"))
                else:
                    self.indices = self.build_blonde()
                    save_path.mkdir(parents=True, exist_ok=True)
                    pickle.dump(self.indices, open(save_path / "indices.pkl", "wb"))
                self.attr = self.celeba.attr[self.indices]
            else:
                self.attr = self.celeba.attr
                self.indices = torch.arange(len(self.celeba))
        elif target_attr == "makeup":
            self.target_idx = 18
            self.attr = self.celeba.attr
            self.indices = torch.arange(len(self.celeba))
        else:
            raise AttributeError(f"Unknown target_attr: {target_attr}")

        if split in ["train", "train_valid"]:
            save_path = Path(
                os.path.join(root, f"clusters/celeba_rand_indices_{target_attr}.pkl")
            )
            if not save_path.exists():
                rand_indices = torch.randperm(len(self.indices))
                save_path.parent.mkdir(parents=True, exist_ok=True)
                pickle.dump(rand_indices, open(save_path, "wb"))
            else:
                rand_indices = pickle.load(open(save_path, "rb"))

            num_total = len(rand_indices)
            num_train = int(0.8 * num_total)
            if split == "train":
                indices = rand_indices[:num_train]
            elif split == "train_valid":
                indices = rand_indices[num_train:]
            self.indices = self.indices[indices]
            self.attr = self.attr[indices]

        self.targets = self.attr[:, self.target_idx]
        self.biases = self.attr[:, self.bias_idx]

        self.confusion_matrix_org, self.confusion_matrix, self.confusion_matrix_by = (
            get_confusion_matrix(
                num_classes=2, targets=self.targets, biases=self.biases
            )
        )

        print(
            f"BiasedCelebASplit: target={target_attr} split={split}\n{self.confusion_matrix_org}"
        )

    def build_blonde(self):
        biases = self.celeba.attr[:, self.bias_idx]
        targets = self.celeba.attr[:, self.target_idx]
        selects = torch.arange(len(self.celeba))[(biases == 0) & (targets == 0)]
        non_selects = torch.arange(len(self.celeba))[~((biases == 0) & (targets == 0))]
        np.random.shuffle(selects)
        indices = torch.cat([selects[:2000], non_selects])
        return indices

    def __getitem__(self, index):
        img, _ = self.celeba.__getitem__(self.indices[index])
        target, bias = self.targets[index], self.biases[index]
        return {"inputs": img, "targets": target, "gender": bias, "index": index}

    def __len__(self):
        return len(self.targets)


def get_celeba(root, batch_size, split, target_attr, img_size=224, transform=None):
    """Load CelebA dataset with given split and return loader + dataset."""
    if transform is None:
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    dataset = BiasedCelebASplit(
        root=root, split=split, transform=transform, target_attr=target_attr
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return loader, dataset


# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════


def get_subgroup_label(target: int, bias: int, target_names: list[str]) -> str:
    """e.g. 'blonde_male', 'non-blonde_female'."""
    return f"{target_names[target]}_{BIAS_NAMES[bias]}"


def encode_images_from_dataset(
    model, preprocess, dataset, device: str, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load images from BiasedCelebASplit using the model's own preprocess,
    encode them, and return (features, targets, biases).

    We re-load images with the CLIP preprocess instead of the dataset's transform.
    """
    all_features = []
    all_targets = []
    all_biases = []

    # Access the underlying CelebA dataset for raw image loading
    celeba = dataset.celeba
    indices = dataset.indices

    for i in range(0, len(dataset), batch_size):
        batch_end = min(i + batch_size, len(dataset))
        images = []
        for j in range(i, batch_end):
            # Load raw image via CelebA (bypassing dataset transform)
            img_idx = (
                indices[j].item()
                if isinstance(indices[j], torch.Tensor)
                else indices[j]
            )
            img_path = os.path.join(
                celeba.root,
                celeba.base_folder,
                "img_align_celeba",
                celeba.filename[img_idx],
            )
            img = Image.open(img_path).convert("RGB")
            images.append(preprocess(img))

        batch = torch.stack(images).to(device)
        with torch.no_grad(), torch.amp.autocast(device):
            features = model.encode_image(batch)
        all_features.append(features.float().cpu())
        del batch, features

        all_targets.append(dataset.targets[i:batch_end])
        all_biases.append(dataset.biases[i:batch_end])

    features = F.normalize(torch.cat(all_features, dim=0), dim=-1)
    targets = torch.cat(all_targets, dim=0)
    biases = torch.cat(all_biases, dim=0)
    return features, targets, biases


def encode_texts(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    """Encode a list of texts, return normalized features."""
    tokens = tokenizer(texts).to(device)
    with torch.no_grad(), torch.amp.autocast(device):
        features = model.encode_text(tokens)
    return F.normalize(features.float().cpu(), dim=-1)


def encode_text_ensemble(
    model, tokenizer, texts_list: list[str], device: str
) -> torch.Tensor:
    """
    Encode multiple texts for one class, return ALL embeddings (not averaged).

    At inference time, we compute similarity to every text in the class
    and take the MAX per image — the image is as similar to the class as
    it is to its best-matching description.

    Args:
        texts_list: list of text descriptions for one class
    Returns:
        (N, D) normalized embeddings, one per text
    """
    return encode_texts(model, tokenizer, texts_list, device)  # (N, D)


def zero_shot_classify(sim_text1: np.ndarray, sim_text2: np.ndarray) -> dict:
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


def cleanup_model(model, device: str) -> None:
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def delete_cached_weights(model_name: str, pretrained_tag: str) -> None:
    home = os.path.expanduser("~")
    deleted = []

    hf_cache = os.path.join(home, ".cache", "huggingface", "hub")
    if os.path.isdir(hf_cache):
        for entry in os.listdir(hf_cache):
            entry_path = os.path.join(hf_cache, entry)
            if not os.path.isdir(entry_path):
                continue
            model_parts = model_name.lower().replace("-", "").replace("_", "")
            tag_parts = pretrained_tag.lower().replace("-", "").replace("_", "")
            entry_clean = entry.lower().replace("-", "").replace("_", "")
            if model_parts in entry_clean and (
                tag_parts in entry_clean or "clip" in entry.lower()
            ):
                try:
                    shutil.rmtree(entry_path)
                    deleted.append(entry_path)
                except Exception as e:
                    print(f"    [warn] Could not delete {entry_path}: {e}")

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

    torch_cache = os.path.join(home, ".cache", "torch", "hub", "checkpoints")
    if os.path.isdir(torch_cache):
        for entry in os.listdir(torch_cache):
            entry_path = os.path.join(torch_cache, entry)
            model_parts = model_name.lower().replace("-", "_")
            if model_parts in entry.lower() or pretrained_tag.lower() in entry.lower():
                try:
                    os.remove(entry_path)
                    deleted.append(entry_path)
                except Exception as e:
                    print(f"    [warn] Could not delete {entry_path}: {e}")

    if deleted:
        for p in deleted:
            print(f"    [deleted] {p}")
        print(f"    [cleanup] Removed {len(deleted)} cached file(s)/dir(s) from disk.")
    else:
        print(f"    [cleanup] No cached weight files found to delete.")


def parse_model_info(model_name: str, pretrained_tag: str) -> dict:
    info = {
        "model_name": model_name,
        "pretrained_tag": pretrained_tag,
        "model_id": f"{model_name}_{pretrained_tag}",
    }

    if model_name.startswith("RN"):
        info["arch_family"] = "ResNet"
        m = re.match(r"RN(\d+)(x\d+)?", model_name)
        if m:
            info["resnet_depth"] = int(m.group(1))
            info["resnet_multiplier"] = m.group(2) or "x1"
        info["patch_size"] = None
    elif "ViT" in model_name or "vit" in model_name.lower():
        info["arch_family"] = "ViT"
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

    img_m = re.search(r"(\d{3,4})$", model_name.replace("-", ""))
    if img_m and int(img_m.group(1)) >= 200:
        info["image_size"] = int(img_m.group(1))
    else:
        info["image_size"] = None

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

    info["quickgelu"] = "quickgelu" in model_name.lower()
    return info


# ═══════════════════════════════════════════════════════════════════
# Per-model plotting
# ═══════════════════════════════════════════════════════════════════


def plot_model_results(
    subgroup_results: dict[str, dict],
    text1: str,
    text2: str,
    model_id: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n_sub = len(subgroup_results)
    labels_short = list(subgroup_results.keys())

    # ── Histograms ───────────────────────────────────────────────
    cols = min(4, n_sub)
    rows = (n_sub + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for idx, (name, data) in enumerate(subgroup_results.items()):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
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
        ax.set_title(f"{name}\nn={data['count']}, Acc={acc:.1f}%", fontsize=9)
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
    fig, ax = plt.subplots(figsize=(10, 5))
    accs = [subgroup_results[n]["accuracy"] * 100 for n in labels_short]
    x_pos = np.arange(n_sub)
    # Color: target=0 (text1 class) blue, target=1 (text2 class) coral
    colors = [
        "steelblue" if subgroup_results[n]["gt_label"] == 0 else "coral"
        for n in labels_short
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
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_short, rotation=30, ha="right", fontsize=9)
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
    os.makedirs(output_dir, exist_ok=True)
    if df.empty:
        return

    for metric_col, metric_label in [
        ("avg_accuracy", "Average Accuracy"),
        ("worst_group_accuracy", "Worst-Group Accuracy"),
    ]:
        df_sorted = df.sort_values(metric_col, ascending=True).copy()
        fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted) * 0.35)))
        y_pos = np.arange(len(df_sorted))
        colors = plt.cm.RdYlGn(df_sorted[metric_col].values)
        ax.barh(
            y_pos,
            df_sorted[metric_col] * 100,
            color=colors,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted["model_id"], fontsize=6)
        ax.set_xlabel(f"{metric_label} (%)")
        ax.set_title(f"CelebA Zero-Shot {metric_label} — All Models")
        ax.axvline(50, color="gray", linewidth=0.8, linestyle="--")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"all_models_{metric_col}.png"), dpi=150)
        plt.close(fig)

    # Avg vs Worst-Group scatter
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
    ax.set_title("CelebA — Average vs Worst-Group Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "avg_vs_worst_group.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Single-model analysis
# ═══════════════════════════════════════════════════════════════════


def analyze_single_model(
    model_name: str,
    pretrained_tag: str,
    dataset,
    texts1: list[str],
    texts2: list[str],
    target_names: list[str],
    device: str,
    batch_size: int,
    output_dir: str,
) -> dict | None:
    model_id = f"{model_name}_{pretrained_tag}"
    model_dir = os.path.join(output_dir, model_id)
    info = parse_model_info(model_name, pretrained_tag)

    print(f"\n{'═' * 70}")
    print(f"  MODEL: {model_name} / {pretrained_tag}")
    print(f"{'═' * 70}")

    model = None
    try:
        # ── Load CLIP model ──────────────────────────────────────
        t0 = time.time()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_tag, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        load_time = time.time() - t0

        # Model info
        if hasattr(model, "visual"):
            visual = model.visual
            if hasattr(visual, "image_size"):
                img_size = visual.image_size
                info["image_size"] = (
                    img_size[0] if isinstance(img_size, (list, tuple)) else img_size
                )
            if hasattr(visual, "output_dim"):
                info["embed_dim"] = visual.output_dim
            elif hasattr(model, "embed_dim"):
                info["embed_dim"] = model.embed_dim

        total_params = sum(p.numel() for p in model.parameters())
        info["total_params_M"] = round(total_params / 1e6, 1)

        print(
            f"  Loaded in {load_time:.1f}s | {info['total_params_M']}M params | "
            f"img_size={info.get('image_size', '?')} | embed_dim={info.get('embed_dim', '?')}"
        )

        # ── Encode text ensembles ────────────────────────────────
        print(f"  Encoding text ensembles: {len(texts1)} + {len(texts2)} prompts …")
        feat_texts1 = encode_text_ensemble(model, tokenizer, texts1, device)  # (N1, D)
        feat_texts2 = encode_text_ensemble(model, tokenizer, texts2, device)  # (N2, D)

        # ── Encode all images ────────────────────────────────────
        print(f"  Encoding {len(dataset)} images …")
        t1 = time.time()
        img_features, targets, biases = encode_images_from_dataset(
            model, preprocess, dataset, device, batch_size
        )
        encode_time = time.time() - t1
        print(f"  Encoded in {encode_time:.1f}s")

        # ── Compute similarities (max over texts per class) ──────
        # img_features: (N_img, D), feat_texts1: (N1, D) → sim_all1: (N_img, N1)
        sim_all1 = (img_features @ feat_texts1.T).numpy()  # (N_img, N1)
        sim_all2 = (img_features @ feat_texts2.T).numpy()  # (N_img, N2)

        # Per image, take the max similarity across all texts in each class
        sim1 = sim_all1.max(axis=1)  # (N_img,)
        sim2 = sim_all2.max(axis=1)  # (N_img,)

        targets_np = targets.numpy()
        biases_np = biases.numpy()

        # ── Group by (target, bias) subgroups ────────────────────
        # target: 0=text1_class (e.g., blonde), 1=text2_class (e.g., non-blonde)
        # For zero-shot: prediction=0 means classified as text1
        # GT: target=1 in CelebA means attribute is present
        # So for "blonde": target=1 → blonde (matches text1), target=0 → non-blonde (matches text2)
        # We need to map: CelebA target=1 → ZS class 0 (text1=blonde)
        #                  CelebA target=0 → ZS class 1 (text2=non-blonde)
        gt_zs = 1 - targets_np  # flip: CelebA 1→ZS 0, CelebA 0→ZS 1

        zs = zero_shot_classify(sim1, sim2)
        predictions = zs["predictions"]

        subgroup_results = {}
        for target_val in [0, 1]:
            for bias_val in [0, 1]:
                mask = (targets_np == target_val) & (biases_np == bias_val)
                if mask.sum() == 0:
                    continue

                sg_name = get_subgroup_label(target_val, bias_val, target_names)
                sg_gt = gt_zs[mask]
                sg_pred = predictions[mask]
                sg_sim1 = sim1[mask]
                sg_sim2 = sim2[mask]
                accuracy = float((sg_pred == sg_gt).mean())

                subgroup_results[sg_name] = {
                    "sim_text1": sg_sim1.tolist(),
                    "sim_text2": sg_sim2.tolist(),
                    "count": int(mask.sum()),
                    "gt_label": int(1 - target_val),  # ZS ground truth
                    "accuracy": accuracy,
                    "target_val": int(target_val),
                    "bias_val": int(bias_val),
                }

                print(f"    {sg_name}: n={mask.sum()}, acc={accuracy:.1%}")

        # ── Aggregate metrics ────────────────────────────────────
        if not subgroup_results:
            print("  WARNING: No subgroup results.")
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

        # Overall ZS accuracy (not grouped)
        overall_acc = float((predictions == gt_zs).mean())

        result = {
            **info,
            "avg_accuracy": avg_acc,
            "overall_accuracy": overall_acc,
            "worst_group_accuracy": worst_group_acc,
            "worst_group": worst_group_name,
            "best_group_accuracy": best_group_acc,
            "best_group": best_group_name,
            "accuracy_gap": avg_acc - worst_group_acc,
            "total_images": total_count,
            "load_time_s": round(load_time, 1),
            "encode_time_s": round(encode_time, 1),
        }

        for name, acc in sorted(accuracies.items()):
            result[f"acc_{name}"] = acc

        # ── Save per-model outputs ───────────────────────────────
        plot_model_results(
            subgroup_results, target_names[0], target_names[1], model_id, model_dir
        )
        with open(os.path.join(model_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"\n  ✓ avg={avg_acc:.1%}  overall={overall_acc:.1%}  "
            f"worst={worst_group_acc:.1%} ({worst_group_name})  gap={avg_acc - worst_group_acc:.1%}"
        )

        return result

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return None

    finally:
        if model is not None:
            cleanup_model(model, device)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  [cleanup] Model memory freed.")
        delete_cached_weights(model_name, pretrained_tag)
        print(f"  [cleanup] Done.")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Run CelebA ZS analysis across all OpenCLIP pretrained models."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--target-attr", default=DEFAULT_TARGET_ATTR, choices=["blonde", "makeup"]
    )
    parser.add_argument(
        "--extra-texts1",
        nargs="*",
        default=None,
        help="Additional text prompts for class 1 (appended to defaults).",
    )
    parser.add_argument(
        "--extra-texts2",
        nargs="*",
        default=None,
        help="Additional text prompts for class 2 (appended to defaults).",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Text prompts (multi-text ensembles)
    text_cfg = TARGET_TEXTS[args.target_attr]
    texts1 = list(text_cfg["texts1"])
    texts2 = list(text_cfg["texts2"])
    target_names = text_cfg["target_names"]

    if args.extra_texts1:
        texts1.extend(args.extra_texts1)
    if args.extra_texts2:
        texts2.extend(args.extra_texts2)

    print(f"Device      : {device}")
    print(f"Data root   : {args.data_root}")
    print(f"Target attr : {args.target_attr}")
    print(f"Texts class1 ({len(texts1)} prompts):")
    for t in texts1:
        print(f'  - "{t}"')
    print(f"Texts class2 ({len(texts2)} prompts):")
    for t in texts2:
        print(f'  - "{t}"')
    print(f"Output dir  : {args.output_dir}\n")

    # ── Load CelebA test set (with dummy transform — will be overridden per model) ──
    print("Loading CelebA test dataset …")
    from torchvision import transforms

    dummy_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = BiasedCelebASplit(
        root=args.data_root,
        split="test",
        transform=dummy_transform,
        target_attr=args.target_attr,
    )
    print(f"Test set size: {len(dataset)}")
    print(f"Subgroup counts:")
    for t in [0, 1]:
        for b in [0, 1]:
            mask = (dataset.targets == t) & (dataset.biases == b)
            sg = get_subgroup_label(t, b, target_names)
            print(f"  {sg}: {mask.sum().item()}")

    # ── Enumerate models ─────────────────────────────────────────
    all_pretrained = open_clip.list_pretrained()
    print(
        f"\nOpenCLIP has {len(all_pretrained)} pretrained model/weight combinations.\n"
    )

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
                print(f"  [skip] {model_id}")
                continue
        filtered.append((model_name, pretrained_tag))

    print(f"Models to run: {len(filtered)}")
    if not filtered:
        print("Nothing to do.")
        if args.resume:
            _generate_summary_from_existing(args.output_dir)
        return

    # ── Run ──────────────────────────────────────────────────────
    all_results = []
    if args.resume:
        all_results = _load_existing_results(args.output_dir)
        print(f"Loaded {len(all_results)} existing results.\n")

    os.makedirs(args.output_dir, exist_ok=True)

    for idx, (model_name, pretrained_tag) in enumerate(filtered):
        print(f"\n[{idx + 1}/{len(filtered)}] ", end="")

        result = analyze_single_model(
            model_name=model_name,
            pretrained_tag=pretrained_tag,
            dataset=dataset,
            texts1=texts1,
            texts2=texts2,
            target_names=target_names,
            device=device,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

        if result is not None:
            all_results.append(result)
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

        print(f"\n\n{'═' * 80}")
        print("FINAL SUMMARY")
        print(f"{'═' * 80}")
        print(f"Models analyzed: {len(df)}")
        cols_show = [
            "model_id",
            "arch_family",
            "total_params_M",
            "training_data",
            "avg_accuracy",
            "worst_group_accuracy",
            "accuracy_gap",
            "worst_group",
        ]
        cols_show = [c for c in cols_show if c in df.columns]
        print(f"\nTop 10 by Average Accuracy:")
        print(df[cols_show].head(10).to_string(index=False))
        print(f"\nTop 10 by Worst-Group Accuracy:")
        print(
            df.sort_values("worst_group_accuracy", ascending=False)[cols_show]
            .head(10)
            .to_string(index=False)
        )
        print(f"\nResults saved to:")
        print(f"  {args.output_dir}/master_results.csv")
        print(f"  {args.output_dir}/master_results.json")
        print(f"  {summary_dir}/")


def _load_existing_results(output_dir: str) -> list[dict]:
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
