#!/usr/bin/env python3
"""
Load UrbanCars test images from all subgroup directories, compute CLIP
similarities to two given texts, and save distribution plots per subgroup.

Directory structure expected:
  data/urbancars/bg-0.5_co_occur_obj-0.5/test/
    obj-{urban,country}_bg-{urban,country}_co_occur_obj-{urban,country}/
      000.jpg, 001.jpg, ...

Usage:
    python urbancars_similarity.py
    python urbancars_similarity.py --text1 "an urban car" --text2 "a country car"
    python urbancars_similarity.py --data-root /path/to/data/urbancars/...

Requirements:
    pip install torch open-clip-torch Pillow matplotlib numpy tqdm
"""

import argparse
import glob
import os
import re
from itertools import product as iterproduct

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip


# ── Defaults ─────────────────────────────────────────────────────
DEFAULT_DATA_ROOT = "../data/urbancars/bg-0.5_co_occur_obj-0.5/test"
DEFAULT_TEXT1 = "a photograph of a compact, sports, sedan car"
DEFAULT_TEXT2 = "a photograph of a truck, jeep, pickup car"
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s32b_b79k"
DEFAULT_BATCH_SIZE = 64
DEFAULT_OUTPUT_DIR = "./plots"

ATTRIBUTES = ["urban", "country"]


def get_subgroup_dirs(data_root: str) -> dict[str, str]:
    """
    Enumerate all 8 subgroup directories from the 3 binary attributes:
      obj-{urban,country}_bg-{urban,country}_co_occur_obj-{urban,country}

    Returns:
        dict mapping subgroup name -> full directory path
    """
    subgroups = {}
    for obj, bg, co in iterproduct(ATTRIBUTES, repeat=3):
        name = f"obj-{obj}_bg-{bg}_co_occur_obj-{co}"
        path = os.path.join(data_root, name)
        if os.path.isdir(path):
            subgroups[name] = path
        else:
            print(f"  [WARNING] Directory not found, skipping: {path}")
    return subgroups


def load_and_encode_images(
    model, preprocess, image_paths: list[str], device: str, batch_size: int
) -> torch.Tensor:
    """Load images in batches, encode, and return normalized features."""
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
        all_features.append(features.float())
    return F.normalize(torch.cat(all_features, dim=0), dim=-1)


def encode_texts(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    """Encode texts and return normalized features."""
    tokens = tokenizer(texts).to(device)
    with torch.no_grad(), torch.amp.autocast(device):
        features = model.encode_text(tokens)
    return F.normalize(features.float(), dim=-1)


def parse_subgroup_label(name: str) -> str:
    """Convert directory name to a readable label, e.g. 'obj=urban, bg=country, co=urban'."""
    m = re.match(r"obj-(\w+)_bg-(\w+)_co_occur_obj-(\w+)", name)
    if m:
        return f"obj={m.group(1)}, bg={m.group(2)}, co={m.group(3)}"
    return name


def get_ground_truth_label(name: str) -> int:
    """
    Extract ground-truth class from the obj-{urban,country} attribute.
    Returns 0 if obj=urban (matches text1), 1 if obj=country (matches text2).
    """
    m = re.match(r"obj-(\w+)_bg-(\w+)_co_occur_obj-(\w+)", name)
    if m:
        return 0 if m.group(1) == "urban" else 1
    return -1


def zero_shot_classify(sim_text1: np.ndarray, sim_text2: np.ndarray) -> dict:
    """
    Zero-shot classification via softmax over the two text similarities.

    Returns dict with:
        predictions: array of 0 (text1) or 1 (text2) per image
        probs_text1: P(text1) per image
        probs_text2: P(text2) per image
    """
    # Stack as (N, 2) and apply softmax
    logits = np.stack([sim_text1, sim_text2], axis=-1)
    # Numerically stable softmax
    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    predictions = np.argmax(probs, axis=-1)
    return {
        "predictions": predictions,
        "probs_text1": probs[:, 0],
        "probs_text2": probs[:, 1],
    }


def plot_distributions(
    subgroup_results: dict[str, dict],
    text1: str,
    text2: str,
    output_dir: str,
) -> None:
    """
    Create and save plots:
      1. Per-subgroup histograms (one figure per subgroup, both texts overlaid)
      2. Combined overview (all subgroups side by side)
      3. Box plot summary
    """
    os.makedirs(output_dir, exist_ok=True)
    n_sub = len(subgroup_results)

    # ── 1. Per-subgroup histograms ───────────────────────────────
    for name, data in subgroup_results.items():
        label = parse_subgroup_label(name)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            data["sim_text1"],
            bins=40,
            alpha=0.6,
            label=f'"{text1}"',
            color="steelblue",
            density=True,
        )
        ax.hist(
            data["sim_text2"],
            bins=40,
            alpha=0.6,
            label=f'"{text2}"',
            color="coral",
            density=True,
        )
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.set_title(f"Similarity Distribution — {label}\n(n={data['count']})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"hist_{name}.png"), dpi=150)
        plt.close(fig)

    # ── 2. Combined grid of histograms ───────────────────────────
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
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Cosine Sim", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    # hide unused axes
    for idx in range(n_sub, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)
    fig.suptitle("UrbanCars — Text Similarity Distributions per Subgroup", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, "combined_histograms.png"), dpi=150)
    plt.close(fig)

    # ── 3. Box plot summary ──────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    labels_short = [parse_subgroup_label(n) for n in subgroup_results]

    data_t1 = [subgroup_results[n]["sim_text1"] for n in subgroup_results]
    data_t2 = [subgroup_results[n]["sim_text2"] for n in subgroup_results]

    bp1 = ax1.boxplot(data_t1, vert=True, patch_artist=True, labels=labels_short)
    for patch in bp1["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax1.set_title(f'Similarity to "{text1}"')
    ax1.set_ylabel("Cosine Similarity")
    ax1.tick_params(axis="x", rotation=45, labelsize=7)
    ax1.grid(True, alpha=0.3)

    bp2 = ax2.boxplot(data_t2, vert=True, patch_artist=True, labels=labels_short)
    for patch in bp2["boxes"]:
        patch.set_facecolor("coral")
        patch.set_alpha(0.6)
    ax2.set_title(f'Similarity to "{text2}"')
    ax2.tick_params(axis="x", rotation=45, labelsize=7)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("UrbanCars — Similarity Box Plots per Subgroup", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "boxplots.png"), dpi=150)
    plt.close(fig)

    # ── 4. Difference plot (text1 - text2) ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    diff_data = [
        np.array(subgroup_results[n]["sim_text1"])
        - np.array(subgroup_results[n]["sim_text2"])
        for n in subgroup_results
    ]
    bp = ax.boxplot(diff_data, vert=True, patch_artist=True, labels=labels_short)
    for patch in bp["boxes"]:
        patch.set_facecolor("mediumpurple")
        patch.set_alpha(0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f'Similarity Difference: "{text1}" − "{text2}"')
    ax.set_ylabel("Cosine Similarity Difference")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "difference_boxplots.png"), dpi=150)
    plt.close(fig)

    # ── 5. Zero-shot classification accuracy per subgroup ────────
    fig, ax = plt.subplots(figsize=(12, 6))
    names_sorted = list(subgroup_results.keys())
    accs = [subgroup_results[n].get("accuracy", 0.0) * 100 for n in names_sorted]
    x_pos = np.arange(len(names_sorted))

    # Color by ground-truth class: steelblue for text1 (urban), coral for text2 (country)
    colors = [
        "steelblue" if subgroup_results[n].get("gt_label", 0) == 0 else "coral"
        for n in names_sorted
    ]
    bars = ax.bar(
        x_pos, accs, color=colors, alpha=0.75, edgecolor="black", linewidth=0.5
    )

    # Add accuracy text on bars
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
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="Chance (50%)")
    ax.set_title(
        f"Zero-Shot Classification Accuracy\n" f'"{text1}" (blue) vs "{text2}" (red)'
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "zeroshot_accuracy.png"), dpi=150)
    plt.close(fig)

    # ── 6. Softmax probability distributions per subgroup ────────
    cols = 4
    rows = (n_sub + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for idx, (name, data) in enumerate(subgroup_results.items()):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        label = parse_subgroup_label(name)
        gt = data.get("gt_label", 0)
        gt_name = f'"{text1}"' if gt == 0 else f'"{text2}"'
        acc = data.get("accuracy", 0.0) * 100

        probs1 = data.get("probs_text1", [])
        if probs1:
            ax.hist(probs1, bins=30, alpha=0.7, color="steelblue", density=True)
        ax.axvline(0.5, color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{label}\nGT={gt_name}, Acc={acc:.1f}%", fontsize=8)
        ax.set_xlabel(f'P("{text1}")', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
    for idx in range(n_sub, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)
    fig.suptitle(f'Zero-Shot P("{text1}") Distribution per Subgroup', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, "zeroshot_prob_distributions.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="UrbanCars subgroup similarity analysis with OpenCLIP."
    )
    parser.add_argument(
        "--data-root", default=DEFAULT_DATA_ROOT, help="Path to test directory."
    )
    parser.add_argument(
        "--text1",
        default=DEFAULT_TEXT1,
        help=f'First text prompt (default: "{DEFAULT_TEXT1}").',
    )
    parser.add_argument(
        "--text2",
        default=DEFAULT_TEXT2,
        help=f'Second text prompt (default: "{DEFAULT_TEXT2}").',
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenCLIP model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--pretrained",
        default=DEFAULT_PRETRAINED,
        help=f"Weights (default: {DEFAULT_PRETRAINED}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for image encoding.",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output plots."
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Model      : {args.model} / {args.pretrained}")
    print(f"Data root  : {args.data_root}")
    print(f'Text 1     : "{args.text1}"')
    print(f'Text 2     : "{args.text2}"')
    print(f"Output dir : {args.output_dir}\n")

    # ── Load model ───────────────────────────────────────────────
    print("Loading model …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    # ── Encode the two texts ─────────────────────────────────────
    print("Encoding texts …")
    text_features = encode_texts(model, tokenizer, [args.text1, args.text2], device)
    feat_text1 = text_features[0]  # (D,)
    feat_text2 = text_features[1]  # (D,)

    # ── Discover subgroups ───────────────────────────────────────
    print("Scanning subgroup directories …")
    subgroups = get_subgroup_dirs(args.data_root)
    print(f"Found {len(subgroups)} subgroups.\n")

    if not subgroups:
        print("ERROR: No subgroup directories found. Check --data-root path.")
        return

    # ── Process each subgroup ────────────────────────────────────
    subgroup_results = {}
    for name, path in sorted(subgroups.items()):
        label = parse_subgroup_label(name)
        image_paths = sorted(glob.glob(os.path.join(path, "*.jpg")))

        if not image_paths:
            print(f"  [{label}] No images found, skipping.")
            continue

        print(f"  [{label}] Encoding {len(image_paths)} images …")
        img_features = load_and_encode_images(
            model, preprocess, image_paths, device, args.batch_size
        )

        # Cosine similarity: (N,D) @ (D,) -> (N,)
        sim1 = (img_features @ feat_text1).cpu().numpy()
        sim2 = (img_features @ feat_text2).cpu().numpy()

        # Zero-shot classification
        zs = zero_shot_classify(sim1, sim2)
        gt_label = get_ground_truth_label(name)  # 0=urban(text1), 1=country(text2)
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
            "pred_text1_count": int((zs["predictions"] == 0).sum()),
            "pred_text2_count": int((zs["predictions"] == 1).sum()),
            "probs_text1": zs["probs_text1"].tolist(),
            "probs_text2": zs["probs_text2"].tolist(),
        }

        gt_name = args.text1 if gt_label == 0 else args.text2
        print(
            f'           sim("{args.text1}"): {sim1.mean():.4f} ± {sim1.std():.4f}  |  '
            f'sim("{args.text2}"): {sim2.mean():.4f} ± {sim2.std():.4f}  |  '
            f"ZS acc: {accuracy:.1%} (GT={gt_name})"
        )

    # ── Summary table ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(
        f"{'Subgroup':<45} {'Text1 mean±std':>18} {'Text2 mean±std':>18} {'ZS Acc':>8} {'N':>6}"
    )
    print("-" * 100)
    total_correct = 0
    total_count = 0
    for name, data in sorted(subgroup_results.items()):
        label = parse_subgroup_label(name)
        gt = data["gt_label"]
        correct = int(data["accuracy"] * data["count"])
        total_correct += correct
        total_count += data["count"]
        print(
            f"{label:<45} "
            f"{data['mean_text1']:>7.4f}±{data['std_text1']:<7.4f}  "
            f"{data['mean_text2']:>7.4f}±{data['std_text2']:<7.4f}  "
            f"{data['accuracy']:>7.1%} "
            f"{data['count']:>5}"
        )
    overall_acc = total_correct / total_count if total_count > 0 else 0
    print("-" * 100)
    print(f"{'OVERALL':<45} {'':>18} {'':>18} {overall_acc:>7.1%} {total_count:>5}")

    # ── Per ground-truth class accuracy ──────────────────────────
    for gt_val, gt_name in [(0, args.text1), (1, args.text2)]:
        sub = {n: d for n, d in subgroup_results.items() if d["gt_label"] == gt_val}
        if sub:
            c = sum(int(d["accuracy"] * d["count"]) for d in sub.values())
            t = sum(d["count"] for d in sub.values())
            print(f'  GT="{gt_name}" subgroups: {c}/{t} = {c / t:.1%}')

    # ── Plot ─────────────────────────────────────────────────────
    print(f"\nGenerating plots in '{args.output_dir}/' …")
    plot_distributions(subgroup_results, args.text1, args.text2, args.output_dir)
    print("Done! Saved:")
    print(
        f"  - {args.output_dir}/hist_<subgroup>.png            (per-subgroup histograms)"
    )
    print(f"  - {args.output_dir}/combined_histograms.png        (grid overview)")
    print(
        f"  - {args.output_dir}/boxplots.png                   (similarity box plots)"
    )
    print(f"  - {args.output_dir}/difference_boxplots.png        (sim difference)")
    print(
        f"  - {args.output_dir}/zeroshot_accuracy.png          (classification accuracy)"
    )
    print(
        f"  - {args.output_dir}/zeroshot_prob_distributions.png (softmax prob distributions)"
    )


if __name__ == "__main__":
    main()
