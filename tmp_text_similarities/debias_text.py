#!/usr/bin/env python3
"""
UrbanCars zero-shot classification with text and image embedding debiasing.

Three classification modes are compared:
  1. RAW:        original text + original image embeddings
  2. TEXT-DEB:   debiased text + original image embeddings
  3. TEXT+IMG-DEB: debiased text + debiased image embeddings

Debiasing = project out the environment direction (e_urban - e_country)
from the embedding and re-normalize, so:
  - For texts:  sim(t_car', e_urban) == sim(t_car', e_country)
  - For images: sim(img', e_urban)   == sim(img', e_country)

Usage:
    python urbancars_similarity.py
    python urbancars_similarity.py --text1 "a compact car" --text2 "a pickup truck"
    python urbancars_similarity.py --env-text1 "a city street" --env-text2 "a rural road"

Requirements:
    pip install torch open-clip-torch Pillow matplotlib numpy
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

import open_clip


# ── Defaults ─────────────────────────────────────────────────────
DEFAULT_DATA_ROOT = "../data/urbancars/bg-0.5_co_occur_obj-0.5/test"
DEFAULT_TEXT1 = "a photograph of a compact, sports, sedan car"
DEFAULT_TEXT2 = "a photograph of a truck, jeep, pickup car"
DEFAULT_ENV_TEXT1 = "a photo of an urban environment"
DEFAULT_ENV_TEXT2 = "a photo of country environment"
DEFAULT_MODEL = "ViT-B-32-quickgelu"
DEFAULT_PRETRAINED = "metaclip_fullcc"
DEFAULT_BATCH_SIZE = 64
DEFAULT_OUTPUT_DIR = "./plots"

ATTRIBUTES = ["urban", "country"]


# ═══════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════


def get_subgroup_dirs(data_root: str) -> dict[str, str]:
    """Enumerate all 8 subgroup directories."""
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
        all_features.append(features.float())
    return F.normalize(torch.cat(all_features, dim=0), dim=-1)


def encode_texts(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    """Encode texts, return normalized features."""
    tokens = tokenizer(texts).to(device)
    with torch.no_grad(), torch.amp.autocast(device):
        features = model.encode_text(tokens)
    return F.normalize(features.float(), dim=-1)


def debias_embedding(v: torch.Tensor, d_hat: torch.Tensor) -> torch.Tensor:
    """
    Project out the environment direction from an embedding (single vector).

    v:     (D,) normalized embedding
    d_hat: (D,) unit environment direction

    Returns (D,) normalized debiased embedding with sim(v', e1) == sim(v', e2).
    """
    projection = (v @ d_hat) * d_hat
    v_deb = v - projection
    return F.normalize(v_deb, dim=0)


def debias_embeddings_batch(V: torch.Tensor, d_hat: torch.Tensor) -> torch.Tensor:
    """
    Project out the environment direction from a batch of embeddings.

    V:     (N, D) normalized embeddings
    d_hat: (D,)   unit environment direction

    Returns (N, D) normalized debiased embeddings.
    Each row v' satisfies: sim(v', e1) == sim(v', e2).
    """
    # (N,) projection scalars
    projections = V @ d_hat  # (N,)
    V_deb = V - 1.5 * projections.unsqueeze(1) * d_hat  # (N, D)
    return F.normalize(V_deb, dim=-1)


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
    """Softmax over two similarities → predictions + probabilities."""
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


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════


def plot_distributions(
    subgroup_results: dict[str, dict],
    text1: str,
    text2: str,
    output_dir: str,
) -> None:
    """Create comparison plots for all 3 modes."""
    os.makedirs(output_dir, exist_ok=True)
    names_sorted = sorted(subgroup_results.keys())
    labels_sorted = [parse_subgroup_label(n) for n in names_sorted]
    n_sub = len(names_sorted)

    modes = [
        ("raw", "Raw", "lightcoral"),
        ("text_deb", "Text-Debiased", "steelblue"),
        ("full_deb", "Text+Image-Debiased", "seagreen"),
    ]

    # ── 1. Histograms grid: 3 columns (raw / text-deb / full-deb) ─
    fig, axes = plt.subplots(n_sub, 3, figsize=(15, 3.5 * n_sub), squeeze=False)
    for idx, name in enumerate(names_sorted):
        data = subgroup_results[name]
        label = parse_subgroup_label(name)
        for col, (mode_key, mode_label, color) in enumerate(modes):
            ax = axes[idx][col]
            s1_key = f"sim_text1_{mode_key}"
            s2_key = f"sim_text2_{mode_key}"
            acc_key = f"accuracy_{mode_key}"
            ax.hist(
                data[s1_key],
                bins=30,
                alpha=0.6,
                label=f'"{text1}"',
                color="steelblue",
                density=True,
            )
            ax.hist(
                data[s2_key],
                bins=30,
                alpha=0.6,
                label=f'"{text2}"',
                color="coral",
                density=True,
            )
            ax.set_title(
                f"{mode_label} — {label}\nAcc={data[acc_key]*100:.1f}%", fontsize=8
            )
            ax.set_xlabel("Cosine Sim", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5)
            ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Similarity Distributions: Raw vs Text-Deb vs Text+Img-Deb", fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(output_dir, "histograms_3modes.png"), dpi=150)
    plt.close(fig)

    # ── 2. Accuracy comparison (grouped bars) ────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))
    x_pos = np.arange(n_sub)
    n_modes = len(modes)
    width = 0.25

    for i, (mode_key, mode_label, color) in enumerate(modes):
        accs = [subgroup_results[n][f"accuracy_{mode_key}"] * 100 for n in names_sorted]
        offset = (i - n_modes / 2 + 0.5) * width
        bars = ax.bar(
            x_pos + offset,
            accs,
            width,
            label=mode_label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )
        for bar, acc in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{acc:.1f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="Chance")
    ax.set_title("Zero-Shot Classification: Raw vs Text-Deb vs Text+Img-Deb")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_3modes.png"), dpi=150)
    plt.close(fig)

    # ── 3. Improvement over raw ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_i, (mode_key, mode_label, color) in enumerate(modes[1:]):
        ax = axes[ax_i]
        imps = [
            (
                subgroup_results[n][f"accuracy_{mode_key}"]
                - subgroup_results[n]["accuracy_raw"]
            )
            * 100
            for n in names_sorted
        ]
        colors = ["green" if imp >= 0 else "red" for imp in imps]
        ax.bar(x_pos, imps, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
        for i, imp in enumerate(imps):
            ax.text(
                i,
                imp + (0.5 if imp >= 0 else -1.5),
                f"{imp:+.1f}",
                ha="center",
                fontsize=7,
                fontweight="bold",
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Accuracy Change (pp)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Improvement: {mode_label} − Raw")
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_improvement.png"), dpi=150)
    plt.close(fig)

    # ── 4. Difference box plots (3 modes) ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax_i, (mode_key, mode_label, color) in enumerate(modes):
        ax = axes[ax_i]
        diffs = [
            np.array(subgroup_results[n][f"sim_text1_{mode_key}"])
            - np.array(subgroup_results[n][f"sim_text2_{mode_key}"])
            for n in names_sorted
        ]
        bp = ax.boxplot(diffs, vert=True, patch_artist=True, labels=labels_sorted)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{mode_label}")
        ax.set_ylabel("sim(text1) − sim(text2)" if ax_i == 0 else "")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Similarity Difference Distributions", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "difference_3modes.png"), dpi=150)
    plt.close(fig)

    # ── 5. Environment prior distribution (how much env bias per subgroup)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), squeeze=False)
    for idx, name in enumerate(names_sorted):
        r, c = divmod(idx, 4)
        ax = axes[r][c]
        data = subgroup_results[name]
        label = parse_subgroup_label(name)

        env_sim1 = np.array(data["sim_env1_raw"])
        env_sim2 = np.array(data["sim_env2_raw"])
        env_diff = env_sim1 - env_sim2

        ax.hist(env_diff, bins=30, alpha=0.7, color="mediumpurple", density=True)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.axvline(
            env_diff.mean(),
            color="red",
            linewidth=1.5,
            label=f"μ={env_diff.mean():.3f}",
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("sim(urban_env) − sim(country_env)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for idx in range(n_sub, 8):
        r, c = divmod(idx, 4)
        axes[r][c].set_visible(False)
    fig.suptitle("Environment Bias per Image (before debiasing)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, "env_bias_distributions.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="UrbanCars ZS classification with text + image embedding debiasing."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--text1", default=DEFAULT_TEXT1, help="Urban car text.")
    parser.add_argument("--text2", default=DEFAULT_TEXT2, help="Country car text.")
    parser.add_argument(
        "--env-text1", default=DEFAULT_ENV_TEXT1, help="Urban environment text."
    )
    parser.add_argument(
        "--env-text2", default=DEFAULT_ENV_TEXT2, help="Country environment text."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Model      : {args.model} / {args.pretrained}")
    print(f"Data root  : {args.data_root}")
    print(f'Text 1     : "{args.text1}"')
    print(f'Text 2     : "{args.text2}"')
    print(f'Env Text 1 : "{args.env_text1}"')
    print(f'Env Text 2 : "{args.env_text2}"')
    print(f"Output dir : {args.output_dir}\n")

    # ── Load model ───────────────────────────────────────────────
    print("Loading model …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    # ── Encode all 4 texts ───────────────────────────────────────
    print("Encoding texts …")
    all_texts = [args.text1, args.text2, args.env_text1, args.env_text2]
    text_features = encode_texts(model, tokenizer, all_texts, device)
    feat_text1 = text_features[0]  # urban car
    feat_text2 = text_features[1]  # country car
    feat_env1 = text_features[2]  # urban environment
    feat_env2 = text_features[3]  # country environment

    # ── Compute environment direction ────────────────────────────
    d_env = feat_env1 - feat_env2  # environment direction
    d_env_hat = d_env / d_env.norm()  # unit vector

    # ── Debias text embeddings ───────────────────────────────────
    print("\nDebiasing text embeddings …")
    print(f"  BEFORE:")
    print(f"    sim(t1_urban_car, e_urban)  = {(feat_text1 @ feat_env1).item():.4f}")
    print(f"    sim(t1_urban_car, e_country)= {(feat_text1 @ feat_env2).item():.4f}")
    print(f"    sim(t2_country_car, e_urban) = {(feat_text2 @ feat_env1).item():.4f}")
    print(f"    sim(t2_country_car, e_country)= {(feat_text2 @ feat_env2).item():.4f}")

    feat_text1_deb = debias_embedding(feat_text1, d_env_hat)
    feat_text2_deb = debias_embedding(feat_text2, d_env_hat)

    print(f"  AFTER:")
    print(f"    sim(t1', e_urban)  = {(feat_text1_deb @ feat_env1).item():.4f}")
    print(f"    sim(t1', e_country)= {(feat_text1_deb @ feat_env2).item():.4f}")
    print(f"    sim(t2', e_urban)  = {(feat_text2_deb @ feat_env1).item():.4f}")
    print(f"    sim(t2', e_country)= {(feat_text2_deb @ feat_env2).item():.4f}")

    diff1 = abs(
        (feat_text1_deb @ feat_env1).item() - (feat_text1_deb @ feat_env2).item()
    )
    diff2 = abs(
        (feat_text2_deb @ feat_env1).item() - (feat_text2_deb @ feat_env2).item()
    )
    print(f"  Text equidistance check (≈0): t1={diff1:.6f}, t2={diff2:.6f}")

    # ── Discover subgroups ───────────────────────────────────────
    print("\nScanning subgroup directories …")
    subgroups = get_subgroup_dirs(args.data_root)
    print(f"Found {len(subgroups)} subgroups.\n")

    if not subgroups:
        print("ERROR: No subgroup directories found.")
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

        # ── Debias image embeddings ──────────────────────────────
        img_features_deb = debias_embeddings_batch(img_features, d_env_hat)

        gt_label = get_ground_truth_label(name)

        # ── Mode 1: RAW (original text, original image) ─────────
        sim1_raw = (img_features @ feat_text1).cpu().numpy()
        sim2_raw = (img_features @ feat_text2).cpu().numpy()
        zs_raw = zero_shot_classify(sim1_raw, sim2_raw)
        acc_raw = float((zs_raw["predictions"] == gt_label).mean())

        # ── Mode 2: TEXT-DEB (debiased text, original image) ─────
        sim1_tdeb = (img_features @ feat_text1_deb).cpu().numpy()
        sim2_tdeb = (img_features @ feat_text2_deb).cpu().numpy()
        zs_tdeb = zero_shot_classify(sim1_tdeb, sim2_tdeb)
        acc_tdeb = float((zs_tdeb["predictions"] == gt_label).mean())

        # ── Mode 3: FULL-DEB (debiased text, debiased image) ────
        sim1_fdeb = (img_features_deb @ feat_text1).cpu().numpy()
        sim2_fdeb = (img_features_deb @ feat_text2).cpu().numpy()
        zs_fdeb = zero_shot_classify(sim1_fdeb, sim2_fdeb)
        acc_fdeb = float((zs_fdeb["predictions"] == gt_label).mean())

        # Environment similarities (for diagnostics)
        sim_env1_raw = (img_features @ feat_env1).cpu().numpy()
        sim_env2_raw = (img_features @ feat_env2).cpu().numpy()
        sim_env1_deb = (img_features_deb @ feat_env1).cpu().numpy()
        sim_env2_deb = (img_features_deb @ feat_env2).cpu().numpy()

        subgroup_results[name] = {
            # Raw
            "sim_text1_raw": sim1_raw.tolist(),
            "sim_text2_raw": sim2_raw.tolist(),
            "accuracy_raw": acc_raw,
            # Text-debiased
            "sim_text1_text_deb": sim1_tdeb.tolist(),
            "sim_text2_text_deb": sim2_tdeb.tolist(),
            "accuracy_text_deb": acc_tdeb,
            # Full-debiased
            "sim_text1_full_deb": sim1_fdeb.tolist(),
            "sim_text2_full_deb": sim2_fdeb.tolist(),
            "accuracy_full_deb": acc_fdeb,
            # Environment (for plots)
            "sim_env1_raw": sim_env1_raw.tolist(),
            "sim_env2_raw": sim_env2_raw.tolist(),
            "sim_env1_deb": sim_env1_deb.tolist(),
            "sim_env2_deb": sim_env2_deb.tolist(),
            # Meta
            "count": len(image_paths),
            "gt_label": gt_label,
        }

        gt_name = args.text1 if gt_label == 0 else args.text2
        env_check = np.abs(sim_env1_deb - sim_env2_deb).max()
        print(
            f"           raw={acc_raw:.1%}  text_deb={acc_tdeb:.1%}  "
            f"full_deb={acc_fdeb:.1%}  "
            f"(GT={gt_name})  img_env_check={env_check:.6f}"
        )

    # ── Summary table ────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(
        f"{'Subgroup':<45} {'Raw':>8} {'Txt-Deb':>8} {'Full-Deb':>9} "
        f"{'Δ(Txt)':>8} {'Δ(Full)':>8} {'N':>6}"
    )
    print("-" * 120)

    totals = {"raw": 0, "text_deb": 0, "full_deb": 0, "count": 0}
    worst = {"raw": 1.0, "text_deb": 1.0, "full_deb": 1.0}

    for name, data in sorted(subgroup_results.items()):
        label = parse_subgroup_label(name)
        n = data["count"]
        r, t, f_ = (
            data["accuracy_raw"],
            data["accuracy_text_deb"],
            data["accuracy_full_deb"],
        )

        totals["raw"] += int(r * n)
        totals["text_deb"] += int(t * n)
        totals["full_deb"] += int(f_ * n)
        totals["count"] += n

        worst["raw"] = min(worst["raw"], r)
        worst["text_deb"] = min(worst["text_deb"], t)
        worst["full_deb"] = min(worst["full_deb"], f_)

        print(
            f"{label:<45} "
            f"{r:>7.1%} "
            f"{t:>7.1%} "
            f"{f_:>8.1%} "
            f"{t - r:>+7.1%} "
            f"{f_ - r:>+7.1%} "
            f"{n:>5}"
        )

    tc = totals["count"]
    o_r = totals["raw"] / tc if tc else 0
    o_t = totals["text_deb"] / tc if tc else 0
    o_f = totals["full_deb"] / tc if tc else 0

    print("-" * 120)
    print(
        f"{'OVERALL':<45} "
        f"{o_r:>7.1%} "
        f"{o_t:>7.1%} "
        f"{o_f:>8.1%} "
        f"{o_t - o_r:>+7.1%} "
        f"{o_f - o_r:>+7.1%} "
        f"{tc:>5}"
    )
    print(
        f"{'WORST-GROUP':<45} "
        f"{worst['raw']:>7.1%} "
        f"{worst['text_deb']:>7.1%} "
        f"{worst['full_deb']:>8.1%} "
        f"{worst['text_deb'] - worst['raw']:>+7.1%} "
        f"{worst['full_deb'] - worst['raw']:>+7.1%}"
    )

    # Per GT-class
    for gt_val, gt_name in [(0, args.text1), (1, args.text2)]:
        sub = {n: d for n, d in subgroup_results.items() if d["gt_label"] == gt_val}
        if sub:
            t = sum(d["count"] for d in sub.values())
            cr = sum(int(d["accuracy_raw"] * d["count"]) for d in sub.values())
            ct = sum(int(d["accuracy_text_deb"] * d["count"]) for d in sub.values())
            cf = sum(int(d["accuracy_full_deb"] * d["count"]) for d in sub.values())
            print(
                f'  GT="{gt_name}": raw={cr/t:.1%}  txt_deb={ct/t:.1%}  full_deb={cf/t:.1%}'
            )

    # ── Plot ─────────────────────────────────────────────────────
    print(f"\nGenerating plots in '{args.output_dir}/' …")
    plot_distributions(subgroup_results, args.text1, args.text2, args.output_dir)
    print("Done! Saved:")
    print(f"  - {args.output_dir}/histograms_3modes.png")
    print(f"  - {args.output_dir}/accuracy_3modes.png")
    print(f"  - {args.output_dir}/accuracy_improvement.png")
    print(f"  - {args.output_dir}/difference_3modes.png")
    print(f"  - {args.output_dir}/env_bias_distributions.png")


if __name__ == "__main__":
    main()
