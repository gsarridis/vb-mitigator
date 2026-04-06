#!/usr/bin/env python3
"""
Comprehensive VLM Robustness Analysis for ACM MM Paper (v3).

Analyzes OpenCLIP models across three datasets:
  - ImageNet: General capability baseline (Top-1 accuracy)
  - UrbanCars: Robustness to spurious correlations (Worst-Group accuracy)
  - CelebA: Robustness to demographic biases (Worst-Group accuracy)

Key metrics:
  - ImageNet Top-1: General visual understanding
  - WG Accuracy: Robustness to distribution shift (higher = better)
  - Gap (Avg - WG): Fairness penalty (lower = more fair)
  - Robustness Gap: ImageNet - WG (lower = better)

NEW in v3: For UrbanCars/CelebA, generates BOTH:
  - WG accuracy plots (absolute robustness)
  - Gap plots (fairness penalty: avg - wg)

Usage:
    python analyze_vlm_robustness_v3.py --imagenet imagenet.csv --urbancars uc.csv --celeba celeba.csv
    python analyze_vlm_robustness_v3.py --imagenet imagenet.csv --urbancars uc.csv  # without CelebA
"""

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kruskal, mannwhitneyu

warnings.filterwarnings("ignore")

try:
    import seaborn as sns

    HAS_SEABORN = True
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
except ImportError:
    HAS_SEABORN = False

try:
    import statsmodels.formula.api as smf

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

COLORS_OBJ = {
    "CLIP": "#3498db",
    "SigLIP": "#2ecc71",
    "SigLIP2": "#27ae60",
    "CoCa": "#e74c3c",
    "CLIPA": "#9b59b6",
    "EVA": "#f39c12",
    "Contrastive": "#3498db",
}

COLORS_ARCH = {
    "CLIP": "#3498db",
    "SigLIP": "#2ecc71",
    "SigLIP2": "#27ae60",
    "CoCa": "#e74c3c",
    "CLIPA": "#9b59b6",
    "EVA": "#f39c12",
    "ConvNeXt": "#e67e22",
    "ResNet": "#c0392b",
    "MobileCLIP": "#1abc9c",
    "MobileCLIP2": "#16a085",
    "ViTamin": "#8e44ad",
    "PE": "#2980b9",
    "NLLB-SigLIP": "#58d68d",
    "NNLB-CLIP": "#5dade2",
    "RoBERTa-CLIP": "#af7ac5",
    "XLM-RoBERTa-CLIP": "#bb8fce",
    "ViT": "#3498db",
    "Other": "#95a5a6",
}

DATASET_SIZES_M = {
    "OpenAI-400m": 400,
    "openai": 400,
    "LAION-400m": 400,
    "laion400m": 400,
    "LAION-2b": 2000,
    "laion2b": 2000,
    "LAION-5b": 5000,
    "LAION-A-900m": 900,
    "DataComp-1b": 1000,
    "datacomp": 1000,
    "DataComp-12.8b": 12800,
    "DataComp-128m": 128,
    "DataComp-13m": 13,
    "CommonPool-12.8b": 12800,
    "CommonPool-1b": 1000,
    "CommonPool-128m": 128,
    "CommonPool-13m": 13,
    "MetaCLIP-400m": 400,
    "metaclip": 400,
    "MetaCLIP-5.4b": 5400,
    "MetaCLIP2-2.5b": 2500,
    "CommonCrawl-2.5b": 2500,
    "DFN-2b": 2000,
    "dfn": 2000,
    "DFN-5b": 5000,
    "DFNDR-2b": 2000,
    "WebLI-10b": 10000,
    "webli": 10000,
    "Merged-2b": 2000,
    "YFCC-15m": 15,
    "CC-12m": 12,
}

VIT_SIZE_ORDER = {
    "T": 0,
    "S": 1,
    "B": 2,
    "L": 3,
    "H": 4,
    "g": 5,
    "G": 6,
    "bigG": 7,
    "e": 8,
}


# ═══════════════════════════════════════════════════════════════════
# Report Writer
# ═══════════════════════════════════════════════════════════════════


class ReportWriter:
    def __init__(self):
        self.sections = []

    def add_header(self, title: str, level: int = 1):
        m = "=" if level == 1 else "-"
        self.sections.extend(
            ["", m * 80, title.upper() if level == 1 else title, m * 80]
        )

    def add_subheader(self, title: str):
        self.sections.extend(["", f"### {title}"])

    def add_line(self, text: str = ""):
        self.sections.append(text)

    def add_kv(self, key: str, value, indent: int = 0):
        p = "  " * indent
        self.sections.append(
            f"{p}{key}: {value:.4f}"
            if isinstance(value, float)
            else f"{p}{key}: {value}"
        )

    def add_table(self, df: pd.DataFrame):
        self.sections.extend(["", df.to_string(), ""])

    def get_report(self) -> str:
        return "\n".join(self.sections)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.get_report())


# ═══════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════


def save_figure(fig, filepath_base):
    """Save figure in PDF, PNG, and pickle formats."""
    fig.savefig(f"{filepath_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{filepath_base}.png", dpi=300, bbox_inches="tight")
    with open(f"{filepath_base}.pkl", "wb") as f:
        pickle.dump(fig, f)


def get_pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return boolean mask for Pareto-optimal points (maximize both x and y)."""
    n = len(x)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if (
                i != j
                and x[j] >= x[i]
                and y[j] >= y[i]
                and (x[j] > x[i] or y[j] > y[i])
            ):
                pareto[i] = False
                break
    return pareto


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════


def load_and_enrich(
    path: str, name: str, report: ReportWriter, is_imagenet: bool = False
) -> pd.DataFrame:
    """Load and enrich a dataset CSV."""
    df = pd.read_csv(path)
    report.add_header(f"DATA LOADING: {name}", level=2)
    report.add_kv("File", path)
    report.add_kv("Models", len(df))

    numeric_cols = [
        "total_params_M",
        "image_size",
        "patch_size",
        "embed_dim",
        "avg_accuracy",
        "worst_group_accuracy",
        "top1_accuracy",
        "top5_accuracy",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if is_imagenet:
        if "top1_accuracy" in df.columns:
            if df["top1_accuracy"].max() > 1:
                df["top1_accuracy"] = df["top1_accuracy"] / 100
            if "top5_accuracy" in df.columns and df["top5_accuracy"].max() > 1:
                df["top5_accuracy"] = df["top5_accuracy"] / 100
    else:
        # Compute gap: avg - wg (lower gap = more fair)
        if "avg_accuracy" in df.columns and "worst_group_accuracy" in df.columns:
            df["accuracy_gap"] = df["avg_accuracy"] - df["worst_group_accuracy"]

    def infer_objective(model_name):
        name = str(model_name)
        if "SigLIP2" in name:
            return "SigLIP2"
        elif "SigLIP" in name:
            return "SigLIP"
        elif "coca" in name.lower():
            return "CoCa"
        elif "CLIPA" in name:
            return "CLIPA"
        elif "EVA" in name:
            return "EVA"
        else:
            return "CLIP"

    df["training_objective"] = df["model_name"].apply(infer_objective)

    def get_data_size(training_data):
        td = str(training_data).lower()
        for key, size in DATASET_SIZES_M.items():
            if key.lower() in td:
                return size
        return 400

    df["training_data_size_M"] = df["training_data"].apply(get_data_size)

    df["patch_size_num"] = pd.to_numeric(df.get("patch_size"), errors="coerce").fillna(
        1
    )
    df["image_size"] = df["image_size"].fillna(224)
    df["num_tokens"] = (df["image_size"] / df["patch_size_num"]) ** 2
    df["log_params"] = np.log10(df["total_params_M"] + 1)

    df["resolution_bucket"] = pd.cut(
        df["image_size"],
        bins=[0, 240, 300, 400, 600],
        labels=["<=240", "241-300", "301-400", ">400"],
    )
    df["params_bucket"] = pd.cut(
        df["total_params_M"],
        bins=[0, 200, 500, 1000, 5000],
        labels=["<200M", "200-500M", "500M-1B", ">1B"],
    )

    if "vit_size" in df.columns:
        df["vit_size_rank"] = df["vit_size"].map(VIT_SIZE_ORDER)

    df["dataset"] = name

    report.add_subheader("Data Summary")
    report.add_kv(
        "Training objectives", df["training_objective"].value_counts().to_dict()
    )
    if "arch_family" in df.columns:
        report.add_kv("Architectures", df["arch_family"].value_counts().to_dict())

    if not is_imagenet and "accuracy_gap" in df.columns:
        report.add_kv(
            "Gap (Avg-WG) range",
            f"{df['accuracy_gap'].min()*100:.2f}pp - {df['accuracy_gap'].max()*100:.2f}pp",
        )
        report.add_kv("Gap (Avg-WG) mean", f"{df['accuracy_gap'].mean()*100:.2f}pp")

    return df


# ═══════════════════════════════════════════════════════════════════
# Statistical Analysis Functions
# ═══════════════════════════════════════════════════════════════════


def analyze_overall_dual(df: pd.DataFrame, name: str, report: ReportWriter):
    """Overall statistics for both WG accuracy and Gap."""
    report.add_header(f"OVERALL STATISTICS: {name}", level=2)

    # WG Accuracy
    if "worst_group_accuracy" in df.columns:
        v = df["worst_group_accuracy"].dropna() * 100
        report.add_subheader("Worst-Group Accuracy (%)")
        for k, fn in [
            ("N", len),
            ("Mean", np.mean),
            ("Std", np.std),
            ("Min", np.min),
            ("Max", np.max),
            ("Median", np.median),
        ]:
            report.add_kv(k, fn(v))

    # Gap
    if "accuracy_gap" in df.columns:
        v = df["accuracy_gap"].dropna() * 100
        report.add_subheader("Accuracy Gap: Avg - WG (pp)")
        for k, fn in [
            ("N", len),
            ("Mean", np.mean),
            ("Std", np.std),
            ("Min", np.min),
            ("Max", np.max),
            ("Median", np.median),
        ]:
            report.add_kv(k, fn(v))


def analyze_categorical_dual(
    df: pd.DataFrame,
    fcol: str,
    flabel: str,
    name: str,
    report: ReportWriter,
    min_n: int = 3,
):
    """Analyze both WG and Gap by categorical factor."""
    report.add_header(f"FACTOR: {flabel} ({name})", level=2)

    if fcol not in df.columns:
        report.add_line(f"Column {fcol} not found.")
        return

    counts = df[fcol].value_counts()
    valid = counts[counts >= min_n].index.tolist()

    if len(valid) < 2:
        report.add_line("Insufficient groups.")
        return

    df_f = df[df[fcol].isin(valid)]

    # WG Accuracy stats
    if "worst_group_accuracy" in df.columns:
        report.add_subheader(f"WG Accuracy by {flabel}")
        g = df_f.groupby(fcol)["worst_group_accuracy"].agg(["mean", "std", "count"])
        g = g.sort_values("mean", ascending=False)
        report.add_line(f"\n{'Level':<30} {'Mean%':>8} {'Std%':>8} {'N':>5}")
        report.add_line("-" * 55)
        for lvl, row in g.iterrows():
            report.add_line(
                f"{str(lvl):<30} {row['mean']*100:>8.2f} {row['std']*100:>8.2f} {int(row['count']):>5}"
            )

    # Gap stats
    if "accuracy_gap" in df.columns:
        report.add_subheader(f"Gap (Avg-WG) by {flabel}")
        g = df_f.groupby(fcol)["accuracy_gap"].agg(["mean", "std", "count"])
        g = g.sort_values("mean", ascending=True)  # Lower gap is better
        report.add_line(f"\n{'Level':<30} {'Mean pp':>8} {'Std pp':>8} {'N':>5}")
        report.add_line("-" * 55)
        for lvl, row in g.iterrows():
            report.add_line(
                f"{str(lvl):<30} {row['mean']*100:>8.2f} {row['std']*100:>8.2f} {int(row['count']):>5}"
            )


def analyze_numeric_dual(
    df: pd.DataFrame, fcol: str, flabel: str, name: str, report: ReportWriter
):
    """Analyze correlation for both WG and Gap."""
    report.add_header(f"NUMERIC: {flabel} ({name})", level=2)

    if fcol not in df.columns:
        report.add_line(f"Column {fcol} not found.")
        return

    # WG Accuracy correlation
    if "worst_group_accuracy" in df.columns:
        mask = df[fcol].notna() & df["worst_group_accuracy"].notna()
        if mask.sum() >= 5:
            x, y = (
                df.loc[mask, fcol].values,
                df.loc[mask, "worst_group_accuracy"].values,
            )
            rs, ps = spearmanr(x, y)
            sig = (
                "***"
                if ps < 0.001
                else "**" if ps < 0.01 else "*" if ps < 0.05 else "ns"
            )
            report.add_line(f"{flabel} → WG Acc: ρ={rs:+.4f}, p={ps:.6f} ({sig})")

    # Gap correlation
    if "accuracy_gap" in df.columns:
        mask = df[fcol].notna() & df["accuracy_gap"].notna()
        if mask.sum() >= 5:
            x, y = df.loc[mask, fcol].values, df.loc[mask, "accuracy_gap"].values
            rs, ps = spearmanr(x, y)
            sig = (
                "***"
                if ps < 0.001
                else "**" if ps < 0.01 else "*" if ps < 0.05 else "ns"
            )
            report.add_line(f"{flabel} → Gap: ρ={rs:+.4f}, p={ps:.6f} ({sig})")


def analyze_top_bottom_dual(
    df: pd.DataFrame, name: str, report: ReportWriter, n: int = 15
):
    """Show top/bottom models for both WG and Gap."""
    report.add_header(f"TOP/BOTTOM MODELS: {name}", level=2)

    cols_base = [
        "model_id",
        "training_objective",
        "arch_family",
        "total_params_M",
        "image_size",
    ]

    # Top by WG (highest = best)
    if "worst_group_accuracy" in df.columns:
        cols = [
            c
            for c in cols_base + ["worst_group_accuracy", "accuracy_gap"]
            if c in df.columns
        ]
        report.add_subheader(f"Top {n} by WG Accuracy (highest)")
        top = df.nlargest(n, "worst_group_accuracy")[cols].copy()
        top["worst_group_accuracy"] = (top["worst_group_accuracy"] * 100).round(2)
        if "accuracy_gap" in top.columns:
            top["accuracy_gap"] = (top["accuracy_gap"] * 100).round(2)
        report.add_table(top)

    # Top by Gap (lowest = best/most fair)
    if "accuracy_gap" in df.columns:
        cols = [
            c
            for c in cols_base + ["accuracy_gap", "worst_group_accuracy"]
            if c in df.columns
        ]
        report.add_subheader(f"Top {n} by Gap (lowest = most fair)")
        top = df.nsmallest(n, "accuracy_gap")[cols].copy()
        top["accuracy_gap"] = (top["accuracy_gap"] * 100).round(2)
        if "worst_group_accuracy" in top.columns:
            top["worst_group_accuracy"] = (top["worst_group_accuracy"] * 100).round(2)
        report.add_table(top)


# ═══════════════════════════════════════════════════════════════════
# Plotting Functions (Dual: WG and Gap)
# ═══════════════════════════════════════════════════════════════════


def plot_landscape_wg_gap(
    df: pd.DataFrame, name: str, out_dir: str, report: ReportWriter
):
    """Scatter plot: WG Accuracy vs Gap (shows tradeoff)."""
    if "worst_group_accuracy" not in df.columns or "accuracy_gap" not in df.columns:
        return

    report.add_header(f"FIGURE: WG vs Gap Landscape ({name})", level=2)

    mask = df["worst_group_accuracy"].notna() & df["accuracy_gap"].notna()
    df_valid = df[mask]

    if len(df_valid) < 5:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    x = df_valid["worst_group_accuracy"].values * 100
    y = df_valid["accuracy_gap"].values * 100

    arch_col = "arch_family" if "arch_family" in df.columns else "training_objective"
    archs = sorted(df_valid[arch_col].dropna().unique())

    for arch in archs:
        m = df_valid[arch_col] == arch
        color = COLORS_ARCH.get(arch, COLORS_OBJ.get(arch, "#95a5a6"))
        ax.scatter(
            x[m.values],
            y[m.values],
            c=color,
            label=f"{arch} (n={m.sum()})",
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidth=0.3,
        )

    # Ideal: high WG, low gap (bottom-right)
    # Highlight best models (Pareto: maximize WG, minimize gap)
    pareto = get_pareto_mask(x, -y)  # Negate gap since we want to minimize it
    ax.scatter(
        x[pareto],
        y[pareto],
        facecolors="none",
        edgecolors="red",
        linewidth=2,
        s=150,
        label=f"Pareto (n={pareto.sum()})",
        zorder=4,
    )

    report.add_line(f"\nPareto-optimal (high WG, low Gap): {pareto.sum()} models")
    for idx in np.where(pareto)[0]:
        model_id = df_valid.iloc[idx]["model_id"]
        ax.annotate(
            model_id[:20],
            (x[idx], y[idx]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
            color="darkred",
            fontweight="bold",
        )
        report.add_line(f"  {model_id}: WG={x[idx]:.1f}%, Gap={y[idx]:.1f}pp")

    ax.set_xlabel("Worst-Group Accuracy (%)")
    ax.set_ylabel("Accuracy Gap: Avg − WG (pp)")
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_wg_vs_gap_{name.lower()}")
    plt.close()


def plot_by_factor_dual(
    df: pd.DataFrame,
    name: str,
    out_dir: str,
    report: ReportWriter,
    factor_col: str,
    factor_label: str,
    min_n: int = 3,
):
    """Bar chart of BOTH WG accuracy AND Gap by categorical factor."""
    if factor_col not in df.columns:
        return

    valid = (
        df.groupby(factor_col).filter(lambda x: len(x) >= min_n)[factor_col].unique()
    )
    df_f = df[df[factor_col].isin(valid)]

    if len(valid) < 2:
        return

    # ─── Plot 1: WG Accuracy ─────────────────────────────────────
    if "worst_group_accuracy" in df.columns:
        report.add_header(f"FIGURE: {factor_label} - WG Accuracy ({name})", level=2)

        stats = df_f.groupby(factor_col)["worst_group_accuracy"].agg(
            ["mean", "std", "count"]
        )
        stats = stats.sort_values("mean", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(5, len(stats) * 0.4)))
        y_pos = np.arange(len(stats))
        colors = [COLORS_ARCH.get(a, COLORS_OBJ.get(a, "#95a5a6")) for a in stats.index]

        ax.barh(
            y_pos,
            stats["mean"] * 100,
            xerr=stats["std"] * 100,
            color=colors,
            edgecolor="black",
            capsize=3,
            alpha=0.8,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{idx} (n={int(row['count'])})" for idx, row in stats.iterrows()]
        )
        ax.set_xlabel("Worst-Group Accuracy (%)")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        save_figure(fig, f"{out_dir}/fig_{factor_col}_wg_{name.lower()}")
        plt.close()

        for idx, row in stats.iterrows():
            report.add_line(
                f"  {idx}: WG={row['mean']*100:.2f}% ± {row['std']*100:.2f}%"
            )

    # ─── Plot 2: Gap ─────────────────────────────────────────────
    if "accuracy_gap" in df.columns:
        report.add_header(f"FIGURE: {factor_label} - Gap ({name})", level=2)

        stats_gap = df_f.groupby(factor_col)["accuracy_gap"].agg(
            ["mean", "std", "count"]
        )
        stats_gap = stats_gap.sort_values("mean", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(5, len(stats_gap) * 0.4)))
        y_pos = np.arange(len(stats_gap))

        gap_vals = stats_gap["mean"].values
        if gap_vals.max() > gap_vals.min():
            colors = plt.cm.RdYlGn_r(
                (gap_vals - gap_vals.min()) / (gap_vals.max() - gap_vals.min())
            )
        else:
            colors = ["#95a5a6"] * len(stats_gap)

        ax.barh(
            y_pos,
            stats_gap["mean"] * 100,
            xerr=stats_gap["std"] * 100,
            color=colors,
            edgecolor="black",
            capsize=3,
            alpha=0.8,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{idx} (n={int(row['count'])})" for idx, row in stats_gap.iterrows()]
        )
        ax.set_xlabel("Accuracy Gap: Avg − WG (pp) [lower = more fair]")
        ax.axvline(0, color="black", linewidth=1)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        save_figure(fig, f"{out_dir}/fig_{factor_col}_gap_{name.lower()}")
        plt.close()

        for idx, row in stats_gap.iterrows():
            report.add_line(
                f"  {idx}: Gap={row['mean']*100:.2f}pp ± {row['std']*100:.2f}pp"
            )


def plot_scale_scatter_dual(
    df: pd.DataFrame,
    name: str,
    out_dir: str,
    report: ReportWriter,
    factor_col: str,
    factor_label: str,
    use_log: bool = False,
):
    """Scatter plot of BOTH WG accuracy AND Gap vs numeric factor."""
    if factor_col not in df.columns:
        return

    arch_col = "arch_family" if "arch_family" in df.columns else "training_objective"

    # ─── Plot 1: WG Accuracy ─────────────────────────────────────
    if "worst_group_accuracy" in df.columns:
        mask = df[factor_col].notna() & df["worst_group_accuracy"].notna()
        if mask.sum() >= 5:
            report.add_header(f"FIGURE: {factor_label} vs WG ({name})", level=2)

            x_all = df.loc[mask, factor_col].values
            y_all = df.loc[mask, "worst_group_accuracy"].values * 100

            fig, ax = plt.subplots(figsize=(8, 6))
            archs = sorted(df.loc[mask, arch_col].dropna().unique())

            for arch in archs:
                m = (df[arch_col] == arch) & mask
                color = COLORS_ARCH.get(arch, COLORS_OBJ.get(arch, "#95a5a6"))
                ax.scatter(
                    df.loc[m, factor_col].values,
                    df.loc[m, "worst_group_accuracy"].values * 100,
                    c=color,
                    label=f"{arch}",
                    alpha=0.6,
                    s=40,
                    edgecolors="white",
                    linewidth=0.3,
                )

            # Trend line
            if use_log and x_all.min() > 0:
                log_x = np.log10(x_all)
                slope, intercept = np.polyfit(log_x, y_all, 1)
                x_line = np.linspace(x_all.min(), x_all.max(), 100)
                y_line = slope * np.log10(x_line) + intercept
            else:
                slope, intercept = np.polyfit(x_all, y_all, 1)
                x_line = np.linspace(x_all.min(), x_all.max(), 100)
                y_line = slope * x_line + intercept

            ax.plot(x_line, y_line, "r-", linewidth=2, alpha=0.7, label="Trend")

            if use_log:
                ax.set_xscale("log")

            r, p = spearmanr(x_all, y_all)
            report.add_line(f"{factor_label} vs WG: ρ={r:.4f}, p={p:.6f}")

            ax.set_xlabel(factor_label)
            ax.set_ylabel("Worst-Group Accuracy (%)")
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fname = factor_col.replace("_", "").replace("M", "").lower()
            save_figure(fig, f"{out_dir}/fig_scale_{fname}_wg_{name.lower()}")
            plt.close()

    # ─── Plot 2: Gap ─────────────────────────────────────────────
    if "accuracy_gap" in df.columns:
        mask = df[factor_col].notna() & df["accuracy_gap"].notna()
        if mask.sum() >= 5:
            report.add_header(f"FIGURE: {factor_label} vs Gap ({name})", level=2)

            x_all = df.loc[mask, factor_col].values
            y_all = df.loc[mask, "accuracy_gap"].values * 100

            fig, ax = plt.subplots(figsize=(8, 6))
            archs = sorted(df.loc[mask, arch_col].dropna().unique())

            for arch in archs:
                m = (df[arch_col] == arch) & mask
                color = COLORS_ARCH.get(arch, COLORS_OBJ.get(arch, "#95a5a6"))
                ax.scatter(
                    df.loc[m, factor_col].values,
                    df.loc[m, "accuracy_gap"].values * 100,
                    c=color,
                    label=f"{arch}",
                    alpha=0.6,
                    s=40,
                    edgecolors="white",
                    linewidth=0.3,
                )

            # Trend line
            if use_log and x_all.min() > 0:
                log_x = np.log10(x_all)
                slope, intercept = np.polyfit(log_x, y_all, 1)
                x_line = np.linspace(x_all.min(), x_all.max(), 100)
                y_line = slope * np.log10(x_line) + intercept
            else:
                slope, intercept = np.polyfit(x_all, y_all, 1)
                x_line = np.linspace(x_all.min(), x_all.max(), 100)
                y_line = slope * x_line + intercept

            ax.plot(x_line, y_line, "r-", linewidth=2, alpha=0.7, label="Trend")

            if use_log:
                ax.set_xscale("log")

            r, p = spearmanr(x_all, y_all)
            report.add_line(f"{factor_label} vs Gap: ρ={r:.4f}, p={p:.6f}")

            ax.set_xlabel(factor_label)
            ax.set_ylabel("Accuracy Gap: Avg − WG (pp)")
            ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fname = factor_col.replace("_", "").replace("M", "").lower()
            save_figure(fig, f"{out_dir}/fig_scale_{fname}_gap_{name.lower()}")
            plt.close()


def plot_scale_bars_dual(
    df: pd.DataFrame,
    name: str,
    out_dir: str,
    report: ReportWriter,
    factor_col: str,
    factor_label: str,
    n_bins: int = 6,
    use_log: bool = False,
):
    """Bar chart of BOTH WG accuracy AND Gap by binned numeric factor."""
    if factor_col not in df.columns:
        return

    # ─── Plot 1: WG Accuracy ─────────────────────────────────────
    if "worst_group_accuracy" in df.columns:
        mask = df[factor_col].notna() & df["worst_group_accuracy"].notna()
        if mask.sum() >= 10:
            x_all = df.loc[mask, factor_col].values

            if use_log and x_all.min() > 0:
                bin_edges = np.logspace(
                    np.log10(x_all.min()), np.log10(x_all.max()), n_bins
                )
            else:
                bin_edges = np.linspace(x_all.min(), x_all.max(), n_bins)

            df_temp = df.loc[mask].copy()
            df_temp["bin"] = pd.cut(
                df_temp[factor_col], bins=bin_edges, include_lowest=True
            )

            bin_stats = df_temp.groupby("bin", observed=True)[
                "worst_group_accuracy"
            ].agg(["mean", "std", "count"])
            bin_stats = bin_stats[bin_stats["count"] >= 2]

            if len(bin_stats) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))

                bar_labels = [
                    f"{interval.left:.0f}-{interval.right:.0f}"
                    for interval in bin_stats.index
                ]
                x_pos = np.arange(len(bin_stats))
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bin_stats)))

                bars = ax.bar(
                    x_pos,
                    bin_stats["mean"] * 100,
                    yerr=bin_stats["std"] * 100,
                    color=colors,
                    edgecolor="black",
                    linewidth=0.5,
                    capsize=4,
                    alpha=0.8,
                )

                for i, (bar, count) in enumerate(zip(bars, bin_stats["count"])):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + bin_stats["std"].iloc[i] * 100 + 1,
                        f"n={int(count)}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                ax.set_xticks(x_pos)
                ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=9)
                ax.set_xlabel(factor_label)
                ax.set_ylabel("Worst-Group Accuracy (%)")
                ax.grid(True, alpha=0.3, axis="y")

                plt.tight_layout()
                fname = factor_col.replace("_", "").replace("M", "").lower()
                save_figure(fig, f"{out_dir}/fig_scale_{fname}_bar_wg_{name.lower()}")
                plt.close()

    # ─── Plot 2: Gap ─────────────────────────────────────────────
    if "accuracy_gap" in df.columns:
        mask = df[factor_col].notna() & df["accuracy_gap"].notna()
        if mask.sum() >= 10:
            x_all = df.loc[mask, factor_col].values

            if use_log and x_all.min() > 0:
                bin_edges = np.logspace(
                    np.log10(x_all.min()), np.log10(x_all.max()), n_bins
                )
            else:
                bin_edges = np.linspace(x_all.min(), x_all.max(), n_bins)

            df_temp = df.loc[mask].copy()
            df_temp["bin"] = pd.cut(
                df_temp[factor_col], bins=bin_edges, include_lowest=True
            )

            bin_stats = df_temp.groupby("bin", observed=True)["accuracy_gap"].agg(
                ["mean", "std", "count"]
            )
            bin_stats = bin_stats[bin_stats["count"] >= 2]

            if len(bin_stats) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))

                bar_labels = [
                    f"{interval.left:.0f}-{interval.right:.0f}"
                    for interval in bin_stats.index
                ]
                x_pos = np.arange(len(bin_stats))

                gap_vals = bin_stats["mean"].values
                if gap_vals.max() > gap_vals.min():
                    colors = plt.cm.RdYlGn_r(
                        (gap_vals - gap_vals.min()) / (gap_vals.max() - gap_vals.min())
                    )
                else:
                    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bin_stats)))

                bars = ax.bar(
                    x_pos,
                    bin_stats["mean"] * 100,
                    yerr=bin_stats["std"] * 100,
                    color=colors,
                    edgecolor="black",
                    linewidth=0.5,
                    capsize=4,
                    alpha=0.8,
                )

                for i, (bar, count) in enumerate(zip(bars, bin_stats["count"])):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + bin_stats["std"].iloc[i] * 100 + 0.5,
                        f"n={int(count)}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                ax.set_xticks(x_pos)
                ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=9)
                ax.set_xlabel(factor_label)
                ax.set_ylabel("Accuracy Gap: Avg − WG (pp)")
                ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
                ax.grid(True, alpha=0.3, axis="y")

                plt.tight_layout()
                fname = factor_col.replace("_", "").replace("M", "").lower()
                save_figure(fig, f"{out_dir}/fig_scale_{fname}_bar_gap_{name.lower()}")
                plt.close()


def plot_correlation_heatmap_dual(
    df: pd.DataFrame, name: str, out_dir: str, report: ReportWriter
):
    """Correlation heatmap for BOTH WG accuracy AND Gap."""
    report.add_header(f"FIGURE: Correlations ({name})", level=2)

    fcols = [
        c
        for c in [
            "total_params_M",
            "image_size",
            "patch_size_num",
            "num_tokens",
            "embed_dim",
            "training_data_size_M",
        ]
        if c in df.columns
    ]

    metrics = []
    metric_labels = []
    if "worst_group_accuracy" in df.columns:
        metrics.append("worst_group_accuracy")
        metric_labels.append("WG Acc")
    if "accuracy_gap" in df.columns:
        metrics.append("accuracy_gap")
        metric_labels.append("Gap")

    if not metrics or not fcols:
        report.add_line("Insufficient data.")
        return

    df_num = df[fcols + metrics].dropna()
    if len(df_num) < 10:
        report.add_line("Insufficient data.")
        return

    corr = pd.DataFrame(index=fcols, columns=metric_labels, dtype=float)

    report.add_line(f"\n{'Factor':<25} {'WG Acc ρ':>12} {'Gap ρ':>12}")
    report.add_line("-" * 55)

    for f in fcols:
        for metric, label in zip(metrics, metric_labels):
            r, p = spearmanr(df_num[f], df_num[metric])
            corr.loc[f, label] = r

        r_wg = corr.loc[f, "WG Acc"] if "WG Acc" in corr.columns else np.nan
        r_gap = corr.loc[f, "Gap"] if "Gap" in corr.columns else np.nan
        report.add_line(f"{f:<25} {r_wg:>+12.4f} {r_gap:>+12.4f}")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6, 8))
    if HAS_SEABORN:
        annot = corr.apply(
            lambda col: col.map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        )
        sns.heatmap(
            corr.astype(float),
            annot=annot,
            fmt="",
            cmap="RdBu_r",
            center=0,
            vmin=-0.5,
            vmax=0.5,
            ax=ax,
            linewidths=0.5,
        )
    ax.set_title(f"Factor Correlations ({name})")
    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_corr_{name.lower()}")
    plt.close()
    corr.to_csv(f"{out_dir}/corr_{name.lower()}.csv")


# ═══════════════════════════════════════════════════════════════════
# Cross-Dataset Analysis
# ═══════════════════════════════════════════════════════════════════


def compute_robustness_metrics(
    df_imagenet: pd.DataFrame,
    df_bias: pd.DataFrame,
    bias_name: str,
    report: ReportWriter,
) -> pd.DataFrame:
    """Compute robustness gap and ratio metrics."""
    report.add_header(f"ROBUSTNESS METRICS: ImageNet vs {bias_name}", level=2)

    merge_cols = [
        "model_id",
        "model_name",
        "training_objective",
        "arch_family",
        "total_params_M",
        "image_size",
        "training_data",
        "top1_accuracy",
    ]
    merge_cols = [c for c in merge_cols if c in df_imagenet.columns]

    merged = df_imagenet[merge_cols].merge(
        df_bias[["model_id", "worst_group_accuracy", "avg_accuracy", "accuracy_gap"]],
        on="model_id",
    )

    report.add_kv("Models in ImageNet", len(df_imagenet))
    report.add_kv(f"Models in {bias_name}", len(df_bias))
    report.add_kv("Overlapping models", len(merged))

    if len(merged) < 10:
        report.add_line("Insufficient overlap.")
        return merged

    merged["imagenet_acc"] = merged["top1_accuracy"]
    merged["wg_acc"] = merged["worst_group_accuracy"]
    merged["robustness_gap"] = merged["imagenet_acc"] - merged["wg_acc"]
    merged["robustness_ratio"] = merged["wg_acc"] / merged["imagenet_acc"].replace(
        0, np.nan
    )

    report.add_subheader("Statistics")
    report.add_kv(
        "Mean ImageNet-WG gap", f"{merged['robustness_gap'].mean()*100:.2f}pp"
    )
    report.add_kv("Mean Avg-WG gap", f"{merged['accuracy_gap'].mean()*100:.2f}pp")

    r, p = spearmanr(merged["imagenet_acc"], merged["wg_acc"])
    report.add_line(f"ImageNet vs WG correlation: ρ={r:.4f}, p={p:.6f}")

    return merged


def plot_imagenet_vs_wg(
    merged: pd.DataFrame, bias_name: str, out_dir: str, report: ReportWriter
):
    """Scatter plot of ImageNet vs WG accuracy."""
    if len(merged) < 5:
        return

    report.add_header(f"FIGURE: ImageNet vs {bias_name} WG", level=2)

    fig, ax = plt.subplots(figsize=(10, 8))
    x = merged["imagenet_acc"].values * 100
    y = merged["wg_acc"].values * 100

    color_col = (
        "arch_family" if "arch_family" in merged.columns else "training_objective"
    )
    for cat in sorted(merged[color_col].dropna().unique()):
        m = merged[color_col] == cat
        color = COLORS_ARCH.get(cat, COLORS_OBJ.get(cat, "#95a5a6"))
        ax.scatter(
            x[m.values],
            y[m.values],
            c=color,
            label=f"{cat}",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.3,
        )

    lims = [min(x.min(), y.min()) - 5, max(x.max(), y.max()) + 5]
    ax.plot(lims, lims, "k--", alpha=0.3, label="No gap")

    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(
        np.linspace(x.min(), x.max(), 100),
        slope * np.linspace(x.min(), x.max(), 100) + intercept,
        "r-",
        linewidth=2,
        alpha=0.7,
        label="Trend",
    )

    pareto = get_pareto_mask(x, y)
    ax.scatter(
        x[pareto],
        y[pareto],
        facecolors="none",
        edgecolors="darkred",
        linewidth=2,
        s=150,
        label=f"Pareto (n={pareto.sum()})",
        zorder=4,
    )

    r, p = spearmanr(x, y)
    report.add_line(f"Correlation: ρ={r:.4f}, p={p:.6f}")

    ax.set_xlabel("ImageNet Top-1 Accuracy (%)")
    ax.set_ylabel(f"{bias_name} Worst-Group Accuracy (%)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_imagenet_vs_wg_{bias_name.lower()}")
    plt.close()


def plot_imagenet_vs_gap(
    merged: pd.DataFrame, bias_name: str, out_dir: str, report: ReportWriter
):
    """Scatter plot of ImageNet vs Accuracy Gap."""
    if len(merged) < 5 or "accuracy_gap" not in merged.columns:
        return

    report.add_header(f"FIGURE: ImageNet vs {bias_name} Gap", level=2)

    fig, ax = plt.subplots(figsize=(10, 8))
    x = merged["imagenet_acc"].values * 100
    y = merged["accuracy_gap"].values * 100

    color_col = (
        "arch_family" if "arch_family" in merged.columns else "training_objective"
    )
    for cat in sorted(merged[color_col].dropna().unique()):
        m = merged[color_col] == cat
        color = COLORS_ARCH.get(cat, COLORS_OBJ.get(cat, "#95a5a6"))
        ax.scatter(
            x[m.values],
            y[m.values],
            c=color,
            label=f"{cat}",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.3,
        )

    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(
        np.linspace(x.min(), x.max(), 100),
        slope * np.linspace(x.min(), x.max(), 100) + intercept,
        "r-",
        linewidth=2,
        alpha=0.7,
        label="Trend",
    )

    r, p = spearmanr(x, y)
    report.add_line(f"ImageNet vs Gap correlation: ρ={r:.4f}, p={p:.6f}")

    ax.set_xlabel("ImageNet Top-1 Accuracy (%)")
    ax.set_ylabel(f"{bias_name} Accuracy Gap: Avg − WG (pp)")
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_imagenet_vs_gap_{bias_name.lower()}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Multivariate Analysis
# ═══════════════════════════════════════════════════════════════════


def run_multivariate_regression_dual(
    df: pd.DataFrame, name: str, out_dir: str, report: ReportWriter
):
    """Multiple OLS regression for BOTH WG accuracy AND Gap."""
    report.add_header(f"MULTIVARIATE REGRESSION: {name}", level=2)

    if not HAS_STATSMODELS:
        report.add_line("statsmodels not available.")
        return

    df_r = df.copy()

    # Architecture dummies
    def get_arch_group(arch):
        arch = str(arch)
        if arch in ["SigLIP", "SigLIP2"]:
            return "SigLIP"
        elif arch == "CoCa":
            return "CoCa"
        elif arch == "EVA":
            return "EVA"
        else:
            return "CLIP"

    df_r["arch_group"] = (
        df_r["arch_family"].apply(get_arch_group)
        if "arch_family" in df_r.columns
        else "CLIP"
    )
    df_r["is_siglip"] = (df_r["arch_group"] == "SigLIP").astype(int)
    df_r["is_coca"] = (df_r["arch_group"] == "CoCa").astype(int)

    # Dataset dummies
    def get_data_group(data):
        data = str(data).lower()
        if "webli" in data:
            return "WebLI"
        elif "datacomp" in data:
            return "DataComp"
        elif "dfn" in data:
            return "DFN"
        else:
            return "LAION"

    df_r["data_group"] = df_r["training_data"].apply(get_data_group)
    df_r["is_webli"] = (df_r["data_group"] == "WebLI").astype(int)
    df_r["is_datacomp"] = (df_r["data_group"] == "DataComp").astype(int)

    # Numeric variables (standardized)
    df_r["log_params"] = np.log10(df_r["total_params_M"].clip(lower=1))
    df_r["log_datasize"] = np.log10(df_r["training_data_size_M"].clip(lower=1))
    df_r["log_patchsize"] = np.log2(df_r["patch_size_num"].clip(lower=1))
    df_r["log_imgsize"] = np.log2(df_r["image_size"].clip(lower=1))

    for col in ["log_params", "log_datasize", "log_patchsize", "log_imgsize"]:
        mean, std = df_r[col].mean(), df_r[col].std()
        df_r[f"{col}_z"] = (df_r[col] - mean) / std if std > 0 else 0

    # Run regression for both metrics
    for metric, label in [
        ("worst_group_accuracy", "WG Accuracy"),
        ("accuracy_gap", "Gap"),
    ]:
        if metric not in df_r.columns:
            continue

        report.add_subheader(f"Regression: {label}")

        df_r["y"] = df_r[metric] * 100
        df_reg = df_r.dropna(
            subset=[
                "y",
                "is_siglip",
                "log_params_z",
                "log_datasize_z",
                "log_patchsize_z",
                "log_imgsize_z",
            ]
        )

        if len(df_reg) < 50:
            report.add_line("Insufficient data.")
            continue

        formula = "y ~ is_siglip + is_coca + is_webli + is_datacomp + log_params_z + log_datasize_z + log_patchsize_z + log_imgsize_z"

        try:
            model = smf.ols(formula, data=df_reg).fit()
            report.add_kv("R²", model.rsquared)
            report.add_kv("Adj R²", model.rsquared_adj)
            report.add_kv("N", len(df_reg))

            report.add_line(f"\n{'Variable':<20} {'Coef':>10} {'p':>12} {'Sig':>5}")
            report.add_line("-" * 50)

            for var in model.params.index:
                c = model.params[var]
                p = model.pvalues[var]
                sig = (
                    "***"
                    if p < 0.001
                    else "**" if p < 0.01 else "*" if p < 0.05 else ""
                )
                report.add_line(f"{var:<20} {c:>+10.3f} {p:>12.4f} {sig:>5}")

        except Exception as e:
            report.add_line(f"Regression failed: {e}")


def compute_variance_partitioning_dual(
    df: pd.DataFrame, name: str, out_dir: str, report: ReportWriter
):
    """Variance partitioning for BOTH WG accuracy AND Gap."""
    report.add_header(f"VARIANCE PARTITIONING: {name}", level=2)

    if not HAS_STATSMODELS:
        report.add_line("statsmodels not available.")
        return

    df_r = df.copy()
    df_r["is_siglip"] = (
        df_r["arch_family"].isin(["SigLIP", "SigLIP2"]).astype(int)
        if "arch_family" in df_r.columns
        else 0
    )
    df_r["log_params"] = np.log10(df_r["total_params_M"].clip(lower=1))
    df_r["log_imgsize"] = np.log2(df_r["image_size"].clip(lower=1))
    df_r["log_datasize"] = np.log10(df_r["training_data_size_M"].clip(lower=1))

    df_r = df_r.dropna(
        subset=["is_siglip", "log_params", "log_imgsize", "log_datasize"]
    )

    if len(df_r) < 50:
        report.add_line("Insufficient data.")
        return

    for metric, label in [
        ("worst_group_accuracy", "WG Accuracy"),
        ("accuracy_gap", "Gap"),
    ]:
        if metric not in df_r.columns:
            continue

        report.add_subheader(f"Variance Partitioning: {label}")
        df_r["y"] = df_r[metric] * 100

        formulas = {
            "arch": "y ~ is_siglip",
            "scale": "y ~ log_params + log_imgsize",
            "data": "y ~ log_datasize",
            "arch_scale": "y ~ is_siglip + log_params + log_imgsize",
            "arch_data": "y ~ is_siglip + log_datasize",
            "scale_data": "y ~ log_params + log_imgsize + log_datasize",
            "full": "y ~ is_siglip + log_params + log_imgsize + log_datasize",
        }

        r2 = {}
        for key, formula in formulas.items():
            try:
                r2[key] = smf.ols(formula, data=df_r).fit().rsquared
            except:
                r2[key] = 0

        unique_arch = r2["full"] - r2["scale_data"]
        unique_scale = r2["full"] - r2["arch_data"]
        unique_data = r2["full"] - r2["arch_scale"]
        shared = r2["full"] - unique_arch - unique_scale - unique_data

        report.add_line(f"\n{'Factor':<20} {'Unique R²':>12} {'% of Total':>12}")
        report.add_line("-" * 50)
        for factor, unique in [
            ("Architecture", unique_arch),
            ("Scale", unique_scale),
            ("Data", unique_data),
            ("Shared", shared),
        ]:
            pct = (unique / r2["full"] * 100) if r2["full"] > 0 else 0
            report.add_line(f"{factor:<20} {unique:>12.4f} {pct:>11.1f}%")
        report.add_line(f"{'TOTAL':<20} {r2['full']:>12.4f}")


# ═══════════════════════════════════════════════════════════════════
# Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════


def analyze_bias_dataset(
    df: pd.DataFrame, name: str, out_dir: str, report: ReportWriter, min_n: int = 3
):
    """Full analysis for a bias dataset (UrbanCars/CelebA) - generates both WG and Gap plots."""
    report.add_header(f"ANALYSIS: {name}", level=1)

    # Overall statistics
    analyze_overall_dual(df, name, report)

    # Categorical factors
    for fcol, flabel in [
        ("training_objective", "Training Objective"),
        ("training_data", "Training Data"),
        ("arch_family", "Architecture"),
        ("resolution_bucket", "Resolution"),
        ("params_bucket", "Params Bucket"),
    ]:
        if fcol in df.columns:
            analyze_categorical_dual(df, fcol, flabel, name, report, min_n)

    # Numeric factors
    for fcol, flabel in [
        ("total_params_M", "Parameters (M)"),
        ("image_size", "Image Size"),
        ("patch_size_num", "Patch Size"),
        ("num_tokens", "Tokens"),
        ("training_data_size_M", "Data Size (M)"),
    ]:
        if fcol in df.columns:
            analyze_numeric_dual(df, fcol, flabel, name, report)

    # Top/bottom models
    analyze_top_bottom_dual(df, name, report)

    # ─── PLOTS: Both WG and Gap ──────────────────────────────────

    # Landscape: WG vs Gap
    plot_landscape_wg_gap(df, name, out_dir, report)

    # Categorical factor plots (dual)
    for fcol, flabel in [
        ("training_objective", "objective"),
        ("arch_family", "arch"),
        ("training_data", "data"),
    ]:
        if fcol in df.columns:
            plot_by_factor_dual(df, name, out_dir, report, fcol, flabel, min_n)

    # Scale scatter plots (dual)
    scale_factors = [
        ("total_params_M", "Parameters (M)", True),
        ("image_size", "Image Size (px)", False),
        ("patch_size_num", "Patch Size (px)", False),
        ("num_tokens", "Number of Tokens", True),
        ("training_data_size_M", "Training Data Size (M)", True),
    ]

    for fcol, flabel, use_log in scale_factors:
        if fcol in df.columns:
            plot_scale_scatter_dual(df, name, out_dir, report, fcol, flabel, use_log)
            n_bins = 8 if fcol == "num_tokens" else 6
            plot_scale_bars_dual(
                df, name, out_dir, report, fcol, flabel, n_bins, use_log
            )

    # Correlation heatmap (dual)
    plot_correlation_heatmap_dual(df, name, out_dir, report)

    # Multivariate analysis (dual)
    run_multivariate_regression_dual(df, name, out_dir, report)
    compute_variance_partitioning_dual(df, name, out_dir, report)


def analyze_imagenet(
    df: pd.DataFrame, out_dir: str, report: ReportWriter, min_n: int = 3
):
    """Analysis for ImageNet (Top-1 only, no gap)."""
    report.add_header("ANALYSIS: ImageNet", level=1)

    metric_col = "top1_accuracy"
    metric_label = "Top-1 Accuracy"

    if metric_col not in df.columns:
        report.add_line("top1_accuracy column not found.")
        return

    # Statistics
    v = df[metric_col].dropna() * 100
    report.add_subheader(metric_label)
    for k, fn in [
        ("N", len),
        ("Mean", np.mean),
        ("Std", np.std),
        ("Min", np.min),
        ("Max", np.max),
    ]:
        report.add_kv(k, fn(v))

    # Top models
    report.add_subheader("Top 15 Models")
    cols = [
        "model_id",
        "training_objective",
        "arch_family",
        "total_params_M",
        "image_size",
        metric_col,
    ]
    cols = [c for c in cols if c in df.columns]
    top = df.nlargest(15, metric_col)[cols].copy()
    top[metric_col] = (top[metric_col] * 100).round(2)
    report.add_table(top)


def main():
    parser = argparse.ArgumentParser(
        description="VLM Robustness Analysis v3 (Dual WG + Gap)"
    )
    parser.add_argument("--imagenet", required=True, help="ImageNet results CSV")
    parser.add_argument("--urbancars", required=True, help="UrbanCars results CSV")
    parser.add_argument("--celeba", default=None, help="CelebA results CSV (optional)")
    parser.add_argument(
        "--output", default="paper_analysis_v3", help="Output directory"
    )
    parser.add_argument("--min-n", type=int, default=3, help="Min samples per group")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    report = ReportWriter()

    report.add_header("VLM ROBUSTNESS ANALYSIS v3 - DUAL WG + GAP", level=1)
    report.add_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.add_line(f"Output: {args.output}")
    report.add_line(
        "\nThis version generates BOTH WG accuracy and Gap plots for all factors."
    )

    # Load datasets
    print("Loading ImageNet...")
    df_imagenet = load_and_enrich(args.imagenet, "ImageNet", report, is_imagenet=True)
    print(f"  Loaded {len(df_imagenet)} models")

    print("Loading UrbanCars...")
    df_uc = load_and_enrich(args.urbancars, "UrbanCars", report, is_imagenet=False)
    print(f"  Loaded {len(df_uc)} models")

    df_celeba = None
    if args.celeba:
        print("Loading CelebA...")
        df_celeba = load_and_enrich(args.celeba, "CelebA", report, is_imagenet=False)
        print(f"  Loaded {len(df_celeba)} models")

    # Analyze datasets
    print("\nAnalyzing ImageNet...")
    analyze_imagenet(df_imagenet, args.output, report, args.min_n)

    print("Analyzing UrbanCars (WG + Gap)...")
    analyze_bias_dataset(df_uc, "UrbanCars", args.output, report, args.min_n)

    if df_celeba is not None:
        print("Analyzing CelebA (WG + Gap)...")
        analyze_bias_dataset(df_celeba, "CelebA", args.output, report, args.min_n)

    # Cross-dataset analysis
    print("\nCross-dataset analysis...")
    merged_uc = compute_robustness_metrics(df_imagenet, df_uc, "UrbanCars", report)
    if len(merged_uc) >= 10:
        plot_imagenet_vs_wg(merged_uc, "UrbanCars", args.output, report)
        plot_imagenet_vs_gap(merged_uc, "UrbanCars", args.output, report)
        merged_uc.to_csv(f"{args.output}/merged_imagenet_urbancars.csv", index=False)

    if df_celeba is not None:
        merged_ca = compute_robustness_metrics(df_imagenet, df_celeba, "CelebA", report)
        if len(merged_ca) >= 10:
            plot_imagenet_vs_wg(merged_ca, "CelebA", args.output, report)
            plot_imagenet_vs_gap(merged_ca, "CelebA", args.output, report)
            merged_ca.to_csv(f"{args.output}/merged_imagenet_celeba.csv", index=False)

    # Save report
    report_path = f"{args.output}/complete_analysis_report.txt"
    report.save(report_path)

    # Summary
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {args.output}/")

    # List generated files
    files = sorted(os.listdir(args.output))
    wg_files = [f for f in files if "_wg_" in f]
    gap_files = [f for f in files if "_gap_" in f]
    print(f"\nWG accuracy plots: {len(wg_files)}")
    print(f"Gap plots: {len(gap_files)}")
    print(f"Total files: {len(files)}")

    print(f"\n*** Report: {report_path}")


if __name__ == "__main__":
    main()
