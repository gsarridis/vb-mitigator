#!/usr/bin/env python3
"""
Comprehensive VLM Robustness Analysis for ACM MM Paper.

Analyzes OpenCLIP models across three datasets:
  - ImageNet: General capability baseline (Top-1 accuracy)
  - UrbanCars: Robustness to spurious correlations (Worst-Group accuracy)
  - CelebA: Robustness to demographic biases (Worst-Group accuracy)

Key metrics:
  - ImageNet Top-1: General visual understanding
  - WG Accuracy: Robustness to distribution shift
  - Robustness Gap: ImageNet - WG (lower = better)
  - Robustness Ratio: WG / ImageNet (higher = better)

Usage:
    python analyze_vlm_robustness_v2.py --imagenet imagenet.csv --urbancars uc.csv --celeba celeba.csv
    python analyze_vlm_robustness_v2.py --imagenet imagenet.csv --urbancars uc.csv  # without CelebA
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

# ═══════════════════════════════════════════════════════════════════
# Global Font Size Configuration
# ═══════════════════════════════════════════════════════════════════

FONT_SIZE_AXIS_LABEL = 16  # x-axis and y-axis labels
FONT_SIZE_AXIS_TICK = 16  # axis tick values / tick labels
FONT_SIZE_TITLE = 0  # figure / subplot titles
FONT_SIZE_LEGEND = 16  # legend text
FONT_SIZE_ANNOTATION = 16  # in-plot annotations (e.g. Pareto labels)
FONT_SIZE_BAR_LABEL = 16  # labels drawn on/above bars (count, value)
FONT_SIZE_COLORBAR = 16  # colorbar tick labels and label
FONT_SIZE_SUPTITLE = 0  # figure-level suptitle
FONT_SIZE_HEATMAP_ANNOT = 16  # seaborn heatmap cell annotations

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": FONT_SIZE_AXIS_TICK,
        "axes.labelsize": FONT_SIZE_AXIS_LABEL,
        "axes.titlesize": FONT_SIZE_TITLE,
        "xtick.labelsize": FONT_SIZE_AXIS_TICK,
        "ytick.labelsize": FONT_SIZE_AXIS_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "figure.titlesize": FONT_SIZE_SUPTITLE,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

# ── Global bar colour palette ────────────────────────────────────────────────
# All bar/barh charts use viridis (same as fig_scale_trainingdatasize_bar_*).
# Call _bar_colors(n) to get n evenly-spaced colours from the ramp.
BAR_CMAP = plt.cm.viridis


def _bar_colors(n: int, vmin: float = 0.15, vmax: float = 0.85):
    """Return *n* colours sampled from BAR_CMAP (viridis by default)."""
    return [BAR_CMAP(v) for v in np.linspace(vmin, vmax, max(n, 1))]


# ── Legend placement helper ──────────────────────────────────────────────────
LEGEND_OUTSIDE_THRESHOLD = 6  # move legend outside when items exceed this


def _place_legend(ax, handles=None, labels=None, **legend_kwargs):
    """Place legend inside axes normally, or outside-right when items > threshold.

    Usage:  _place_legend(ax)
            _place_legend(ax, loc="lower right")   # hint used when staying inside
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    n_items = len(handles)
    fs = legend_kwargs.pop("fontsize", FONT_SIZE_LEGEND)
    inside_loc = legend_kwargs.pop("loc", "best")  # always consume loc first
    if n_items > LEGEND_OUTSIDE_THRESHOLD:
        ax.legend(
            handles,
            labels,
            fontsize=fs,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            **legend_kwargs,
        )
    else:
        ax.legend(handles, labels, fontsize=fs, loc=inside_loc, **legend_kwargs)


# Colors for training objectives
COLORS_OBJ = {
    "CLIP": "#3498db",
    "SigLIP": "#2ecc71",
    "SigLIP2": "#27ae60",
    "CoCa": "#e74c3c",
    "CLIPA": "#9b59b6",
    "EVA": "#f39c12",
    "Contrastive": "#3498db",  # Alias for CLIP
}

# Colors for architecture families
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

# Training data sizes in millions
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


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled = np.sqrt(((n1 - 1) * g1.var() + (n2 - 1) * g2.var()) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 0 else 0


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

    # Ensure numeric columns
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

    # Normalize metric column name
    if is_imagenet:
        # ImageNet uses top1_accuracy as primary metric
        if "top1_accuracy" in df.columns:
            # Ensure it's in 0-1 scale like other datasets
            if df["top1_accuracy"].max() > 1:
                df["top1_accuracy"] = df["top1_accuracy"] / 100
            df["primary_metric"] = df["top1_accuracy"]
            df["secondary_metric"] = df.get("top5_accuracy", df["top1_accuracy"])
            if "top5_accuracy" in df.columns and df["top5_accuracy"].max() > 1:
                df["top5_accuracy"] = df["top5_accuracy"] / 100
                df["secondary_metric"] = df["top5_accuracy"]
    else:
        # UrbanCars/CelebA use worst_group_accuracy as primary
        df["primary_metric"] = df.get("worst_group_accuracy", df.get("avg_accuracy"))
        df["secondary_metric"] = df.get("avg_accuracy", df["primary_metric"])

    # Training objective inference
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

    # Training data size
    def get_data_size(training_data):
        td = str(training_data).lower()
        for key, size in DATASET_SIZES_M.items():
            if key.lower() in td:
                return size
        return 400  # default

    df["training_data_size_M"] = df["training_data"].apply(get_data_size)

    # Derived features
    # NOTE: Keep NaN for models without patch size (ResNets, ConvNeXt, MobileCLIP, etc.)
    # These architectures don't use ViT-style patching, so patch_size is meaningless
    df["patch_size_num"] = pd.to_numeric(df.get("patch_size"), errors="coerce")
    df["image_size"] = df["image_size"].fillna(224)
    # Only compute tokens where patch_size is valid
    df["num_tokens"] = np.where(
        df["patch_size_num"].notna(),
        (df["image_size"] / df["patch_size_num"]) ** 2,
        np.nan,
    )
    df["log_params"] = np.log10(df["total_params_M"] + 1)

    # Track which models have valid patch size for reporting
    n_with_patch = df["patch_size_num"].notna().sum()
    n_without_patch = df["patch_size_num"].isna().sum()
    report.add_kv(
        "Models with valid patch_size",
        f"{n_with_patch} ({n_with_patch/len(df)*100:.1f}%)",
    )
    report.add_kv(
        "Models without patch_size (ResNet, ConvNeXt, etc.)",
        f"{n_without_patch} ({n_without_patch/len(df)*100:.1f}%)",
    )

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

    # Report summary
    report.add_subheader("Data Summary")
    report.add_kv(
        "Training objectives", df["training_objective"].value_counts().to_dict()
    )
    if "arch_family" in df.columns:
        report.add_kv("Architectures", df["arch_family"].value_counts().to_dict())
    report.add_kv(
        "Training data", df["training_data"].value_counts().head(10).to_dict()
    )
    report.add_kv("Image sizes", sorted(df["image_size"].dropna().unique().tolist()))
    report.add_kv(
        "Params range",
        f"{df['total_params_M'].min():.1f} - {df['total_params_M'].max():.1f}",
    )

    metric_col = "top1_accuracy" if is_imagenet else "worst_group_accuracy"
    if metric_col in df.columns:
        report.add_kv(
            f"{metric_col} range",
            f"{df[metric_col].min()*100:.1f}% - {df[metric_col].max()*100:.1f}%",
        )

    return df


# ═══════════════════════════════════════════════════════════════════
# Statistical Analysis Functions
# ═══════════════════════════════════════════════════════════════════


def analyze_overall(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    report: ReportWriter,
):
    """Overall statistics for a metric."""
    report.add_header(f"OVERALL STATISTICS: {name}", level=2)

    if metric_col not in df.columns:
        report.add_line(f"Column {metric_col} not found.")
        return

    v = df[metric_col].dropna() * 100  # Convert to percentage
    report.add_subheader(metric_label)
    for k, fn in [
        ("N", len),
        ("Mean", np.mean),
        ("Std", np.std),
        ("Min", np.min),
        ("Max", np.max),
        ("Median", np.median),
    ]:
        report.add_kv(k, fn(v))
    for p in [5, 10, 25, 75, 90, 95]:
        report.add_kv(f"P{p}", np.percentile(v, p))


def analyze_categorical(
    df: pd.DataFrame,
    fcol: str,
    flabel: str,
    metric_col: str,
    metric_label: str,
    name: str,
    report: ReportWriter,
    min_n: int = 3,
):
    """Analyze metric by categorical factor."""
    report.add_header(f"FACTOR: {flabel} ({name})", level=2)

    if fcol not in df.columns:
        report.add_line(f"Column {fcol} not found.")
        return
    if metric_col not in df.columns:
        report.add_line(f"Metric {metric_col} not found.")
        return

    counts = df[fcol].value_counts()
    valid = counts[counts >= min_n].index.tolist()

    report.add_subheader("Group Counts")
    for lvl, cnt in counts.items():
        report.add_line(f"  {lvl}: n={cnt}" + ("" if lvl in valid else " [EXCLUDED]"))

    if len(valid) < 2:
        report.add_line("Insufficient groups.")
        return

    df_f = df[df[fcol].isin(valid)]

    report.add_subheader(f"{metric_label} by {flabel}")
    g = df_f.groupby(fcol)[metric_col].agg(
        ["mean", "std", "min", "max", "median", "count"]
    )
    g = g.sort_values("mean", ascending=False)
    g_pct = g.copy()
    for col in ["mean", "std", "min", "max", "median"]:
        g_pct[col] = g_pct[col] * 100

    report.add_line(
        f"\n{'Level':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Med':>8} {'N':>5}"
    )
    report.add_line("-" * 85)
    for lvl, row in g_pct.iterrows():
        report.add_line(
            f"{str(lvl):<35} {row['mean']:>8.2f} {row['std']:>8.2f} {row['min']:>8.2f} {row['max']:>8.2f} {row['median']:>8.2f} {int(row['count']):>5}"
        )

    # Kruskal-Wallis test
    gdata = [
        grp[metric_col].dropna().values
        for _, grp in df_f.groupby(fcol)
        if len(grp) >= 2
    ]
    if len(gdata) >= 2:
        h, p = kruskal(*gdata)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        report.add_line(f"\nKruskal-Wallis: H={h:.4f}, p={p:.6f} ({sig})")


def analyze_numeric(
    df: pd.DataFrame,
    fcol: str,
    flabel: str,
    metric_col: str,
    metric_label: str,
    name: str,
    report: ReportWriter,
):
    """Analyze correlation between numeric factor and metric."""
    report.add_header(f"NUMERIC: {flabel} ({name})", level=2)

    if fcol not in df.columns or metric_col not in df.columns:
        report.add_line(f"Column not found.")
        return

    mask = df[fcol].notna() & df[metric_col].notna()
    if mask.sum() < 5:
        report.add_line("Insufficient data.")
        return

    x, y = df.loc[mask, fcol].values, df.loc[mask, metric_col].values

    report.add_subheader(f"{flabel} → {metric_label}")
    rp, pp = pearsonr(x, y)
    rs, ps = spearmanr(x, y)
    report.add_line(
        f"  Pearson: r={rp:+.4f}, p={pp:.6f} ({'***' if pp<0.001 else '**' if pp<0.01 else '*' if pp<0.05 else 'ns'})"
    )
    report.add_line(
        f"  Spearman: ρ={rs:+.4f}, p={ps:.6f} ({'***' if ps<0.001 else '**' if ps<0.01 else '*' if ps<0.05 else 'ns'})"
    )
    report.add_kv("N", len(x))


def analyze_top_bottom(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    report: ReportWriter,
    n: int = 20,
):
    """Show top and bottom models."""
    report.add_header(f"TOP/BOTTOM MODELS: {name}", level=2)

    cols = [
        "model_id",
        "training_objective",
        "training_data",
        "arch_family",
        "total_params_M",
        "image_size",
        "patch_size",
        metric_col,
    ]
    cols = [c for c in cols if c in df.columns]

    df_s = df.sort_values(metric_col, ascending=False)

    for subset, label in [(df_s.head(n), f"Top {n}"), (df_s.tail(n), f"Bottom {n}")]:
        report.add_subheader(f"{label} by {metric_label}")
        t = subset[cols].copy()
        if metric_col in t.columns:
            t[metric_col] = (t[metric_col] * 100).round(2)
        report.add_table(t)


# ═══════════════════════════════════════════════════════════════════
# Plotting Functions
# ═══════════════════════════════════════════════════════════════════


def plot_landscape(
    df: pd.DataFrame,
    name: str,
    out_dir: str,
    report: ReportWriter,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
):
    """Scatter plot with Pareto front."""
    report.add_header(f"FIGURE: Landscape ({name})", level=2)

    mask = df[x_col].notna() & df[y_col].notna()
    df_valid = df[mask]

    if len(df_valid) < 5:
        report.add_line("Insufficient data.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    x = df_valid[x_col].values * 100
    y = df_valid[y_col].values * 100

    # Color by architecture
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

    # Diagonal reference
    lims = [min(x.min(), y.min()) - 5, max(x.max(), y.max()) + 5]
    ax.plot(lims, lims, "k--", alpha=0.3)

    # Pareto front
    pareto = get_pareto_mask(x, y)
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

    # Label Pareto models
    report.add_line(f"\nPareto-optimal ({pareto.sum()}):")
    pareto_indices = np.where(pareto)[0]
    for idx in pareto_indices:
        model_id = df_valid.iloc[idx]["model_id"]
        short_name = (
            model_id.replace("ViT-", "")
            .replace("-quickgelu", "")
            .replace("_", " ")[:25]
        )
        ax.annotate(
            short_name,
            (x[idx], y[idx]),
            fontsize=FONT_SIZE_ANNOTATION,
            xytext=(5, 5),
            textcoords="offset points",
            color="darkred",
            fontweight="bold",
        )
        report.add_line(f"  {model_id}: x={x[idx]:.1f}%, y={y[idx]:.1f}%")

    ax.set_xlabel(x_label, fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_AXIS_LABEL)
    _place_legend(ax, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_landscape_{name.lower()}")
    plt.close()


def plot_by_factor(
    df: pd.DataFrame,
    name: str,
    out_dir: str,
    report: ReportWriter,
    factor_col: str,
    factor_label: str,
    metric_col: str,
    metric_label: str,
    min_n: int = 3,
):
    """Bar chart of metric by categorical factor."""
    report.add_header(f"FIGURE: {factor_label} ({name})", level=2)

    if factor_col not in df.columns or metric_col not in df.columns:
        report.add_line("Column not found.")
        return

    # Filter to groups with enough samples
    valid = (
        df.groupby(factor_col).filter(lambda x: len(x) >= min_n)[factor_col].unique()
    )
    df_f = df[df[factor_col].isin(valid)]

    if len(valid) < 2:
        report.add_line("Insufficient groups.")
        return

    stats = df_f.groupby(factor_col)[metric_col].agg(["mean", "std", "count"])
    stats = stats.sort_values("mean", ascending=True)

    # Log stats
    for idx, row in stats.iterrows():
        report.add_line(
            f"  {idx}: mean={row['mean']*100:.2f}%, std={row['std']*100:.2f}%, n={int(row['count'])}"
        )

    fig, ax = plt.subplots(figsize=(10, max(5, len(stats) * 0.4)))
    y_pos = np.arange(len(stats))

    # All bar charts use the viridis palette (consistent with plot_scale_bars).
    colors = _bar_colors(len(stats))

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
        [f"{idx} (n={int(row['count'])})" for idx, row in stats.iterrows()],
        fontsize=FONT_SIZE_AXIS_TICK,
    )
    ax.set_xlabel(f"{metric_label} (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_{factor_col}_{name.lower()}")
    plt.close()


def plot_scale_scatter(
    df: pd.DataFrame,
    name: str,
    out_dir: str,
    report: ReportWriter,
    factor_col: str,
    factor_label: str,
    metric_col: str,
    metric_label: str,
    use_log: bool = False,
):
    """Scatter plot of metric vs numeric factor with trend line."""
    report.add_header(f"FIGURE: {factor_label} Scale ({name})", level=2)

    if factor_col not in df.columns or metric_col not in df.columns:
        report.add_line("Column not found.")
        return

    mask = df[factor_col].notna() & df[metric_col].notna()
    if mask.sum() < 5:
        report.add_line("Insufficient data.")
        return

    x_all = df.loc[mask, factor_col].values
    y_all = df.loc[mask, metric_col].values * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by architecture
    arch_col = "arch_family" if "arch_family" in df.columns else "training_objective"
    archs = sorted(df.loc[mask, arch_col].dropna().unique())

    for arch in archs:
        m = (df[arch_col] == arch) & mask
        x_vals = df.loc[m, factor_col].values
        y_vals = df.loc[m, metric_col].values * 100
        color = COLORS_ARCH.get(arch, COLORS_OBJ.get(arch, "#95a5a6"))
        ax.scatter(
            x_vals,
            y_vals,
            c=color,
            label=f"{arch} (n={m.sum()})",
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

    ax.set_xlabel(factor_label, fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel(f"{metric_label} (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    _place_legend(ax, loc="best")
    ax.grid(True, alpha=0.3)

    # Log correlation
    r, p = spearmanr(x_all, y_all)
    report.add_line(f"{factor_label} vs {metric_label}: ρ={r:.4f}, p={p:.6f}")

    plt.tight_layout()
    fname = factor_col.replace("_", "").replace("M", "").lower()
    save_figure(fig, f"{out_dir}/fig_scale_{fname}_{name.lower()}")
    plt.close()


def plot_scale_bars(
    df: pd.DataFrame,
    name: str,
    out_dir: str,
    report: ReportWriter,
    factor_col: str,
    factor_label: str,
    metric_col: str,
    metric_label: str,
    n_bins: int = 6,
    use_log: bool = False,
):
    """Bar chart of metric by binned numeric factor."""

    if factor_col not in df.columns or metric_col not in df.columns:
        return

    mask = df[factor_col].notna() & df[metric_col].notna()
    if mask.sum() < 10:
        return

    x_all = df.loc[mask, factor_col].values

    # Create bins
    if use_log and x_all.min() > 0:
        log_min, log_max = np.log10(x_all.min()), np.log10(x_all.max())
        bin_edges = np.logspace(log_min, log_max, n_bins)
    else:
        bin_edges = np.linspace(x_all.min(), x_all.max(), n_bins)

    df_temp = df.loc[mask].copy()
    df_temp["bin"] = pd.cut(df_temp[factor_col], bins=bin_edges, include_lowest=True)

    bin_stats = df_temp.groupby("bin", observed=True)[metric_col].agg(
        ["mean", "std", "count"]
    )
    bin_stats = bin_stats[bin_stats["count"] >= 2]

    if len(bin_stats) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_labels = [
        f"{interval.left:.0f}-{interval.right:.0f}" for interval in bin_stats.index
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
            fontsize=FONT_SIZE_BAR_LABEL,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        bar_labels, rotation=45, ha="right", fontsize=FONT_SIZE_AXIS_TICK
    )
    ax.set_xlabel(factor_label, fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel(f"{metric_label} (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fname = factor_col.replace("_", "").replace("M", "").lower()
    save_figure(fig, f"{out_dir}/fig_scale_{fname}_bar_{name.lower()}")
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    name: str,
    out_dir: str,
    report: ReportWriter,
    metric_col: str,
    metric_label: str,
):
    """Correlation heatmap between numeric factors and metric."""
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

    if metric_col not in df.columns:
        report.add_line("Metric column not found.")
        return

    df_num = df[fcols + [metric_col]].dropna()
    if len(df_num) < 10:
        report.add_line("Insufficient data.")
        return

    corr = pd.DataFrame(index=fcols, columns=[metric_label], dtype=float)
    pval = pd.DataFrame(index=fcols, columns=[metric_label], dtype=float)

    report.add_line(f"\n{'Factor':<25} {'Corr':>10} {'p-value':>12}")
    report.add_line("-" * 50)

    for f in fcols:
        r, p = spearmanr(df_num[f], df_num[metric_col])
        corr.loc[f, metric_label] = r
        pval.loc[f, metric_label] = p
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        report.add_line(f"{f:<25} {r:>+10.4f}{sig} {p:>12.6f}")

    fig, ax = plt.subplots(figsize=(6, 8))
    if HAS_SEABORN:
        annot = corr.apply(lambda col: col.map(lambda x: f"{x:.2f}"))
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

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_corr_{name.lower()}")
    plt.close()
    corr.to_csv(f"{out_dir}/corr_{name.lower()}.csv")


# ═══════════════════════════════════════════════════════════════════
# Cross-Dataset Analysis (ImageNet as Baseline)
# ═══════════════════════════════════════════════════════════════════


def compute_robustness_metrics(
    df_imagenet: pd.DataFrame,
    df_bias: pd.DataFrame,
    bias_name: str,
    report: ReportWriter,
) -> pd.DataFrame:
    """Compute robustness gap and ratio metrics."""
    report.add_header(f"ROBUSTNESS METRICS: ImageNet vs {bias_name}", level=2)

    # Merge on model_id
    merged = df_imagenet[
        [
            "model_id",
            "model_name",
            "training_objective",
            "arch_family",
            "total_params_M",
            "image_size",
            "training_data",
            "top1_accuracy",
        ]
    ].merge(
        df_bias[["model_id", "worst_group_accuracy", "avg_accuracy"]],
        on="model_id",
        suffixes=("", f"_{bias_name.lower()}"),
    )

    report.add_kv("Models in ImageNet", len(df_imagenet))
    report.add_kv(f"Models in {bias_name}", len(df_bias))
    report.add_kv("Overlapping models", len(merged))

    if len(merged) < 10:
        report.add_line("Insufficient overlap for analysis.")
        return merged

    # Compute metrics (ensure same scale)
    merged["imagenet_acc"] = merged["top1_accuracy"]
    merged["wg_acc"] = merged["worst_group_accuracy"]

    # Robustness gap: ImageNet - WG (lower = more robust)
    merged["robustness_gap"] = merged["imagenet_acc"] - merged["wg_acc"]

    # Robustness ratio: WG / ImageNet (higher = more robust)
    merged["robustness_ratio"] = merged["wg_acc"] / merged["imagenet_acc"].replace(
        0, np.nan
    )

    # Relative drop: (ImageNet - WG) / ImageNet * 100
    merged["relative_drop_pct"] = (
        merged["robustness_gap"] / merged["imagenet_acc"].replace(0, np.nan)
    ) * 100

    report.add_subheader("Robustness Gap Statistics")
    gap = merged["robustness_gap"] * 100
    report.add_kv("Mean gap", f"{gap.mean():.2f}pp")
    report.add_kv("Std gap", f"{gap.std():.2f}pp")
    report.add_kv("Min gap", f"{gap.min():.2f}pp")
    report.add_kv("Max gap", f"{gap.max():.2f}pp")

    report.add_subheader("Robustness Ratio Statistics")
    ratio = merged["robustness_ratio"]
    report.add_kv("Mean ratio", f"{ratio.mean():.3f}")
    report.add_kv("Std ratio", f"{ratio.std():.3f}")
    report.add_kv("Min ratio", f"{ratio.min():.3f}")
    report.add_kv("Max ratio", f"{ratio.max():.3f}")

    # Correlation between ImageNet and WG
    report.add_subheader("Correlation: ImageNet vs WG")
    r_pearson, p_pearson = pearsonr(merged["imagenet_acc"], merged["wg_acc"])
    r_spearman, p_spearman = spearmanr(merged["imagenet_acc"], merged["wg_acc"])
    report.add_line(f"  Pearson: r={r_pearson:.4f}, p={p_pearson:.6f}")
    report.add_line(f"  Spearman: ρ={r_spearman:.4f}, p={p_spearman:.6f}")

    return merged


def plot_imagenet_vs_wg(
    merged: pd.DataFrame, bias_name: str, out_dir: str, report: ReportWriter
):
    """Scatter plot of ImageNet accuracy vs WG accuracy."""
    report.add_header(f"FIGURE: ImageNet vs {bias_name} WG", level=2)

    if len(merged) < 5:
        report.add_line("Insufficient data.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    x = merged["imagenet_acc"].values * 100
    y = merged["wg_acc"].values * 100

    # Color by architecture or objective
    color_col = (
        "arch_family" if "arch_family" in merged.columns else "training_objective"
    )
    categories = sorted(merged[color_col].dropna().unique())

    for cat in categories:
        m = merged[color_col] == cat
        color = COLORS_ARCH.get(cat, COLORS_OBJ.get(cat, "#95a5a6"))
        ax.scatter(
            x[m.values],
            y[m.values],
            c=color,
            label=f"{cat} (n={m.sum()})",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.3,
        )

    # Diagonal (perfect robustness)
    lims = [min(x.min(), y.min()) - 5, max(x.max(), y.max()) + 5]
    ax.plot(lims, lims, "k--", alpha=0.3, label="No robustness gap")

    # Trend line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line, slope * x_line + intercept, "r-", linewidth=2, alpha=0.7, label="Trend"
    )

    # Pareto front
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

    # Label Pareto models
    for idx in np.where(pareto)[0]:
        model_id = merged.iloc[idx]["model_id"]
        short_name = (
            model_id.replace("ViT-", "")
            .replace("-quickgelu", "")
            .replace("_", " ")[:20]
        )
        ax.annotate(
            short_name,
            (x[idx], y[idx]),
            fontsize=FONT_SIZE_ANNOTATION,
            xytext=(5, 5),
            textcoords="offset points",
            color="darkred",
            fontweight="bold",
        )

    r, p = spearmanr(x, y)
    ax.set_xlabel("ImageNet Top-1 Accuracy (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel(
        f"{bias_name} Worst-Group Accuracy (%)", fontsize=FONT_SIZE_AXIS_LABEL
    )
    _place_legend(ax, loc="lower right")
    ax.grid(True, alpha=0.3)

    report.add_line(f"Correlation: ρ={r:.4f}, p={p:.6f}")

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_imagenet_vs_wg_{bias_name.lower()}")
    plt.close()


def plot_robustness_gap(
    merged: pd.DataFrame, bias_name: str, out_dir: str, report: ReportWriter
):
    """Bar chart of robustness gap by model."""
    report.add_header(f"FIGURE: Robustness Gap ({bias_name})", level=2)

    if len(merged) < 5:
        report.add_line("Insufficient data.")
        return

    # Sort by gap
    df_sorted = merged.sort_values("robustness_gap", ascending=True).copy()

    # Show top and bottom 30
    n_show = min(30, len(df_sorted))
    df_plot = pd.concat(
        [df_sorted.head(n_show), df_sorted.tail(n_show)]
    ).drop_duplicates()
    df_plot = df_plot.sort_values("robustness_gap", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(df_plot) * 0.25)))

    y_pos = np.arange(len(df_plot))
    gaps = df_plot["robustness_gap"].values * 100

    # Color: green for small gap, red for large gap
    colors = plt.cm.RdYlGn_r((gaps - gaps.min()) / (gaps.max() - gaps.min() + 1e-6))

    ax.barh(y_pos, gaps, color=colors, edgecolor="black", linewidth=0.3, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["model_id"].str[:40], fontsize=FONT_SIZE_AXIS_TICK - 2)
    ax.set_xlabel(
        f"Robustness Gap: ImageNet − {bias_name} WG (pp)", fontsize=FONT_SIZE_AXIS_LABEL
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(True, alpha=0.3, axis="x")

    # Stats
    report.add_line(f"\nTop 5 most robust (smallest gap):")
    for _, row in df_sorted.head(5).iterrows():
        report.add_line(f"  {row['model_id']}: gap={row['robustness_gap']*100:.2f}pp")

    report.add_line(f"\nTop 5 least robust (largest gap):")
    for _, row in df_sorted.tail(5).iterrows():
        report.add_line(f"  {row['model_id']}: gap={row['robustness_gap']*100:.2f}pp")

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_robustness_gap_{bias_name.lower()}")
    plt.close()


def plot_gap_by_factor(
    merged: pd.DataFrame,
    factor_col: str,
    factor_label: str,
    bias_name: str,
    out_dir: str,
    report: ReportWriter,
    min_n: int = 3,
):
    """Bar chart of robustness gap by categorical factor."""
    report.add_header(f"FIGURE: Gap by {factor_label} ({bias_name})", level=2)

    if factor_col not in merged.columns:
        report.add_line(f"Column {factor_col} not found.")
        return

    valid = (
        merged.groupby(factor_col)
        .filter(lambda x: len(x) >= min_n)[factor_col]
        .unique()
    )
    df_f = merged[merged[factor_col].isin(valid)]

    if len(valid) < 2:
        report.add_line("Insufficient groups.")
        return

    stats = df_f.groupby(factor_col)["robustness_gap"].agg(["mean", "std", "count"])
    stats = stats.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(stats) * 0.4)))
    y_pos = np.arange(len(stats))

    colors = _bar_colors(len(stats))

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
        [f"{idx} (n={int(row['count'])})" for idx, row in stats.iterrows()],
        fontsize=FONT_SIZE_AXIS_TICK,
    )
    ax.set_xlabel(
        f"Robustness Gap: ImageNet − {bias_name} WG (pp)", fontsize=FONT_SIZE_AXIS_LABEL
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(True, alpha=0.3, axis="x")

    for idx, row in stats.iterrows():
        report.add_line(
            f"  {idx}: gap={row['mean']*100:.2f}pp ± {row['std']*100:.2f}pp (n={int(row['count'])})"
        )

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_gap_by_{factor_col}_{bias_name.lower()}")
    plt.close()


def plot_3way_correlation(
    df_imagenet: pd.DataFrame,
    df_uc: pd.DataFrame,
    df_celeba: pd.DataFrame,
    out_dir: str,
    report: ReportWriter,
):
    """3-way scatter matrix: ImageNet, UrbanCars WG, CelebA WG."""
    report.add_header("FIGURE: 3-Way Correlation Matrix", level=2)

    # Merge all three
    merged = (
        df_imagenet[["model_id", "arch_family", "training_objective", "top1_accuracy"]]
        .merge(df_uc[["model_id", "worst_group_accuracy"]], on="model_id")
        .merge(
            df_celeba[["model_id", "worst_group_accuracy"]],
            on="model_id",
            suffixes=("_uc", "_celeba"),
        )
    )

    merged.rename(
        columns={
            "top1_accuracy": "ImageNet",
            "worst_group_accuracy_uc": "UrbanCars_WG",
            "worst_group_accuracy_celeba": "CelebA_WG",
        },
        inplace=True,
    )

    report.add_kv("Models in all 3 datasets", len(merged))

    if len(merged) < 10:
        report.add_line("Insufficient overlap.")
        return merged

    # Convert to percentage
    for col in ["ImageNet", "UrbanCars_WG", "CelebA_WG"]:
        merged[col] = merged[col] * 100

    # Correlation matrix
    metrics = ["ImageNet", "UrbanCars_WG", "CelebA_WG"]
    corr_matrix = pd.DataFrame(index=metrics, columns=metrics, dtype=float)

    report.add_subheader("Pairwise Correlations (Spearman)")
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i <= j:
                r, p = spearmanr(merged[m1], merged[m2])
                corr_matrix.loc[m1, m2] = r
                corr_matrix.loc[m2, m1] = r
                if i < j:
                    report.add_line(f"  {m1} vs {m2}: ρ={r:.4f}, p={p:.6f}")

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(
            corr_matrix.astype(float),
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            linewidths=0.5,
            square=True,
        )
    ax.set_title("Cross-Dataset Correlation (Spearman)", fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_3way_correlation")
    plt.close()

    # Scatter matrix
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [
        ("ImageNet", "UrbanCars_WG"),
        ("ImageNet", "CelebA_WG"),
        ("UrbanCars_WG", "CelebA_WG"),
    ]

    for ax, (m1, m2) in zip(axes, pairs):
        x, y = merged[m1].values, merged[m2].values

        # Color by architecture
        archs = sorted(merged["arch_family"].dropna().unique())
        for arch in archs:
            m = merged["arch_family"] == arch
            color = COLORS_ARCH.get(arch, "#95a5a6")
            ax.scatter(
                x[m.values],
                y[m.values],
                c=color,
                alpha=0.6,
                s=40,
                edgecolors="white",
                linewidth=0.3,
                label=arch,
            )

        # Diagonal
        lims = [min(x.min(), y.min()) - 5, max(x.max(), y.max()) + 5]
        ax.plot(lims, lims, "k--", alpha=0.3)

        # Trend
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot(
            np.linspace(x.min(), x.max(), 100),
            slope * np.linspace(x.min(), x.max(), 100) + intercept,
            "r-",
            linewidth=2,
            alpha=0.7,
        )

        r, _ = spearmanr(x, y)
        ax.set_xlabel(f"{m1} (%)", fontsize=FONT_SIZE_AXIS_LABEL)
        ax.set_ylabel(f"{m2} (%)", fontsize=FONT_SIZE_AXIS_LABEL)
        ax.set_title(f"ρ = {r:.3f}", fontsize=FONT_SIZE_TITLE)
        ax.grid(True, alpha=0.3)

    _place_legend(axes[0], loc="lower right")
    # (only axes[0] gets a legend; it typically has few arch entries)
    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_3way_scatter")
    plt.close()

    return merged


# ═══════════════════════════════════════════════════════════════════
# Isolated Factor Analysis (Natural Experiments)
# ═══════════════════════════════════════════════════════════════════


def find_isolated_factor_groups(
    df: pd.DataFrame,
    vary_factor: str,
    fixed_factors: List[str],
    min_group_size: int = 2,
) -> Dict:
    """
    Find groups of models where only one factor varies while others are fixed.

    This enables "natural experiment" analysis where we can make stronger
    causal claims about the effect of a single factor.

    Args:
        df: DataFrame with model data
        vary_factor: The factor that should vary (e.g., 'patch_size_num')
        fixed_factors: Factors that should be held constant (e.g., ['arch_family', 'training_data'])
        min_group_size: Minimum models with different values of vary_factor in a group

    Returns:
        Dict with group keys and their member models
    """
    # Drop rows with missing values in relevant columns
    cols = [vary_factor] + fixed_factors
    df_valid = df.dropna(subset=cols)

    if len(df_valid) == 0:
        return {}

    # Group by fixed factors
    groups = {}
    for key, group in df_valid.groupby(fixed_factors, dropna=False):
        # Check if the vary_factor has multiple unique values in this group
        unique_values = group[vary_factor].nunique()
        if unique_values >= min_group_size:
            # Convert key to string for dict
            key_str = str(key) if isinstance(key, tuple) else str((key,))
            groups[key_str] = {
                "fixed_values": dict(
                    zip(fixed_factors, key if isinstance(key, tuple) else (key,))
                ),
                "models": group,
                "vary_values": sorted(group[vary_factor].unique()),
                "n_models": len(group),
                "n_values": unique_values,
            }

    return groups


def analyze_isolated_patch_size(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    out_dir: str,
    report: ReportWriter,
):
    """
    Analyze effect of patch size while holding other factors constant.

    Finds model pairs/groups like:
    - ViT-B-14 vs ViT-B-16 vs ViT-B-32 on same data
    """
    report.add_header(f"ISOLATED FACTOR: Patch Size ({name})", level=2)

    # Only analyze models with valid patch_size
    df_valid = df[df["patch_size_num"].notna()].copy()
    n_excluded = len(df) - len(df_valid)
    report.add_line(f"\nAnalyzing {len(df_valid)} models with valid patch_size")
    report.add_line(
        f"Excluded {n_excluded} models without patch_size (ResNet, ConvNeXt, etc.)"
    )

    if len(df_valid) < 10:
        report.add_line("Insufficient models with patch_size.")
        return None

    # Define what should be held constant
    fixed_factors = ["arch_family", "training_data", "vit_size"]
    fixed_factors = [f for f in fixed_factors if f in df_valid.columns]

    groups = find_isolated_factor_groups(
        df_valid, "patch_size_num", fixed_factors, min_group_size=2
    )

    if not groups:
        report.add_line("No matched groups found where only patch size varies.")
        return None

    report.add_line(
        f"\nFound {len(groups)} matched groups where only patch size varies:"
    )

    # Collect all comparisons for summary statistics
    all_comparisons = []

    # Table header
    report.add_line(
        f"\n{'Base Configuration':<50} | {'Patch Sizes':<20} | {metric_label:<40}"
    )
    report.add_line("-" * 115)

    for key, group_info in sorted(groups.items(), key=lambda x: -x[1]["n_models"]):
        models = group_info["models"]
        fixed = group_info["fixed_values"]

        # Create readable base config name
        base_name_parts = [str(v) for v in fixed.values() if pd.notna(v)]
        base_name = " / ".join(base_name_parts)[:48]

        # Get metrics for each patch size
        patch_metrics = []
        for ps in sorted(models["patch_size_num"].unique()):
            m = models[models["patch_size_num"] == ps]
            if len(m) > 0:
                metric_val = m[metric_col].mean() * 100
                patch_metrics.append((int(ps), metric_val, len(m)))

        # Format patch size results
        ps_str = ", ".join([f"{ps}" for ps, _, _ in patch_metrics])
        metric_str = ", ".join([f"{ps}→{v:.1f}%" for ps, v, _ in patch_metrics])

        report.add_line(f"{base_name:<50} | {ps_str:<20} | {metric_str:<40}")

        # Compute pairwise effects
        for i in range(len(patch_metrics)):
            for j in range(i + 1, len(patch_metrics)):
                ps1, m1, n1 = patch_metrics[i]
                ps2, m2, n2 = patch_metrics[j]
                effect = m2 - m1  # Effect of going from smaller to larger patch
                all_comparisons.append(
                    {
                        "base_config": base_name,
                        "from_patch": ps1,
                        "to_patch": ps2,
                        "from_metric": m1,
                        "to_metric": m2,
                        "effect_pp": effect,
                        "direction": "increase" if ps2 > ps1 else "decrease",
                    }
                )

    # Summary statistics
    if all_comparisons:
        comp_df = pd.DataFrame(all_comparisons)

        report.add_subheader("Summary: Patch Size Effect")

        # Group by direction (14→16, 16→32, 14→32, etc.)
        for (from_ps, to_ps), grp in comp_df.groupby(["from_patch", "to_patch"]):
            effects = grp["effect_pp"].values
            report.add_line(
                f"\nPatch {int(from_ps)} → {int(to_ps)} (n={len(effects)} comparisons):"
            )
            report.add_line(f"  Mean effect: {effects.mean():+.2f} pp")
            report.add_line(f"  Std: {effects.std():.2f} pp")
            report.add_line(f"  Range: [{effects.min():.2f}, {effects.max():.2f}] pp")
            report.add_line(
                f"  Direction: {(effects > 0).sum()} positive, {(effects < 0).sum()} negative, {(effects == 0).sum()} no change"
            )

        # Overall trend: larger patch → higher or lower metric?
        # Compare 14 vs 16 and 16 vs 32
        increasing_patch = comp_df[comp_df["to_patch"] > comp_df["from_patch"]]
        if len(increasing_patch) > 0:
            mean_effect = increasing_patch["effect_pp"].mean()
            report.add_line(
                f"\n** Overall: Increasing patch size → {mean_effect:+.2f} pp {metric_label} on average **"
            )
            if mean_effect < 0:
                report.add_line("   (Smaller patches = better robustness)")
            else:
                report.add_line("   (Larger patches = better robustness)")

        # Save comparisons
        comp_df.to_csv(f"{out_dir}/isolated_patch_size_{name.lower()}.csv", index=False)

        # Create ladder/slope plot
        _plot_isolated_factor(
            comp_df, "patch_size_num", "Patch Size", metric_label, name, out_dir, report
        )

        return comp_df

    return None


def analyze_isolated_image_size(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    out_dir: str,
    report: ReportWriter,
):
    """
    Analyze effect of image/input resolution while holding other factors constant.

    Finds model pairs like:
    - ViT-L-14 at 224 vs 336 px
    """
    report.add_header(f"ISOLATED FACTOR: Image Size ({name})", level=2)

    # All models should have image_size
    df_valid = df[df["image_size"].notna()].copy()

    report.add_line(f"\nAnalyzing {len(df_valid)} models")

    # Define what should be held constant
    # For image size, we want same architecture, same data, same patch
    fixed_factors = ["arch_family", "training_data", "patch_size_num"]
    # Also consider vit_size if available
    if "vit_size" in df_valid.columns:
        fixed_factors.append("vit_size")
    fixed_factors = [f for f in fixed_factors if f in df_valid.columns]

    # Filter to models with valid fixed factors (drop if any fixed factor is NaN)
    for f in fixed_factors:
        df_valid = df_valid[df_valid[f].notna()]

    groups = find_isolated_factor_groups(
        df_valid, "image_size", fixed_factors, min_group_size=2
    )

    if not groups:
        report.add_line("No matched groups found where only image size varies.")
        return None

    report.add_line(
        f"\nFound {len(groups)} matched groups where only image size varies:"
    )

    all_comparisons = []

    report.add_line(
        f"\n{'Base Configuration':<55} | {'Resolutions':<20} | {metric_label:<40}"
    )
    report.add_line("-" * 120)

    for key, group_info in sorted(groups.items(), key=lambda x: -x[1]["n_models"]):
        models = group_info["models"]
        fixed = group_info["fixed_values"]

        base_name_parts = [str(v) for v in fixed.values() if pd.notna(v)]
        base_name = " / ".join(base_name_parts)[:53]

        res_metrics = []
        for res in sorted(models["image_size"].unique()):
            m = models[models["image_size"] == res]
            if len(m) > 0:
                metric_val = m[metric_col].mean() * 100
                res_metrics.append((int(res), metric_val, len(m)))

        res_str = ", ".join([f"{r}px" for r, _, _ in res_metrics])
        metric_str = ", ".join([f"{r}→{v:.1f}%" for r, v, _ in res_metrics])

        report.add_line(f"{base_name:<55} | {res_str:<20} | {metric_str:<40}")

        for i in range(len(res_metrics)):
            for j in range(i + 1, len(res_metrics)):
                r1, m1, n1 = res_metrics[i]
                r2, m2, n2 = res_metrics[j]
                effect = m2 - m1
                all_comparisons.append(
                    {
                        "base_config": base_name,
                        "from_res": r1,
                        "to_res": r2,
                        "from_metric": m1,
                        "to_metric": m2,
                        "effect_pp": effect,
                    }
                )

    if all_comparisons:
        comp_df = pd.DataFrame(all_comparisons)

        report.add_subheader("Summary: Image Size Effect")

        # Group by resolution transitions
        for (from_r, to_r), grp in comp_df.groupby(["from_res", "to_res"]):
            effects = grp["effect_pp"].values
            report.add_line(
                f"\n{int(from_r)}px → {int(to_r)}px (n={len(effects)} comparisons):"
            )
            report.add_line(f"  Mean effect: {effects.mean():+.2f} pp")
            report.add_line(f"  Std: {effects.std():.2f} pp")
            report.add_line(f"  Range: [{effects.min():.2f}, {effects.max():.2f}] pp")

        # Overall trend
        increasing_res = comp_df[comp_df["to_res"] > comp_df["from_res"]]
        if len(increasing_res) > 0:
            mean_effect = increasing_res["effect_pp"].mean()
            report.add_line(
                f"\n** Overall: Increasing resolution → {mean_effect:+.2f} pp {metric_label} on average **"
            )

        comp_df.to_csv(f"{out_dir}/isolated_image_size_{name.lower()}.csv", index=False)

        _plot_isolated_factor(
            comp_df,
            "image_size",
            "Image Size (px)",
            metric_label,
            name,
            out_dir,
            report,
            from_col="from_res",
            to_col="to_res",
        )

        return comp_df

    return None


def analyze_isolated_training_data(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    out_dir: str,
    report: ReportWriter,
):
    """
    Analyze effect of training data while holding architecture constant.

    Finds model groups like:
    - ViT-B-16 trained on LAION vs DataComp vs OpenAI
    """
    report.add_header(f"ISOLATED FACTOR: Training Data ({name})", level=2)

    df_valid = df[df["training_data"].notna()].copy()

    report.add_line(f"\nAnalyzing {len(df_valid)} models")

    # For training data, hold architecture constant
    fixed_factors = ["arch_family", "image_size"]
    if "vit_size" in df_valid.columns:
        fixed_factors.append("vit_size")
    if "patch_size_num" in df_valid.columns:
        # Only use patch_size if it's available (skip for ResNet etc.)
        df_with_patch = df_valid[df_valid["patch_size_num"].notna()]
        if len(df_with_patch) > len(df_valid) * 0.5:  # If most models have patch size
            fixed_factors.append("patch_size_num")

    fixed_factors = [f for f in fixed_factors if f in df_valid.columns]

    # Filter valid fixed factors
    for f in fixed_factors:
        df_valid = df_valid[df_valid[f].notna()]

    groups = find_isolated_factor_groups(
        df_valid, "training_data", fixed_factors, min_group_size=2
    )

    if not groups:
        report.add_line("No matched groups found where only training data varies.")
        return None

    report.add_line(
        f"\nFound {len(groups)} matched groups where only training data varies:"
    )

    all_comparisons = []

    report.add_line(
        f"\n{'Architecture Config':<45} | {'Datasets':<35} | {metric_label}"
    )
    report.add_line("-" * 130)

    for key, group_info in sorted(groups.items(), key=lambda x: -x[1]["n_values"]):
        models = group_info["models"]
        fixed = group_info["fixed_values"]

        base_name_parts = [str(v) for v in fixed.values() if pd.notna(v)]
        base_name = " / ".join(base_name_parts)[:43]

        data_metrics = []
        for data in sorted(models["training_data"].unique()):
            m = models[models["training_data"] == data]
            if len(m) > 0:
                metric_val = m[metric_col].mean() * 100
                data_metrics.append((data, metric_val, len(m)))

        # Sort by metric value to see ranking
        data_metrics_sorted = sorted(data_metrics, key=lambda x: -x[1])

        data_str = ", ".join([d[:12] for d, _, _ in data_metrics_sorted])
        metric_str = " > ".join(
            [f"{d[:10]}={v:.1f}%" for d, v, _ in data_metrics_sorted]
        )

        report.add_line(f"{base_name:<45} | {data_str:<35} | {metric_str}")

        # Store pairwise comparisons
        for i in range(len(data_metrics_sorted)):
            for j in range(i + 1, len(data_metrics_sorted)):
                d1, m1, n1 = data_metrics_sorted[i]
                d2, m2, n2 = data_metrics_sorted[j]
                all_comparisons.append(
                    {
                        "base_config": base_name,
                        "data_1": d1,
                        "data_2": d2,
                        "metric_1": m1,
                        "metric_2": m2,
                        "effect_pp": m1 - m2,  # How much better is data_1 vs data_2
                    }
                )

    if all_comparisons:
        comp_df = pd.DataFrame(all_comparisons)

        report.add_subheader("Summary: Training Data Effect")

        # Aggregate by dataset pairs
        report.add_line("\nHead-to-head comparisons (positive = first dataset wins):")

        # Create win/loss record for each dataset
        datasets = set(comp_df["data_1"].unique()) | set(comp_df["data_2"].unique())
        win_record = {
            d: {"wins": 0, "losses": 0, "ties": 0, "total_diff": 0.0} for d in datasets
        }

        for _, row in comp_df.iterrows():
            d1, d2, diff = row["data_1"], row["data_2"], row["effect_pp"]
            if diff > 0.5:  # d1 wins by >0.5pp
                win_record[d1]["wins"] += 1
                win_record[d2]["losses"] += 1
            elif diff < -0.5:  # d2 wins
                win_record[d1]["losses"] += 1
                win_record[d2]["wins"] += 1
            else:  # Tie
                win_record[d1]["ties"] += 1
                win_record[d2]["ties"] += 1
            win_record[d1]["total_diff"] += diff
            win_record[d2]["total_diff"] -= diff

        # Sort by win rate
        records = [
            (d, r["wins"], r["losses"], r["ties"], r["total_diff"])
            for d, r in win_record.items()
        ]
        records_sorted = sorted(
            records, key=lambda x: (-x[1], x[2])
        )  # Most wins, fewest losses

        report.add_line(
            f"\n{'Dataset':<25} {'Wins':>6} {'Losses':>8} {'Ties':>6} {'Net Δ (pp)':>12}"
        )
        report.add_line("-" * 60)
        for d, w, l, t, diff in records_sorted:
            n_comparisons = w + l + t
            if n_comparisons > 0:
                report.add_line(
                    f"{d:<25} {w:>6} {l:>8} {t:>6} {diff/n_comparisons:>+12.2f}"
                )

        comp_df.to_csv(
            f"{out_dir}/isolated_training_data_{name.lower()}.csv", index=False
        )

        # Create bar chart of dataset rankings
        _plot_isolated_training_data(
            records_sorted, metric_label, name, out_dir, report
        )

        return comp_df

    return None


def analyze_isolated_model_scale(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    out_dir: str,
    report: ReportWriter,
):
    """
    Analyze effect of model scale (vit_size: S, B, L, H, g, G) while holding other factors constant.
    """
    report.add_header(f"ISOLATED FACTOR: Model Scale ({name})", level=2)

    if "vit_size" not in df.columns:
        report.add_line("vit_size column not available.")
        return None

    df_valid = df[df["vit_size"].notna()].copy()

    report.add_line(f"\nAnalyzing {len(df_valid)} models with vit_size")

    # Hold constant: architecture family, training data, patch size, image size
    fixed_factors = ["arch_family", "training_data", "image_size"]
    if "patch_size_num" in df_valid.columns:
        df_with_patch = df_valid[df_valid["patch_size_num"].notna()]
        if len(df_with_patch) > len(df_valid) * 0.3:
            fixed_factors.append("patch_size_num")
            df_valid = df_with_patch

    fixed_factors = [f for f in fixed_factors if f in df_valid.columns]

    for f in fixed_factors:
        df_valid = df_valid[df_valid[f].notna()]

    groups = find_isolated_factor_groups(
        df_valid, "vit_size", fixed_factors, min_group_size=2
    )

    if not groups:
        report.add_line("No matched groups found where only model scale varies.")
        return None

    report.add_line(f"\nFound {len(groups)} matched groups where only vit_size varies:")

    all_comparisons = []

    # ViT size ordering
    size_order = {
        "S": 0,
        "M": 1,
        "B": 2,
        "L": 3,
        "H": 4,
        "g": 5,
        "G": 6,
        "e": 7,
        "E": 8,
    }

    report.add_line(f"\n{'Base Configuration':<55} | {'Sizes':<15} | {metric_label}")
    report.add_line("-" * 120)

    for key, group_info in sorted(groups.items(), key=lambda x: -x[1]["n_values"]):
        models = group_info["models"]
        fixed = group_info["fixed_values"]

        base_name_parts = [str(v) for v in fixed.values() if pd.notna(v)]
        base_name = " / ".join(base_name_parts)[:53]

        size_metrics = []
        for size in models["vit_size"].unique():
            m = models[models["vit_size"] == size]
            if len(m) > 0:
                metric_val = m[metric_col].mean() * 100
                order = size_order.get(size, 99)
                size_metrics.append((size, metric_val, order, len(m)))

        # Sort by size order
        size_metrics_sorted = sorted(size_metrics, key=lambda x: x[2])

        size_str = " < ".join([s for s, _, _, _ in size_metrics_sorted])
        metric_str = " → ".join([f"{s}={v:.1f}%" for s, v, _, _ in size_metrics_sorted])

        report.add_line(f"{base_name:<55} | {size_str:<15} | {metric_str}")

        # Pairwise comparisons (larger vs smaller)
        for i in range(len(size_metrics_sorted)):
            for j in range(i + 1, len(size_metrics_sorted)):
                s1, m1, o1, n1 = size_metrics_sorted[i]
                s2, m2, o2, n2 = size_metrics_sorted[j]
                effect = m2 - m1  # Effect of going larger
                all_comparisons.append(
                    {
                        "base_config": base_name,
                        "from_size": s1,
                        "to_size": s2,
                        "from_metric": m1,
                        "to_metric": m2,
                        "effect_pp": effect,
                    }
                )

    if all_comparisons:
        comp_df = pd.DataFrame(all_comparisons)

        report.add_subheader("Summary: Model Scale Effect")

        # Overall: does scaling up help?
        report.add_line(
            f"\nOverall effect of scaling up (n={len(comp_df)} comparisons):"
        )
        effects = comp_df["effect_pp"].values
        report.add_line(f"  Mean effect: {effects.mean():+.2f} pp")
        report.add_line(f"  Std: {effects.std():.2f} pp")
        report.add_line(
            f"  {(effects > 0).sum()} positive, {(effects < 0).sum()} negative"
        )

        # By specific transitions
        report.add_line("\nBy size transition:")
        for (from_s, to_s), grp in comp_df.groupby(["from_size", "to_size"]):
            eff = grp["effect_pp"].values
            if len(eff) >= 2:
                report.add_line(
                    f"  {from_s} → {to_s}: {eff.mean():+.2f} pp (n={len(eff)}, std={eff.std():.2f})"
                )

        comp_df.to_csv(
            f"{out_dir}/isolated_model_scale_{name.lower()}.csv", index=False
        )

        return comp_df

    return None


def analyze_isolated_objective(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    out_dir: str,
    report: ReportWriter,
):
    """
    Analyze effect of training objective (CLIP vs SigLIP) while holding other factors constant.
    """
    report.add_header(f"ISOLATED FACTOR: Training Objective ({name})", level=2)

    if "training_objective" not in df.columns:
        report.add_line("training_objective column not available.")
        return None

    df_valid = df[df["training_objective"].notna()].copy()

    report.add_line(f"\nAnalyzing {len(df_valid)} models")

    # Hold constant: architecture components, training data, scale
    fixed_factors = ["training_data", "image_size", "total_params_M"]
    if "vit_size" in df_valid.columns:
        fixed_factors.append("vit_size")
    if "patch_size_num" in df_valid.columns:
        df_with_patch = df_valid[df_valid["patch_size_num"].notna()]
        if len(df_with_patch) > len(df_valid) * 0.3:
            fixed_factors.append("patch_size_num")
            df_valid = df_with_patch

    fixed_factors = [f for f in fixed_factors if f in df_valid.columns]

    # For params, we need to bucket since exact match is rare
    if "total_params_M" in fixed_factors:
        # Create param buckets
        df_valid["params_bucket_fine"] = pd.cut(
            df_valid["total_params_M"],
            bins=[0, 100, 200, 400, 600, 1000, 2000, 5000, 50000],
            labels=[
                "<100M",
                "100-200M",
                "200-400M",
                "400-600M",
                "600M-1B",
                "1-2B",
                "2-5B",
                ">5B",
            ],
        )
        fixed_factors = [
            f if f != "total_params_M" else "params_bucket_fine" for f in fixed_factors
        ]

    for f in fixed_factors:
        df_valid = df_valid[df_valid[f].notna()]

    groups = find_isolated_factor_groups(
        df_valid, "training_objective", fixed_factors, min_group_size=2
    )

    if not groups:
        report.add_line("No matched groups found where only training objective varies.")
        return None

    report.add_line(
        f"\nFound {len(groups)} matched groups where only training objective varies:"
    )

    all_comparisons = []

    report.add_line(f"\n{'Configuration':<60} | {'Objectives':<25} | {metric_label}")
    report.add_line("-" * 130)

    for key, group_info in sorted(groups.items(), key=lambda x: -x[1]["n_values"]):
        models = group_info["models"]
        fixed = group_info["fixed_values"]

        base_name_parts = [str(v) for v in fixed.values() if pd.notna(v)]
        base_name = " / ".join(base_name_parts)[:58]

        obj_metrics = []
        for obj in models["training_objective"].unique():
            m = models[models["training_objective"] == obj]
            if len(m) > 0:
                metric_val = m[metric_col].mean() * 100
                obj_metrics.append((obj, metric_val, len(m)))

        obj_metrics_sorted = sorted(obj_metrics, key=lambda x: -x[1])

        obj_str = ", ".join([o for o, _, _ in obj_metrics_sorted])
        metric_str = " > ".join([f"{o}={v:.1f}%" for o, v, _ in obj_metrics_sorted])

        report.add_line(f"{base_name:<60} | {obj_str:<25} | {metric_str}")

        for i in range(len(obj_metrics_sorted)):
            for j in range(i + 1, len(obj_metrics_sorted)):
                o1, m1, n1 = obj_metrics_sorted[i]
                o2, m2, n2 = obj_metrics_sorted[j]
                all_comparisons.append(
                    {
                        "base_config": base_name,
                        "obj_1": o1,
                        "obj_2": o2,
                        "metric_1": m1,
                        "metric_2": m2,
                        "effect_pp": m1 - m2,
                    }
                )

    if all_comparisons:
        comp_df = pd.DataFrame(all_comparisons)

        report.add_subheader("Summary: Training Objective Effect")

        # Head-to-head: SigLIP vs CLIP
        siglip_vs_clip = comp_df[
            (
                (comp_df["obj_1"].isin(["SigLIP", "SigLIP2"]))
                & (comp_df["obj_2"] == "CLIP")
            )
            | (
                (comp_df["obj_2"].isin(["SigLIP", "SigLIP2"]))
                & (comp_df["obj_1"] == "CLIP")
            )
        ].copy()

        if len(siglip_vs_clip) > 0:
            # Normalize so positive = SigLIP better
            def normalize_effect(row):
                if row["obj_1"] in ["SigLIP", "SigLIP2"]:
                    return row["effect_pp"]
                else:
                    return -row["effect_pp"]

            siglip_vs_clip["siglip_advantage"] = siglip_vs_clip.apply(
                normalize_effect, axis=1
            )

            effects = siglip_vs_clip["siglip_advantage"].values
            report.add_line(
                f"\nSigLIP vs CLIP (n={len(effects)} head-to-head comparisons):"
            )
            report.add_line(
                f"  SigLIP advantage: {effects.mean():+.2f} pp (std={effects.std():.2f})"
            )
            report.add_line(
                f"  SigLIP wins: {(effects > 0.5).sum()}, CLIP wins: {(effects < -0.5).sum()}, Ties: {((effects >= -0.5) & (effects <= 0.5)).sum()}"
            )

        comp_df.to_csv(f"{out_dir}/isolated_objective_{name.lower()}.csv", index=False)

        return comp_df

    return None


def _plot_isolated_factor(
    comp_df: pd.DataFrame,
    factor_col: str,
    factor_label: str,
    metric_label: str,
    name: str,
    out_dir: str,
    report: ReportWriter,
    from_col: str = "from_patch",
    to_col: str = "to_patch",
):
    """Create slope/ladder plot for isolated factor analysis."""

    report.add_subheader(f"Figure: {factor_label} Isolated Effect")

    # Get unique base configs
    configs = comp_df["base_config"].unique()

    if len(configs) == 0:
        return

    # Limit to top 15 configs by number of comparisons
    config_counts = comp_df["base_config"].value_counts()
    top_configs = config_counts.head(15).index.tolist()
    plot_df = comp_df[comp_df["base_config"].isin(top_configs)]

    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Get all unique factor values
    all_values = sorted(set(plot_df[from_col].unique()) | set(plot_df[to_col].unique()))

    # Color by effect direction
    cmap = plt.cm.RdYlGn

    y_offset = 0
    y_labels = []

    for config in top_configs:
        cfg_data = plot_df[plot_df["base_config"] == config]
        if len(cfg_data) == 0:
            continue

        # Get unique values and metrics for this config
        values_metrics = {}
        for _, row in cfg_data.iterrows():
            values_metrics[row[from_col]] = row["from_metric"]
            values_metrics[row[to_col]] = row["to_metric"]

        # Sort by factor value
        sorted_items = sorted(values_metrics.items())

        xs = [v for v, _ in sorted_items]
        ys = [m for _, m in sorted_items]

        # Plot line
        ax.plot(xs, [y_offset] * len(xs), "k-", alpha=0.3, linewidth=1)

        # Plot points with color based on metric value
        for x, metric in zip(xs, ys):
            color = cmap((metric - 20) / 60)  # Normalize to 0-1 assuming 20-80% range
            ax.scatter(
                x,
                y_offset,
                c=[color],
                s=100,
                edgecolors="black",
                linewidth=0.5,
                zorder=3,
            )
            ax.annotate(
                f"{metric:.1f}%",
                (x, y_offset),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=FONT_SIZE_ANNOTATION,
            )

        y_labels.append(config[:40])
        y_offset += 1

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=FONT_SIZE_AXIS_TICK)
    ax.set_xlabel(factor_label, fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title(
        f"Isolated {factor_label} Effect on {metric_label}\n{name}",
        fontsize=FONT_SIZE_TITLE,
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(20, 80))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label(metric_label + " (%)", fontsize=FONT_SIZE_COLORBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_COLORBAR)

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_isolated_{factor_col}_{name.lower()}")
    plt.close()


def _plot_isolated_training_data(
    records: List[Tuple],
    metric_label: str,
    name: str,
    out_dir: str,
    report: ReportWriter,
):
    """Create bar chart ranking datasets by head-to-head performance."""

    report.add_subheader("Figure: Training Data Rankings")

    if len(records) < 2:
        return

    # Filter to datasets with at least 2 comparisons
    records = [(d, w, l, t, diff) for d, w, l, t, diff in records if (w + l + t) >= 2]

    if len(records) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Win-Loss record
    ax = axes[0]
    datasets = [r[0][:20] for r in records]
    wins = [r[1] for r in records]
    losses = [r[2] for r in records]

    x = np.arange(len(datasets))
    width = 0.35

    ax.barh(
        x - width / 2,
        wins,
        width,
        label="Wins",
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.barh(
        x + width / 2,
        losses,
        width,
        label="Losses",
        color="#e74c3c",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(x)
    ax.set_yticklabels(datasets, fontsize=FONT_SIZE_AXIS_TICK)
    ax.set_xlabel("Count", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title("Head-to-Head Record", fontsize=FONT_SIZE_TITLE)
    _place_legend(ax, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    # Plot 2: Net advantage
    ax = axes[1]
    n_comparisons = [w + l + t for d, w, l, t, _ in records]
    net_diff = [
        diff / n if n > 0 else 0
        for (_, _, _, _, diff), n in zip(records, n_comparisons)
    ]

    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in net_diff]
    ax.barh(x, net_diff, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(x)
    ax.set_yticklabels(datasets, fontsize=FONT_SIZE_AXIS_TICK)
    ax.set_xlabel(f"Mean {metric_label} Advantage (pp)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title("Average Performance Advantage", fontsize=FONT_SIZE_TITLE)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.suptitle(
        f"Training Data Comparison: {name}",
        fontsize=FONT_SIZE_SUPTITLE,
        fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_isolated_training_data_{name.lower()}")
    plt.close()


def run_isolated_factor_analysis(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    out_dir: str,
    report: ReportWriter,
):
    """
    Run all isolated factor analyses for a dataset.

    This section identifies "natural experiments" where models differ in only
    one factor, enabling stronger causal claims.
    """
    report.add_header(f"ISOLATED FACTOR ANALYSIS: {name}", level=1)
    report.add_line(
        "\nThis section identifies model pairs/groups where only ONE factor varies,"
    )
    report.add_line("enabling stronger causal claims than correlational analysis.")
    report.add_line(
        "\nNote: Models without patch_size (ResNet, ConvNeXt, MobileCLIP) are excluded"
    )
    report.add_line("from patch size and token analyses.")

    results = {}

    # 1. Patch size (excluding non-ViT models)
    results["patch_size"] = analyze_isolated_patch_size(
        df, name, metric_col, metric_label, out_dir, report
    )

    # 2. Image size (all models)
    results["image_size"] = analyze_isolated_image_size(
        df, name, metric_col, metric_label, out_dir, report
    )

    # 3. Training data (all models)
    results["training_data"] = analyze_isolated_training_data(
        df, name, metric_col, metric_label, out_dir, report
    )

    # 4. Model scale (vit_size)
    results["model_scale"] = analyze_isolated_model_scale(
        df, name, metric_col, metric_label, out_dir, report
    )

    # 5. Training objective (CLIP vs SigLIP)
    results["objective"] = analyze_isolated_objective(
        df, name, metric_col, metric_label, out_dir, report
    )

    # Summary table
    report.add_header("ISOLATED FACTORS SUMMARY", level=2)

    summary_lines = []
    for factor, comp_df in results.items():
        if comp_df is not None and len(comp_df) > 0:
            n_comparisons = len(comp_df)
            summary_lines.append(
                f"  {factor}: {n_comparisons} matched comparisons found"
            )
        else:
            summary_lines.append(f"  {factor}: No matched comparisons")

    report.add_line("\n".join(summary_lines))

    return results


# ═══════════════════════════════════════════════════════════════════
# Multivariate Analysis (Confounder Control)
# ═══════════════════════════════════════════════════════════════════


def run_multivariate_regression(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    out_dir: str,
    report: ReportWriter,
):
    """
    Multiple OLS regression to estimate independent effects of each factor,
    controlling for confounders.

    Factors analyzed:
    - Architecture family (categorical): SigLIP, SigLIP2, CoCa, EVA, CLIPA, ConvNeXt, ResNet, etc.
    - Model parameters (numeric): total_params_M
    - Dataset type (categorical): training_data
    - Dataset size (numeric): training_data_size_M
    - Patch size (numeric): patch_size_num
    - Image size (numeric): image_size
    """
    report.add_header(f"MULTIVARIATE REGRESSION: {name}", level=2)

    if not HAS_STATSMODELS:
        report.add_line(
            "statsmodels not available. Install with: pip install statsmodels"
        )
        return None

    # Prepare data
    df_r = df.copy()

    # Convert metric to percentage for interpretability
    df_r["y"] = df_r[metric_col] * 100

    # ─── Architecture family dummies (reference: CLIP/ViT) ─────────
    # Group into major architecture families
    def get_arch_group(arch):
        arch = str(arch)
        if arch in ["SigLIP", "SigLIP2"]:
            return "SigLIP"
        elif arch == "CoCa":
            return "CoCa"
        elif arch == "EVA":
            return "EVA"
        elif arch == "CLIPA":
            return "CLIPA"
        elif arch in ["ConvNeXt"]:
            return "ConvNeXt"
        elif arch in ["ResNet"]:
            return "ResNet"
        elif arch in ["MobileCLIP", "MobileCLIP2"]:
            return "MobileCLIP"
        else:
            return "CLIP"  # Default/reference category

    df_r["arch_group"] = df_r["arch_family"].apply(get_arch_group)

    # Create dummy variables (reference: CLIP)
    for arch in ["SigLIP", "CoCa", "EVA", "CLIPA", "ConvNeXt", "ResNet", "MobileCLIP"]:
        df_r[f"is_{arch.lower()}"] = (df_r["arch_group"] == arch).astype(int)

    # ─── Dataset type dummies (reference: LAION) ───────────────────
    def get_data_group(data):
        data = str(data).lower()
        if "webli" in data:
            return "WebLI"
        elif "datacomp" in data:
            return "DataComp"
        elif "dfn" in data:
            return "DFN"
        elif "openai" in data:
            return "OpenAI"
        elif "metaclip" in data:
            return "MetaCLIP"
        elif "commonpool" in data:
            return "CommonPool"
        else:
            return "LAION"  # Default/reference

    df_r["data_group"] = df_r["training_data"].apply(get_data_group)

    for dset in ["WebLI", "DataComp", "DFN", "OpenAI", "MetaCLIP"]:
        df_r[f"is_{dset.lower()}"] = (df_r["data_group"] == dset).astype(int)

    # ─── Numeric variables (log-transform and standardize) ─────────
    df_r["log_params"] = np.log10(df_r["total_params_M"].clip(lower=1))
    df_r["log_datasize"] = np.log10(df_r["training_data_size_M"].clip(lower=1))
    # NOTE: patch_size_num can be NaN for non-ViT models (ResNet, ConvNeXt, MobileCLIP)
    # We keep NaN here - models without patch size will be excluded from regression
    df_r["log_patchsize"] = np.where(
        df_r["patch_size_num"].notna(),
        np.log2(df_r["patch_size_num"].clip(lower=1)),
        np.nan,
    )
    df_r["log_imgsize"] = np.log2(df_r["image_size"].clip(lower=1))

    # Report how many models will be excluded due to missing patch_size
    n_with_patch = df_r["log_patchsize"].notna().sum()
    n_without_patch = df_r["log_patchsize"].isna().sum()
    report.add_line(
        f"\nNote: {n_without_patch} models lack patch_size (ResNet, ConvNeXt, MobileCLIP, etc.)"
    )
    report.add_line(
        f"These will be EXCLUDED from the regression which includes patch_size as a factor."
    )
    report.add_line(f"Regression will use {n_with_patch} models with valid patch_size.")

    # Standardize for comparable coefficients
    for col in ["log_params", "log_datasize", "log_patchsize", "log_imgsize"]:
        if col in df_r.columns:
            mean, std = df_r[col].mean(), df_r[col].std()
            if std > 0:
                df_r[f"{col}_z"] = (df_r[col] - mean) / std
            else:
                df_r[f"{col}_z"] = 0

    # ─── Drop missing values ───────────────────────────────────────
    reg_cols = [
        "y",
        "is_siglip",
        "is_coca",
        "is_eva",
        "log_params_z",
        "log_datasize_z",
        "log_patchsize_z",
        "log_imgsize_z",
    ]
    df_r = df_r.dropna(subset=[c for c in reg_cols if c in df_r.columns])

    report.add_kv("Samples for regression", len(df_r))

    if len(df_r) < 50:
        report.add_line("Insufficient samples for robust regression.")
        return None

    # Report factor distributions
    report.add_subheader("Factor Distributions in Regression Sample")
    report.add_line(f"\nArchitecture groups:")
    for arch, cnt in df_r["arch_group"].value_counts().items():
        report.add_line(f"  {arch}: n={cnt}")
    report.add_line(f"\nDataset groups:")
    for dset, cnt in df_r["data_group"].value_counts().items():
        report.add_line(f"  {dset}: n={cnt}")

    # ═══════════════════════════════════════════════════════════════
    # Model 1: Full model with all factors
    # ═══════════════════════════════════════════════════════════════
    report.add_subheader("Model 1: Full Model (all factors)")

    # Build formula with available architecture dummies
    arch_terms = []
    for arch in ["siglip", "coca", "eva", "clipa", "convnext", "resnet", "mobileclip"]:
        col = f"is_{arch}"
        if col in df_r.columns and df_r[col].sum() >= 3:  # At least 3 samples
            arch_terms.append(col)

    data_terms = []
    for dset in ["webli", "datacomp", "dfn", "openai", "metaclip"]:
        col = f"is_{dset}"
        if col in df_r.columns and df_r[col].sum() >= 3:
            data_terms.append(col)

    formula_parts = ["y ~ "]
    formula_parts.append(" + ".join(arch_terms) if arch_terms else "1")
    formula_parts.append(" + " + " + ".join(data_terms) if data_terms else "")
    formula_parts.append(
        " + log_params_z + log_datasize_z + log_patchsize_z + log_imgsize_z"
    )
    formula = "".join(formula_parts)

    try:
        model_full = smf.ols(formula, data=df_r).fit()

        report.add_line(f"\nFormula: {formula}")
        report.add_kv("R²", model_full.rsquared)
        report.add_kv("Adj R²", model_full.rsquared_adj)
        report.add_kv("F-statistic", model_full.fvalue)
        report.add_kv("F p-value", model_full.f_pvalue)

        report.add_line(
            f"\n{'Variable':<25} {'Coef':>10} {'SE':>10} {'t':>10} {'p':>12} {'95% CI':>20} {'Sig':>5}"
        )
        report.add_line("-" * 95)

        for var in model_full.params.index:
            c = model_full.params[var]
            se = model_full.bse[var]
            t = model_full.tvalues[var]
            p = model_full.pvalues[var]
            ci_low, ci_high = model_full.conf_int().loc[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            report.add_line(
                f"{var:<25} {c:>10.3f} {se:>10.3f} {t:>10.2f} {p:>12.4f} [{ci_low:>7.2f}, {ci_high:>6.2f}] {sig:>5}"
            )

        report.add_subheader("Interpretation Guide")
        report.add_line("\n• Architecture dummies: effect vs CLIP (reference category)")
        report.add_line("• Dataset dummies: effect vs LAION (reference category)")
        report.add_line("• Numeric variables: standardized (1 unit = 1 SD)")
        report.add_line("  - log_params_z: model size (log10 of params)")
        report.add_line("  - log_datasize_z: training data size (log10)")
        report.add_line("  - log_patchsize_z: patch size (log2)")
        report.add_line("  - log_imgsize_z: image resolution (log2)")

        report.add_subheader("Significant Effects (p < 0.05)")
        for var in model_full.params.index:
            if var == "Intercept":
                continue
            c = model_full.params[var]
            p = model_full.pvalues[var]
            if p < 0.05:
                direction = "↑" if c > 0 else "↓"
                report.add_line(f"  {direction} {var}: {c:+.2f}pp (p={p:.4f})")

    except Exception as e:
        report.add_line(f"Regression failed: {e}")
        import traceback

        report.add_line(traceback.format_exc())
        return None

    # ═══════════════════════════════════════════════════════════════
    # Model 2: Hierarchical regression (incremental R²)
    # ═══════════════════════════════════════════════════════════════
    report.add_subheader("Model 2: Hierarchical Regression (Incremental R²)")
    report.add_line("\nHow much variance does each factor GROUP add?")

    # Define factor groups
    arch_formula = " + ".join(arch_terms) if arch_terms else "1"
    data_formula = " + ".join(data_terms) if data_terms else ""
    scale_formula = "log_params_z + log_imgsize_z + log_patchsize_z"
    datasize_formula = "log_datasize_z"

    # Order 1: Architecture → Scale → Dataset Type → Dataset Size
    model_steps = [
        ("1. Architecture", f"y ~ {arch_formula}"),
        ("2. + Scale (params, img, patch)", f"y ~ {arch_formula} + {scale_formula}"),
        (
            "3. + Dataset Type",
            f"y ~ {arch_formula} + {scale_formula}"
            + (f" + {data_formula}" if data_formula else ""),
        ),
        (
            "4. + Dataset Size",
            f"y ~ {arch_formula} + {scale_formula}"
            + (f" + {data_formula}" if data_formula else "")
            + f" + {datasize_formula}",
        ),
    ]

    prev_r2 = 0
    report.add_line(f"\n{'Step':<40} {'R²':>10} {'ΔR²':>10} {'F':>10} {'p':>12}")
    report.add_line("-" * 85)

    for step_name, formula in model_steps:
        try:
            model = smf.ols(formula, data=df_r).fit()
            delta_r2 = model.rsquared - prev_r2
            report.add_line(
                f"{step_name:<40} {model.rsquared:>10.4f} {delta_r2:>10.4f} {model.fvalue:>10.2f} {model.f_pvalue:>12.6f}"
            )
            prev_r2 = model.rsquared
        except Exception as e:
            report.add_line(f"{step_name:<40} FAILED: {e}")

    # Alternative order: Scale → Dataset Size → Dataset Type → Architecture
    report.add_line("\n\nAlternative order (robustness check):")
    report.add_line(f"\n{'Step':<40} {'R²':>10} {'ΔR²':>10}")
    report.add_line("-" * 65)

    alt_steps = [
        ("1. Scale", f"y ~ {scale_formula}"),
        ("2. + Dataset Size", f"y ~ {scale_formula} + {datasize_formula}"),
        (
            "3. + Dataset Type",
            f"y ~ {scale_formula} + {datasize_formula}"
            + (f" + {data_formula}" if data_formula else ""),
        ),
        (
            "4. + Architecture",
            f"y ~ {scale_formula} + {datasize_formula}"
            + (f" + {data_formula}" if data_formula else "")
            + f" + {arch_formula}",
        ),
    ]

    prev_r2 = 0
    for step_name, formula in alt_steps:
        try:
            model = smf.ols(formula, data=df_r).fit()
            delta_r2 = model.rsquared - prev_r2
            report.add_line(
                f"{step_name:<40} {model.rsquared:>10.4f} {delta_r2:>10.4f}"
            )
            prev_r2 = model.rsquared
        except:
            pass

    return model_full


def compute_partial_correlations(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    out_dir: str,
    report: ReportWriter,
):
    """
    Compute partial correlations: correlation between each factor and the metric,
    controlling for all other factors.

    Factors: arch_family, params, dataset_type, dataset_size, patch_size, image_size

    NOTE: Models without patch_size are excluded from this analysis.
    """
    report.add_header(f"PARTIAL CORRELATIONS: {name}", level=2)

    # Prepare numeric data
    df_num = df.copy()
    df_num["y"] = df_num[metric_col]
    df_num["log_params"] = np.log10(df_num["total_params_M"].clip(lower=1))
    df_num["log_datasize"] = np.log10(df_num["training_data_size_M"].clip(lower=1))
    # Keep NaN for models without patch size
    df_num["log_patchsize"] = np.where(
        df_num["patch_size_num"].notna(),
        np.log2(df_num["patch_size_num"].clip(lower=1)),
        np.nan,
    )
    df_num["log_imgsize"] = np.log2(df_num["image_size"].clip(lower=1))

    # Encode architecture family as numeric (for partial correlation)
    df_num["is_siglip"] = (
        df_num["arch_family"].isin(["SigLIP", "SigLIP2"]).astype(float)
    )

    factors = [
        "is_siglip",
        "log_params",
        "log_datasize",
        "log_patchsize",
        "log_imgsize",
    ]
    factor_labels = {
        "is_siglip": "Architecture (SigLIP)",
        "log_params": "Model Parameters",
        "log_datasize": "Dataset Size",
        "log_patchsize": "Patch Size",
        "log_imgsize": "Image Size",
    }

    # Report exclusions
    n_excluded = df_num["log_patchsize"].isna().sum()
    report.add_line(
        f"\nNote: Excluding {n_excluded} models without patch_size (ResNet, ConvNeXt, etc.)"
    )

    df_num = df_num[["y"] + factors].dropna()

    report.add_kv("Samples", len(df_num))

    if len(df_num) < 30:
        report.add_line("Insufficient data.")
        return

    report.add_subheader("Zero-Order vs Partial Correlations")
    report.add_line("\nZero-order: raw correlation (potentially confounded)")
    report.add_line("Partial: correlation controlling for all other factors")
    report.add_line(
        f"\n{'Factor':<25} {'Zero-Order':>12} {'Partial':>12} {'Δ':>10} {'Interpretation':<20}"
    )
    report.add_line("-" * 85)

    results = []
    for factor in factors:
        # Zero-order correlation
        r_zero, p_zero = spearmanr(df_num[factor], df_num["y"])

        # Partial correlation using residuals method
        other_factors = [f for f in factors if f != factor]

        if HAS_STATSMODELS and len(other_factors) > 0:
            try:
                # Regress factor on other factors, get residuals
                formula_x = f"{factor} ~ " + " + ".join(other_factors)
                formula_y = "y ~ " + " + ".join(other_factors)

                resid_x = smf.ols(formula_x, data=df_num).fit().resid
                resid_y = smf.ols(formula_y, data=df_num).fit().resid
                r_partial, p_partial = pearsonr(resid_x, resid_y)
            except:
                r_partial = np.nan
        else:
            r_partial = r_zero

        diff = r_partial - r_zero if not np.isnan(r_partial) else np.nan

        # Interpretation
        if np.isnan(r_partial):
            interp = "N/A"
        elif abs(diff) < 0.05:
            interp = "Independent effect"
        elif diff < -0.1:
            interp = "Confounded ↓"
        elif diff > 0.1:
            interp = "Suppressed ↑"
        else:
            interp = "Slight confounding"

        results.append(
            {
                "factor": factor,
                "label": factor_labels.get(factor, factor),
                "r_zero": r_zero,
                "r_partial": r_partial,
                "diff": diff,
                "interp": interp,
            }
        )

        label = factor_labels.get(factor, factor)
        report.add_line(
            f"{label:<25} {r_zero:>+12.4f} {r_partial:>+12.4f} {diff:>+10.4f} {interp:<20}"
        )

    report.add_subheader("Interpretation Guide")
    report.add_line(
        "\n• Partial ≈ Zero-order: Factor has independent effect (not confounded)"
    )
    report.add_line(
        "• Partial << Zero-order: Effect was confounded (inflated by correlations)"
    )
    report.add_line(
        "• Partial >> Zero-order: Suppressor effect (other factors were masking it)"
    )

    # Highlight key findings
    report.add_subheader("Key Findings")
    for r in sorted(results, key=lambda x: abs(x.get("diff", 0) or 0), reverse=True):
        if r["diff"] is not None and not np.isnan(r["diff"]):
            if abs(r["diff"]) > 0.1:
                report.add_line(f"• {r['label']}: {r['interp']} (Δ = {r['diff']:+.3f})")

    return results


def compute_variance_partitioning(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    out_dir: str,
    report: ReportWriter,
):
    """
    Variance partitioning: decompose R² into unique and shared contributions.

    Factor groups:
    - Architecture: arch_family dummies
    - Scale: params, image_size, patch_size
    - Data: dataset_type, dataset_size
    """
    report.add_header(f"VARIANCE PARTITIONING: {name}", level=2)

    if not HAS_STATSMODELS:
        report.add_line("statsmodels not available.")
        return

    # Prepare data
    df_r = df.copy()
    df_r["y"] = df_r[metric_col] * 100

    # Architecture dummies
    df_r["is_siglip"] = df_r["arch_family"].isin(["SigLIP", "SigLIP2"]).astype(int)
    df_r["is_coca"] = (df_r["arch_family"] == "CoCa").astype(int)
    df_r["is_eva"] = (df_r["arch_family"] == "EVA").astype(int)

    # Dataset type dummies
    def get_data_group(data):
        data = str(data).lower()
        if "webli" in data:
            return "WebLI"
        elif "datacomp" in data:
            return "DataComp"
        elif "dfn" in data:
            return "DFN"
        else:
            return "Other"

    df_r["data_group"] = df_r["training_data"].apply(get_data_group)
    df_r["is_webli"] = (df_r["data_group"] == "WebLI").astype(int)
    df_r["is_datacomp"] = (df_r["data_group"] == "DataComp").astype(int)
    df_r["is_dfn"] = (df_r["data_group"] == "DFN").astype(int)

    # Scale variables
    df_r["log_params"] = np.log10(df_r["total_params_M"].clip(lower=1))
    df_r["log_imgsize"] = np.log2(df_r["image_size"].clip(lower=1))
    # Keep NaN for models without patch size
    df_r["log_patchsize"] = np.where(
        df_r["patch_size_num"].notna(),
        np.log2(df_r["patch_size_num"].clip(lower=1)),
        np.nan,
    )
    df_r["log_datasize"] = np.log10(df_r["training_data_size_M"].clip(lower=1))

    # Report exclusions
    n_excluded = df_r["log_patchsize"].isna().sum()
    report.add_line(
        f"\nNote: Excluding {n_excluded} models without patch_size (ResNet, ConvNeXt, etc.)"
    )

    df_r = df_r.dropna(
        subset=[
            "y",
            "is_siglip",
            "is_coca",
            "log_params",
            "log_imgsize",
            "log_patchsize",
            "log_datasize",
        ]
    )

    if len(df_r) < 50:
        report.add_line("Insufficient data.")
        return

    report.add_kv("Samples", len(df_r))

    # Define factor groups
    arch_vars = "is_siglip + is_coca + is_eva"
    scale_vars = "log_params + log_imgsize + log_patchsize"
    data_vars = "is_webli + is_datacomp + is_dfn + log_datasize"

    # Fit all possible models for variance decomposition
    models = {}
    formulas = {
        "none": "y ~ 1",
        "arch": f"y ~ {arch_vars}",
        "scale": f"y ~ {scale_vars}",
        "data": f"y ~ {data_vars}",
        "arch_scale": f"y ~ {arch_vars} + {scale_vars}",
        "arch_data": f"y ~ {arch_vars} + {data_vars}",
        "scale_data": f"y ~ {scale_vars} + {data_vars}",
        "full": f"y ~ {arch_vars} + {scale_vars} + {data_vars}",
    }

    for key, formula in formulas.items():
        try:
            models[key] = smf.ols(formula, data=df_r).fit().rsquared
        except:
            models[key] = 0

    # Calculate unique contributions (Type III: what each adds when entered last)
    unique_arch = models["full"] - models["scale_data"]
    unique_scale = models["full"] - models["arch_data"]
    unique_data = models["full"] - models["arch_scale"]

    # Shared variance
    total_r2 = models["full"]
    total_unique = unique_arch + unique_scale + unique_data
    shared = total_r2 - total_unique

    report.add_subheader("Individual Model R² Values")
    report.add_line(f"\n{'Model':<40} {'R²':>10}")
    report.add_line("-" * 55)
    for key in [
        "none",
        "arch",
        "scale",
        "data",
        "arch_scale",
        "arch_data",
        "scale_data",
        "full",
    ]:
        report.add_line(f"{key:<40} {models.get(key, 0):>10.4f}")

    report.add_subheader("Unique Contributions (Type III)")
    report.add_line("\nUnique = variance explained ONLY by this factor group,")
    report.add_line("not shared with any other factor group.")
    report.add_line(
        f"\n{'Factor Group':<25} {'Unique R²':>12} {'% of Total':>12} {'Variables':<30}"
    )
    report.add_line("-" * 85)

    factor_info = [
        ("Architecture", unique_arch, "SigLIP, CoCa, EVA dummies"),
        ("Scale", unique_scale, "params, img_size, patch_size"),
        ("Data", unique_data, "dataset_type, dataset_size"),
    ]

    for factor, unique, variables in factor_info:
        pct = (unique / total_r2 * 100) if total_r2 > 0 else 0
        report.add_line(f"{factor:<25} {unique:>12.4f} {pct:>11.1f}% {variables:<30}")

    pct_shared = (shared / total_r2 * 100) if total_r2 > 0 else 0
    report.add_line(f"{'Shared/Confounded':<25} {shared:>12.4f} {pct_shared:>11.1f}%")
    report.add_line("-" * 85)
    report.add_line(f"{'TOTAL EXPLAINED':<25} {total_r2:>12.4f} {'100.0':>11}%")
    report.add_line(f"{'Unexplained':<25} {1-total_r2:>12.4f}")

    report.add_subheader("Interpretation")

    # Rank factor importance
    ranked = sorted(factor_info, key=lambda x: x[1], reverse=True)
    report.add_line("\nFactor importance ranking (by unique variance):")
    for i, (factor, unique, _) in enumerate(ranked, 1):
        pct = (unique / total_r2 * 100) if total_r2 > 0 else 0
        report.add_line(f"  {i}. {factor}: {pct:.1f}% of explained variance")

    if pct_shared > 30:
        report.add_line(
            f"\n⚠ Warning: {pct_shared:.1f}% of variance is shared (confounded)."
        )
        report.add_line(
            "  Factors are highly correlated; interpret individual effects with caution."
        )

    # Plot variance partitioning
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Stacked bar showing unique + shared
    ax = axes[0]
    categories = ["Architecture", "Scale", "Data", "Shared"]
    values = [unique_arch * 100, unique_scale * 100, unique_data * 100, shared * 100]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        if val > 2:  # Only label if visible
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_BAR_LABEL,
                fontweight="bold",
            )

    ax.set_ylabel(
        f"Variance Explained (% of Total R² = {total_r2*100:.1f}%)",
        fontsize=FONT_SIZE_AXIS_LABEL,
    )
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Unique vs Shared Variance", fontsize=FONT_SIZE_TITLE)

    # Plot 2: Pie chart
    ax = axes[1]
    sizes = [unique_arch, unique_scale, unique_data, shared, 1 - total_r2]
    labels = [
        "Architecture\n(unique)",
        "Scale\n(unique)",
        "Data\n(unique)",
        "Shared",
        "Unexplained",
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6", "#ecf0f1"]
    explode = (0.02, 0.02, 0.02, 0.02, 0)

    # Filter out tiny slices
    filtered = [
        (s, l, c, e) for s, l, c, e in zip(sizes, labels, colors, explode) if s > 0.01
    ]
    if filtered:
        sizes_f, labels_f, colors_f, explode_f = zip(*filtered)
        ax.pie(
            sizes_f,
            labels=labels_f,
            colors=colors_f,
            explode=explode_f,
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            startangle=90,
            counterclock=False,
        )
    ax.set_title("Variance Decomposition")

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_variance_partition_{name.lower()}")
    plt.close()

    return {
        "unique_arch": unique_arch,
        "unique_scale": unique_scale,
        "unique_data": unique_data,
        "shared": shared,
        "total_r2": total_r2,
    }


def compute_feature_importance_rf(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    out_dir: str,
    report: ReportWriter,
):
    """
    Random Forest feature importance with cross-validation.

    Features: arch_family, params, dataset_type, dataset_size, patch_size, image_size
    """
    report.add_header(f"RANDOM FOREST IMPORTANCE: {name}", level=2)

    if not HAS_SKLEARN:
        report.add_line("sklearn not available.")
        return

    # Prepare features
    df_ml = df.copy()

    # Encode categorical variables
    encoders = {}
    for col in ["arch_family", "training_data"]:
        if col in df_ml.columns:
            le = LabelEncoder()
            df_ml[f"{col}_enc"] = le.fit_transform(
                df_ml[col].fillna("Unknown").astype(str)
            )
            encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Feature columns with clear labels
    feature_config = [
        ("arch_family_enc", "Architecture Family"),
        ("total_params_M", "Model Parameters (M)"),
        ("training_data_enc", "Dataset Type"),
        ("training_data_size_M", "Dataset Size (M)"),
        ("patch_size_num", "Patch Size"),
        ("image_size", "Image Size"),
    ]

    # Additional features if available
    extra_features = [
        ("embed_dim", "Embedding Dim"),
        ("num_tokens", "Num Tokens"),
    ]

    feature_cols = []
    feature_labels = []

    for col, label in feature_config + extra_features:
        if col in df_ml.columns:
            feature_cols.append(col)
            feature_labels.append(label)

    # Report exclusions before dropna
    n_before = len(df_ml)
    if "patch_size_num" in feature_cols:
        n_without_patch = df_ml["patch_size_num"].isna().sum()
        report.add_line(
            f"\nNote: {n_without_patch} models lack patch_size and will be excluded."
        )

    df_ml = df_ml.dropna(subset=[metric_col] + feature_cols)

    report.add_kv("Samples", len(df_ml))
    report.add_kv("Features", len(feature_cols))

    if len(df_ml) < 50:
        report.add_line("Insufficient data.")
        return

    X = df_ml[feature_cols].values
    y = df_ml[metric_col].values

    # Fit Random Forest
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    report.add_kv("CV R² mean", cv_scores.mean())
    report.add_kv("CV R² std", cv_scores.std())
    report.add_kv("Training R²", rf.score(X, y))

    # Feature importance
    importance = pd.DataFrame(
        {
            "Feature": feature_labels,
            "Column": feature_cols,
            "Importance": rf.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    report.add_subheader("Feature Importance Ranking")
    report.add_line("\nImportance = Mean Decrease in Impurity (Gini importance)")
    report.add_line(
        f"\n{'Rank':<5} {'Feature':<25} {'Importance':>12} {'Cumulative':>12}"
    )
    report.add_line("-" * 60)

    cumsum = 0
    for i, (_, row) in enumerate(importance.iterrows()):
        cumsum += row["Importance"]
        report.add_line(
            f"{i+1:<5} {row['Feature']:<25} {row['Importance']:>12.4f} {cumsum:>12.4f}"
        )

    # Group importance by factor category
    report.add_subheader("Importance by Factor Category")

    arch_imp = importance[importance["Column"].str.contains("arch")]["Importance"].sum()
    scale_imp = importance[
        importance["Column"].isin(
            [
                "total_params_M",
                "image_size",
                "patch_size_num",
                "num_tokens",
                "embed_dim",
            ]
        )
    ]["Importance"].sum()
    data_imp = importance[importance["Column"].str.contains("training_data|datasize")][
        "Importance"
    ].sum()

    report.add_line(f"\n{'Category':<20} {'Importance':>12}")
    report.add_line("-" * 35)
    for cat, imp in [
        ("Architecture", arch_imp),
        ("Scale", scale_imp),
        ("Data", data_imp),
    ]:
        report.add_line(f"{cat:<20} {imp:>12.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Individual features
    ax = axes[0]
    y_pos = np.arange(len(importance))
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(importance)))

    ax.barh(
        y_pos, importance["Importance"], color=colors, edgecolor="black", linewidth=0.5
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance["Feature"], fontsize=FONT_SIZE_AXIS_TICK)
    ax.set_xlabel("Feature Importance", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title("Individual Features", fontsize=FONT_SIZE_TITLE)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    # Plot 2: Category-level
    ax = axes[1]
    categories = ["Architecture", "Scale", "Data"]
    values = [arch_imp, scale_imp, data_imp]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_BAR_LABEL,
            fontweight="bold",
        )
    ax.set_ylabel("Total Importance", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title("By Factor Category", fontsize=FONT_SIZE_TITLE)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_rf_importance_{name.lower()}")
    plt.close()

    # Save importance table
    importance.to_csv(f"{out_dir}/feature_importance_{name.lower()}.csv", index=False)

    # Save encodings for reference
    if encoders:
        report.add_subheader("Categorical Encodings")
        for col, mapping in encoders.items():
            report.add_line(f"\n{col}:")
            for cat, code in sorted(mapping.items(), key=lambda x: x[1]):
                report.add_line(f"  {code}: {cat}")

    return importance


def analyze_interactions(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    out_dir: str,
    report: ReportWriter,
):
    """
    Analyze interaction effects between factors.

    Key interactions to test:
    - Architecture × Scale: Do larger models benefit more from certain architectures?
    - Architecture × Data: Do certain architectures benefit more from certain datasets?
    - Scale × Data: Do larger models benefit more from larger datasets?
    """
    report.add_header(f"INTERACTION ANALYSIS: {name}", level=2)

    if not HAS_STATSMODELS:
        report.add_line("statsmodels not available.")
        return

    # Prepare data
    df_r = df.copy()
    df_r["y"] = df_r[metric_col] * 100

    # Architecture dummy
    df_r["is_siglip"] = df_r["arch_family"].isin(["SigLIP", "SigLIP2"]).astype(int)

    # Scale variables (standardized)
    df_r["log_params"] = np.log10(df_r["total_params_M"].clip(lower=1))
    df_r["log_imgsize"] = np.log2(df_r["image_size"].clip(lower=1))
    df_r["log_datasize"] = np.log10(df_r["training_data_size_M"].clip(lower=1))

    for col in ["log_params", "log_imgsize", "log_datasize"]:
        mean, std = df_r[col].mean(), df_r[col].std()
        if std > 0:
            df_r[f"{col}_z"] = (df_r[col] - mean) / std

    df_r = df_r.dropna(
        subset=["y", "is_siglip", "log_params_z", "log_imgsize_z", "log_datasize_z"]
    )

    if len(df_r) < 50:
        report.add_line("Insufficient data.")
        return

    report.add_kv("Samples", len(df_r))

    # ─── Test interaction effects ──────────────────────────────────
    report.add_subheader("Testing Interaction Effects")

    # Model without interactions
    formula_main = "y ~ is_siglip + log_params_z + log_imgsize_z + log_datasize_z"
    model_main = smf.ols(formula_main, data=df_r).fit()

    # Model with key interactions
    formula_int = """y ~ is_siglip + log_params_z + log_imgsize_z + log_datasize_z 
                     + is_siglip:log_params_z + is_siglip:log_datasize_z 
                     + log_params_z:log_datasize_z"""
    model_int = smf.ols(formula_int, data=df_r).fit()

    report.add_line(
        f"\nMain effects only: R² = {model_main.rsquared:.4f}, Adj R² = {model_main.rsquared_adj:.4f}"
    )
    report.add_line(
        f"With interactions: R² = {model_int.rsquared:.4f}, Adj R² = {model_int.rsquared_adj:.4f}"
    )
    report.add_line(
        f"Improvement: ΔR² = {model_int.rsquared - model_main.rsquared:.4f}"
    )

    # F-test for interaction terms
    from scipy.stats import f as f_dist

    df1 = model_int.df_model - model_main.df_model
    df2 = model_int.df_resid
    ssr_main = model_main.ssr
    ssr_int = model_int.ssr

    if df1 > 0 and ssr_int > 0:
        f_stat = ((ssr_main - ssr_int) / df1) / (ssr_int / df2)
        p_value = 1 - f_dist.cdf(f_stat, df1, df2)
        report.add_line(
            f"\nF-test for interactions: F={f_stat:.3f}, df=({df1}, {df2}), p={p_value:.4f}"
        )

        if p_value < 0.05:
            report.add_line("→ Interaction effects are statistically significant!")
        else:
            report.add_line("→ No significant interaction effects detected.")

    report.add_subheader("Interaction Coefficients")
    report.add_line(f"\n{'Term':<40} {'Coef':>10} {'SE':>10} {'p':>12} {'Sig':>5}")
    report.add_line("-" * 80)

    for var in model_int.params.index:
        c = model_int.params[var]
        se = model_int.bse[var]
        p = model_int.pvalues[var]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        is_interaction = ":" in var
        prefix = "  → " if is_interaction else ""
        report.add_line(f"{prefix}{var:<38} {c:>10.3f} {se:>10.3f} {p:>12.4f} {sig:>5}")

    # ─── Interpretation ────────────────────────────────────────────
    report.add_subheader("Interaction Interpretation")

    for var in model_int.params.index:
        if ":" in var and model_int.pvalues[var] < 0.05:
            c = model_int.params[var]
            parts = var.split(":")

            if "is_siglip" in var and "log_params" in var:
                if c > 0:
                    report.add_line(
                        f"\n• SigLIP × Model Size: SigLIP benefits MORE from larger models"
                    )
                    report.add_line(
                        f"  (Each 1 SD increase in size gives SigLIP an extra {c:.2f}pp)"
                    )
                else:
                    report.add_line(
                        f"\n• SigLIP × Model Size: SigLIP benefits LESS from larger models"
                    )

            elif "is_siglip" in var and "log_datasize" in var:
                if c > 0:
                    report.add_line(
                        f"\n• SigLIP × Data Size: SigLIP benefits MORE from larger datasets"
                    )
                else:
                    report.add_line(
                        f"\n• SigLIP × Data Size: SigLIP benefits LESS from larger datasets"
                    )

            elif "log_params" in var and "log_datasize" in var:
                if c > 0:
                    report.add_line(
                        f"\n• Size × Data: Larger models benefit MORE from larger datasets"
                    )
                else:
                    report.add_line(
                        f"\n• Size × Data: Larger models benefit LESS from larger datasets"
                    )

    # ─── Visualization ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Metric by params, split by architecture
    ax = axes[0]
    for is_sig, color, label in [
        (0, "#3498db", "CLIP/Other"),
        (1, "#2ecc71", "SigLIP"),
    ]:
        mask = df_r["is_siglip"] == is_sig
        if mask.sum() > 5:
            x = df_r.loc[mask, "log_params_z"]
            y_vals = df_r.loc[mask, "y"]
            ax.scatter(x, y_vals, c=color, alpha=0.4, s=30, label=label)

            # Trend line
            slope, intercept = np.polyfit(x, y_vals, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, c=color, linewidth=2.5)

    ax.set_xlabel("Model Size (standardized)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel(f"{metric_label} (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title("Architecture × Model Size", fontsize=FONT_SIZE_TITLE)
    _place_legend(ax, loc="best")
    ax.grid(True, alpha=0.3)

    # Plot 2: Metric by data size, split by architecture
    ax = axes[1]
    for is_sig, color, label in [
        (0, "#3498db", "CLIP/Other"),
        (1, "#2ecc71", "SigLIP"),
    ]:
        mask = df_r["is_siglip"] == is_sig
        if mask.sum() > 5:
            x = df_r.loc[mask, "log_datasize_z"]
            y_vals = df_r.loc[mask, "y"]
            ax.scatter(x, y_vals, c=color, alpha=0.4, s=30, label=label)

            slope, intercept = np.polyfit(x, y_vals, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, c=color, linewidth=2.5)

    ax.set_xlabel("Dataset Size (standardized)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel(f"{metric_label} (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title("Architecture × Dataset Size", fontsize=FONT_SIZE_TITLE)
    _place_legend(ax, loc="best")
    ax.grid(True, alpha=0.3)

    # Plot 3: Metric by params, split by data size (median split)
    ax = axes[2]
    median_data = df_r["log_datasize_z"].median()
    for is_large, color, label in [
        (False, "#9b59b6", "Smaller Data"),
        (True, "#f39c12", "Larger Data"),
    ]:
        if is_large:
            mask = df_r["log_datasize_z"] >= median_data
        else:
            mask = df_r["log_datasize_z"] < median_data

        if mask.sum() > 5:
            x = df_r.loc[mask, "log_params_z"]
            y_vals = df_r.loc[mask, "y"]
            ax.scatter(x, y_vals, c=color, alpha=0.4, s=30, label=label)

            slope, intercept = np.polyfit(x, y_vals, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, c=color, linewidth=2.5)

    ax.set_xlabel("Model Size (standardized)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel(f"{metric_label} (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title("Model Size × Dataset Size", fontsize=FONT_SIZE_TITLE)
    _place_legend(ax, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_interactions_{name.lower()}")
    plt.close()


def run_full_multivariate_analysis(
    df: pd.DataFrame,
    name: str,
    metric_col: str,
    metric_label: str,
    out_dir: str,
    report: ReportWriter,
):
    """
    Run all multivariate analyses for a dataset.
    """
    report.add_header(f"MULTIVARIATE ANALYSIS: {name}", level=1)
    report.add_line("\nThis section analyzes confounded relationships between factors.")
    report.add_line("Goal: Identify the independent contribution of each factor,")
    report.add_line("controlling for other factors that may be correlated.")

    # 1. Multiple regression
    run_multivariate_regression(df, name, metric_col, metric_label, out_dir, report)

    # 2. Partial correlations
    compute_partial_correlations(df, name, metric_col, metric_label, out_dir, report)

    # 3. Variance partitioning
    compute_variance_partitioning(df, name, metric_col, metric_label, out_dir, report)

    # 4. Random Forest importance
    compute_feature_importance_rf(df, name, metric_col, metric_label, out_dir, report)

    # 5. Interaction analysis
    analyze_interactions(df, name, metric_col, metric_label, out_dir, report)


def plot_pareto_3way(merged_3way: pd.DataFrame, out_dir: str, report: ReportWriter):
    """Pareto-optimal models across all 3 metrics."""
    report.add_header("FIGURE: 3-Way Pareto Analysis", level=2)

    if len(merged_3way) < 10:
        report.add_line("Insufficient data.")
        return

    # Find Pareto-optimal in 3D (maximize all three)
    metrics = ["ImageNet", "UrbanCars_WG", "CelebA_WG"]
    values = merged_3way[metrics].values

    n = len(values)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                # j dominates i if j >= i on all and j > i on at least one
                if all(values[j] >= values[i]) and any(values[j] > values[i]):
                    pareto[i] = False
                    break

    report.add_line(f"\n3-Way Pareto-optimal models ({pareto.sum()}):")
    pareto_df = merged_3way[pareto][
        ["model_id", "training_objective", "arch_family"] + metrics
    ]
    pareto_df = pareto_df.sort_values("ImageNet", ascending=False)

    for _, row in pareto_df.iterrows():
        report.add_line(
            f"  {row['model_id']}: IN={row['ImageNet']:.1f}%, UC={row['UrbanCars_WG']:.1f}%, CA={row['CelebA_WG']:.1f}%"
        )

    # Plot: ImageNet vs average WG
    merged_3way["avg_wg"] = (merged_3way["UrbanCars_WG"] + merged_3way["CelebA_WG"]) / 2

    fig, ax = plt.subplots(figsize=(10, 8))

    x = merged_3way["ImageNet"].values
    y = merged_3way["avg_wg"].values

    archs = sorted(merged_3way["arch_family"].dropna().unique())
    for arch in archs:
        m = merged_3way["arch_family"] == arch
        color = COLORS_ARCH.get(arch, "#95a5a6")
        ax.scatter(
            x[m.values],
            y[m.values],
            c=color,
            label=f"{arch}",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.3,
        )

    # Highlight 3-way Pareto
    ax.scatter(
        x[pareto],
        y[pareto],
        facecolors="none",
        edgecolors="darkred",
        linewidth=2.5,
        s=180,
        label=f"3-Way Pareto (n={pareto.sum()})",
        zorder=4,
    )

    # Label Pareto
    for idx in np.where(pareto)[0]:
        model_id = merged_3way.iloc[idx]["model_id"]
        short_name = (
            model_id.replace("ViT-", "")
            .replace("-quickgelu", "")
            .replace("_", " ")[:20]
        )
        ax.annotate(
            short_name,
            (x[idx], y[idx]),
            fontsize=FONT_SIZE_ANNOTATION,
            xytext=(5, 5),
            textcoords="offset points",
            color="darkred",
            fontweight="bold",
        )

    ax.plot([0, 100], [0, 100], "k--", alpha=0.3)
    ax.set_xlabel("ImageNet Top-1 Accuracy (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel(
        "Average WG Accuracy (UrbanCars + CelebA) (%)", fontsize=FONT_SIZE_AXIS_LABEL
    )
    _place_legend(ax, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_pareto_3way")
    plt.close()

    # Save Pareto models
    pareto_df.to_csv(f"{out_dir}/pareto_3way_models.csv", index=False)


# ═══════════════════════════════════════════════════════════════════
# Cross-Dataset Scale Barplots (all datasets in one figure)
# ═══════════════════════════════════════════════════════════════════


def _compute_scale_bin_stats(
    df: pd.DataFrame,
    factor_col: str,
    metric_col: str,
    n_bins: int = 6,
    use_log: bool = False,
    shared_edges: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Helper: bin a numeric factor and compute mean±std per bin.

    Returns (bin_stats DataFrame, bin_edges used).
    """
    mask = df[factor_col].notna() & df[metric_col].notna()
    if mask.sum() < 5:
        return pd.DataFrame(), np.array([])

    x_all = df.loc[mask, factor_col].values

    if shared_edges is not None:
        bin_edges = shared_edges
    elif use_log and x_all.min() > 0:
        log_min, log_max = np.log10(x_all.min()), np.log10(x_all.max())
        bin_edges = np.logspace(log_min, log_max, n_bins)
    else:
        bin_edges = np.linspace(x_all.min(), x_all.max(), n_bins)

    df_temp = df.loc[mask].copy()
    df_temp["bin"] = pd.cut(df_temp[factor_col], bins=bin_edges, include_lowest=True)
    bin_stats = df_temp.groupby("bin", observed=True)[metric_col].agg(
        ["mean", "std", "count"]
    )
    bin_stats = bin_stats[bin_stats["count"] >= 2]
    return bin_stats, bin_edges


def plot_scale_patchsize_bar_all_datasets(
    datasets: Dict[str, Tuple[pd.DataFrame, str, str]],
    out_dir: str,
    report: ReportWriter,
    n_bins: int = 6,
):
    """Grouped barplot of metric vs binned patch size for all datasets in one figure.

    Args:
        datasets: {name: (df, metric_col, metric_label)}
    """
    report.add_header("FIGURE: Patch Size Bars – All Datasets", level=2)

    factor_col = "patch_size_num"
    factor_label = "Patch Size (px)"

    # Determine shared bin edges from the union of all data
    all_x = []
    for name, (df, metric_col, _) in datasets.items():
        if factor_col in df.columns and metric_col in df.columns:
            mask = df[factor_col].notna() & df[metric_col].notna()
            all_x.extend(df.loc[mask, factor_col].tolist())

    if len(all_x) < 10:
        report.add_line("Insufficient data for cross-dataset patch size plot.")
        return

    all_x = np.array(all_x)
    bin_edges = np.linspace(all_x.min(), all_x.max(), n_bins)

    # Collect per-dataset stats
    dataset_stats = {}
    for name, (df, metric_col, metric_label) in datasets.items():
        if factor_col not in df.columns or metric_col not in df.columns:
            continue
        stats, _ = _compute_scale_bin_stats(
            df, factor_col, metric_col, n_bins=n_bins, shared_edges=bin_edges
        )
        if len(stats) > 0:
            dataset_stats[name] = (stats, metric_label)

    if len(dataset_stats) == 0:
        report.add_line("No valid data found.")
        return

    # Build common x-axis labels from the shared bin edges
    bin_labels = [
        f"{bin_edges[i]:.0f}–{bin_edges[i+1]:.0f}" for i in range(len(bin_edges) - 1)
    ]

    n_datasets = len(dataset_stats)
    bar_width = 0.8 / n_datasets
    dataset_colors = _bar_colors(n_datasets)  # viridis, consistent with scale bar plots

    fig, ax = plt.subplots(figsize=(8, 6))

    x_pos = np.arange(len(bin_labels))

    for i, (ds_name, (stats, metric_label)) in enumerate(dataset_stats.items()):
        # Map each bin label to an index
        offsets = (i - (n_datasets - 1) / 2) * bar_width
        # Align stats bins to the shared label list
        means = []
        stds = []
        for interval in stats.index:
            means.append(stats.loc[interval, "mean"] * 100)
            stds.append(stats.loc[interval, "std"] * 100)

        # Map bins to correct x positions (stats may not cover all bins)
        stats_labels = [
            f"{interval.left:.0f}–{interval.right:.0f}" for interval in stats.index
        ]
        bar_x = []
        bar_means = []
        bar_stds = []
        for j, bl in enumerate(bin_labels):
            if bl in stats_labels:
                idx = stats_labels.index(bl)
                bar_x.append(x_pos[j] + offsets)
                bar_means.append(means[idx])
                bar_stds.append(stds[idx])

        if bar_x:
            ax.bar(
                bar_x,
                bar_means,
                width=bar_width,
                yerr=bar_stds,
                label=ds_name,
                color=dataset_colors[i],
                edgecolor="black",
                linewidth=0.2,
                capsize=3,
                alpha=0.55,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        bin_labels, rotation=45, ha="right", fontsize=FONT_SIZE_AXIS_TICK
    )
    ax.set_xlabel(factor_label, fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel("Metric (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title("Effect of Patch Size across Datasets", fontsize=FONT_SIZE_TITLE)
    _place_legend(ax, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_scale_patchsizenum_bar_all_datasets")
    plt.close()
    report.add_line("Saved: fig_scale_patchsizenum_bar_all_datasets")


def plot_scale_imagesize_bar_all_datasets(
    datasets: Dict[str, Tuple[pd.DataFrame, str, str]],
    out_dir: str,
    report: ReportWriter,
    n_bins: int = 6,
):
    """Grouped barplot of metric vs binned image size for all datasets in one figure.

    Args:
        datasets: {name: (df, metric_col, metric_label)}
    """
    report.add_header("FIGURE: Image Size Bars – All Datasets", level=2)

    factor_col = "image_size"
    factor_label = "Image Size (px)"

    # Shared bin edges from union of all data
    all_x = []
    for name, (df, metric_col, _) in datasets.items():
        if factor_col in df.columns and metric_col in df.columns:
            mask = df[factor_col].notna() & df[metric_col].notna()
            all_x.extend(df.loc[mask, factor_col].tolist())

    if len(all_x) < 10:
        report.add_line("Insufficient data for cross-dataset image size plot.")
        return

    all_x = np.array(all_x)
    bin_edges = np.linspace(all_x.min(), all_x.max(), n_bins)

    dataset_stats = {}
    for name, (df, metric_col, metric_label) in datasets.items():
        if factor_col not in df.columns or metric_col not in df.columns:
            continue
        stats, _ = _compute_scale_bin_stats(
            df, factor_col, metric_col, n_bins=n_bins, shared_edges=bin_edges
        )
        if len(stats) > 0:
            dataset_stats[name] = (stats, metric_label)

    if len(dataset_stats) == 0:
        report.add_line("No valid data found.")
        return

    bin_labels = [
        f"{bin_edges[i]:.0f}–{bin_edges[i+1]:.0f}" for i in range(len(bin_edges) - 1)
    ]

    n_datasets = len(dataset_stats)
    bar_width = 0.8 / n_datasets
    dataset_colors = _bar_colors(n_datasets)  # viridis, consistent with scale bar plots

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(bin_labels))

    for i, (ds_name, (stats, metric_label)) in enumerate(dataset_stats.items()):
        offsets = (i - (n_datasets - 1) / 2) * bar_width

        stats_labels = [
            f"{interval.left:.0f}–{interval.right:.0f}" for interval in stats.index
        ]
        bar_x = []
        bar_means = []
        bar_stds = []
        for j, bl in enumerate(bin_labels):
            if bl in stats_labels:
                idx = stats_labels.index(bl)
                bar_x.append(x_pos[j] + offsets)
                bar_means.append(stats.iloc[idx]["mean"] * 100)
                bar_stds.append(stats.iloc[idx]["std"] * 100)

        if bar_x:
            ax.bar(
                bar_x,
                bar_means,
                width=bar_width,
                yerr=bar_stds,
                label=ds_name,
                color=dataset_colors[i],
                edgecolor="black",
                linewidth=0.2,
                capsize=3,
                alpha=0.55,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        bin_labels, rotation=45, ha="right", fontsize=FONT_SIZE_AXIS_TICK
    )
    ax.set_xlabel(factor_label, fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel("Metric (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title("Effect of Image Size across Datasets", fontsize=FONT_SIZE_TITLE)
    _place_legend(ax, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_scale_imagesize_bar_all_datasets")
    plt.close()
    report.add_line("Saved: fig_scale_imagesize_bar_all_datasets")


def analyze_single_dataset(
    df: pd.DataFrame,
    name: str,
    out_dir: str,
    report: ReportWriter,
    min_n: int = 3,
    metric_col: str = "worst_group_accuracy",
    metric_label: str = "Worst-Group Accuracy",
):
    """Full analysis for a single dataset."""
    report.add_header(f"ANALYSIS: {name}", level=1)

    # Overall statistics
    analyze_overall(df, name, metric_col, metric_label, report)

    # Categorical factors
    for fcol, flabel in [
        ("training_objective", "Training Objective"),
        ("training_data", "Training Data"),
        ("arch_family", "Architecture"),
        ("resolution_bucket", "Resolution Bucket"),
        ("params_bucket", "Params Bucket"),
    ]:
        if fcol in df.columns:
            analyze_categorical(
                df, fcol, flabel, metric_col, metric_label, name, report, min_n
            )

    # Numeric factors
    for fcol, flabel in [
        ("total_params_M", "Parameters (M)"),
        ("image_size", "Image Size"),
        ("patch_size_num", "Patch Size"),
        ("num_tokens", "Tokens"),
        ("embed_dim", "Embed Dim"),
        ("training_data_size_M", "Data Size (M)"),
    ]:
        if fcol in df.columns:
            analyze_numeric(df, fcol, flabel, metric_col, metric_label, name, report)

    # Top/bottom models
    analyze_top_bottom(df, name, metric_col, metric_label, report, n=20)

    # Plots
    # Landscape
    if name == "ImageNet":
        plot_landscape(
            df,
            name,
            out_dir,
            report,
            "top1_accuracy",
            "top5_accuracy",
            "Top-1 Accuracy (%)",
            "Top-5 Accuracy (%)",
        )
    else:
        plot_landscape(
            df,
            name,
            out_dir,
            report,
            "avg_accuracy",
            "worst_group_accuracy",
            "Average Accuracy (%)",
            "Worst-Group Accuracy (%)",
        )

    # Factor plots
    for fcol, flabel in [
        ("training_objective", "objective"),
        ("arch_family", "arch"),
        ("training_data", "data"),
    ]:
        if fcol in df.columns:
            plot_by_factor(
                df, name, out_dir, report, fcol, flabel, metric_col, metric_label, min_n
            )

    # Scale plots
    scale_factors = [
        ("total_params_M", "Parameters (M)", True),
        ("image_size", "Image Size (px)", False),
        ("patch_size_num", "Patch Size (px)", False),
        ("num_tokens", "Number of Tokens", True),
        ("training_data_size_M", "Training Data Size (M)", True),
    ]

    for fcol, flabel, use_log in scale_factors:
        if fcol in df.columns:
            plot_scale_scatter(
                df,
                name,
                out_dir,
                report,
                fcol,
                flabel,
                metric_col,
                metric_label,
                use_log,
            )
            n_bins = 9 if fcol == "num_tokens" else 6
            plot_scale_bars(
                df,
                name,
                out_dir,
                report,
                fcol,
                flabel,
                metric_col,
                metric_label,
                n_bins,
                use_log,
            )

    # Correlation heatmap
    plot_correlation_heatmap(df, name, out_dir, report, metric_col, metric_label)

    # Isolated factor analysis (natural experiments)
    run_isolated_factor_analysis(df, name, metric_col, metric_label, out_dir, report)

    # Multivariate analysis (confounder control)
    run_full_multivariate_analysis(df, name, metric_col, metric_label, out_dir, report)


def main():
    parser = argparse.ArgumentParser(
        description="VLM Robustness Analysis with ImageNet Baseline"
    )
    parser.add_argument("--imagenet", required=True, help="ImageNet results CSV")
    parser.add_argument("--urbancars", required=True, help="UrbanCars results CSV")
    parser.add_argument("--celeba", default=None, help="CelebA results CSV (optional)")
    parser.add_argument("--output", default="paper_analysis", help="Output directory")
    parser.add_argument("--min-n", type=int, default=3, help="Min samples per group")
    args = parser.parse_args()

    # print(paok)
    os.makedirs(args.output, exist_ok=True)
    report = ReportWriter()

    report.add_header("VLM ROBUSTNESS ANALYSIS - COMPLETE REPORT", level=1)
    report.add_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.add_line(f"Output: {args.output}")
    report.add_line("\nThis report analyzes VLM robustness across:")
    report.add_line("  - ImageNet: General capability (Top-1 accuracy)")
    report.add_line("  - UrbanCars: Robustness to spurious correlations (WG accuracy)")
    if args.celeba:
        report.add_line("  - CelebA: Robustness to demographic biases (WG accuracy)")

    # ═══════════════════════════════════════════════════════════════
    # Load datasets
    # ═══════════════════════════════════════════════════════════════

    print("=" * 80)
    print("Loading ImageNet...")
    df_imagenet = load_and_enrich(args.imagenet, "ImageNet", report, is_imagenet=True)
    print(f"Loaded {len(df_imagenet)} models")

    print("\nLoading UrbanCars...")
    df_uc = load_and_enrich(args.urbancars, "UrbanCars", report, is_imagenet=False)
    print(f"Loaded {len(df_uc)} models")

    df_celeba = None
    if args.celeba:
        print("\nLoading CelebA...")
        df_celeba = load_and_enrich(args.celeba, "CelebA", report, is_imagenet=False)
        print(f"Loaded {len(df_celeba)} models")

    # ═══════════════════════════════════════════════════════════════
    # Single dataset analyses
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("Analyzing ImageNet...")
    analyze_single_dataset(
        df_imagenet,
        "ImageNet",
        args.output,
        report,
        args.min_n,
        metric_col="top1_accuracy",
        metric_label="Top-1 Accuracy",
    )

    print("\nAnalyzing UrbanCars...")
    analyze_single_dataset(
        df_uc,
        "UrbanCars",
        args.output,
        report,
        args.min_n,
        metric_col="worst_group_accuracy",
        metric_label="Worst-Group Accuracy",
    )

    if df_celeba is not None:
        print("\nAnalyzing CelebA...")
        analyze_single_dataset(
            df_celeba,
            "CelebA",
            args.output,
            report,
            args.min_n,
            metric_col="worst_group_accuracy",
            metric_label="Worst-Group Accuracy",
        )

    # ═══════════════════════════════════════════════════════════════
    # Cross-dataset analyses: ImageNet as baseline
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("Cross-dataset analysis: ImageNet vs UrbanCars...")

    merged_uc = compute_robustness_metrics(df_imagenet, df_uc, "UrbanCars", report)
    if len(merged_uc) >= 10:
        plot_imagenet_vs_wg(merged_uc, "UrbanCars", args.output, report)
        plot_robustness_gap(merged_uc, "UrbanCars", args.output, report)
        for fcol, flabel in [
            ("training_objective", "Objective"),
            ("arch_family", "Architecture"),
        ]:
            if fcol in merged_uc.columns:
                plot_gap_by_factor(
                    merged_uc,
                    fcol,
                    flabel,
                    "UrbanCars",
                    args.output,
                    report,
                    args.min_n,
                )
        merged_uc.to_csv(f"{args.output}/merged_imagenet_urbancars.csv", index=False)

    merged_celeba = None
    if df_celeba is not None:
        print("\nCross-dataset analysis: ImageNet vs CelebA...")
        merged_celeba = compute_robustness_metrics(
            df_imagenet, df_celeba, "CelebA", report
        )
        if len(merged_celeba) >= 10:
            plot_imagenet_vs_wg(merged_celeba, "CelebA", args.output, report)
            plot_robustness_gap(merged_celeba, "CelebA", args.output, report)
            for fcol, flabel in [
                ("training_objective", "Objective"),
                ("arch_family", "Architecture"),
            ]:
                if fcol in merged_celeba.columns:
                    plot_gap_by_factor(
                        merged_celeba,
                        fcol,
                        flabel,
                        "CelebA",
                        args.output,
                        report,
                        args.min_n,
                    )
            merged_celeba.to_csv(
                f"{args.output}/merged_imagenet_celeba.csv", index=False
            )

    # 3-way analysis
    if df_celeba is not None and len(merged_uc) >= 10 and len(merged_celeba) >= 10:
        print("\n3-way analysis: ImageNet × UrbanCars × CelebA...")
        merged_3way = plot_3way_correlation(
            df_imagenet, df_uc, df_celeba, args.output, report
        )
        if len(merged_3way) >= 10:
            plot_pareto_3way(merged_3way, args.output, report)
            merged_3way.to_csv(f"{args.output}/merged_3way.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    # Cross-dataset scale barplots (patch size & image size)
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("Cross-dataset scale barplots (patch size & image size)...")

    # Build the datasets dict: name -> (df, metric_col, metric_label)
    all_datasets_for_scale: Dict[str, Tuple[pd.DataFrame, str, str]] = {
        "ImageNet": (df_imagenet, "top1_accuracy", "Top-1 Accuracy"),
        # "UrbanCars": (df_uc, "worst_group_accuracy", "Worst-Group Accuracy"),
    }
    if df_celeba is not None:
        all_datasets_for_scale["CelebA"] = (
            df_celeba,
            "worst_group_accuracy",
            "Worst-Group Accuracy",
        )

    all_datasets_for_scale["UrbanCars"] = (
        df_uc,
        "worst_group_accuracy",
        "Worst-Group Accuracy",
    )
    plot_scale_patchsize_bar_all_datasets(
        all_datasets_for_scale, args.output, report, n_bins=3
    )
    plot_scale_imagesize_bar_all_datasets(
        all_datasets_for_scale, args.output, report, n_bins=6
    )

    # ═══════════════════════════════════════════════════════════════
    # Executive Summary
    # ═══════════════════════════════════════════════════════════════

    report.add_header("EXECUTIVE SUMMARY", level=1)

    report.add_subheader("ImageNet Key Findings")
    top5_in = df_imagenet.nlargest(5, "top1_accuracy")
    report.add_line("\nTop 5 by Top-1 Accuracy:")
    for _, r in top5_in.iterrows():
        report.add_line(
            f"  {r['model_id']}: Top-1={r['top1_accuracy']*100:.1f}%, {r['training_objective']}"
        )

    report.add_subheader("UrbanCars Key Findings")
    top5_uc = df_uc.nlargest(5, "worst_group_accuracy")
    report.add_line("\nTop 5 by WG Accuracy:")
    for _, r in top5_uc.iterrows():
        report.add_line(
            f"  {r['model_id']}: WG={r['worst_group_accuracy']*100:.1f}%, {r['training_objective']}"
        )

    if len(merged_uc) >= 10:
        report.add_subheader("Robustness Analysis (UrbanCars)")
        r, p = spearmanr(merged_uc["imagenet_acc"], merged_uc["wg_acc"])
        report.add_line(f"\nImageNet vs WG correlation: ρ={r:.3f} (p={p:.2e})")
        report.add_line(
            f"Mean robustness gap: {merged_uc['robustness_gap'].mean()*100:.1f}pp"
        )
        report.add_line(
            f"Models with <10pp gap: {(merged_uc['robustness_gap'] < 0.1).sum()}"
        )

    if df_celeba is not None:
        report.add_subheader("CelebA Key Findings")
        top5_ca = df_celeba.nlargest(5, "worst_group_accuracy")
        report.add_line("\nTop 5 by WG Accuracy:")
        for _, r in top5_ca.iterrows():
            report.add_line(
                f"  {r['model_id']}: WG={r['worst_group_accuracy']*100:.1f}%, {r['training_objective']}"
            )

    # ═══════════════════════════════════════════════════════════════
    # Save outputs
    # ═══════════════════════════════════════════════════════════════

    report_path = os.path.join(args.output, "complete_analysis_report.txt")
    report.save(report_path)

    # Save top models tables
    for df, name, metric in [
        (df_imagenet, "imagenet", "top1_accuracy"),
        (df_uc, "urbancars", "worst_group_accuracy"),
    ]:
        cols = [
            "model_id",
            "training_objective",
            "training_data",
            "arch_family",
            "total_params_M",
            "image_size",
            metric,
        ]
        cols = [c for c in cols if c in df.columns]
        top = df.nlargest(30, metric)[cols].copy()
        top[metric] = (top[metric] * 100).round(2)
        top.to_csv(os.path.join(args.output, f"top_models_{name}.csv"), index=False)

    if df_celeba is not None:
        cols = [
            "model_id",
            "training_objective",
            "training_data",
            "arch_family",
            "total_params_M",
            "image_size",
            "worst_group_accuracy",
        ]
        cols = [c for c in cols if c in df_celeba.columns]
        top = df_celeba.nlargest(30, "worst_group_accuracy")[cols].copy()
        top["worst_group_accuracy"] = (top["worst_group_accuracy"] * 100).round(2)
        top.to_csv(os.path.join(args.output, "top_models_celeba.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════
    # Print summary
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {args.output}/")
    for f in sorted(os.listdir(args.output)):
        print(f"  {f}")
    print(f"\n*** Report: {report_path}")

    print("\n" + "-" * 80)
    print("HIGHLIGHTS")
    print("-" * 80)

    print(f"\nImageNet ({len(df_imagenet)} models):")
    print(
        f"  Top-1 Range: {df_imagenet['top1_accuracy'].min()*100:.1f}% - {df_imagenet['top1_accuracy'].max()*100:.1f}%"
    )

    print(f"\nUrbanCars ({len(df_uc)} models):")
    print(
        f"  WG Range: {df_uc['worst_group_accuracy'].min()*100:.1f}% - {df_uc['worst_group_accuracy'].max()*100:.1f}%"
    )

    if len(merged_uc) >= 10:
        r, _ = spearmanr(merged_uc["imagenet_acc"], merged_uc["wg_acc"])
        print(f"\nImageNet↔UrbanCars WG correlation: ρ={r:.3f}")
        print(f"Mean robustness gap: {merged_uc['robustness_gap'].mean()*100:.1f}pp")


if __name__ == "__main__":
    main()
