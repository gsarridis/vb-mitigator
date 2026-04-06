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
    python full_analysis_paper_v4.py --urbancars ./urbancars/results/master_results.csv --celeba ./celeba/results_celeba/master_results.csv --imagenet ./imagenet/results_imagenet/master_results.csv --output paper_analysis_v3
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
            fontsize=9,
            xytext=(5, 5),
            textcoords="offset points",
            color="darkred",
            fontweight="bold",
        )
        report.add_line(f"  {model_id}: x={x[idx]:.1f}%, y={y[idx]:.1f}%")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="lower right", fontsize=8)
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
    ax.set_xlabel(f"{metric_label} (%)")
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

    fig, ax = plt.subplots(figsize=(8, 6))

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

    ax.set_xlabel(factor_label)
    ax.set_ylabel(f"{metric_label} (%)")
    ax.legend(loc="best", fontsize=8)
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
            fontsize=8,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel(factor_label)
    ax.set_ylabel(f"{metric_label} (%)")
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
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
            color="darkred",
            fontweight="bold",
        )

    r, p = spearmanr(x, y)
    ax.set_xlabel("ImageNet Top-1 Accuracy (%)")
    ax.set_ylabel(f"{bias_name} Worst-Group Accuracy (%)")
    ax.legend(loc="lower right", fontsize=8)
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
    ax.set_yticklabels(df_plot["model_id"].str[:40], fontsize=7)
    ax.set_xlabel(f"Robustness Gap: ImageNet − {bias_name} WG (pp)")
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
    ax.set_xlabel(f"Robustness Gap: ImageNet − {bias_name} WG (pp)")
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
    ax.set_title("Cross-Dataset Correlation (Spearman)")
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
        ax.set_xlabel(f"{m1} (%)")
        ax.set_ylabel(f"{m2} (%)")
        ax.set_title(f"ρ = {r:.3f}")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="lower right", fontsize=6)
    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_3way_scatter")
    plt.close()

    return merged


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
    df_r["log_patchsize"] = np.log2(df_r["patch_size_num"].clip(lower=1))
    df_r["log_imgsize"] = np.log2(df_r["image_size"].clip(lower=1))

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
    """
    report.add_header(f"PARTIAL CORRELATIONS: {name}", level=2)

    # Prepare numeric data
    df_num = df.copy()
    df_num["y"] = df_num[metric_col]
    df_num["log_params"] = np.log10(df_num["total_params_M"].clip(lower=1))
    df_num["log_datasize"] = np.log10(df_num["training_data_size_M"].clip(lower=1))
    df_num["log_patchsize"] = np.log2(df_num["patch_size_num"].clip(lower=1))
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
    df_r["log_patchsize"] = np.log2(df_r["patch_size_num"].clip(lower=1))
    df_r["log_datasize"] = np.log10(df_r["training_data_size_M"].clip(lower=1))

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
                fontsize=10,
                fontweight="bold",
            )

    ax.set_ylabel(f"Variance Explained (% of Total R² = {total_r2*100:.1f}%)")
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Unique vs Shared Variance")

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
    Non-linear, captures interactions automatically.
    """
    report.add_header(f"RANDOM FOREST IMPORTANCE: {name}", level=2)

    if not HAS_SKLEARN:
        report.add_line("sklearn not available.")
        return

    # Prepare features
    df_ml = df.copy()

    # Encode categorical variables
    encoders = {}
    for col in ["training_objective", "arch_family", "training_data"]:
        if col in df_ml.columns:
            le = LabelEncoder()
            df_ml[f"{col}_enc"] = le.fit_transform(
                df_ml[col].fillna("Unknown").astype(str)
            )
            encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Feature columns
    feature_cols = []
    feature_labels = []

    for col, label in [
        ("total_params_M", "Params (M)"),
        ("image_size", "Image Size"),
        ("patch_size_num", "Patch Size"),
        ("num_tokens", "Tokens"),
        ("training_data_size_M", "Data Size (M)"),
        ("embed_dim", "Embed Dim"),
        ("training_objective_enc", "Objective"),
        ("arch_family_enc", "Architecture"),
        ("training_data_enc", "Training Data"),
    ]:
        if col in df_ml.columns:
            feature_cols.append(col)
            feature_labels.append(label)

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

    # Feature importance
    importance = pd.DataFrame(
        {"Feature": feature_labels, "Importance": rf.feature_importances_}
    ).sort_values("Importance", ascending=False)

    report.add_subheader("Feature Importance Ranking")
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

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(importance))
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(importance)))

    ax.barh(
        y_pos, importance["Importance"], color=colors, edgecolor="black", linewidth=0.5
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance["Feature"])
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_rf_importance_{name.lower()}")
    plt.close()

    # Save importance table
    importance.to_csv(f"{out_dir}/feature_importance_{name.lower()}.csv", index=False)

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
    """
    report.add_header(f"INTERACTION ANALYSIS: {name}", level=2)

    if not HAS_STATSMODELS:
        report.add_line("statsmodels not available.")
        return

    # Prepare data
    df_r = df.copy()
    df_r["y"] = df_r[metric_col] * 100
    df_r["is_siglip"] = (
        df_r["training_objective"].isin(["SigLIP", "SigLIP2"]).astype(int)
    )
    df_r["log_params"] = np.log10(df_r["total_params_M"].clip(lower=1))
    df_r["log_data"] = np.log10(df_r["training_data_size_M"].clip(lower=1))

    # Standardize
    for col in ["log_params", "log_data"]:
        mean, std = df_r[col].mean(), df_r[col].std()
        if std > 0:
            df_r[f"{col}_z"] = (df_r[col] - mean) / std

    df_r = df_r.dropna(subset=["y", "is_siglip", "log_params_z", "log_data_z"])

    if len(df_r) < 50:
        report.add_line("Insufficient data.")
        return

    report.add_subheader("Testing Interaction Effects")

    # Model without interactions
    formula_main = "y ~ is_siglip + log_params_z + log_data_z"
    model_main = smf.ols(formula_main, data=df_r).fit()

    # Model with interactions
    formula_int = "y ~ is_siglip * log_params_z + is_siglip * log_data_z + log_params_z * log_data_z"
    model_int = smf.ols(formula_int, data=df_r).fit()

    report.add_line(f"\nMain effects only: R² = {model_main.rsquared:.4f}")
    report.add_line(f"With interactions: R² = {model_int.rsquared:.4f}")
    report.add_line(
        f"Improvement: ΔR² = {model_int.rsquared - model_main.rsquared:.4f}"
    )

    # Test if interactions are significant (F-test)
    from scipy.stats import f as f_dist

    df1 = model_int.df_model - model_main.df_model
    df2 = model_int.df_resid
    ssr_main = model_main.ssr
    ssr_int = model_int.ssr

    if df1 > 0 and ssr_int > 0:
        f_stat = ((ssr_main - ssr_int) / df1) / (ssr_int / df2)
        p_value = 1 - f_dist.cdf(f_stat, df1, df2)
        report.add_line(f"F-test for interactions: F={f_stat:.3f}, p={p_value:.4f}")

        if p_value < 0.05:
            report.add_line("→ Interactions are statistically significant!")
        else:
            report.add_line("→ No significant interaction effects.")

    report.add_subheader("Interaction Coefficients")
    report.add_line(f"\n{'Term':<35} {'Coef':>10} {'p':>12} {'Sig':>5}")
    report.add_line("-" * 65)

    for var in model_int.params.index:
        if ":" in var:  # Interaction term
            c = model_int.params[var]
            p = model_int.pvalues[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            report.add_line(f"{var:<35} {c:>10.3f} {p:>12.4f} {sig:>5}")

    # Visualize key interaction: SigLIP × Scale
    report.add_subheader("SigLIP × Model Size Interaction")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Metric by params, split by objective
    ax = axes[0]
    for obj, color, label in [
        ("CLIP", "#3498db", "CLIP"),
        ("SigLIP", "#2ecc71", "SigLIP/SigLIP2"),
    ]:
        if obj == "CLIP":
            mask = ~df_r["training_objective"].isin(["SigLIP", "SigLIP2", "CoCa"])
        else:
            mask = df_r["training_objective"].isin(["SigLIP", "SigLIP2"])

        if mask.sum() > 5:
            x = df_r.loc[mask, "log_params_z"]
            y_vals = df_r.loc[mask, "y"]
            ax.scatter(x, y_vals, c=color, alpha=0.5, s=30, label=label)

            # Trend line
            if len(x) > 5:
                slope, intercept = np.polyfit(x, y_vals, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, slope * x_line + intercept, c=color, linewidth=2)

    ax.set_xlabel("Model Size (standardized log params)")
    ax.set_ylabel(f"{metric_label} (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Metric by data size, split by objective
    ax = axes[1]
    for obj, color, label in [
        ("CLIP", "#3498db", "CLIP"),
        ("SigLIP", "#2ecc71", "SigLIP/SigLIP2"),
    ]:
        if obj == "CLIP":
            mask = ~df_r["training_objective"].isin(["SigLIP", "SigLIP2", "CoCa"])
        else:
            mask = df_r["training_objective"].isin(["SigLIP", "SigLIP2"])

        if mask.sum() > 5:
            x = df_r.loc[mask, "log_data_z"]
            y_vals = df_r.loc[mask, "y"]
            ax.scatter(x, y_vals, c=color, alpha=0.5, s=30, label=label)

            if len(x) > 5:
                slope, intercept = np.polyfit(x, y_vals, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, slope * x_line + intercept, c=color, linewidth=2)

    ax.set_xlabel("Training Data Size (standardized log)")
    ax.set_ylabel(f"{metric_label} (%)")
    ax.legend()
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
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
            color="darkred",
            fontweight="bold",
        )

    ax.plot([0, 100], [0, 100], "k--", alpha=0.3)
    ax.set_xlabel("ImageNet Top-1 Accuracy (%)")
    ax.set_ylabel("Average WG Accuracy (UrbanCars + CelebA) (%)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_pareto_3way")
    plt.close()

    # Save Pareto models
    pareto_df.to_csv(f"{out_dir}/pareto_3way_models.csv", index=False)


# ═══════════════════════════════════════════════════════════════════
# Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════


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
