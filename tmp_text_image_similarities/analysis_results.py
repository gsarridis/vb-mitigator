#!/usr/bin/env python3
"""
Analyze the impact of model properties (architecture, image size, patch size,
training data, model size, etc.) on UrbanCars zero-shot classification
performance (average accuracy, worst-group accuracy, fairness gap).

Reads the master_results.csv produced by run_all_models.py.

Outputs:
  - analysis/  — all analysis plots
  - analysis/statistical_summary.txt — text report
  - analysis/correlation_table.csv — numeric correlations

Usage:
    python analyze_results.py
    python analyze_results.py --input results/master_results.csv --output analysis
    python analyze_results.py --min-models 3  # min models per group for analysis

Requirements:
    pip install pandas matplotlib numpy scipy seaborn
"""

import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print(
        "[INFO] seaborn not installed — falling back to matplotlib. pip install seaborn for nicer plots."
    )


# ── Defaults ─────────────────────────────────────────────────────
DEFAULT_INPUT = "results/master_results.csv"
DEFAULT_OUTPUT = "analysis"
DEFAULT_MIN_MODELS = 2

# Metrics to analyze
METRICS = [
    ("avg_accuracy", "Average Accuracy"),
    ("worst_group_accuracy", "Worst-Group Accuracy"),
    ("accuracy_gap", "Fairness Gap (Avg − Worst)"),
]

# Categorical factors
CATEGORICAL_FACTORS = [
    ("arch_family", "Architecture Family"),
    ("training_data_label", "Training Data"),
    ("quickgelu", "QuickGELU"),
]

# Numeric factors
NUMERIC_FACTORS = [
    ("total_params_M", "Model Size (M params)"),
    ("image_size", "Image Size (px)"),
    ("patch_size", "Patch Size (px)"),
    ("embed_dim", "Embedding Dim"),
    ("training_samples_M", "Training Samples (M)"),
]

# ── Dataset sizes (approximate, in millions of image-text pairs) ──
# Sources: CLIP paper, LAION papers, DataComp paper, DFN paper, MetaCLIP paper
DATASET_SIZES_M = {
    "OpenAI WIT (400M)": 400,
    "LAION-400M": 400,
    "LAION-2B": 2000,
    "DataComp": 1000,  # DataComp-1B (XL scale); varies by filtering
    "MetaCLIP": 400,  # MetaCLIP-400M base; 2.5B scaled version exists
    "DFN": 2000,  # DFN-2B (ViT-H used DFN-5B ≈ 5000)
    "CommonPool": 12800,  # DataComp-12.8B raw pool
    "YFCC-15M": 15,
    "CC-12M": 12,
    "WebLI": 10000,  # Google's WebLI ~10B
    "Merged": 2000,  # merged datasets, approximate
    "MS-COCO (finetuned)": 0.6,  # ~600K, finetuning dataset
}


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV and clean up data types."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} models from {path}")

    # Ensure numeric columns
    for col in [
        "total_params_M",
        "image_size",
        "patch_size",
        "embed_dim",
        "avg_accuracy",
        "worst_group_accuracy",
        "accuracy_gap",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute additional columns
    if "accuracy_gap" not in df.columns and "avg_accuracy" in df.columns:
        df["accuracy_gap"] = df["avg_accuracy"] - df["worst_group_accuracy"]

    # Compute tokens if patch_size and image_size available
    if "patch_size" in df.columns and "image_size" in df.columns:
        mask = (
            df["patch_size"].notna() & df["image_size"].notna() & (df["patch_size"] > 0)
        )
        df.loc[mask, "n_tokens"] = (
            df.loc[mask, "image_size"] / df.loc[mask, "patch_size"]
        ) ** 2

    # ViT size ordering for plots
    vit_size_order = {
        "S": 0,
        "B": 1,
        "L": 2,
        "H": 3,
        "g": 4,
        "G": 5,
        "bigG": 6,
        "e": 7,
        "E": 8,
    }
    if "vit_size" in df.columns:
        df["vit_size_rank"] = df["vit_size"].map(vit_size_order)

    # ── Map training data to sample counts ───────────────────────
    if "training_data" in df.columns:
        df["training_samples_M"] = df["training_data"].map(DATASET_SIZES_M)

        # Create label with sample count: e.g. "LAION-2B (2000M)"
        def _make_label(name):
            size = DATASET_SIZES_M.get(name)
            if size is not None:
                if size >= 1000:
                    return f"{name} ({size/1000:.1f}B)"
                else:
                    return f"{name} ({size:.0f}M)"
            return name

        df["training_data_label"] = df["training_data"].apply(_make_label)

    print(f"Columns: {list(df.columns)}")
    print(f"Architecture families: {df['arch_family'].value_counts().to_dict()}")
    print(
        f"Training data: {df.get('training_data_label', df.get('training_data', pd.Series())).value_counts().to_dict()}"
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# Statistical analysis
# ═══════════════════════════════════════════════════════════════════


def analyze_categorical(
    df: pd.DataFrame,
    factor_col: str,
    factor_label: str,
    min_models: int,
    report_lines: list,
) -> dict:
    """Run grouped analysis for a categorical factor. Returns per-metric summary dicts."""
    results = {}
    report_lines.append(f"\n{'─' * 70}")
    report_lines.append(f"Factor: {factor_label} ({factor_col})")
    report_lines.append(f"{'─' * 70}")

    groups = df.groupby(factor_col)
    group_counts = groups.size()
    valid_groups = group_counts[group_counts >= min_models].index.tolist()

    report_lines.append(
        f"Groups with >= {min_models} models: {len(valid_groups)} / {len(group_counts)}"
    )
    for g, c in group_counts.items():
        report_lines.append(f"  {g}: n={c}")

    if len(valid_groups) < 2:
        report_lines.append("  → Insufficient groups for comparison.")
        return results

    df_filtered = df[df[factor_col].isin(valid_groups)]

    for metric_col, metric_label in METRICS:
        if metric_col not in df.columns:
            continue

        report_lines.append(f"\n  Metric: {metric_label}")
        grouped = df_filtered.groupby(factor_col)[metric_col]
        summary = grouped.agg(["mean", "std", "min", "max", "count"])
        summary = summary.sort_values("mean", ascending=False)

        for idx, row in summary.iterrows():
            report_lines.append(
                f"    {idx:<30} mean={row['mean']:.4f}  std={row['std']:.4f}  "
                f"min={row['min']:.4f}  max={row['max']:.4f}  n={int(row['count'])}"
            )

        # Kruskal-Wallis test (non-parametric ANOVA)
        groups_data = [
            g[metric_col].dropna().values
            for _, g in df_filtered.groupby(factor_col)
            if len(g) >= min_models
        ]
        if len(groups_data) >= 2 and all(len(g) >= 2 for g in groups_data):
            try:
                stat, p_value = stats.kruskal(*groups_data)
                report_lines.append(
                    f"    Kruskal-Wallis: H={stat:.3f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}"
                )
            except Exception:
                pass

        results[metric_col] = summary

    return results


def analyze_numeric(
    df: pd.DataFrame, factor_col: str, factor_label: str, report_lines: list
) -> dict:
    """Compute correlations between a numeric factor and metrics."""
    results = {}
    report_lines.append(f"\n{'─' * 70}")
    report_lines.append(f"Factor: {factor_label} ({factor_col})")
    report_lines.append(f"{'─' * 70}")

    if factor_col not in df.columns:
        report_lines.append("  → Column not found.")
        return results

    valid = df[factor_col].notna()
    report_lines.append(f"Models with valid {factor_col}: {valid.sum()} / {len(df)}")

    if valid.sum() < 5:
        report_lines.append("  → Too few valid values for correlation analysis.")
        return results

    for metric_col, metric_label in METRICS:
        if metric_col not in df.columns:
            continue

        mask = valid & df[metric_col].notna()
        x = df.loc[mask, factor_col].values
        y = df.loc[mask, metric_col].values

        if len(x) < 5:
            continue

        # Pearson
        r_pearson, p_pearson = stats.pearsonr(x, y)
        # Spearman
        r_spearman, p_spearman = stats.spearmanr(x, y)

        report_lines.append(
            f"  {metric_label}:\n"
            f"    Pearson:  r={r_pearson:+.4f}  p={p_pearson:.4f}\n"
            f"    Spearman: ρ={r_spearman:+.4f}  p={p_spearman:.4f}"
        )

        results[metric_col] = {
            "pearson_r": r_pearson,
            "pearson_p": p_pearson,
            "spearman_r": r_spearman,
            "spearman_p": p_spearman,
            "n": int(mask.sum()),
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# Plotting functions
# ═══════════════════════════════════════════════════════════════════


def plot_categorical_bars(
    df: pd.DataFrame,
    factor_col: str,
    factor_label: str,
    min_models: int,
    output_dir: str,
) -> None:
    """Grouped bar charts for each metric by a categorical factor."""
    valid_groups = (
        df.groupby(factor_col)
        .filter(lambda g: len(g) >= min_models)[factor_col]
        .unique()
    )
    df_f = df[df[factor_col].isin(valid_groups)]
    if len(valid_groups) < 2:
        return

    for metric_col, metric_label in METRICS:
        if metric_col not in df.columns:
            continue

        grouped = df_f.groupby(factor_col)[metric_col].agg(["mean", "std", "count"])
        grouped = grouped.sort_values("mean", ascending=True)

        fig, ax = plt.subplots(figsize=(max(8, len(grouped) * 0.8), 6))
        y_pos = np.arange(len(grouped))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(grouped)))

        bars = ax.barh(
            y_pos,
            grouped["mean"] * 100,
            xerr=grouped["std"] * 100,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
            capsize=3,
        )
        ax.set_yticks(y_pos)
        labels = [f"{idx} (n={int(row['count'])})" for idx, row in grouped.iterrows()]
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f"{metric_label} (%)")
        ax.set_title(f"{metric_label} by {factor_label}")
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for bar, (_, row) in zip(bars, grouped.iterrows()):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{row['mean']*100:.1f}%",
                va="center",
                fontsize=7,
            )

        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"bar_{factor_col}_{metric_col}.png"), dpi=150
        )
        plt.close(fig)


def plot_categorical_box(
    df: pd.DataFrame,
    factor_col: str,
    factor_label: str,
    min_models: int,
    output_dir: str,
) -> None:
    """Box plots for each metric by a categorical factor."""
    valid_groups = (
        df.groupby(factor_col)
        .filter(lambda g: len(g) >= min_models)[factor_col]
        .unique()
    )
    df_f = df[df[factor_col].isin(valid_groups)].copy()
    if len(valid_groups) < 2:
        return

    # Sort groups by median avg_accuracy
    if "avg_accuracy" in df_f.columns:
        order = (
            df_f.groupby(factor_col)["avg_accuracy"]
            .median()
            .sort_values()
            .index.tolist()
        )
    else:
        order = sorted(valid_groups)

    for metric_col, metric_label in METRICS:
        if metric_col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.2), 6))

        if HAS_SEABORN:
            sns.boxplot(
                data=df_f,
                x=factor_col,
                y=metric_col,
                order=order,
                ax=ax,
                palette="Set2",
                showfliers=True,
            )
            sns.stripplot(
                data=df_f,
                x=factor_col,
                y=metric_col,
                order=order,
                ax=ax,
                color="black",
                alpha=0.4,
                size=3,
                jitter=True,
            )
        else:
            groups_data = [
                df_f[df_f[factor_col] == g][metric_col].dropna().values for g in order
            ]
            bp = ax.boxplot(groups_data, labels=order, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("steelblue")
                patch.set_alpha(0.6)

        ax.set_xlabel(factor_label)
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by {factor_label}")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"box_{factor_col}_{metric_col}.png"), dpi=150
        )
        plt.close(fig)


def plot_numeric_scatter(
    df: pd.DataFrame, factor_col: str, factor_label: str, output_dir: str
) -> None:
    """Scatter plots + regression line for numeric factor vs each metric."""
    if factor_col not in df.columns:
        return

    for metric_col, metric_label in METRICS:
        if metric_col not in df.columns:
            continue

        mask = df[factor_col].notna() & df[metric_col].notna()
        if mask.sum() < 5:
            continue

        x = df.loc[mask, factor_col].values
        y = df.loc[mask, metric_col].values * 100

        fig, ax = plt.subplots(figsize=(10, 7))

        # Color by architecture family if available
        if "arch_family" in df.columns:
            families = df.loc[mask, "arch_family"].values
            unique_fam = sorted(set(families))
            cmap = plt.cm.get_cmap("tab10", len(unique_fam))
            fam_to_color = {f: cmap(i) for i, f in enumerate(unique_fam)}
            colors = [fam_to_color[f] for f in families]
            ax.scatter(
                x, y, c=colors, alpha=0.6, edgecolors="black", linewidth=0.5, s=50
            )
            # Legend
            for fam in unique_fam:
                ax.scatter(
                    [],
                    [],
                    c=[fam_to_color[fam]],
                    label=fam,
                    edgecolors="black",
                    linewidth=0.5,
                )
            ax.legend(fontsize=7, title="Architecture", title_fontsize=8)
        else:
            ax.scatter(
                x,
                y,
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5,
                s=50,
                color="steelblue",
            )

        # Regression line
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "r--", alpha=0.7, label=f"r={r_val:.3f}, p={p_val:.3f}")

        # Annotate outliers (top/bottom 3)
        sorted_idx = np.argsort(y)
        for i in list(sorted_idx[:3]) + list(sorted_idx[-3:]):
            model_id = df.loc[mask].iloc[i].get("model_id", "")
            ax.annotate(
                model_id,
                (x[i], y[i]),
                fontsize=4,
                alpha=0.6,
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_xlabel(factor_label)
        ax.set_ylabel(f"{metric_label} (%)")
        ax.set_title(f"{metric_label} vs {factor_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"scatter_{factor_col}_{metric_col}.png"), dpi=150
        )
        plt.close(fig)


def plot_heatmap_correlation(df: pd.DataFrame, output_dir: str) -> None:
    """Correlation heatmap between all numeric factors and metrics."""
    cols_factors = [c for c, _ in NUMERIC_FACTORS if c in df.columns]
    cols_metrics = [c for c, _ in METRICS if c in df.columns]

    if "n_tokens" in df.columns:
        cols_factors.append("n_tokens")
    if "vit_size_rank" in df.columns:
        cols_factors.append("vit_size_rank")

    all_cols = cols_factors + cols_metrics
    df_num = df[all_cols].dropna()

    if len(df_num) < 5 or len(all_cols) < 3:
        return

    corr = df_num.corr(method="spearman")

    # Plot only factor→metric correlations
    corr_sub = corr.loc[cols_factors, cols_metrics]

    fig, ax = plt.subplots(
        figsize=(max(8, len(cols_metrics) * 2), max(6, len(cols_factors) * 0.8))
    )

    if HAS_SEABORN:
        sns.heatmap(
            corr_sub,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            linewidths=0.5,
        )
    else:
        im = ax.imshow(corr_sub.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols_metrics)))
        ax.set_xticklabels(cols_metrics, rotation=45, ha="right")
        ax.set_yticks(range(len(cols_factors)))
        ax.set_yticklabels(cols_factors)
        for i in range(len(cols_factors)):
            for j in range(len(cols_metrics)):
                ax.text(
                    j,
                    i,
                    f"{corr_sub.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        plt.colorbar(im, ax=ax)

    ax.set_title("Spearman Correlation: Model Properties → Performance Metrics")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    # Save correlation table
    corr_sub.to_csv(os.path.join(output_dir, "correlation_table.csv"))


def plot_interaction_heatmap(
    df: pd.DataFrame, output_dir: str, min_models: int
) -> None:
    """Heatmaps: arch_family × training_data → each metric."""
    td_col = (
        "training_data_label"
        if "training_data_label" in df.columns
        else "training_data"
    )
    if "arch_family" not in df.columns or td_col not in df.columns:
        return

    for metric_col, metric_label in METRICS:
        if metric_col not in df.columns:
            continue

        pivot = df.pivot_table(
            values=metric_col, index="arch_family", columns=td_col, aggfunc="mean"
        )
        count_pivot = df.pivot_table(
            values=metric_col, index="arch_family", columns=td_col, aggfunc="count"
        )

        # Mask cells with too few models
        pivot = pivot.where(count_pivot >= min_models)

        if pivot.dropna(how="all").empty:
            continue

        fig, ax = plt.subplots(
            figsize=(max(10, pivot.shape[1] * 1.5), max(5, pivot.shape[0] * 0.8))
        )

        if HAS_SEABORN:
            sns.heatmap(
                pivot * 100,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                ax=ax,
                linewidths=0.5,
                cbar_kws={"label": f"{metric_label} (%)"},
            )
        else:
            im = ax.imshow(pivot.values * 100, cmap="YlGnBu", aspect="auto")
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(
                            j, i, f"{val*100:.1f}", ha="center", va="center", fontsize=8
                        )
            plt.colorbar(im, ax=ax, label=f"{metric_label} (%)")

        ax.set_title(f"{metric_label}: Architecture × Training Data")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"interaction_{metric_col}.png"), dpi=150)
        plt.close(fig)


def plot_pareto_front(df: pd.DataFrame, output_dir: str) -> None:
    """Pareto front: avg accuracy vs worst-group accuracy (or vs fairness gap)."""
    if "avg_accuracy" not in df.columns or "worst_group_accuracy" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Avg vs Worst-Group
    ax = axes[0]
    x = df["avg_accuracy"].values * 100
    y = df["worst_group_accuracy"].values * 100

    if "arch_family" in df.columns:
        families = df["arch_family"].values
        unique_fam = sorted(set(families))
        cmap = plt.cm.get_cmap("tab10", len(unique_fam))
        fam_to_color = {f: cmap(i) for i, f in enumerate(unique_fam)}
        colors = [fam_to_color[f] for f in families]
        ax.scatter(x, y, c=colors, alpha=0.6, edgecolors="black", linewidth=0.5, s=50)
        for fam in unique_fam:
            ax.scatter(
                [],
                [],
                c=[fam_to_color[fam]],
                label=fam,
                edgecolors="black",
                linewidth=0.5,
            )
        ax.legend(fontsize=7, title="Architecture")
    else:
        ax.scatter(x, y, alpha=0.6, edgecolors="black", linewidth=0.5, s=50)

    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="avg=worst")
    ax.set_xlabel("Average Accuracy (%)")
    ax.set_ylabel("Worst-Group Accuracy (%)")
    ax.set_title("Average vs Worst-Group Accuracy")
    ax.grid(True, alpha=0.3)

    # Highlight Pareto-optimal models (max avg AND max worst-group)
    pareto_mask = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if (
                i != j
                and x[j] >= x[i]
                and y[j] >= y[i]
                and (x[j] > x[i] or y[j] > y[i])
            ):
                pareto_mask[i] = False
                break
    if pareto_mask.any():
        ax.scatter(
            x[pareto_mask],
            y[pareto_mask],
            facecolors="none",
            edgecolors="red",
            linewidth=2,
            s=120,
            label="Pareto front",
            zorder=5,
        )
        for i in np.where(pareto_mask)[0]:
            ax.annotate(
                df.iloc[i].get("model_id", ""),
                (x[i], y[i]),
                fontsize=5,
                color="red",
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.legend(fontsize=7)

    # Avg Accuracy vs Fairness Gap
    ax = axes[1]
    if "accuracy_gap" in df.columns:
        gap = df["accuracy_gap"].values * 100
        if "total_params_M" in df.columns:
            sizes = df["total_params_M"].fillna(100).values
            sizes = np.clip(sizes / sizes.max() * 200, 10, 300)
        else:
            sizes = 50
        ax.scatter(
            x, gap, s=sizes, alpha=0.6, edgecolors="black", linewidth=0.5, c="steelblue"
        )
        ax.set_xlabel("Average Accuracy (%)")
        ax.set_ylabel("Fairness Gap: Avg − Worst (%)")
        ax.set_title("Accuracy vs Fairness Gap (size = model params)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.grid(True, alpha=0.3)

        for i in np.argsort(gap)[-5:]:
            ax.annotate(
                df.iloc[i].get("model_id", ""),
                (x[i], gap[i]),
                fontsize=5,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pareto_and_gap.png"), dpi=150)
    plt.close(fig)


def plot_scaling_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """How does performance scale with model size, for each architecture?"""
    if "total_params_M" not in df.columns or "arch_family" not in df.columns:
        return

    for metric_col, metric_label in METRICS:
        if metric_col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))

        families = df["arch_family"].dropna().unique()
        cmap = plt.cm.get_cmap("tab10", len(families))

        for i, fam in enumerate(sorted(families)):
            sub = df[df["arch_family"] == fam].dropna(
                subset=["total_params_M", metric_col]
            )
            if len(sub) < 2:
                ax.scatter(
                    sub["total_params_M"],
                    sub[metric_col] * 100,
                    color=cmap(i),
                    label=fam,
                    s=50,
                    edgecolors="black",
                    linewidth=0.5,
                )
                continue

            ax.scatter(
                sub["total_params_M"],
                sub[metric_col] * 100,
                color=cmap(i),
                label=fam,
                s=50,
                edgecolors="black",
                linewidth=0.5,
                alpha=0.7,
            )

            # Fit trend line per family
            x = sub["total_params_M"].values
            y = sub[metric_col].values * 100
            if len(x) >= 3:
                z = np.polyfit(np.log10(x + 1), y, 1)
                x_line = np.linspace(x.min(), x.max(), 50)
                y_line = z[0] * np.log10(x_line + 1) + z[1]
                ax.plot(x_line, y_line, color=cmap(i), alpha=0.5, linestyle="--")

        ax.set_xscale("log")
        ax.set_xlabel("Model Size (M params, log scale)")
        ax.set_ylabel(f"{metric_label} (%)")
        ax.set_title(f"Scaling: {metric_label} vs Model Size by Architecture")
        ax.legend(fontsize=7, title="Architecture")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"scaling_{metric_col}.png"), dpi=150)
        plt.close(fig)


def plot_top_bottom_table(df: pd.DataFrame, output_dir: str) -> None:
    """Save a visual table of top-10 and bottom-10 models."""
    cols_to_show = [
        "model_id",
        "arch_family",
        "total_params_M",
        "image_size",
        "training_data",
        "avg_accuracy",
        "worst_group_accuracy",
        "accuracy_gap",
    ]
    cols_available = [c for c in cols_to_show if c in df.columns]

    for metric_col, metric_label in METRICS:
        if metric_col not in df.columns:
            continue

        df_sorted = df.sort_values(metric_col, ascending=False)

        top = df_sorted.head(15)[cols_available].copy()
        bottom = df_sorted.tail(15)[cols_available].copy()

        for sub_df, label, fname in [
            (top, "Top 15", "top"),
            (bottom, "Bottom 15", "bottom"),
        ]:
            sub_df = sub_df.copy()
            for c in ["avg_accuracy", "worst_group_accuracy", "accuracy_gap"]:
                if c in sub_df.columns:
                    sub_df[c] = (sub_df[c] * 100).round(2)

            fig, ax = plt.subplots(figsize=(16, max(4, len(sub_df) * 0.4 + 1)))
            ax.axis("off")
            table = ax.table(
                cellText=sub_df.values,
                colLabels=sub_df.columns,
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.auto_set_column_width(list(range(len(sub_df.columns))))
            ax.set_title(
                f"{label} Models by {metric_label}", fontsize=12, fontweight="bold"
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(output_dir, f"table_{fname}_{metric_col}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Analyze run_all_models results.")
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="Path to master_results.csv"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Output directory for analysis"
    )
    parser.add_argument(
        "--min-models",
        type=int,
        default=DEFAULT_MIN_MODELS,
        help="Minimum models per group for categorical analysis",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = load_and_clean(args.input)

    if df.empty:
        print("ERROR: No data to analyze.")
        return

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("URBANCARS MODEL ANALYSIS REPORT")
    report_lines.append(f"Models analyzed: {len(df)}")
    report_lines.append("=" * 70)

    # ── Basic stats ──────────────────────────────────────────────
    report_lines.append("\nOVERALL STATISTICS:")
    for metric_col, metric_label in METRICS:
        if metric_col in df.columns:
            vals = df[metric_col].dropna()
            report_lines.append(
                f"  {metric_label}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                f"min={vals.min():.4f}  max={vals.max():.4f}"
            )

    # ── Categorical analysis ─────────────────────────────────────
    print("\nRunning categorical analysis …")
    for factor_col, factor_label in CATEGORICAL_FACTORS:
        if factor_col in df.columns:
            analyze_categorical(
                df, factor_col, factor_label, args.min_models, report_lines
            )
            plot_categorical_bars(
                df, factor_col, factor_label, args.min_models, args.output
            )
            plot_categorical_box(
                df, factor_col, factor_label, args.min_models, args.output
            )

    # ViT size as special categorical
    if "vit_size" in df.columns:
        vit_df = df[df["vit_size"].notna()]
        if len(vit_df) >= 5:
            analyze_categorical(
                vit_df,
                "vit_size",
                "ViT Size (S/B/L/H/g/G)",
                args.min_models,
                report_lines,
            )
            plot_categorical_bars(
                vit_df, "vit_size", "ViT Size", args.min_models, args.output
            )
            plot_categorical_box(
                vit_df, "vit_size", "ViT Size", args.min_models, args.output
            )

    # ── Numeric analysis ─────────────────────────────────────────
    print("Running numeric/correlation analysis …")
    all_correlations = {}
    for factor_col, factor_label in NUMERIC_FACTORS:
        results = analyze_numeric(df, factor_col, factor_label, report_lines)
        if results:
            all_correlations[factor_col] = results
        plot_numeric_scatter(df, factor_col, factor_label, args.output)

    # n_tokens as additional numeric factor
    if "n_tokens" in df.columns:
        results = analyze_numeric(
            df, "n_tokens", "Number of Tokens (img_size/patch_size)²", report_lines
        )
        if results:
            all_correlations["n_tokens"] = results
        plot_numeric_scatter(df, "n_tokens", "Number of Tokens", args.output)

    # ── Correlation heatmap ──────────────────────────────────────
    print("Generating correlation heatmap …")
    plot_heatmap_correlation(df, args.output)

    # ── Interaction analysis ─────────────────────────────────────
    print("Generating interaction heatmaps …")
    plot_interaction_heatmap(df, args.output, args.min_models)

    # ── Pareto front ─────────────────────────────────────────────
    print("Generating Pareto front …")
    plot_pareto_front(df, args.output)

    # ── Scaling analysis ─────────────────────────────────────────
    print("Generating scaling analysis …")
    plot_scaling_analysis(df, args.output)

    # ── Top/bottom tables ────────────────────────────────────────
    print("Generating top/bottom tables …")
    plot_top_bottom_table(df, args.output)

    # ── Save report ──────────────────────────────────────────────
    report_text = "\n".join(report_lines)
    report_path = os.path.join(args.output, "statistical_summary.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Output directory: {args.output}/")
    print(f"  - statistical_summary.txt  (full text report)")
    print(f"  - correlation_table.csv    (Spearman correlations)")
    print(f"  - correlation_heatmap.png")
    print(f"  - bar_*.png, box_*.png     (categorical analysis)")
    print(f"  - scatter_*.png            (numeric correlations)")
    print(f"  - interaction_*.png        (arch × training data)")
    print(f"  - scaling_*.png            (model size scaling)")
    print(f"  - pareto_and_gap.png       (Pareto front)")
    print(f"  - table_top_*.png, table_bottom_*.png")

    # Print key findings
    print(f"\nKEY FINDINGS:")
    print(report_text[-2000:] if len(report_text) > 2000 else report_text)


if __name__ == "__main__":
    main()
