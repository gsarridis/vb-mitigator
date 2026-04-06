#!/usr/bin/env python3
"""
Comprehensive VLM Robustness Analysis for ACM MM Paper.

Analyzes OpenCLIP models across UrbanCars and CelebA datasets.

Usage:
    python analyze_vlm_robustness.py --urbancars uc_results.csv --celeba celeba_results.csv
    python analyze_vlm_robustness.py --urbancars uc_results.csv  # single dataset
"""

import argparse
import os
import warnings
from typing import Dict, Tuple
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

# Configuration
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
    "CLIP": "#3498db",  # Blue
    "SigLIP": "#2ecc71",  # Green
    "SigLIP2": "#27ae60",  # Darker green
    "CoCa": "#e74c3c",  # Red
    "CLIPA": "#9b59b6",  # Purple
    "EVA": "#f39c12",  # Orange
}

COLORS_ARCH = {
    "CLIP": "#3498db",  # Blue
    "SigLIP": "#2ecc71",  # Green
    "SigLIP2": "#27ae60",  # Darker green
    "CoCa": "#e74c3c",  # Red
    "CLIPA": "#9b59b6",  # Purple
    "EVA": "#f39c12",  # Orange
    "ConvNeXt": "#e67e22",  # Dark orange
    "ResNet": "#c0392b",  # Dark red
    "MobileCLIP": "#1abc9c",  # Teal
    "MobileCLIP2": "#16a085",  # Darker teal
    "ViTamin": "#8e44ad",  # Dark purple
    "PE": "#2980b9",  # Dark blue
    "NLLB-SigLIP": "#58d68d",  # Light green
    "NNLB-CLIP": "#5dade2",  # Light blue
    "RoBERTa-CLIP": "#af7ac5",  # Light purple
    "XLM-RoBERTa-CLIP": "#bb8fce",  # Lighter purple
    "Other": "#95a5a6",  # Gray
}

DATASET_SIZES_M = {
    # OpenAI
    "OpenAI-400m": 400,
    # LAION variants
    "LAION-400m": 400,
    "LAION-2b": 2000,
    "LAION-5b": 5000,
    "LAION-A-900m": 900,
    # DataComp variants
    "DataComp-1b": 1000,
    "DataComp-12.8b": 12800,
    "DataComp-128m": 128,
    "DataComp-13m": 13,
    # CommonPool variants
    "CommonPool-12.8b": 12800,
    "CommonPool-1b": 1000,
    "CommonPool-128m": 128,
    "CommonPool-13m": 13,
    # MetaCLIP variants
    "MetaCLIP-400m": 400,
    "MetaCLIP-5.4b": 5400,
    "MetaCLIP2-2.5b": 2500,
    "CommonCrawl-2.5b": 2500,  # MetaCLIP fullcc
    # DFN variants
    "DFN-2b": 2000,
    "DFN-5b": 5000,
    "DFNDR-2b": 2000,
    # WebLI
    "WebLI-10b": 10000,
    # Merged (EVA models)
    "Merged-2b": 2000,
    # Small datasets
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
METRICS = [
    ("avg_accuracy", "Average Accuracy"),
    ("worst_group_accuracy", "Worst-Group Accuracy"),
]


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


def save_figure(fig, filepath_base):
    """
    Save figure in multiple formats: PDF, PNG, and pickle (editable).

    Args:
        fig: matplotlib figure object
        filepath_base: base path without extension (e.g., 'output/fig_landscape_urbancars')

    Usage to reload and edit:
        import pickle
        import matplotlib.pyplot as plt

        with open('fig_landscape_urbancars.pkl', 'rb') as f:
            fig = pickle.load(f)

        # Edit the figure
        ax = fig.axes[0]
        ax.set_xlabel('New Label')
        ax.set_xlim(0, 100)

        # Save again
        fig.savefig('edited_figure.pdf')
        plt.show()
    """
    # Save as PDF (vector)
    fig.savefig(f"{filepath_base}.pdf")

    # Save as PNG (raster)
    fig.savefig(f"{filepath_base}.png", dpi=300)

    # Save as pickle (editable)
    with open(f"{filepath_base}.pkl", "wb") as f:
        pickle.dump(fig, f)


def load_and_enrich(path: str, name: str, report: ReportWriter) -> pd.DataFrame:
    df = pd.read_csv(path)
    report.add_header(f"DATA LOADING: {name}", level=2)
    report.add_kv("File", path)
    report.add_kv("Models", len(df))

    for col in [
        "total_params_M",
        "image_size",
        "patch_size",
        "embed_dim",
        "avg_accuracy",
        "worst_group_accuracy",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["training_objective"] = df["model_name"].apply(
        lambda x: (
            "SigLIP"
            if "SigLIP" in str(x)
            else "CoCa" if "coca" in str(x).lower() else "Contrastive"
        )
    )
    df["training_data_size_M"] = df["training_data"].map(DATASET_SIZES_M).fillna(400)
    df["patch_size_num"] = pd.to_numeric(df["patch_size"], errors="coerce").fillna(16)
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
    report.add_kv("Architectures", df["arch_family"].value_counts().to_dict())
    report.add_kv("Training data", df["training_data"].value_counts().to_dict())
    report.add_kv("Image sizes", sorted(df["image_size"].dropna().unique().tolist()))
    report.add_kv(
        "Patch sizes", sorted(df["patch_size_num"].dropna().unique().tolist())
    )
    report.add_kv(
        "Params range",
        f"{df['total_params_M'].min():.1f} - {df['total_params_M'].max():.1f}",
    )
    return df


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled = np.sqrt(((n1 - 1) * g1.var() + (n2 - 1) * g2.var()) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 0 else 0


def analyze_overall(df: pd.DataFrame, name: str, report: ReportWriter):
    report.add_header(f"OVERALL STATISTICS: {name}", level=2)
    for mc, ml in METRICS:
        if mc not in df.columns:
            continue
        v = df[mc].dropna()
        report.add_subheader(ml)
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
    name: str,
    report: ReportWriter,
    min_n: int = 3,
):
    report.add_header(f"FACTOR: {flabel} ({name})", level=2)
    if fcol not in df.columns:
        report.add_line(f"Column {fcol} not found.")
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
    for mc, ml in METRICS:
        if mc not in df.columns:
            continue
        report.add_subheader(f"{ml} by {flabel}")
        g = (
            df_f.groupby(fcol)[mc]
            .agg(["mean", "std", "min", "max", "median", "count"])
            .sort_values("mean", ascending=False)
        )
        report.add_line(
            f"\n{'Level':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Med':>8} {'N':>5}"
        )
        report.add_line("-" * 85)
        for lvl, row in g.iterrows():
            report.add_line(
                f"{str(lvl):<35} {row['mean']:>8.4f} {row['std']:>8.4f} {row['min']:>8.4f} {row['max']:>8.4f} {row['median']:>8.4f} {int(row['count']):>5}"
            )

        gdata = [
            grp[mc].dropna().values for _, grp in df_f.groupby(fcol) if len(grp) >= 2
        ]
        if len(gdata) >= 2:
            h, p = kruskal(*gdata)
            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            report.add_line(f"\nKruskal-Wallis: H={h:.4f}, p={p:.6f} ({sig})")

        if fcol == "training_objective":
            report.add_line("\nPairwise comparisons:")
            for o1, o2 in [
                ("SigLIP", "Contrastive"),
                ("SigLIP", "CoCa"),
                ("Contrastive", "CoCa"),
            ]:
                g1 = df_f[df_f[fcol] == o1][mc].dropna().values
                g2 = df_f[df_f[fcol] == o2][mc].dropna().values
                if len(g1) >= 2 and len(g2) >= 2:
                    u, p = mannwhitneyu(g1, g2, alternative="two-sided")
                    d = cohens_d(g1, g2)
                    sig = (
                        "***"
                        if p < 0.001
                        else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    )
                    report.add_line(
                        f"  {o1} vs {o2}: U={u:.0f}, p={p:.6f} ({sig}), d={d:+.3f}, Δ={g1.mean()-g2.mean():+.4f}"
                    )


def analyze_numeric(
    df: pd.DataFrame, fcol: str, flabel: str, name: str, report: ReportWriter
):
    report.add_header(f"NUMERIC: {flabel} ({name})", level=2)
    if fcol not in df.columns:
        report.add_line(f"Column {fcol} not found.")
        return

    valid = df[fcol].notna()
    report.add_kv("Valid samples", valid.sum())
    if valid.sum() < 5:
        report.add_line("Insufficient data.")
        return

    v = df.loc[valid, fcol]
    report.add_subheader(f"{flabel} Distribution")
    for k, fn in [
        ("Mean", np.mean),
        ("Std", np.std),
        ("Min", np.min),
        ("Max", np.max),
        ("Median", np.median),
    ]:
        report.add_kv(k, fn(v))

    for mc, ml in METRICS:
        if mc not in df.columns:
            continue
        mask = valid & df[mc].notna()
        x, y = df.loc[mask, fcol].values, df.loc[mask, mc].values
        if len(x) < 5:
            continue

        report.add_subheader(f"{flabel} → {ml}")
        rp, pp = pearsonr(x, y)
        rs, ps = spearmanr(x, y)
        report.add_line(
            f"  Pearson: r={rp:+.4f}, p={pp:.6f} ({'***' if pp<0.001 else '**' if pp<0.01 else '*' if pp<0.05 else 'ns'})"
        )
        report.add_line(
            f"  Spearman: ρ={rs:+.4f}, p={ps:.6f} ({'***' if ps<0.001 else '**' if ps<0.01 else '*' if ps<0.05 else 'ns'})"
        )
        report.add_kv("N", len(x))
        slope, intercept = np.polyfit(x, y, 1)
        report.add_line(f"  Linear: y = {slope:.6f}*x + {intercept:.4f}")


def run_regression(df: pd.DataFrame, name: str, report: ReportWriter):
    report.add_header(f"REGRESSION: {name}", level=2)
    if not HAS_STATSMODELS:
        report.add_line("statsmodels not available.")
        return

    df_r = df.copy()
    df_r["wg_pct"] = df_r["worst_group_accuracy"] * 100
    df_r["is_siglip"] = (df_r["training_objective"] == "SigLIP").astype(int)
    df_r["is_coca"] = (df_r["training_objective"] == "CoCa").astype(int)
    df_r["log_params"] = np.log10(df_r["total_params_M"] + 1)
    df_r["log_data"] = np.log10(df_r["training_data_size_M"] + 1)
    df_r["log_tokens"] = np.log2(df_r["num_tokens"] + 1)

    cols = [
        "wg_pct",
        "is_siglip",
        "is_coca",
        "log_params",
        "log_data",
        "image_size",
        "log_tokens",
    ]
    df_r = df_r.dropna(subset=[c for c in cols if c in df_r.columns])
    report.add_kv("Samples", len(df_r))
    if len(df_r) < 30:
        report.add_line("Insufficient data.")
        return

    formula = (
        "wg_pct ~ is_siglip + is_coca + log_params + log_data + image_size + log_tokens"
    )
    try:
        model = smf.ols(formula, data=df_r).fit()
        report.add_line(f"\nFormula: {formula}")
        report.add_kv("R-squared", model.rsquared)
        report.add_kv("Adj R-squared", model.rsquared_adj)
        report.add_kv("F-statistic", model.fvalue)
        report.add_kv("F p-value", model.f_pvalue)

        report.add_line(
            f"\n{'Variable':<20} {'Coef':>10} {'SE':>10} {'t':>10} {'p':>12} {'Sig':>5}"
        )
        report.add_line("-" * 70)
        for var in model.params.index:
            c, se, t, p = (
                model.params[var],
                model.bse[var],
                model.tvalues[var],
                model.pvalues[var],
            )
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            report.add_line(
                f"{var:<20} {c:>10.4f} {se:>10.4f} {t:>10.3f} {p:>12.6f} {sig:>5}"
            )

        report.add_subheader("Interpretation")
        for var in ["is_siglip", "log_params", "image_size"]:
            if var in model.params.index and model.pvalues[var] < 0.05:
                c = model.params[var]
                report.add_line(
                    f"  {var}: {'increases' if c > 0 else 'decreases'} WG by {abs(c):.2f}pp (p={model.pvalues[var]:.4f})"
                )
    except Exception as e:
        report.add_line(f"Regression failed: {e}")


def compute_feature_importance(
    df: pd.DataFrame, name: str, report: ReportWriter, out_dir: str
):
    report.add_header(f"FEATURE IMPORTANCE: {name}", level=2)
    if not HAS_SKLEARN:
        report.add_line("sklearn not available.")
        return

    df_ml = df.copy()
    encoders = {}
    for col in ["training_objective", "arch_family", "training_data"]:
        if col in df_ml.columns:
            le = LabelEncoder()
            df_ml[f"{col}_enc"] = le.fit_transform(
                df_ml[col].fillna("Unknown").astype(str)
            )
            encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    fcols = [
        "total_params_M",
        "image_size",
        "patch_size_num",
        "num_tokens",
        "training_data_size_M",
        "embed_dim",
    ]
    fcols += [
        f"{c}_enc"
        for c in ["training_objective", "arch_family", "training_data"]
        if f"{c}_enc" in df_ml.columns
    ]
    fcols = [c for c in fcols if c in df_ml.columns]
    df_ml = df_ml.dropna(subset=["worst_group_accuracy"] + fcols)

    report.add_kv("Samples", len(df_ml))
    if len(df_ml) < 30:
        report.add_line("Insufficient data.")
        return

    X, y = df_ml[fcols].values, df_ml["worst_group_accuracy"].values
    rf = RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=6, n_jobs=-1
    )
    rf.fit(X, y)
    cv = cross_val_score(rf, X, y, cv=5, scoring="r2")
    report.add_kv("CV R² mean", cv.mean())
    report.add_kv("CV R² std", cv.std())

    imp = pd.DataFrame(
        {"Feature": fcols, "Importance": rf.feature_importances_}
    ).sort_values("Importance", ascending=False)
    report.add_subheader("Ranking")
    report.add_line(f"{'Rank':<5} {'Feature':<30} {'Importance':>12}")
    report.add_line("-" * 50)
    for i, (_, row) in enumerate(imp.iterrows()):
        report.add_line(f"{i+1:<5} {row['Feature']:<30} {row['Importance']:>12.4f}")

    report.add_subheader("Encodings")
    for col, mapping in encoders.items():
        report.add_line(f"\n{col}:")
        for cat, code in sorted(mapping.items(), key=lambda x: x[1]):
            report.add_line(f"  {code}: {cat}")


def analyze_interactions(df: pd.DataFrame, name: str, report: ReportWriter):
    report.add_header(f"INTERACTIONS: {name}", level=2)

    for idx_col, col_col, title in [
        ("arch_family", "training_objective", "Arch × Objective"),
        ("training_data", "training_objective", "Data × Objective"),
    ]:
        report.add_subheader(title)
        pivot = df.pivot_table(
            values="worst_group_accuracy",
            index=idx_col,
            columns=col_col,
            aggfunc=["mean", "count"],
        )
        if not pivot.empty:
            report.add_line("\nMean WG Accuracy (%):")
            report.add_table((pivot["mean"] * 100).round(2))
            report.add_line("\nCounts:")
            report.add_table(pivot["count"])

    report.add_subheader("Image Size × Patch Size")
    df_r = df.dropna(subset=["image_size", "patch_size_num"])
    if len(df_r) > 0:
        pivot = df_r.pivot_table(
            values="worst_group_accuracy",
            index="image_size",
            columns="patch_size_num",
            aggfunc=["mean", "count"],
        )
        if not pivot.empty:
            report.add_line("\nMean WG Accuracy (%):")
            report.add_table((pivot["mean"] * 100).round(1))
            report.add_line("\nCounts:")
            report.add_table(pivot["count"])


def analyze_top_bottom(df: pd.DataFrame, name: str, report: ReportWriter, n: int = 20):
    report.add_header(f"TOP/BOTTOM MODELS: {name}", level=2)
    cols = [
        "model_id",
        "training_objective",
        "training_data",
        "arch_family",
        "total_params_M",
        "image_size",
        "patch_size",
        "avg_accuracy",
        "worst_group_accuracy",
    ]
    cols = [c for c in cols if c in df.columns]
    df_s = df.sort_values("worst_group_accuracy", ascending=False)

    for subset, label in [(df_s.head(n), f"Top {n}"), (df_s.tail(n), f"Bottom {n}")]:
        report.add_subheader(f"{label} by WG Accuracy")
        t = subset[cols].copy()
        t["avg_accuracy"] = (t["avg_accuracy"] * 100).round(2)
        t["worst_group_accuracy"] = (t["worst_group_accuracy"] * 100).round(2)
        report.add_table(t)
        report.add_line(f"\nCharacteristics:")
        report.add_kv("Objectives", t["training_objective"].value_counts().to_dict())
        report.add_kv("Data", t["training_data"].value_counts().to_dict())
        report.add_kv("Architectures", t["arch_family"].value_counts().to_dict())
        report.add_kv("Mean params", t["total_params_M"].mean())
        report.add_kv("Mean img size", t["image_size"].mean())


def analyze_subgroups(df: pd.DataFrame, name: str, report: ReportWriter):
    report.add_header(f"SUBGROUPS: {name}", level=2)
    sg_cols = [
        c
        for c in df.columns
        if c.startswith("acc_obj=")
        or (c.startswith("acc_") and c not in ["accuracy_gap", "avg_accuracy"])
    ]
    if not sg_cols:
        report.add_line("No subgroup columns found.")
        return

    report.add_kv("Subgroup columns", len(sg_cols))
    report.add_subheader("Mean Accuracy per Subgroup")
    stats = {
        c: {
            "mean": df[c].mean(),
            "std": df[c].std(),
            "min": df[c].min(),
            "max": df[c].max(),
        }
        for c in sg_cols
        if df[c].notna().sum() > 0
    }
    sorted_sg = sorted(stats.items(), key=lambda x: x[1]["mean"])
    report.add_line(f"\n{'Subgroup':<50} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    report.add_line("-" * 85)
    for sg, st in sorted_sg:
        report.add_line(
            f"{sg.replace('acc_', ''):<50} {st['mean']:>8.4f} {st['std']:>8.4f} {st['min']:>8.4f} {st['max']:>8.4f}"
        )

    report.add_subheader("Hardest Subgroup per Model")
    hardest = defaultdict(int)
    for _, row in df.iterrows():
        sg_accs = {c: row[c] for c in sg_cols if pd.notna(row[c])}
        if sg_accs:
            hardest[min(sg_accs, key=sg_accs.get)] += 1
    for sg, cnt in sorted(hardest.items(), key=lambda x: -x[1]):
        report.add_line(f"  {sg.replace('acc_', '')}: {cnt} ({cnt/len(df)*100:.1f}%)")


def cross_dataset(
    df1: pd.DataFrame, df2: pd.DataFrame, n1: str, n2: str, report: ReportWriter
):
    report.add_header(f"CROSS-DATASET: {n1} vs {n2}", level=1)
    merged = df1[
        [
            "model_id",
            "model_name",
            "training_objective",
            "arch_family",
            "avg_accuracy",
            "worst_group_accuracy",
        ]
    ].merge(
        df2[["model_id", "avg_accuracy", "worst_group_accuracy"]],
        on="model_id",
        suffixes=(f"_{n1.lower()}", f"_{n2.lower()}"),
    )

    report.add_kv(f"Models in {n1}", len(df1))
    report.add_kv(f"Models in {n2}", len(df2))
    report.add_kv("Overlapping", len(merged))

    if len(merged) < 10:
        report.add_line("Insufficient overlap.")
        return

    report.add_subheader("Ranking Correlations")
    for m in ["avg_accuracy", "worst_group_accuracy"]:
        c1, c2 = f"{m}_{n1.lower()}", f"{m}_{n2.lower()}"
        rs, ps = spearmanr(merged[c1], merged[c2])
        rp, pp = pearsonr(merged[c1], merged[c2])
        ml = "Avg Acc" if m == "avg_accuracy" else "WG Acc"
        report.add_line(f"\n{ml}:")
        report.add_line(f"  Spearman: ρ={rs:.4f} (p={ps:.6f})")
        report.add_line(f"  Pearson: r={rp:.4f} (p={pp:.6f})")

    report.add_subheader("Largest Rank Differences")
    merged["r1"] = merged[f"worst_group_accuracy_{n1.lower()}"].rank(ascending=False)
    merged["r2"] = merged[f"worst_group_accuracy_{n2.lower()}"].rank(ascending=False)
    merged["rdiff"] = abs(merged["r1"] - merged["r2"])
    top_diff = merged.nlargest(10, "rdiff")[
        [
            "model_id",
            f"worst_group_accuracy_{n1.lower()}",
            f"worst_group_accuracy_{n2.lower()}",
            "r1",
            "r2",
            "rdiff",
        ]
    ]
    report.add_table(top_diff.round(3))

    report.add_subheader("Consistent Top Performers")
    top_both = merged[(merged["r1"] <= 20) & (merged["r2"] <= 20)].sort_values("r1")
    report.add_line(f"\nModels in top-20 on BOTH: {len(top_both)}")
    if len(top_both) > 0:
        report.add_table(
            top_both[
                [
                    "model_id",
                    "training_objective",
                    f"worst_group_accuracy_{n1.lower()}",
                    f"worst_group_accuracy_{n2.lower()}",
                ]
            ]
            .head(15)
            .round(3)
        )

    report.add_subheader("Factor Consistency")
    for f in ["training_objective", "arch_family"]:
        report.add_line(f"\n{f}:")
        for lvl in merged[f].dropna().unique():
            mask = merged[f] == lvl
            if mask.sum() >= 3:
                m1 = merged.loc[mask, f"worst_group_accuracy_{n1.lower()}"].mean()
                m2 = merged.loc[mask, f"worst_group_accuracy_{n2.lower()}"].mean()
                report.add_line(
                    f"  {lvl}: {n1}={m1:.3f}, {n2}={m2:.3f}, Δ={m1-m2:+.3f}"
                )


# Plotting functions
def plot_landscape(df, name, out_dir, report):
    report.add_header(f"FIGURE: Landscape ({name})", level=2)
    fig, ax = plt.subplots(figsize=(10, 8))
    x, y = df["avg_accuracy"].values * 100, df["worst_group_accuracy"].values * 100

    # Color by architecture
    archs = sorted(df["arch_family"].dropna().unique())
    for arch in archs:
        m = df["arch_family"] == arch
        ax.scatter(
            x[m],
            y[m],
            c=COLORS_ARCH.get(arch, "#95a5a6"),
            label=f"{arch} (n={m.sum()})",
            alpha=0.7,
            s=50,
        )
        report.add_line(
            f"{arch}: n={m.sum()}, avg={x[m].mean():.1f}%, wg={y[m].mean():.1f}%"
        )

    ax.plot([40, 100], [40, 100], "k--", alpha=0.3)

    # Pareto front
    pareto = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if (
                i != j
                and x[j] >= x[i]
                and y[j] >= y[i]
                and (x[j] > x[i] or y[j] > y[i])
            ):
                pareto[i] = False
                break
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

    # Add text labels for Pareto models with larger font
    report.add_line(f"\nPareto-optimal ({pareto.sum()}):")
    for idx in np.where(pareto)[0]:
        model_id = df.iloc[idx]["model_id"]
        # Shorten model name for display
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
        report.add_line(f"  {model_id}: avg={x[idx]:.1f}%, wg={y[idx]:.1f}%")

    ax.set_xlabel("Average Accuracy (%)")
    ax.set_ylabel("Worst-Group Accuracy (%)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(45, 100)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_landscape_{name.lower()}")
    plt.close()


def plot_objective(df, name, out_dir, report):
    report.add_header(f"FIGURE: Objective ({name})", level=2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = ["Contrastive", "CoCa", "SigLIP"]

    for ax, (mc, ml) in zip(axes, METRICS):
        data = [df[df["training_objective"] == o][mc].dropna() * 100 for o in order]
        for o, d in zip(order, data):
            report.add_line(
                f"{ml} - {o}: mean={d.mean():.2f}, std={d.std():.2f}, n={len(d)}"
            )
        bp = ax.boxplot(data, labels=order, patch_artist=True)
        for patch, o in zip(bp["boxes"], order):
            patch.set_facecolor(COLORS_OBJ[o])
            patch.set_alpha(0.7)
        for i, d in enumerate(data):
            ax.scatter(
                np.ones(len(d)) * (i + 1) + np.random.normal(0, 0.04, len(d)),
                d,
                alpha=0.4,
                s=15,
                c="black",
            )
        ax.set_ylabel(f"{ml} (%)")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_objective_{name.lower()}")
    plt.close()


def plot_training_data(df, name, out_dir, report, min_n=3):
    report.add_header(f"FIGURE: Training Data ({name})", level=2)
    stats = df.groupby("training_data").agg(
        {
            "worst_group_accuracy": ["mean", "std", "count"],
            "avg_accuracy": "mean",
            "training_data_size_M": "first",
        }
    )
    stats.columns = ["wg_mean", "wg_std", "count", "avg_mean", "data_size"]
    stats = stats[stats["count"] >= min_n].sort_values("wg_mean", ascending=True)

    if len(stats) < 2:
        report.add_line("Insufficient data.")
        return

    report.add_line(
        f"{'Data':<30} {'WG Mean':>8} {'WG Std':>8} {'Avg':>8} {'Size':>10} {'N':>5}"
    )
    report.add_line("-" * 75)
    for idx, row in stats.iterrows():
        report.add_line(
            f"{idx:<30} {row['wg_mean']*100:>8.2f} {row['wg_std']*100:>8.2f} {row['avg_mean']*100:>8.2f} {row['data_size']:>10.0f} {int(row['count']):>5}"
        )

    fig, ax = plt.subplots(figsize=(10, max(5, len(stats) * 0.4)))
    y_pos = np.arange(len(stats))
    ax.barh(
        y_pos,
        stats["wg_mean"] * 100,
        xerr=stats["wg_std"] * 100,
        color=plt.cm.RdYlGn(stats["wg_mean"].values),
        edgecolor="black",
        capsize=3,
        alpha=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{idx} (n={int(row['count'])})" for idx, row in stats.iterrows()], fontsize=8
    )
    ax.set_xlabel("Worst-Group Accuracy (%)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_data_{name.lower()}")
    plt.close()


def plot_architecture(df, name, out_dir, report, min_n=3):
    report.add_header(f"FIGURE: Architecture ({name})", level=2)
    valid = (
        df.groupby("arch_family")
        .filter(lambda x: len(x) >= min_n)["arch_family"]
        .unique()
    )
    df_f = df[df["arch_family"].isin(valid)]
    if len(valid) < 2:
        report.add_line("Insufficient architectures.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (mc, ml) in zip(axes, METRICS):
        stats = (
            df_f.groupby("arch_family")[mc]
            .agg(["mean", "std", "count"])
            .sort_values("mean", ascending=True)
        )
        report.add_line(f"\n{ml}:")
        for idx, row in stats.iterrows():
            report.add_line(
                f"  {idx}: mean={row['mean']*100:.2f}%, std={row['std']*100:.2f}%, n={int(row['count'])}"
            )
        y_pos = np.arange(len(stats))
        ax.barh(
            y_pos,
            stats["mean"] * 100,
            xerr=stats["std"] * 100,
            color=[COLORS_ARCH.get(a, "#95a5a6") for a in stats.index],
            edgecolor="black",
            capsize=3,
            alpha=0.8,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{idx} (n={int(row['count'])})" for idx, row in stats.iterrows()]
        )
        ax.set_xlabel(f"{ml} (%)")
        ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_arch_{name.lower()}")
    plt.close()


def plot_scale(df, name, out_dir, report):
    report.add_header(f"FIGURE: Scale ({name})", level=2)

    # Define all scale factors including patch_size
    factors = [
        ("total_params_M", "Parameters (M)", "params", True),
        ("image_size", "Image Size (px)", "imgsize", False),
        ("patch_size_num", "Patch Size (px)", "patchsize", False),
        ("num_tokens", "Number of Tokens", "tokens", True),
        ("training_data_size_M", "Training Data Size (M)", "datasize", True),
    ]

    # Get unique architectures for coloring
    archs = sorted(df["arch_family"].dropna().unique())

    for fcol, flabel, fname, use_log in factors:
        if fcol not in df.columns:
            continue

        # Get valid data
        mask = df[fcol].notna() & df["worst_group_accuracy"].notna()
        if mask.sum() < 5:
            continue

        x_all = df.loc[mask, fcol].values
        y_all = df.loc[mask, "worst_group_accuracy"].values * 100

        # ========================================
        # SCATTER PLOT with trend line
        # ========================================
        fig, ax = plt.subplots(figsize=(8, 6))

        for arch in archs:
            m = (df["arch_family"] == arch) & mask
            x_vals = df.loc[m, fcol].values
            y_vals = df.loc[m, "worst_group_accuracy"].values * 100

            # Add jitter for non-log scales to avoid overlap
            if not use_log and len(x_vals) > 0:
                jitter = np.random.normal(
                    0, 0.02 * (x_all.max() - x_all.min() + 1), len(x_vals)
                )
                x_vals = x_vals + jitter

            ax.scatter(
                x_vals,
                y_vals,
                c=COLORS_ARCH.get(arch, "#95a5a6"),
                label=f"{arch} (n={m.sum()})",
                alpha=0.6,
                s=40,
                edgecolors="white",
                linewidth=0.3,
            )

        # Add trend line
        if use_log:
            # For log scale, fit on log-transformed x
            log_x = np.log10(x_all + 1)
            slope, intercept = np.polyfit(log_x, y_all, 1)
            x_line = np.linspace(x_all.min(), x_all.max(), 100)
            y_line = slope * np.log10(x_line + 1) + intercept
        else:
            slope, intercept = np.polyfit(x_all, y_all, 1)
            x_line = np.linspace(x_all.min(), x_all.max(), 100)
            y_line = slope * x_line + intercept

        ax.plot(x_line, y_line, "r-", linewidth=2, alpha=0.7, label="Trend")

        if use_log:
            ax.set_xscale("log")

        ax.set_xlabel(flabel)
        ax.set_ylabel("Worst-Group Accuracy (%)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, f"{out_dir}/fig_scale_{fname}_{name.lower()}")
        plt.close()

        # Log correlation
        r, p = spearmanr(x_all, y_all)
        report.add_line(f"{flabel} vs WG: ρ={r:.4f}, p={p:.6f}")

        # ========================================
        # BAR CHART version (binned)
        # ========================================
        fig, ax = plt.subplots(figsize=(10, 6))

        # Number of bins: 8 for tokens, 5 for others
        n_bins = (
            9 if fcol == "num_tokens" else 6
        )  # n_bins = n_edges, so 9 edges = 8 bars

        # Create bins
        if use_log:
            # Log-scale bins
            log_min, log_max = np.log10(x_all.min() + 1), np.log10(x_all.max() + 1)
            bin_edges = np.logspace(log_min, log_max, n_bins)
        else:
            # Linear bins
            bin_edges = np.linspace(x_all.min(), x_all.max(), n_bins)

        # Assign bins
        df_temp = df.loc[mask].copy()
        df_temp["bin"] = pd.cut(df_temp[fcol], bins=bin_edges, include_lowest=True)

        # Compute stats per bin
        bin_stats = df_temp.groupby("bin", observed=True)["worst_group_accuracy"].agg(
            ["mean", "std", "count"]
        )
        bin_stats = bin_stats[bin_stats["count"] >= 2]  # Filter bins with < 2 samples

        if len(bin_stats) < 2:
            continue

        # Create bar labels
        bar_labels = []
        for interval in bin_stats.index:
            if use_log:
                bar_labels.append(f"{interval.left:.0f}-{interval.right:.0f}")
            else:
                bar_labels.append(f"{interval.left:.0f}-{interval.right:.0f}")

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

        # Add count labels on bars
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
        ax.set_xlabel(flabel)
        ax.set_ylabel("Worst-Group Accuracy (%)")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        save_figure(fig, f"{out_dir}/fig_scale_{fname}_bar_{name.lower()}")
        plt.close()

        # Log bin stats
        report.add_line(f"\n{flabel} binned stats:")
        for interval, row in bin_stats.iterrows():
            report.add_line(
                f"  {interval}: mean={row['mean']*100:.1f}%, std={row['std']*100:.1f}%, n={int(row['count'])}"
            )


def plot_resolution_patch(df, name, out_dir, report):
    report.add_header(f"FIGURE: Resolution×Patch ({name})", level=2)
    df_f = df.dropna(subset=["image_size", "patch_size_num", "worst_group_accuracy"])
    pivot = df_f.pivot_table(
        values="worst_group_accuracy",
        index="image_size",
        columns="patch_size_num",
        aggfunc=["mean", "count"],
    )

    report.add_line("\nMean WG (%):")
    report.add_table((pivot["mean"] * 100).round(1))
    report.add_line("\nCounts:")
    report.add_table(pivot["count"])

    mean_masked = (pivot["mean"] * 100).where(pivot["count"] >= 2)
    if mean_masked.dropna(how="all").empty:
        report.add_line("Insufficient data.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    if HAS_SEABORN:
        sns.heatmap(
            mean_masked,
            annot=True,
            fmt=".0f",
            cmap="RdYlGn",
            ax=ax,
            linewidths=0.5,
            vmin=40,
            vmax=90,
            cbar_kws={"label": "WG Accuracy (%)"},
        )
    ax.set_xlabel("Patch Size")
    ax.set_ylabel("Image Size")
    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_respatch_{name.lower()}")
    plt.close()


def plot_correlation(df, name, out_dir, report):
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
    mcols = ["avg_accuracy", "worst_group_accuracy"]
    df_num = df[fcols + mcols].dropna()

    if len(df_num) < 10:
        report.add_line("Insufficient data.")
        return

    corr = pd.DataFrame(index=fcols, columns=mcols, dtype=float)
    pval = pd.DataFrame(index=fcols, columns=mcols, dtype=float)

    report.add_line(
        f"\n{'Factor':<25} {'Avg':>10} {'WG':>10} {'p(Avg)':>12} {'p(WG)':>12}"
    )
    report.add_line("-" * 75)
    for f in fcols:
        for m in mcols:
            r, p = spearmanr(df_num[f], df_num[m])
            corr.loc[f, m], pval.loc[f, m] = r, p
        ra, rw = corr.loc[f, "avg_accuracy"], corr.loc[f, "worst_group_accuracy"]
        pa, pw = pval.loc[f, "avg_accuracy"], pval.loc[f, "worst_group_accuracy"]
        sa = "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else ""
        sw = "***" if pw < 0.001 else "**" if pw < 0.01 else "*" if pw < 0.05 else ""
        report.add_line(
            f"{f:<25} {ra:>+10.4f}{sa} {rw:>+10.4f}{sw} {pa:>12.6f} {pw:>12.6f}"
        )

    fig, ax = plt.subplots(figsize=(8, 6))
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


def plot_cross(df1, df2, n1, n2, out_dir, report):
    report.add_header(f"FIGURE: Cross-Dataset", level=2)
    merged = df1[
        ["model_id", "arch_family", "avg_accuracy", "worst_group_accuracy"]
    ].merge(
        df2[["model_id", "avg_accuracy", "worst_group_accuracy"]],
        on="model_id",
        suffixes=("_1", "_2"),
    )

    if len(merged) < 10:
        report.add_line("Insufficient overlap.")
        return

    archs = sorted(merged["arch_family"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, m in zip(axes, ["avg_accuracy", "worst_group_accuracy"]):
        x, y = merged[f"{m}_1"].values * 100, merged[f"{m}_2"].values * 100

        # Plot by architecture
        for arch in archs:
            mask = merged["arch_family"] == arch
            ax.scatter(
                x[mask],
                y[mask],
                c=COLORS_ARCH.get(arch, "#95a5a6"),
                label=f"{arch} (n={mask.sum()})",
                alpha=0.6,
                s=40,
                edgecolors="white",
                linewidth=0.3,
            )

        # Diagonal line
        lims = [min(x.min(), y.min()) - 5, max(x.max(), y.max()) + 5]
        ax.plot(lims, lims, "k--", alpha=0.3)

        # Regression line
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot(
            np.linspace(x.min(), x.max(), 100),
            slope * np.linspace(x.min(), x.max(), 100) + intercept,
            "r-",
            alpha=0.5,
        )

        r, p = spearmanr(x, y)
        ml = "Average Accuracy" if m == "avg_accuracy" else "Worst-Group Accuracy"
        ax.set_xlabel(f"{ml} - {n1} (%)")
        ax.set_ylabel(f"{ml} - {n2} (%)")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        report.add_line(f"{ml}: ρ={r:.4f}, p={p:.6f}")

    plt.tight_layout()
    save_figure(fig, f"{out_dir}/fig_cross")
    plt.close()


def analyze_dataset(df, name, out_dir, report, min_n=3):
    report.add_header(f"ANALYSIS: {name}", level=1)
    analyze_overall(df, name, report)

    for fcol, flabel in [
        ("training_objective", "Objective"),
        ("training_data", "Data"),
        ("arch_family", "Architecture"),
        ("resolution_bucket", "Resolution"),
        ("params_bucket", "Params"),
    ]:
        analyze_categorical(df, fcol, flabel, name, report, min_n)
    if "vit_size" in df.columns:
        analyze_categorical(df, "vit_size", "ViT Size", name, report, min_n)
    if "quickgelu" in df.columns:
        analyze_categorical(df, "quickgelu", "QuickGELU", name, report, min_n)

    for fcol, flabel in [
        ("total_params_M", "Params"),
        ("image_size", "Image Size"),
        ("patch_size_num", "Patch Size"),
        ("num_tokens", "Tokens"),
        ("embed_dim", "Embed Dim"),
        ("training_data_size_M", "Data Size"),
    ]:
        analyze_numeric(df, fcol, flabel, name, report)

    run_regression(df, name, report)
    compute_feature_importance(df, name, report, out_dir)
    analyze_interactions(df, name, report)
    analyze_top_bottom(df, name, report, 20)
    analyze_subgroups(df, name, report)

    plot_landscape(df, name, out_dir, report)
    # plot_objective(df, name, out_dir, report)
    plot_training_data(df, name, out_dir, report, min_n)
    plot_architecture(df, name, out_dir, report, min_n)
    plot_scale(df, name, out_dir, report)
    plot_resolution_patch(df, name, out_dir, report)
    plot_correlation(df, name, out_dir, report)


def main():
    parser = argparse.ArgumentParser(description="VLM Robustness Analysis")
    parser.add_argument("--urbancars", required=True, help="UrbanCars CSV")
    parser.add_argument("--celeba", default=None, help="CelebA CSV")
    parser.add_argument("--output", default="paper_analysis", help="Output dir")
    parser.add_argument("--min-n", type=int, default=3, help="Min samples per group")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    report = ReportWriter()

    report.add_header("VLM ROBUSTNESS ANALYSIS - COMPLETE REPORT", level=1)
    report.add_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.add_line(f"Output: {args.output}")
    report.add_line("\nThis report contains ALL numerical results for paper writing.")

    # UrbanCars
    print("=" * 80)
    print("Loading UrbanCars...")
    df_uc = load_and_enrich(args.urbancars, "UrbanCars", report)
    print(f"Loaded {len(df_uc)} models")
    print("\nAnalyzing...")
    analyze_dataset(df_uc, "UrbanCars", args.output, report, args.min_n)

    # CelebA
    df_celeba = None
    if args.celeba:
        print("\n" + "=" * 80)
        print("Loading CelebA...")
        df_celeba = load_and_enrich(args.celeba, "CelebA", report)
        print(f"Loaded {len(df_celeba)} models")
        print("\nAnalyzing...")
        analyze_dataset(df_celeba, "CelebA", args.output, report, args.min_n)

        print("\nCross-dataset analysis...")
        cross_dataset(df_uc, df_celeba, "UrbanCars", "CelebA", report)
        plot_cross(df_uc, df_celeba, "UrbanCars", "CelebA", args.output, report)

    # Summary
    report.add_header("EXECUTIVE SUMMARY", level=1)
    report.add_subheader("UrbanCars Key Findings")
    top5 = df_uc.nlargest(5, "worst_group_accuracy")
    report.add_line("\nTop 5 by WG:")
    for _, r in top5.iterrows():
        report.add_line(
            f"  {r['model_id']}: WG={r['worst_group_accuracy']*100:.1f}%, Avg={r['avg_accuracy']*100:.1f}%, {r['training_objective']}"
        )

    report.add_line("\nObjective Summary:")
    for obj in ["SigLIP", "Contrastive", "CoCa"]:
        m = df_uc["training_objective"] == obj
        if m.sum() > 0:
            wg = df_uc.loc[m, "worst_group_accuracy"]
            report.add_line(
                f"  {obj}: WG={wg.mean()*100:.1f}% ± {wg.std()*100:.1f}% (n={m.sum()})"
            )

    if df_celeba is not None:
        report.add_subheader("CelebA Key Findings")
        top5c = df_celeba.nlargest(5, "worst_group_accuracy")
        report.add_line("\nTop 5 by WG:")
        for _, r in top5c.iterrows():
            report.add_line(
                f"  {r['model_id']}: WG={r['worst_group_accuracy']*100:.1f}%, {r['training_objective']}"
            )

        report.add_subheader("Cross-Dataset")
        merged = df_uc[["model_id", "worst_group_accuracy"]].merge(
            df_celeba[["model_id", "worst_group_accuracy"]],
            on="model_id",
            suffixes=("_uc", "_celeba"),
        )
        if len(merged) >= 10:
            r, p = spearmanr(
                merged["worst_group_accuracy_uc"], merged["worst_group_accuracy_celeba"]
            )
            report.add_line(
                f"\nWG Ranking Correlation: ρ={r:.3f} (p={p:.2e}), n={len(merged)}"
            )

    # Save
    report_path = os.path.join(args.output, "complete_analysis_report.txt")
    report.save(report_path)

    # Tables
    cols = [
        "model_id",
        "training_objective",
        "training_data",
        "arch_family",
        "total_params_M",
        "image_size",
        "avg_accuracy",
        "worst_group_accuracy",
    ]
    cols = [c for c in cols if c in df_uc.columns]
    top_uc = df_uc.nlargest(30, "worst_group_accuracy")[cols].copy()
    top_uc["avg_accuracy"] = (top_uc["avg_accuracy"] * 100).round(2)
    top_uc["worst_group_accuracy"] = (top_uc["worst_group_accuracy"] * 100).round(2)
    top_uc.to_csv(os.path.join(args.output, "top_models_urbancars.csv"), index=False)

    if df_celeba is not None:
        cols_c = [c for c in cols if c in df_celeba.columns]
        top_c = df_celeba.nlargest(30, "worst_group_accuracy")[cols_c].copy()
        top_c["avg_accuracy"] = (top_c["avg_accuracy"] * 100).round(2)
        top_c["worst_group_accuracy"] = (top_c["worst_group_accuracy"] * 100).round(2)
        top_c.to_csv(os.path.join(args.output, "top_models_celeba.csv"), index=False)

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
    print(f"\nUrbanCars ({len(df_uc)} models):")
    print(
        f"  WG Range: {df_uc['worst_group_accuracy'].min()*100:.1f}% - {df_uc['worst_group_accuracy'].max()*100:.1f}%"
    )
    print(
        f"  WG Mean: {df_uc['worst_group_accuracy'].mean()*100:.1f}% ± {df_uc['worst_group_accuracy'].std()*100:.1f}%"
    )

    sig = df_uc[df_uc["training_objective"] == "SigLIP"]["worst_group_accuracy"]
    con = df_uc[df_uc["training_objective"] == "Contrastive"]["worst_group_accuracy"]
    if len(sig) > 0 and len(con) > 0:
        print(
            f"\n  SigLIP: {sig.mean()*100:.1f}% ± {sig.std()*100:.1f}% (n={len(sig)})"
        )
        print(
            f"  Contrastive: {con.mean()*100:.1f}% ± {con.std()*100:.1f}% (n={len(con)})"
        )
        print(f"  Δ: {(sig.mean()-con.mean())*100:+.1f}pp")


if __name__ == "__main__":
    main()
