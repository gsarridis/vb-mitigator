"""
Plot mean performance per dataset, grouped by category, for two axes:
  - Axis B: how the bias signal is acquired
  - Axis C: what the algorithm does once it has the signal

Some methods belong to more than one category per axis. In that case the
method's performance contributes to *each* of the categories it belongs
to (so the per-category mean is computed over all methods that include
that category as one of their labels).

Inputs:
  - performances.csv                      (method,dataset,mean,std)
  - categorization/method_categories.csv  (method,B,C). The B and C
    cells contain a single category code (e.g. "B1") or a comma-separated
    list (e.g. "B1,B2") for methods that belong to multiple categories.

Outputs:
  - perf_vs_dataset_axis_B_signal.pdf
  - perf_vs_dataset_axis_C_intervention.pdf
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# Global Style Settings
# ============================
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk", font_scale=1.1)

UNIFIED_PALETTE = sns.color_palette("tab10", n_colors=10)

# ============================
# Pretty labels for the category codes
# ============================
AXIS_B_LABELS = {
    "B1": "Direct bias labels access",
    "B2": "Auxiliary models w/ bias labels access",
    "B3": "Auxiliary models w/o bias labels access",
    "B4": "Pseudo-labels through main model",
    "B5": "No bias signal",
}

AXIS_C_LABELS = {
    "C1": "Loss reweighting",
    "C2": "Dataset resampling",
    "C3": "Logit-space intervention",
    "C4": "Representation space regularizer",
    "C5": "Bias injection",
    "C6": "Architectural separation",
}

# ============================
# Load CSVs
# ============================
perf_df = pd.read_csv("performances.csv")  # method,dataset,mean,std
method_cat = pd.read_csv("categorization/method_attr_v2.csv")  # method,B,C


# ============================
# Helper: parse a comma-separated category cell into a list of codes
# ============================
def parse_categories(cell):
    if pd.isna(cell):
        return []
    return [c.strip() for c in str(cell).split(",") if c.strip()]


# ============================
# Separate Vanilla from the methods
# ============================
vanilla = perf_df[perf_df["method"] == "Vanilla"][["dataset", "mean"]].rename(
    columns={"mean": "vanilla_mean"}
)
perf_df = perf_df[perf_df["method"] != "Vanilla"].reset_index(drop=True)


# ============================
# Helper: explode the (method, axis) cell into one row per category
# ============================
def build_long_frame(method_cat, perf_df, axis_col, label_map):
    """Return a long-format frame with one row per (method, dataset, category).

    A method whose `axis_col` cell contains "B1,B2" will produce two rows
    per dataset (one for B1, one for B2) so the per-category mean
    aggregates all member methods.
    """
    cat_long = method_cat[["method", axis_col]].copy()
    cat_long["category_codes"] = cat_long[axis_col].apply(parse_categories)
    cat_long = cat_long.explode("category_codes").rename(
        columns={"category_codes": "category_code"}
    )
    cat_long = cat_long.dropna(subset=["category_code"])
    cat_long["category"] = cat_long["category_code"].map(label_map)
    cat_long = cat_long[["method", "category"]]

    long_df = perf_df.merge(cat_long, on="method", how="inner")
    return long_df


# ============================
# Helper: draw the grouped bar plot for one axis
# ============================
def plot_axis(long_df, label_map, title, out_path):
    category_order = list(label_map.values())

    agg = (
        long_df.groupby(["dataset", "category"], as_index=False)["mean"]
        .mean()
        .rename(columns={"mean": "category_mean"})
    )

    dataset_order = list(perf_df["dataset"].drop_duplicates())

    plt.figure(figsize=(18, 6.5))
    ax = sns.barplot(
        data=agg,
        x="dataset",
        y="category_mean",
        hue="category",
        hue_order=category_order,
        order=dataset_order,
        palette=UNIFIED_PALETTE[: len(category_order)],
    )

    plt.xticks(rotation=15)
    plt.xlabel("Dataset")
    plt.ylabel("Mean performance")
    plt.title(title)
    leg = plt.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        title=None,
    )
    leg.get_frame().set_facecolor("lightgrey")
    leg.get_frame().set_edgecolor("black")

    # Vanilla horizontal dotted lines per dataset
    xticks = ax.get_xticks()
    xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
    for x_pos, ds in zip(xticks, xtick_labels):
        v = vanilla.loc[vanilla["dataset"] == ds, "vanilla_mean"].values
        if len(v) > 0:
            ax.hlines(
                y=v[0],
                xmin=x_pos - 0.4,
                xmax=x_pos + 0.4,
                linestyles="--",
                color="red",
                linewidth=2,
                alpha=0.6,
            )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ============================
# Build long frames and plot
# ============================
long_B = build_long_frame(method_cat, perf_df, "B", AXIS_B_LABELS)
long_C = build_long_frame(method_cat, perf_df, "C", AXIS_C_LABELS)

plot_axis(
    long_B,
    AXIS_B_LABELS,
    title=None,
    out_path="perf_vs_dataset_axis_B_signal.pdf",
)
plot_axis(
    long_C,
    AXIS_C_LABELS,
    title=None,
    out_path="perf_vs_dataset_axis_C_intervention.pdf",
)


# ============================
# Print a small summary so you can sanity-check membership
# ============================
def print_membership(method_cat, axis_col, label_map, axis_name):
    print(f"\n=== {axis_name} membership ===")
    members_by_code = {code: [] for code in label_map}
    for _, row in method_cat.iterrows():
        for code in parse_categories(row[axis_col]):
            if code in members_by_code:
                members_by_code[code].append(row["method"])
    for code, label in label_map.items():
        print(f"  {label:36s} -> {members_by_code[code]}")


print_membership(method_cat, "B", AXIS_B_LABELS, "Axis B")
print_membership(method_cat, "C", AXIS_C_LABELS, "Axis C")
