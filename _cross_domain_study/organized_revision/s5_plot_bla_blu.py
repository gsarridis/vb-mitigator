import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# Global Style Settings
# ============================

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk", font_scale=1.1)

# Unified discrete color palette used everywhere
UNIFIED_PALETTE = sns.color_palette("tab10", n_colors=10)

# ============================
# Load CSVs
# ============================
perf_df = pd.read_csv("performances.csv")  # method,dataset,mean,std
data_attr = pd.read_csv(
    "./categorization/data_attr.csv"
)  # dataset,classes,bias_intensity
method_attr = pd.read_csv(
    "./categorization/method_attr.csv"
)  # method,bias_labels,bcc,multi_obj,reweight,fundamental

# ============================
# Remove Vanilla (but keep for gain computation)
# ============================
vanilla = perf_df[perf_df["method"] == "Vanilla"][["dataset", "mean"]]
vanilla = vanilla.rename(columns={"mean": "vanilla_mean"})

perf_df = perf_df[perf_df["method"] != "Vanilla"].reset_index(drop=True)

# ============================
# Merge performance with dataset attributes
# ============================
df = perf_df.merge(data_attr, on="dataset", how="left")

# ============================
# Compute performance gain vs vanilla
# ============================
df = df.merge(vanilla, on="dataset", how="left")
df["gain"] = df["mean"] - df["vanilla_mean"]

# ============================
# Convert method attributes into long-format group labels
# ============================
group_cols = ["bias_labels", "bcc", "multi_obj", "reweight", "fundamental"]

method_groups = method_attr.melt(
    id_vars="method", value_vars=group_cols, var_name="group", value_name="in_group"
)

method_groups = method_groups[method_groups["in_group"] == 1].drop(columns="in_group")


df_compare = None  # filled later


method_attr["any_other_group"] = (
    method_attr[group_cols].sum(axis=1) - method_attr["bias_labels"]
)

method_attr["bias_group_final"] = method_attr["bias_labels"].apply(
    lambda x: "BLA" if x == 1 else "BLU"
)

df_compare = df.merge(
    method_attr[["method", "bias_group_final"]], on="method", how="left"
)


# ===========================================
# FIGURE 4 — Performance vs Dataset (bias_labels vs not)
# ===========================================
# ===========================================
# FIGURE 4 — Performance vs Dataset (bias_labels vs not)
# ===========================================
plt.figure(figsize=(14, 6))
ax = sns.barplot(
    data=df_compare,
    x="dataset",
    y="mean",
    hue="bias_group_final",
    palette=UNIFIED_PALETTE,
)

plt.xticks(rotation=15)
plt.xlabel("Dataset")
plt.ylabel("Mean Performance")
leg = plt.legend()
leg.get_frame().set_facecolor("lightgrey")
leg.get_frame().set_edgecolor("black")

# ===========================================
# Add dotted horizontal lines for Vanilla performance
# ===========================================
# Get tick labels and corresponding x positions
xticks = ax.get_xticks()
datasets = ax.get_xticklabels()
datasets = [t.get_text() for t in datasets]

# For each dataset, draw a horizontal dotted line
for x_pos, ds in zip(xticks, datasets):
    vanilla_val = vanilla.loc[vanilla["dataset"] == ds, "vanilla_mean"].values
    if len(vanilla_val) > 0:
        ax.hlines(
            y=vanilla_val[0],
            xmin=x_pos - 0.4,  # left edge of bar group
            xmax=x_pos + 0.4,  # right edge of bar group
            linestyles="--",
            color="red",
            linewidth=2,
            alpha=0.5,
        )

plt.tight_layout()
plt.savefig("perf_vs_dataset_bias_labels_vs_not.pdf")
plt.show()
