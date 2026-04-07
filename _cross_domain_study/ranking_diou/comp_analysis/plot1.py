import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# Global Style Settings
# ============================

plt.style.use("seaborn-v0_8-whitegrid")
# sns.set_context("talk", font_scale=1.1)

# Unified discrete color palette used everywhere
UNIFIED_PALETTE = sns.color_palette("tab10", n_colors=10)

# ============================
# Load CSVs
# ============================
perf_df = pd.read_csv("performances.csv")  # method,dataset,mean,std
data_attr = pd.read_csv("data_attr.csv")  # dataset,classes,bias_intensity
method_attr = pd.read_csv(
    "method_attr.csv"
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

df_long = df.merge(method_groups, on="method", how="left")

# ============================
# Treat classes and bias_intensity as categorical (no bins!)
# ============================
df_long["classes_cat"] = df_long["classes"].astype(str)
df_long["bias_intensity_cat"] = df_long["bias_intensity"].astype(str)
# Force bias_intensity order for legend
bias_order = [6.6, 10.55, 20.0, 25.4, 29.6, 30.2]
df_long["bias_intensity_cat"] = pd.Categorical(
    df_long["bias_intensity"], categories=bias_order, ordered=True
)
df_compare = None  # filled later

# ============================
# PART A — PLOTS EXCLUDING bias_labels
# ============================
df_no_bias_labels = df_long[df_long["group"] != "bias_labels"]

# --- Figure 1: Gain vs Group (color = bias intensity categorical)
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_no_bias_labels,
    x="group",
    y="gain",
    hue="bias_intensity_cat",
    palette=UNIFIED_PALETTE,
)
leg = plt.legend(title="Bias Intensity")
leg.get_frame().set_facecolor("lightgrey")  # Set legend box color
leg.get_frame().set_edgecolor("black")  # Optional: add black border

plt.tight_layout()
plt.savefig("gain_bias_intensity_cat.pdf")
plt.show()

# --- Figure 2: Gain vs Group (color = num classes categorical)
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_no_bias_labels,
    x="group",
    y="gain",
    hue="classes_cat",
    palette=UNIFIED_PALETTE,
)
leg = plt.legend(title="#Classes")
leg.get_frame().set_facecolor("lightgrey")  # Set legend box color
leg.get_frame().set_edgecolor("black")  # Optional: add black border

plt.tight_layout()
plt.savefig("gain_num_classes_cat.pdf")
plt.show()

# ============================
# PART B — Bias_labels vs Not Bias_labels
# ============================
method_attr["any_other_group"] = (
    method_attr[group_cols].sum(axis=1) - method_attr["bias_labels"]
)

method_attr["bias_group_final"] = method_attr["bias_labels"].apply(
    lambda x: "bias_labels" if x == 1 else "not_bias_labels"
)

df_compare = df.merge(
    method_attr[["method", "bias_group_final"]], on="method", how="left"
)

# ===========================================
# FIGURE 3 — Performance vs Dataset (groups excluding bias_labels)
# ===========================================
df_no_bias_labels_plot = df_no_bias_labels.copy()

plt.figure(figsize=(14, 6))
sns.barplot(
    data=df_no_bias_labels_plot,
    x="dataset",
    y="mean",
    hue="group",
    palette=UNIFIED_PALETTE,
)
plt.xticks(rotation=30, ha="right")
leg = plt.legend(title="Group")
leg.get_frame().set_facecolor("lightgrey")  # Set legend box color
leg.get_frame().set_edgecolor("black")  # Optional: add black border

plt.tight_layout()
plt.savefig("perf_vs_dataset_groups.pdf")
plt.show()

# ===========================================
# FIGURE 4 — Performance vs Dataset (bias_labels vs not)
# ===========================================
plt.figure(figsize=(14, 6))
sns.barplot(
    data=df_compare,
    x="dataset",
    y="mean",
    hue="bias_group_final",
    palette=UNIFIED_PALETTE,
)
plt.xticks(rotation=30, ha="right")
leg = plt.legend(title="Group")
leg.get_frame().set_facecolor("lightgrey")  # Set legend box color
leg.get_frame().set_edgecolor("black")  # Optional: add black border

plt.tight_layout()
plt.savefig("perf_vs_dataset_bias_labels_vs_not.pdf")
plt.show()
