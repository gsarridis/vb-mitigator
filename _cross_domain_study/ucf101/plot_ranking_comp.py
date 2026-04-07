import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_other = pd.read_csv("mean_method_ranking.csv")  # has 'mean_rank' and 'std_rank'
df_visual = pd.read_csv(
    "method_ranking_stats_visual.csv"
)  # has 'MeanRank_visual' and 'StdRank_visual'

# Rename columns for merge
df_other = df_other.rename(
    columns={
        "method": "Method",
        "mean_rank": "MeanRank_other",
        "std_rank": "StdRank_other",
    }
)

# Merge on Method
df = pd.merge(df_other, df_visual, on="Method")

plt.figure(figsize=(8, 6))

# Detect overlapping coordinates
coords = df.groupby(["MeanRank_visual", "MeanRank_other"])

for (x, y), group in coords:
    methods = group["Method"].tolist()
    n = len(methods)
    xerr = group["StdRank_visual"].values[0]  # x-axis std
    yerr = group["StdRank_other"].values[0]  # y-axis std

    # Draw circle with error bars
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color="purple", capsize=3)

    # Add stacked labels if multiple methods overlap
    for i, method in enumerate(methods):
        offset = (i - (n - 1) / 2) * 0.25
        plt.text(x + 0.15, y + offset, method, fontsize=9)

# Add diagonal line
lims = [
    min(df["MeanRank_visual"].min(), df["MeanRank_other"].min()) - 0.5,
    max(df["MeanRank_visual"].max(), df["MeanRank_other"].max()) + 0.5,
]
plt.plot(lims, lims, "k--", alpha=0.2, label="y = x")

plt.xlabel("Mean Rank (Visual Domain)")
plt.ylabel("Mean Rank (Other Domains)")
plt.title("Method Rankings: Visual vs Other Domains")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ranking_comparison_with_std_circles.png", dpi=300)
plt.show()
