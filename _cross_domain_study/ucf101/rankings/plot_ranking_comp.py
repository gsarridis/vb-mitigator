import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_other = pd.read_csv("mean_method_ranking.csv")
df_visual = pd.read_csv("method_ranking_stats_visual.csv")
df_other = df_other.rename(columns={"method": "Method"})

# Merge on Method
df = pd.merge(df_other, df_visual, on="Method", suffixes=("_other", "_visual"))

plt.figure(figsize=(8, 6))

# Detect overlapping coordinates
coords = df.groupby(["MeanRank_visual", "MeanRank_other"])

for (x, y), group in coords:
    methods = group["Method"].tolist()
    n = len(methods)
    if n == 1:
        plt.scatter(x, y, s=60, marker="o", color="purple")
        plt.text(x + 0.15, y, methods[0], fontsize=9)
    else:
        # Draw a single point
        plt.scatter(x, y, s=60, marker="o", color="purple")
        # Stack labels vertically
        for i, method in enumerate(methods):
            offset = (i - (n - 1) / 2) * 0.25  # spacing
            plt.text(x + 0.15, y, method, fontsize=9)

# Add diagonal line
lims = [
    min(df["MeanRank_visual"].min(), df["MeanRank_other"].min()) - 0.5,
    max(df["MeanRank_visual"].max(), df["MeanRank_other"].max()) + 0.5,
]
plt.plot(lims, lims, "k--", alpha=0.2, label="y = x")

plt.xlabel("Mean Rank (Images)")
plt.ylabel("Mean Rank (Other Domains)")
# plt.title("Method Rankings: Visual vs Other Domains")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(
    "ranking_comparison.svg", bbox_inches="tight"
)  # SVG format, best for Inkscape

plt.savefig("ranking_comparison.png", dpi=300)
plt.show()
