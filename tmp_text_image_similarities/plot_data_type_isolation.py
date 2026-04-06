"""
Generate fig_data_isolation_dotplot.pdf

Datasets on x-axis, sorted by UrbanCars WG mean (descending).
Three lines: ImageNet top-1 (green), CelebA WG (blue), UrbanCars WG (orange).
Small semi-transparent dots = individual architecture config observations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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

# (imagenet, uc_wg, ca_wg) per matched architecture config
dataset_obs = {
    "DataComp-12.8b": [(69.2, 72.0, 83.3)],
    "DFNDR-2b": [(76.3, 70.4, 73.3)],
    "DFN-5b": [(83.1, 73.6, 86.6), (83.7, 80.4, 85.8)],
    "DFN-2b": [(75.5, 65.6, 66.7), (81.3, 61.9, 75.6)],
    "LAION-2b": [
        (66.1, 66.0, 80.3),
        (70.2, 67.2, 81.1),
        (75.2, 72.8, 83.3),
        (77.9, 48.8, 84.4),
        (80.1, 78.4, 87.6),
        (79.1, 70.4, 87.4),
    ],
    "DataComp-1b": [(68.3, 61.2, 75.6), (79.2, 64.8, 80.6), (81.8, 78.4, 84.2)],
    "CommonPool-1b": [(53.96, 40.7, 44.4), (55.7, 62.0, 52.7), (57.8, 60.8, 82.1)],
    "MetaCLIP2-2.5b": [(73.1, 46.1, 83.0), (79.9, 52.0, 86.1), (82.3, 76.8, 84.4)],
    "CommonCrawl-2.5b": [
        (66.2, 69.2, 76.4),
        (70.4, 70.8, 69.4),
        (79.5, 26.8, 79.2),
        (80.9, 55.6, 78.9),
    ],
    "LAION-400m": [(61.6, 52.0, 81.9), (67.0, 70.4, 80.3), (72.7, 54.4, 72.5)],
    "MetaCLIP-400m": [(63.6, 54.4, 70.8), (69.0, 71.2, 70.3)],
    "OpenAI-400m": [(61.5, 29.6, 75.8), (66.4, 51.2, 72.2), (74.2, 45.6, 80.6)],
    "DataComp-128m": [(29.7, 43.2, 48.3)],
    "CommonPool-13m": [(3.84, 21.2, 40.6)],
    "DataComp-13m": [(3.91, 15.2, 32.7)],
}

# Sort by UC mean descending
sorted_datasets = sorted(
    dataset_obs, key=lambda d: np.mean([o[1] for o in dataset_obs[d]]), reverse=True
)
n = len(sorted_datasets)
x_pos = np.arange(n)

IN_COLOR = "#2ca02c"  # green
UC_COLOR = "#E07B39"  # orange
CA_COLOR = "#4C8EC4"  # blue
RNG = np.random.default_rng(42)

fig, ax = plt.subplots(figsize=(14, 5.4))
in_means, uc_means, ca_means = [], [], []

for xi, ds in enumerate(sorted_datasets):
    obs = dataset_obs[ds]
    in_vals = [o[0] for o in obs]
    uc_vals = [o[1] for o in obs]
    ca_vals = [o[2] for o in obs]
    jitter = RNG.uniform(-0.18, 0.18, size=len(obs))

    ax.scatter(
        xi + jitter,
        in_vals,
        color=IN_COLOR,
        alpha=0.28,
        s=36,
        zorder=2,
        edgecolors="none",
    )
    ax.scatter(
        xi + jitter,
        uc_vals,
        color=UC_COLOR,
        alpha=0.28,
        s=36,
        zorder=2,
        edgecolors="none",
    )
    ax.scatter(
        xi + jitter,
        ca_vals,
        color=CA_COLOR,
        alpha=0.28,
        s=36,
        zorder=2,
        edgecolors="none",
    )

    in_m = np.mean(in_vals)
    uc_m = np.mean(uc_vals)
    ca_m = np.mean(ca_vals)

    ax.scatter(
        xi,
        in_m,
        color=IN_COLOR,
        s=80,
        zorder=4,
        edgecolors="white",
        linewidths=0.8,
        marker="s",
    )
    ax.scatter(
        xi,
        uc_m,
        color=UC_COLOR,
        s=80,
        zorder=4,
        edgecolors="white",
        linewidths=0.8,
        marker="o",
    )
    ax.scatter(
        xi,
        ca_m,
        color=CA_COLOR,
        s=80,
        zorder=4,
        edgecolors="white",
        linewidths=0.8,
        marker="D",
    )

    in_means.append(in_m)
    uc_means.append(uc_m)
    ca_means.append(ca_m)

ax.plot(x_pos, in_means, color=IN_COLOR, linewidth=2.0, zorder=3, linestyle="dotted")
ax.plot(x_pos, uc_means, color=UC_COLOR, linewidth=2.2, zorder=3)
ax.plot(x_pos, ca_means, color=CA_COLOR, linewidth=2.2, zorder=3, linestyle="--")

ax.set_xticks(x_pos)
ax.set_xticklabels(sorted_datasets, rotation=35, ha="right")
ax.set_ylabel("Metric (%)")
ax.set_ylim(0, 100)
ax.set_xlim(-0.6, n - 0.4)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title(
    "Isolated training data effect — sorted by UrbanCars WG mean\n"
    "(dots = per architecture config; square/dotted = ImageNet mean; "
    "circle/solid = UC WG mean; diamond/dashed = CA WG mean)",
    pad=8,
)

patch_in = mpatches.Patch(color=IN_COLOR, label="ImageNet top-1 acc mean")
patch_uc = mpatches.Patch(color=UC_COLOR, label="UrbanCars WGA mean")
patch_ca = mpatches.Patch(color=CA_COLOR, label="CelebA WGA mean")
dot_any = plt.scatter(
    [], [], color="gray", alpha=0.4, s=36, label="Individual observation"
)
ax.legend(
    handles=[patch_in, patch_ca, patch_uc, dot_any],
    loc="lower left",
    frameon=True,
)

plt.tight_layout()
plt.savefig("figures/fig_data_isolation_dotplot.pdf", bbox_inches="tight", dpi=300)
plt.savefig("figures/fig_data_isolation_dotplot.png", bbox_inches="tight", dpi=200)
print("Saved.")
