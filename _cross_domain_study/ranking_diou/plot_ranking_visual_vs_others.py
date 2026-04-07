import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_other = pd.read_csv("method_ranks_others.csv")
df_visual = pd.read_csv("method_ranks_visual.csv")

# Rename columns for merge
df_other = df_other.rename(
    columns={
        "method": "Method",
        "MeanRank": "MeanRank_other",
        "StdRank": "StdRank_other",
    }
)

df_visual = df_visual.rename(
    columns={
        "method": "Method",
        "MeanRank": "MeanRank_visual",
        "StdRank": "StdRank_visual",
    }
)

# Merge on Method
df = pd.merge(df_other, df_visual, on="Method")

method_names = {
    "mavias": "MAVias",
    "badd": "BAdd",
    "flac": "FLAC",
    "di": "DI",
    "lff": "LfF",
    "bb": "BB",
    "end": "EnD",
    "debian": "Debian",
    "sd": "SD",
    "groupdro": "GroupDro",
    "erm": "Vanilla",
    "jtt": "JTT",
}
df["Method"] = df["Method"].map(method_names)
df = df[df["Method"] != "Vanilla"]


plt.figure(figsize=(8, 6))

# Determine axis limits
xmin = min(df["MeanRank_visual"].min(), df["MeanRank_other"].min()) - 0.5
xmax = max(df["MeanRank_visual"].max(), df["MeanRank_other"].max()) + 0.5
x_vals = np.linspace(xmin, xmax, 500)

# === Add ±1 diagonal band (shaded region) ===
plt.fill_between(
    x_vals, x_vals - 2, x_vals + 2, color="grey", alpha=0.15, label="±1 range from y=x"
)

# Group overlapping coordinates
coords = df.groupby(["MeanRank_visual", "MeanRank_other"])

for (x, y), group in coords:
    methods = group["Method"].tolist()
    n = len(methods)

    # Plot point with error bars
    plt.errorbar(x, y, fmt="o", color="purple", capsize=5)

    # # Add stacked method names
    # for i, method in enumerate(methods):
    #     offset = (i - (n - 1) / 2) * 0.25
    #     plt.text(x + 0.15, y + offset, method, fontsize=20)
    for i, method in enumerate(methods):
        if method == "EnD":
            # Force EnD label to appear below the scatter point
            plt.text(x + 0.85, y + 0.15, method, fontsize=20, va="top")
        elif method == "LfF":
            # Force EnD label to appear below the scatter point
            plt.text(x + 0.15, y + 0.15, method, fontsize=20, va="top")
        elif method == "MAVias":
            # Force EnD label to appear below the scatter point
            plt.text(x + 1.6, y - 0.15, method, fontsize=20, va="top")
        else:
            offset = - (i - (n - 1) / 2) * 0.25
            plt.text(x - 0.15, y + offset, method, fontsize=20)


# Add diagonal y = x line
plt.plot(x_vals, x_vals, "k--", alpha=0.25)

plt.xlim(xmin, xmax)
plt.ylim(xmin, xmax)

# Reverse axes (high → low)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()


plt.xlabel("Mean Rank (Natural Images)", fontsize=19)
plt.ylabel("Mean Rank (Other Domains/Modalities)", fontsize=19)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ranking_comparison.pdf")
plt.show()
