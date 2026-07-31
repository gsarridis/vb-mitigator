import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("performances.csv")


# Vanilla baseline
vanilla = df[df["method"] == "Vanilla"][["dataset", "mean"]].set_index("dataset")[
    "mean"
]

methods = sorted(df["method"].unique())
methods = [m for m in methods if m != "Vanilla"]  # exclude Vanilla

threshold = 5.0  # ignore differences smaller than ±1

better_counts = []
worse_counts = []

for method in methods:
    sub = df[df["method"] == method][["dataset", "mean"]].set_index("dataset")["mean"]
    compared = sub - vanilla
    # Apply threshold
    better_counts.append((compared > threshold).sum())
    worse_counts.append((compared < -threshold).sum())

# Compute net score and sort methods by ascending net score
net_scores = [w - b for b, w in zip(better_counts, worse_counts)]
sorted_indices = sorted(
    range(len(methods)),
    key=lambda i: (net_scores[i], -better_counts[i]),
)
methods = [methods[i] for i in sorted_indices]
better_counts = [better_counts[i] for i in sorted_indices]
worse_counts = [worse_counts[i] for i in sorted_indices]

# Plot
plt.figure(figsize=(10, 5))
y_pos = range(len(methods))

# Red bars (left)
plt.barh(
    y_pos,
    [-x for x in worse_counts],
    color="red",
    alpha=0.7,
    label=">5% performance drop",
)

# Green bars (right)
plt.barh(y_pos, better_counts, color="green", alpha=0.7, label=">5% performance gain")

# Vertical center line
plt.axvline(0, linestyle="--", color="black")

plt.yticks(y_pos, methods, fontsize=14)
plt.xlabel("#Datasets", fontsize=14)
plt.legend(fontsize=14)

# Set x-axis limits separately
x_left = max(worse_counts) + 1
x_right = max(better_counts) + 1
plt.xlim(-x_left, x_right)

# Optional: x-axis absolute values
plt.xticks(
    range(-x_left, x_right + 1),
    [abs(i) for i in range(-x_left, x_right + 1)],
    fontsize=14,
)

plt.tight_layout()
plt.savefig("plot_binary_gains_drops_05.pdf")
plt.show()
