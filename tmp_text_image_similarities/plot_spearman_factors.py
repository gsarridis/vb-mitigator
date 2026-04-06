import matplotlib.pyplot as plt
import numpy as np

# Data from the table
datasets = [
    "ImageNet\n(overall)",
    "CelebA WG\n(single-attr.)",
    "UrbanCars WG\n(multi-attr.)",
]
x = np.arange(len(datasets))

data = {
    "Parameters": [0.68, 0.48, 0.05],
    "Data size": [0.60, 0.53, 0.39],
    # "Tokens": [0.63, 0.45, 0.25],
    "Patch size": [-0.54, -0.36, -0.20],
    "Image size": [0.42, 0.32, 0.19],
}

# Create figure
plt.figure(figsize=(8, 5))

# Plot each factor
for label, values in data.items():
    plt.plot(x, values, marker="o", linewidth=2, label=label)

# Formatting
plt.xticks(x, datasets)
plt.ylabel("Spearman ρ")
plt.xlabel("Dataset (increasing bias complexity)")
plt.title("Factor–Performance Correlations Across Bias Complexity")

plt.axhline(0, linestyle="--", linewidth=1)  # zero line
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

# Save figure
plt.savefig("correlation_decay_plot.pdf")
plt.savefig("correlation_decay_plot.png", dpi=300)

plt.close()
