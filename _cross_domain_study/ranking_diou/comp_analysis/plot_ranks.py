import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("performances.csv")

# Exclude Vanilla
df_methods = df[df["method"] != "Vanilla"].copy()

# List of datasets and methods
datasets = df["dataset"].unique()
methods = df_methods["method"].unique()

# Compute ranks per dataset (1 = best)
ranks = pd.DataFrame(index=methods, columns=datasets, dtype=float)

for dataset in datasets:
    sub = df_methods[df_methods["dataset"] == dataset].copy()
    # Higher mean = better rank
    sub["rank"] = sub["mean"].rank(ascending=False, method="average")
    for _, row in sub.iterrows():
        ranks.loc[row["method"], dataset] = row["rank"]

# Compute mean and std of ranks per method
mean_ranks = ranks.mean(axis=1)
std_ranks = ranks.std(axis=1)

# Sort methods by mean rank (ascending, best at top)
sorted_indices = np.argsort(mean_ranks.values)
methods_sorted = mean_ranks.index[sorted_indices]
means_sorted = mean_ranks.values[sorted_indices]
stds_sorted = std_ranks.values[sorted_indices]

# Plot
plt.figure(figsize=(12, 8))
y_pos = np.arange(len(methods_sorted))
plt.barh(
    y_pos, means_sorted, xerr=stds_sorted, color="skyblue", ecolor="black", capsize=5
)
plt.yticks(y_pos, methods_sorted, fontsize=14)
plt.xlabel("Mean rank across datasets", fontsize=14)
plt.title("Method Rankings Across Datasets", fontsize=16)
plt.gca().invert_yaxis()  # Best rank at top
plt.tight_layout()
plt.savefig("ranks.pdf")

plt.show()
