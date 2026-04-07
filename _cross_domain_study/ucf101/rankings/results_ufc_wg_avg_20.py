import pandas as pd

# Input/output files
input_file = "metrics_summary_ucf.csv"
output_file = "ranking_ucf101_wg_avg.csv"

# Load data
df = pd.read_csv(input_file)

# Filter for threshold = 20
df_thr = df[df["threshold"] == 20]

# Compute mean and std for both worst_group_acc and avg_acc per method
stats = (
    df_thr.groupby("method")[["worst_group_acc", "avg_acc"]]
    .agg(["mean", "std"])
    .reset_index()
)

# Flatten MultiIndex columns
stats.columns = [
    "method",
    "worst_group_acc_mean",
    "worst_group_acc_std",
    "avg_acc_mean",
    "avg_acc_std",
]

# Sort by mean worst_group_acc descending
stats = stats.sort_values(by="worst_group_acc_mean", ascending=False).reset_index(
    drop=True
)

# Add rank based on worst_group_acc_mean
stats["rank"] = (
    stats["worst_group_acc_mean"].rank(method="min", ascending=False).astype(int)
)

# Save results
stats.to_csv(output_file, index=False)

ranking_ucf = stats
print(f"Saved ranking (mean ± std for worst_group_acc and avg_acc) to {output_file}")
