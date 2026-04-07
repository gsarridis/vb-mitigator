import pandas as pd

# Input/output files
input_file = "metrics_summary_ucf.csv"
output_file = "ranking_ucf101.csv"


# Load data
df = pd.read_csv(input_file)

# Filter for threshold = 20
df_thr = df[df["threshold"] == 20]

# Compute mean worst_group_acc per method across seeds
mean_wg = df_thr.groupby("method")["worst_group_acc"].mean().reset_index()
mean_wg = mean_wg.sort_values(by="worst_group_acc", ascending=False).reset_index(
    drop=True
)

# Add rank column
mean_wg["rank"] = (
    mean_wg["worst_group_acc"].rank(method="min", ascending=False).astype(int)
)

# Save ranking
mean_wg.to_csv(output_file, index=False)
ranking_ucf = mean_wg
print(f"Saved ranking to {output_file}")


# Input/output files
input_file = "full_results_bias_in_bios.csv"
output_file = "ranking_bias_in_bios.csv"

# Load data
df = pd.read_csv(input_file)

# Compute mean worst-group accuracy per method across seeds
mean_wg = df.groupby("Method")["Test Worst Group Accuracy"].mean().reset_index()
mean_wg = mean_wg.rename(columns={"Method": "method"})

# Sort descending
mean_wg = mean_wg.sort_values(
    by="Test Worst Group Accuracy", ascending=False
).reset_index(drop=True)

# Add rank column
mean_wg["rank"] = (
    mean_wg["Test Worst Group Accuracy"].rank(method="min", ascending=False).astype(int)
)

# Save ranking
mean_wg.to_csv(output_file, index=False)
ranking_bias_in_bios = mean_wg

print(f"Saved ranking to {output_file}")


# Input/output files
input_file = "full_results_speech_accent_archive.csv"
output_file = "ranking_speech_accent_archive.csv"

# Load data
df = pd.read_csv(input_file)

# Compute mean worst-group accuracy per method across seeds
mean_wg = df.groupby("Method")["Test Worst Group Accuracy"].mean().reset_index()
mean_wg = mean_wg.rename(columns={"Method": "method"})

# Sort descending
mean_wg = mean_wg.sort_values(
    by="Test Worst Group Accuracy", ascending=False
).reset_index(drop=True)

# Add rank column
mean_wg["rank"] = (
    mean_wg["Test Worst Group Accuracy"].rank(method="min", ascending=False).astype(int)
)

# Save ranking
mean_wg.to_csv(output_file, index=False)
ranking_speech = mean_wg

print(f"Saved ranking to {output_file}")


# Input/output files
input_file = "full_results_urbansounds58.csv"
output_file = "ranking_urbansounds58.csv"

# Load data
df = pd.read_csv(input_file)

# Compute mean worst-group accuracy per method across seeds
mean_wg = df.groupby("Method")["Test Worst Group Accuracy"].mean().reset_index()

# Sort descending
mean_wg = mean_wg.sort_values(
    by="Test Worst Group Accuracy", ascending=False
).reset_index(drop=True)
mean_wg = mean_wg.rename(columns={"Method": "method"})

# Add rank column
mean_wg["rank"] = (
    mean_wg["Test Worst Group Accuracy"].rank(method="min", ascending=False).astype(int)
)

# Save ranking
mean_wg.to_csv(output_file, index=False)
ranking_urbansounds58 = mean_wg

print(f"Saved ranking to {output_file}")


# Input/output files
input_file = "full_results_toxic.csv"
output_file = "ranking_toxic.csv"

# Load data
df = pd.read_csv(input_file)

# Compute mean worst-group accuracy per method across seeds
mean_wg = df.groupby("Method")["Accuracy Generated Set"].mean().reset_index()
mean_wg = mean_wg.rename(columns={"Method": "method"})

# Sort descending
mean_wg = mean_wg.sort_values(by="Accuracy Generated Set", ascending=False).reset_index(
    drop=True
)

# Add rank column
mean_wg["rank"] = (
    mean_wg["Accuracy Generated Set"].rank(method="min", ascending=False).astype(int)
)

# Save ranking
mean_wg.to_csv(output_file, index=False)
ranking_toxic = mean_wg

print(f"Saved ranking to {output_file}")


# Input/output files
input_file = "full_results_chexpert_nih.csv"
output_file = "ranking_chexpert_nih.csv"

# Load data
df = pd.read_csv(input_file)

# Compute mean worst-group accuracy per method across seeds
mean_wg = df.groupby("Method")["Test Worst Group Accuracy"].mean().reset_index()
mean_wg = mean_wg.rename(columns={"Method": "method"})

# Sort descending
mean_wg = mean_wg.sort_values(
    by="Test Worst Group Accuracy", ascending=False
).reset_index(drop=True)

# Add rank column
mean_wg["rank"] = (
    mean_wg["Test Worst Group Accuracy"].rank(method="min", ascending=False).astype(int)
)

# Save ranking
mean_wg.to_csv(output_file, index=False)
ranking_toxic = mean_wg

print(f"Saved ranking to {output_file}")


# Suppose your 5 ranking CSVs are:
files = [
    "ranking_ucf101.csv",
    "ranking_bias_in_bios.csv",
    "ranking_speech_accent_archive.csv",
    "ranking_urbansounds58.csv",
    "ranking_toxic.csv",
    "ranking_chexpert_nih.csv",
]


# Load all ranking DataFrames
dfs = [pd.read_csv(f) for f in files]

# Keep only method and rank columns, rename rank to be unique per file
for i, df in enumerate(dfs):
    df.rename(columns={"rank": f"rank_{i+1}"}, inplace=True)
    dfs[i] = df[["method", f"rank_{i+1}"]]

# Merge all rankings on 'method'
merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="method")

# Compute mean rank
rank_cols = [col for col in merged.columns if col.startswith("rank_")]
merged["MeanRank"] = merged[rank_cols].mean(axis=1)
merged["StdRank"] = merged[rank_cols].std(axis=1)

# Sort by mean rank (ascending = best)
merged = merged.sort_values("MeanRank").reset_index(drop=True)

# Save to CSV
merged.to_csv("mean_method_ranking.csv", index=False)

print("Saved mean method ranking to mean_method_ranking.csv")
