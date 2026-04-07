import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
from statsmodels.stats import multitest
from scipy.stats import rankdata, chi2, friedmanchisquare, wilcoxon


def aligned_rank_posthoc(aligned, methods, correction="holm"):
    n, k = aligned.shape
    pairs, raw_pvals = [], []

    # Rank globally again (on aligned data)
    aligned_ranks = rankdata(aligned.ravel()).reshape(n, k)

    for i in range(k):
        for j in range(i + 1, k):
            diff = aligned_ranks[:, i] - aligned_ranks[:, j]
            stat, p = wilcoxon(diff)
            pairs.append((methods[i], methods[j]))
            raw_pvals.append(p)

    reject, pvals_corr, _, _ = multitest.multipletests(
        raw_pvals, alpha=0.05, method=correction
    )

    results = []
    for (m1, m2), p_raw, p_corr, rej in zip(pairs, raw_pvals, pvals_corr, reject):
        results.append(
            {
                "Method1": m1,
                "Method2": m2,
                "Raw_p": p_raw,
                "Corrected_p": p_corr,
                "Significant": rej,
            }
        )
    return pd.DataFrame(results)


def friedman_aligned_rank_test(matrix):
    n, k = matrix.shape

    # Align: subtract row means and add grand mean
    row_means = matrix.mean(axis=1, keepdims=True)
    col_means = matrix.mean(axis=0, keepdims=True)
    grand_mean = matrix.mean()

    aligned = matrix - row_means - col_means + grand_mean

    # Rank all aligned values globally
    ranks = rankdata(aligned.ravel())
    ranks = ranks.reshape(n, k)

    # Column rank sums
    Rj = ranks.sum(axis=0)

    # Test statistic
    T = (12 / (n * k * (k + 1))) * np.sum((Rj - (n * (k + 1) / 2)) ** 2)
    pval = 1 - chi2.cdf(T, k - 1)

    return T, pval, Rj / n, aligned


# Map datasets to the column used for ranking
METRIC_COLS = {
    "bias_in_bios": "Test Worst Group Accuracy",
    "speech_accent_archive": "Test Worst Group Accuracy",
    "urbansounds58": "Test Worst Group Accuracy",
    "toxic": "Accuracy Generated Set",
}

# List of CSV files (adjust paths if needed)
csv_files = {
    "bias_in_bios": "full_results_bias_in_bios.csv",
    "speech_accent_archive": "full_results_speech_accent_archive.csv",
    "urbansounds58": "full_results_urbansounds58.csv",
    "toxic": "full_results_toxic.csv",
}

# Collect mean metric per dataset-method
# results = []
# for dataset, file in csv_files.items():
#     df = pd.read_csv(file)
#     df["Dataset"] = df["Dataset"] + "_" + df["Seed"]
#     metric_col = METRIC_COLS[dataset]

#     # group by method
#     grouped = df[["Method", metric_col, "Dataset"]].copy()
#     grouped = grouped.rename(columns={metric_col: "Metric"})
#     print(grouped)

#     results.append(grouped)

# Collect mean metric per dataset-method
results = []
for dataset, file in csv_files.items():
    df = pd.read_csv(file)
    metric_col = METRIC_COLS[dataset]

    # group by method
    grouped = df.groupby("Method")[metric_col].mean().reset_index()
    grouped = grouped.rename(columns={metric_col: "Metric"})
    grouped["Dataset"] = dataset

    results.append(grouped)

all_results = pd.concat(results, ignore_index=True)
# Pivot into matrix: rows = datasets, cols = methods, values = metric
matrix = all_results.pivot(index="Dataset", columns="Method", values="Metric")
print(matrix)

# Drop methods missing for some datasets
# matrix = matrix.dropna(axis=1, how="any")
print("\nMatrix:\n", matrix)


methods = matrix.columns.tolist()
scores = matrix.to_numpy()  # shape N x k
print("Scores shape:", scores.shape)
print("Scores array:", scores)
# Friedman test
stat, pval = stats.friedmanchisquare(*[scores[:, j] for j in range(scores.shape[1])])
print(f"Friedman chi2 = {stat:.3f}, p = {pval:.4f}")

# Compute ranks per dataset
ranks = np.array(
    [stats.rankdata(-row, method="average") for row in scores]
)  # minus => higher is better

# Mean and std of ranks
mean_ranks = ranks.mean(axis=0)
std_ranks = ranks.std(axis=0)

# Save to CSV
rank_df = pd.DataFrame(
    {"Method": methods, "MeanRank": mean_ranks, "StdRank": std_ranks}
).sort_values("MeanRank")

rank_df.to_csv("method_ranking_stats.csv", index=False)

print("\nMean and std of ranks across datasets:")
print(rank_df)


# Nemenyi post-hoc
# scores = scores[:, -4:]

nemenyi_p = sp.posthoc_nemenyi_friedman(scores).to_numpy()
print(nemenyi_p)
print("Nemenyi p-values:")
nemenyi_df = pd.DataFrame(nemenyi_p, index=methods, columns=methods)
print(nemenyi_df)
# Optional: Pairwise Wilcoxon with Holm correction
pairs, raw_pvals = [], []
for i in range(scores.shape[1]):
    for j in range(i + 1, scores.shape[1]):
        stat, p = stats.wilcoxon(scores[:, i], scores[:, j])
        pairs.append((methods[i], methods[j]))
        raw_pvals.append(p)

reject, pvals_corr, _, _ = multitest.multipletests(raw_pvals, alpha=0.05, method="holm")
for (m1, m2), p_raw, p_corr, rej in zip(pairs, raw_pvals, pvals_corr, reject):
    print(
        f"{m1} vs {m2}: raw p={p_raw:.4f}, holm-corr p={p_corr:.4f}, significant={rej}"
    )


# 2. Aligned Friedman test
T, pval_aligned, mean_aligned_ranks, aligned = friedman_aligned_rank_test(scores)
print(f"\nFriedman Aligned Rank test: chi2 = {T:.3f}, p = {pval_aligned:.4f}")
print("Mean aligned ranks:", mean_aligned_ranks)


# # 4. Pairwise Aligned Rank Post-hoc
# aligned_df = aligned_rank_posthoc(aligned, methods, correction="fdr_bh")
# print("\nPairwise Aligned Rank post-hoc with Holm correction:")
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)

# print(aligned_df)

# Create a DataFrame for plotting
rank_df = pd.DataFrame({"Method": methods, "MeanAlignedRank": mean_aligned_ranks})

# Sort methods by rank for better visualization
rank_df = rank_df.sort_values("MeanAlignedRank", ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="MeanAlignedRank", y="Method", data=rank_df, palette="viridis")
plt.xlabel("Mean Aligned Rank")
plt.ylabel("Method")
plt.title("Mean Aligned Ranks per Method (Friedman Aligned Rank Test)")
plt.xlim(rank_df["MeanAlignedRank"].min() - 1, rank_df["MeanAlignedRank"].max() + 1)
plt.tight_layout()
plt.savefig("mean_aligned_ranks.png")
plt.show()
