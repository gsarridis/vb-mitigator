import numpy as np
import pandas as pd
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from scipy import stats

# Activate automatic pandas conversion
pandas2ri.activate()

# Import R packages
stats = importr("stats")
pmcmr = importr("PMCMRplus")  # You may need to install this in R first

# Example data: rows = datasets, columns = methods
# Replace this with your actual performance rankings
methods = [
    "badd",
    "flac",
    "maviasb",
    "debian",
    "bb",
    "lff",
    "end",
    "di",
    "sd",
    "groupdro",
    "erm",
    "jtt",
    "bias_ensemble",
    "bpa",
    # "nsf",
    "george",
]
datasets = [
    "bias_in_bios",
    "CelebA",
    "chexpert_nih",
    "speech_accent_archive",
    "toxic",
    "ucf101",
    "UrbanCars",
    "urbansounds58",
    "Waterbirds",
]

all_rankings = []
for d in datasets:
    df = pd.read_csv(f"rankings/ranking_{d}.csv")
    if "dataset" in df.columns:
        df.drop("dataset", axis=1, inplace=True)
    df["dataset"] = d
    df["method"] = df["method"].str.lower()
    all_rankings.append(df)

combined = pd.concat(all_rankings, ignore_index=True)
dups = combined[combined.duplicated(subset=["dataset", "method"], keep=False)]
print(dups.sort_values(["dataset", "method"]))
rankings = combined.pivot(index="dataset", columns="method", values="rank")
rankings["mavias"][rankings["mavias"].isna()] = rankings["maviasb"]
rankings.drop("maviasb", axis=1, inplace=True)

print("Rankings (lower is better):")
print(rankings)

# datasets to keep
selected = [
    "bias_in_bios",
    "chexpert_nih",
    "speech_accent_archive",
    "toxic",
    "ucf101",
    "urbansounds58",
]

# filter rows
df_sel = rankings.loc[selected]

# compute mean and std for each method (i.e., each column)
mean_rank = df_sel.mean(axis=0)
std_rank = df_sel.std(axis=0)

# build final dataframe
out = pd.DataFrame(
    {
        "Method": mean_rank.index,
        "MeanRank": mean_rank.values,
        "StdRank": std_rank.values,
    }
)

# sort if you prefer (optional)
out = out.sort_values("MeanRank")

# save to CSV
out.to_csv("rankings/method_ranks_others.csv", index=False)


# datasets to keep
selected = ["CelebA", "Waterbirds", "UrbanCars"]

# filter rows
df_sel = rankings.loc[selected]

# compute mean and std for each method (i.e., each column)
mean_rank = df_sel.mean(axis=0)
std_rank = df_sel.std(axis=0)

# build final dataframe
out = pd.DataFrame(
    {
        "Method": mean_rank.index,
        "MeanRank": mean_rank.values,
        "StdRank": std_rank.values,
    }
)

# sort if you prefer (optional)
out = out.sort_values("MeanRank")

# save to CSV
out.to_csv("rankings/method_ranks_visual.csv", index=False)


# Convert to R dataframe
r_rankings = pandas2ri.py2rpy(rankings)


# (a) Friedman Test using R code directly
print("\n=== Friedman Test ===")
r_code_friedman = """
perform_friedman <- function(data) {
    # Convert to matrix if needed
    data_matrix <- as.matrix(data)
    result <- friedman.test(data_matrix)
    return(result)
}
"""
r(r_code_friedman)
friedman_result = r["perform_friedman"](r_rankings)
print(f"Chi-squared: {friedman_result[0][0]:.4f}")
print(f"p-value: {friedman_result[2][0]:.10f}")

if friedman_result[2][0] < 0.05:
    print("Significant differences detected among methods!")

    # (b) Nemenyi post-hoc test
    print("\n=== Nemenyi Post-hoc Test ===")

    r_code_nemenyi = """
    library(PMCMRplus)

    perform_nemenyi <- function(data) {
        # Convert wide to long format
        data_long <- stack(as.data.frame(data))
        colnames(data_long) <- c("rank", "method")
        data_long$dataset <- rep(1:nrow(data), ncol(data))
        # Uncomment if you want to check intermediate table
        # print(data_long)

        # Perform Nemenyi test
        result <- frdAllPairsNemenyiTest(
            y = data_long$rank,
            groups = data_long$method,
            blocks = data_long$dataset
        )
        return(result)
    }
    """
    r(r_code_nemenyi)
    nemenyi_result = r["perform_nemenyi"](r_rankings)
    print(nemenyi_result)

    # Extract p-value matrix and its dimension names from R
    r_code_extract = """
    extract_pvalues <- function(result) {
        mat <- result$p.value
        return(list(
            values = mat,
            rownames = rownames(mat),
            colnames = colnames(mat)
        ))
    }
    """
    r(r_code_extract)
    extracted = r["extract_pvalues"](nemenyi_result)

    pval_values = np.array(extracted[0])
    row_names = list(extracted[1])
    col_names = list(extracted[2])

    pval_df = pd.DataFrame(pval_values, index=row_names, columns=col_names)

    erm_pvals = {}

    # ERM's row: p-values vs methods that are columns (badd, bb, ..., end)
    if "erm" in pval_df.index:
        for method, p in pval_df.loc["erm"].items():
            if not np.isnan(p):
                erm_pvals[method] = p

    # ERM's column: p-values vs methods that come after ERM (flac, george, ...)
    if "erm" in pval_df.columns:
        for method, p in pval_df["erm"].items():
            if not np.isnan(p):
                erm_pvals[method] = p

    erm_pvals = pd.Series(erm_pvals).sort_values()
    print(f"Total ERM pairs found: {len(erm_pvals)}")  # should be 14
    print(erm_pvals)

    # Calculate average ranks
    avg_ranks = rankings.mean(axis=0).sort_values()
    print("\n=== Average Ranks (lower is better) ===")
    print(avg_ranks)
    print(f"\nBest method: {avg_ranks.index[0]} (avg rank: {avg_ranks.iloc[0]:.2f})")
else:
    print("No significant differences detected among methods.")
