import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fs = 30
for i in ["visual", "other", "all"]:
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
        "nsf",
        "bpa",
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
        df = pd.read_csv(f"./rankings/ranking_{d}.csv")
        if "dataset" in df.columns:
            df.drop("dataset", axis=1, inplace=True)
        df["dataset"] = d
        df["method"] = df["method"].str.lower()
        all_rankings.append(df)

    combined = pd.concat(all_rankings, ignore_index=True)
    rankings = combined.pivot(index="dataset", columns="method", values="rank")
    rankings["mavias"][rankings["mavias"].isna()] = rankings["maviasb"]
    rankings.drop("maviasb", axis=1, inplace=True)

    print("Rankings (lower is better):")
    # print(rankings)
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
        "bias_ensemble": "BE",
        "nsf": "NSF",
        "bpa": "BPA",
        "george": "GEORGE",
    }
    # Keep only existing columns
    rename_map = {c: method_names[c] for c in rankings.columns if c in method_names}

    rankings = rankings.rename(columns=rename_map)

    # Drop Vanilla if it exists
    # rankings = rankings.drop(columns=["Vanilla"], errors="ignore")
    # print(rankings)

    # datasets to keep

    if i == "visual":
        selected = [
            "Waterbirds",
            "UrbanCars",
            "CelebA",
        ]
        df_sel = rankings.loc[selected]
    elif i == "other":
        selected = [
            "bias_in_bios",
            "chexpert_nih",
            "speech_accent_archive",
            "toxic",
            "ucf101",
            "urbansounds58",
        ]
        df_sel = rankings.loc[selected]
    else:
        df_sel = rankings
    # filter rows

    # print(df_sel)

    # Ensure we only keep the methods in your list (some may be missing)
    # df_sel = df_sel[[m for m in methods if m in df_sel.columns]]
    print(df_sel)

    # Bootstrapping
    n_bootstrap = 10000
    mean_ranks = {}
    std_ranks = {}

    for method in df_sel.columns:
        # resample datasets with replacement
        boot_samples = np.random.choice(
            df_sel.loc[:, method], size=(n_bootstrap, len(df_sel)), replace=True
        )
        boot_means = boot_samples.mean(axis=1)
        mean_ranks[method] = boot_means.mean()
        std_ranks[method] = boot_means.std()

    # Sort methods by mean rank
    methods_sorted = sorted(mean_ranks, key=lambda x: mean_ranks[x])
    means_sorted = [mean_ranks[m] for m in methods_sorted]
    stds_sorted = [std_ranks[m] for m in methods_sorted]

    # Plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(methods_sorted))
    # plt.grid(axis="x", linestyle="--", alpha=0.6)

    plt.barh(y_pos, means_sorted, xerr=stds_sorted, ecolor="black", capsize=5)
    plt.yticks(y_pos, methods_sorted, fontsize=fs)
    plt.xlabel("rank", fontsize=fs)
    plt.xticks(fontsize=fs)

    plt.gca().invert_yaxis()  # Best rank at top
    plt.tight_layout()
    plt.savefig(f"ranks_bootstarp_{i}.pdf")
    plt.show()
