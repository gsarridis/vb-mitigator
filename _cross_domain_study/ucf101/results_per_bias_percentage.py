import argparse
import json
import os
import pandas as pd
import numpy as np


def main(method_name, seed, json_file):
    # === Load bias percentages ===
    with open(json_file, "r") as f:
        data = json.load(f)
    bias_per_label = data.get("bias_per_label", {})
    if not bias_per_label:
        raise ValueError("No bias_per_label found in the provided JSON")

    # === Construct CSV path (template, adjust as needed) ===
    csv_path = (
        f"./output/ucf101_baselines/dev_in_out/{method_name}/test_full_results_{seed}"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV at {csv_path}")

    # === Load full results (class × bias-groups) ===
    df = pd.read_csv(csv_path, index_col=0)
    df = df.dropna(axis=0, how="any")

    # Ensure class names (index) match the bias_per_label keys
    df.index = df.index.astype(str)

    # === Evaluate across thresholds ===
    thresholds = range(0, 50, 5)
    results = []

    for t in thresholds:
        # Select classes with bias ≤ threshold
        selected_classes = [lbl for lbl, bias in bias_per_label.items() if bias >= t]
        if not selected_classes:
            results.append((t, np.nan, np.nan))
            continue

        # Subset dataframe
        sub_df = df.loc[df.index.intersection(selected_classes)]

        # Flatten accuracies into list (ignore NaNs)
        vals = sub_df.values.flatten()
        vals = vals[~np.isnan(vals)]

        if len(vals) == 0:
            wga, avg_acc = np.nan, np.nan
        else:
            wga = np.min(vals)
            avg_acc = np.mean(vals)

        results.append((t, wga, avg_acc))

    # === Print results ===
    print("Threshold | Worst Group Acc | Avg Group Acc")
    for t, wga, avg_acc in results:
        print(
            f"{t:3d}        | {wga:.3f}           | {avg_acc:.3f}"
            if not np.isnan(wga)
            else f"{t:3d}        | NaN             | NaN"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate group accuracies vs bias thresholds"
    )
    parser.add_argument("--method", type=str, required=True, help="Method name")
    parser.add_argument("--seed", type=int, required=True, help="Seed")
    parser.add_argument(
        "--json",
        type=str,
        default="/mnt/cephfs/home/common/datasets/UCF101/ucf101_01.json",
        help="Path to JSON file with bias_per_label",
    )
    args = parser.parse_args()

    main(args.method, args.seed, args.json)
