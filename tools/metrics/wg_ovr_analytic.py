import numpy as np
from itertools import product
from collections import defaultdict
import pandas as pd

wg_ovr_analytic_dict = {"best": "high", "performance": "overall"}


def wg_ovr_analytic(data):
    groups = defaultdict(lambda: {"correct": 0, "total": 0})
    targets = data["targets"]
    predictions = data["predictions"]
    sensitive_attrs = [
        data[key] for key in data if key not in {"targets", "predictions", "ba_groups"}
    ]

    for i in range(len(targets)):
        group_key = (targets[i],) + tuple(attr[i] for attr in sensitive_attrs)
        groups[group_key]["total"] += 1
        if targets[i] == predictions[i]:
            groups[group_key]["correct"] += 1

    accuracies = {
        key: val["correct"] / val["total"]
        for key, val in groups.items()
        if val["total"] > 0
    }
    # Build DataFrame: rows = classes, cols = biases
    classes = sorted(set(targets))
    bias_groups = sorted(
        set(tuple(attr[i] for attr in sensitive_attrs) for i in range(len(targets)))
    )

    df_data = {}
    for bias in bias_groups:
        col_vals = []
        for cls in classes:
            key = (cls,) + bias
            col_vals.append(accuracies.get(key, np.nan))
        df_data[bias] = col_vals

    acc_df = pd.DataFrame(df_data, index=classes)
    worst_group_acc = min(accuracies.values(), default=None)
    avg_group_acc = sum(accuracies.values()) / len(accuracies) if accuracies else None

    out = {
        "worst_group_accuracy": round(worst_group_acc, 3),
        "overall": round(avg_group_acc, 3),
        "full_results": acc_df,
    }
    return out


if __name__ == "__main__":

    data_dict = {
        "predictions": np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1]),
        "targets": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        "background": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0]),
        # "object": np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
    }
    out = wg_ovr_analytic(data_dict)
    print(out)
    # df = out["full_results"]
    # target2name = {
    #     0: "ApplyEyeMakeup",
    #     1: "ApplyLipstick",
    # }

    # # Replace row names
    # df.rename(index=target2name, inplace=True)

    # print("\nAfter renaming:")
    # print(df)
