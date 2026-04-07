import os
import csv
import tensorflow as tf
from pathlib import Path
import pandas as pd

# Paths
target_base_dir = Path(
    "/mnt/cephfs/home/gsarridis/projects/vb-mitigator/output"
)  # Update this with your actual logs directory
csv_output_path = "full_results_bias_in_bios.csv"

# Define methods and datasets
methods = [
    "erm",
    "debian",
    "jtt",
    "lff",
    "sd",
    "di",
    "badd",
    "flac",
    "bb",
    "end",
    "groupdro",
    "maviasb",
]
datasets = ["bias_in_bios"]
# CSV Header
csv_data = [
    [
        "Dataset",
        "Method",
        "Seed",
        "Best Epoch",
        "Test Overall Accuracy",
        "Test Worst Group Accuracy",
    ]
]


# def extract_best_metrics(event_file):
#     """Extracts the epoch with the best 'test_overall_accuracy' and its corresponding 'test_worst_group_accuracy'."""
#     best_epoch = 0
#     best_overall_acc = 0.0
#     best_worst_group_acc = 0.0

#     try:
#         for event in tf.compat.v1.train.summary_iterator(str(event_file)):
#             epoch = event.step
#             overall_acc = None
#             for value in event.summary.value:
#                 # print(value)
#                 if value.tag == "test_overall":
#                     overall_acc = value.simple_value
#                 elif value.tag == "test_worst_group_accuracy" and best_epoch == epoch:
#                     best_worst_group_acc = value.simple_value
#             # If both metrics are found in this epoch
#             if overall_acc is not None:

#                 if overall_acc > best_overall_acc:
#                     best_overall_acc = overall_acc
#                     best_epoch = epoch

#     except Exception as e:
#         print(f"Error processing {event_file}: {e}")

#     return best_epoch, best_overall_acc, best_worst_group_acc


def extract_best_metrics(event_file):
    """Extracts the epoch with the best 'test_overall_accuracy' and its corresponding 'test_worst_group_accuracy'."""
    best_epoch = 0
    best_acc = 0.0
    best_overall_acc = 0.0
    best_worst_group_acc = 0.0

    try:
        for event in tf.compat.v1.train.summary_iterator(str(event_file)):
            epoch = event.step
            overall_acc = None
            for value in event.summary.value:
                if value.tag == "test_overall":
                    overall_acc = value.simple_value
                elif value.tag == "test_worst_group_accuracy":
                    worst_group_acc = value.simple_value
            # If both metrics are found in this epoch
            if overall_acc is not None and worst_group_acc is not None:
                acc = (overall_acc + worst_group_acc) / 2
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    best_overall_acc = overall_acc
                    best_worst_group_acc = worst_group_acc
                overall_acc = None
                worst_group_acc = None

    except Exception as e:
        print(f"Error processing {event_file}: {e}")

    return best_epoch, best_overall_acc, best_worst_group_acc


# Iterate through all datasets and methods
for dataset in datasets:
    dataset_baselines_dir = target_base_dir / f"{dataset}_baselines"

    # Check possible subdirectories for dataset (e.g., "dev", "blonde" for CelebA)
    possible_subdirs = ["dev"]

    for dataset_folder in possible_subdirs:
        for method in methods:
            method_dir = (
                dataset_baselines_dir
                / dataset_folder
                / method
                / "train.events.organized"
            )

            if not method_dir.exists():
                print(f"Skipping missing folder: {method_dir}")
                continue

            for seed_dir in method_dir.iterdir():
                if not seed_dir.is_dir():
                    continue

                seed = seed_dir.name  # Seed folder name (e.g., "seed_0")

                # Find event files
                event_files = list(seed_dir.glob("events.out.tfevents.*"))
                if not event_files:
                    print(f"No events found in {seed_dir}")
                    continue

                # Process the first event file (assuming one per seed)
                event_file = event_files[0]
                best_epoch, best_acc, best_worst_acc = extract_best_metrics(event_file)

                if best_epoch is not None:
                    csv_data.append(
                        [
                            dataset,
                            method,
                            seed,
                            best_epoch,
                            best_acc * 100,
                            best_worst_acc * 100,
                        ]
                    )
                    print(
                        f"✅ {dataset}-{method}-Seed {seed}: Best Epoch {best_epoch} | Accuracy: {best_acc:.4f} | Worst Group: {best_worst_acc:.4f}"
                    )

# Save to CSV
with open(csv_output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print(f"📁 Results saved to {csv_output_path}")

# print mean+-std
csv_path = "full_results_bias_in_bios.csv"  # Change if needed
df = pd.read_csv(csv_path)

# Group by Dataset and Method
grouped = df.groupby(["Dataset", "Method"])

# Compute Mean and Std
results = grouped.agg(
    Test_Overall_Mean=("Test Overall Accuracy", "mean"),
    Test_Overall_Std=("Test Overall Accuracy", "std"),
    Test_Worst_Group_Mean=("Test Worst Group Accuracy", "mean"),
    Test_Worst_Group_Std=("Test Worst Group Accuracy", "std"),
)

# Print results in Markdown table format
print(
    "| Dataset | Method | Test Overall Accuracy (Mean ± Std) | Test Worst Group Accuracy (Mean ± Std) |"
)
print(
    "|---------|--------|------------------------------------|--------------------------------------|"
)

for (dataset, method), row in results.iterrows():
    overall = f"{row['Test_Overall_Mean']:.1f} ± {row['Test_Overall_Std']:.1f}"
    worst_group = (
        f"{row['Test_Worst_Group_Mean']:.1f} ± {row['Test_Worst_Group_Std']:.1f}"
    )
    print(f"| {dataset} | {method} | {overall} | {worst_group} |")
