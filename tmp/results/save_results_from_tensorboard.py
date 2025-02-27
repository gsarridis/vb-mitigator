import os
import csv
import tensorflow as tf
from pathlib import Path
import pandas as pd

# # Paths
# target_base_dir = Path(
#     "/mnt/cephfs/home/gsarridis/projects/vb-mitigator/output"
# )  # Update this with your actual logs directory
# csv_output_path = "full_results.csv"

# # Define methods and datasets
# methods = [
#     "erm",
#     "jtt",
#     "softcon",
#     "debian",
#     "lff",
#     "sd",
#     "flacb",
#     "mavias",
# ]
# datasets = ["celeba", "waterbirds", "urbancars"]

# groups = {
#     "celeba": {
#         "00": "test_blonde_has_tag",
#         "01": "test_blonde_no_tag",
#         "10": "test_non_has_tag",
#         "11": "test_non_no_tag",
#     },
#     "waterbirds": {
#         "00": "test_waterbird_has_tag",
#         "01": "test_waterbird_no_tag",
#         "10": "test_landbird_has_tag",
#         "11": "test_landbird_no_tag",
#     },
#     "urbancars": {
#         "00": "test_urban_has_tag",
#         "01": "test_urban_no_tag",
#         "10": "test_country_has_tag",
#         "11": "test_country_no_tag",
#     },
# }

# # CSV Header
# csv_data = [
#     [
#         "Dataset",
#         "Method",
#         "Seed",
#         "Best Epoch",
#         "Test Overall Accuracy",
#         "Test Worst Group Accuracy",
#         "Test Avg Accuracy",
#     ]
# ]


# def extract_best_metrics(event_file, dataset):
#     """Extracts the epoch with the best 'test_overall_accuracy' and its corresponding 'test_worst_group_accuracy'."""
#     best_epoch = 0
#     best_overall_acc = 0.0
#     best_worst_group_acc = 0.0
#     best_avg_acc = 0.0
#     c00 = 0.0
#     c01 = 0.0
#     c10 = 0.0
#     c11 = 0.0

#     try:
#         for event in tf.compat.v1.train.summary_iterator(str(event_file)):
#             epoch = event.step
#             overall_acc = None
#             for value in event.summary.value:
#                 if value.tag == "test_overall_accuracy":
#                     overall_acc = value.simple_value
#                 elif value.tag == "test_worst_group_accuracy" and best_epoch == epoch:
#                     best_worst_group_acc = value.simple_value
#                 elif value.tag == groups[dataset]["00"] and best_epoch == epoch:
#                     c00 = value.simple_value
#                 elif value.tag == groups[dataset]["01"] and best_epoch == epoch:
#                     c01 = value.simple_value
#                 elif value.tag == groups[dataset]["10"] and best_epoch == epoch:
#                     c10 = value.simple_value
#                 elif value.tag == groups[dataset]["11"] and best_epoch == epoch:
#                     c11 = value.simple_value
#             if best_epoch == epoch:
#                 best_avg_acc = (c00 + c01 + c10 + c11) / 4
#             # If both metrics are found in this epoch
#             if overall_acc is not None:

#                 if overall_acc > best_overall_acc:
#                     best_overall_acc = overall_acc
#                     best_epoch = epoch

#     except Exception as e:
#         print(f"Error processing {event_file}: {e}")

#     return best_epoch, best_overall_acc, best_worst_group_acc, best_avg_acc


# # Iterate through all datasets and methods
# for dataset in datasets:
#     dataset_baselines_dir = target_base_dir / f"{dataset}_baselines"

#     # Check possible subdirectories for dataset (e.g., "dev", "blonde" for CelebA)
#     possible_subdirs = ["dev", "blonde"] if dataset == "celeba" else ["dev"]

#     for dataset_folder in possible_subdirs:
#         for method in methods:
#             method_dir = (
#                 dataset_baselines_dir
#                 / dataset_folder
#                 / method
#                 / "train.events.organized"
#             )

#             if not method_dir.exists():
#                 print(f"Skipping missing folder: {method_dir}")
#                 continue

#             for seed_dir in method_dir.iterdir():
#                 if not seed_dir.is_dir():
#                     continue

#                 seed = seed_dir.name  # Seed folder name (e.g., "seed_0")

#                 # Find event files
#                 event_files = list(seed_dir.glob("events.out.tfevents.*"))
#                 if not event_files:
#                     print(f"No events found in {seed_dir}")
#                     continue

#                 # Process the first event file (assuming one per seed)
#                 event_file = event_files[0]
#                 best_epoch, best_acc, best_worst_acc, best_avg_acc = (
#                     extract_best_metrics(event_file, dataset)
#                 )

#                 if best_epoch is not None:
#                     csv_data.append(
#                         [
#                             dataset,
#                             method,
#                             seed,
#                             best_epoch,
#                             best_acc,
#                             best_worst_acc,
#                             best_avg_acc,
#                         ]
#                     )
#                     print(
#                         f"✅ {dataset}-{method}-Seed {seed}: Best Epoch {best_epoch} | Accuracy: {best_acc:.4f} | Worst Group: {best_worst_acc:.4f} | Avg Accuracy: {best_avg_acc:.4f}"
#                     )

# # Save to CSV
# with open(csv_output_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(csv_data)

# print(f"📁 Results saved to {csv_output_path}")

# print mean+-std
csv_path = "full_results.csv"  # Change if needed
df = pd.read_csv(csv_path)

# Group by Dataset and Method
grouped = df.groupby(["Dataset", "Method"])

# Compute Mean and Std
results = grouped.agg(
    Test_Overall_Mean=("Test Overall Accuracy", "mean"),
    Test_Overall_Std=("Test Overall Accuracy", "std"),
    Test_Worst_Group_Mean=("Test Worst Group Accuracy", "mean"),
    Test_Worst_Group_Std=("Test Worst Group Accuracy", "std"),
    Test_Avg_Accuracy_Mean=("Test Avg Accuracy", "mean"),
    Test_Avg_Accuracy_Std=("Test Avg Accuracy", "std"),
)

# Print results in Markdown table format
print("| Dataset | Method | WG Acc. | Avg. Acc. | Acc. |")
print("|---------|--------|---------|-----------|------|")

for (dataset, method), row in results.iterrows():
    overall = f"{row['Test_Overall_Mean']:.1f} ± {row['Test_Overall_Std']:.1f}"
    worst_group = (
        f"{row['Test_Worst_Group_Mean']:.1f} ± {row['Test_Worst_Group_Std']:.1f}"
    )
    avg_acc = (
        f"{row['Test_Avg_Accuracy_Mean']:.1f} ± {row['Test_Avg_Accuracy_Std']:.1f}"
    )
    print(f"| {dataset} | {method} | {worst_group} | {avg_acc} | {overall}")
