import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Path to your JSON file (update accordingly)
json_path = "/mnt/cephfs/home/common/datasets/UCF101/ucf101_01.json"

# Load JSON file
with open(json_path, "r") as f:
    data = json.load(f)

# Extract database section
database = data["database"]

# Collect co-occurrence counts into a DataFrame
records = []
for vid, info in database.items():
    subset = info["subset"]
    label = info["annotations"]["label"]
    coarse_scene = info["annotations"]["coarse_scene_label"]
    records.append((subset, label, coarse_scene))

df = pd.DataFrame(records, columns=["subset", "label", "coarse_scene"])

all_labels = df["label"].unique()
all_scenes = df["coarse_scene"].unique()


# Function to plot confusion matrix-like heatmap (no numbers in cells)
def plot_confusion_matrix(subset):
    subset_df = df[df["subset"] == subset]
    if subset_df.empty:
        print("empty")
        return

    matrix = pd.crosstab(subset_df["label"], subset_df["coarse_scene"]).T

    n_rows, n_cols = matrix.shape
    plt.figure(figsize=(n_cols * 0.4, n_rows * 0.4))  # adjust scaling factors if needed
    sns.heatmap(matrix, annot=False, cmap="Blues", cbar=True)
    plt.title(f"Label vs Coarse Scene Co-occurrence ({subset} set)")
    plt.ylabel("Coarse Scene Label")
    plt.xlabel("Action Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{subset}.png")
    plt.show()
    plt.close()


# Plot for training and validation
plot_confusion_matrix("training")
plot_confusion_matrix("validation")
