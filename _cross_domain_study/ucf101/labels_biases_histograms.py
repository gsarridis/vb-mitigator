import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Path to your JSON file (update this with the actual file path)
json_path = "/mnt/cephfs/home/common/datasets/UCF101/ucf101_01.json"

# Load JSON file
with open(json_path, "r") as f:
    data = json.load(f)

# The database dictionary
database = data["database"]

# Organize counts: {subset -> {class -> {coarse_scene_label -> count}}}
counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for vid, info in database.items():
    subset = info["subset"]
    label = info["annotations"]["label"]
    coarse_scene = info["annotations"]["coarse_scene_label"]
    counts[subset][label][coarse_scene] += 1

# Plot histograms for training and validation separately
for subset in ["training", "validation"]:
    if subset not in counts:
        continue

    for label, scenes in counts[subset].items():
        plt.figure(figsize=(8, 4))
        plt.bar(scenes.keys(), scenes.values())
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{label} - {subset} set")
        plt.xlabel("Coarse Scene Label")
        plt.ylabel("Number of Samples")
        plt.tight_layout()
        plt.savefig(f"{subset}-{label}.png")
        plt.show()
        plt.close()
