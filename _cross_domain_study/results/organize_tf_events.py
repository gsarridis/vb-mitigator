import os
import shutil
from pathlib import Path

# Define methods and datasets
methods = ["erm", "flacb", "mavias", "softcon", "sd", "jtt", "lff", "debian"]
datasets = ["celeba", "waterbirds", "urbancars"]

# Base directory containing TensorBoard logs
source_dir = Path(
    "/mnt/cephfs/home/gsarridis/projects/vb-mitigator/output"
)  # Update this path
target_base_dir = Path(
    "/mnt/cephfs/home/gsarridis/projects/vb-mitigator/output"
)  # Where logs should be structured

for method in methods:
    for dataset in datasets:
        # Define dataset-specific folder naming
        dataset_folder = "blonde" if dataset == "celeba" else "dev"

        # Path where the current logs are stored
        method_log_dir = (
            source_dir
            / f"{dataset}_baselines"
            / dataset_folder
            / method
            / "train.events"
        )

        if not method_log_dir.exists():
            print(f"Skipping missing folder: {method_log_dir}")
            continue

        # Identify all event files for different seeds
        for idx, event_file in enumerate(method_log_dir.glob("./*tfevents*")):
            # Extract seed from filename if possible (assuming they differ)
            seed = f"seed_{idx}"

            # Destination folder: tensorboard_logs/<method>/<dataset>/seed_<seed>/
            dest_dir = (
                target_base_dir
                / f"{dataset}_baselines"
                / dataset_folder
                / method
                / "train.events.organized"
                / seed
            )
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Move event file to structured folder
            shutil.copy(str(event_file), str(dest_dir / event_file.name))

            print(f"Moved {event_file} -> {dest_dir / event_file.name}")

print("✅ TensorBoard event files have been organized.")
