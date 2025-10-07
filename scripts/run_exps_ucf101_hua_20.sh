#!/bin/bash
#SBATCH --partition=anakin
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --job-name="ucf101"
#SBATCH --output=slurm/slurm_%x_%A_%a.out
#SBATCH --error=slurm/slurm_%x_%A_%a.err
#SBATCH --array=0-59   # 12 methods × 5 seeds
#SBATCH --time=03:00:00  # 3-hour slots
# Environment setup

# Environment setup
export PYTHONPATH="/home/isarridis/projects/vb-mitigator/"
cd /home/isarridis/projects/vb-mitigator

source /home/isarridis/anaconda3/etc/profile.d/conda.sh
conda activate dl310

# Define datasets and methods
DATASETS=("ucf101")
METHODS=("badd" "flac" "maviasb" "erm" "debian" "jtt" "lff" "sd" "di" "bb" "end" "groupdro")
SEEDS=(0 1 2 3 4)


# Compute indices from SLURM_ARRAY_TASK_ID
TOTAL_EXPERIMENTS=$(( ${#DATASETS[@]} * ${#METHODS[@]} * ${#SEEDS[@]} ))
EXPERIMENT_ID=$SLURM_ARRAY_TASK_ID

# Compute dataset, method, and seed indices
DATASET_IDX=0  # Only one dataset: chexpert_nih
METHOD_IDX=$(( EXPERIMENT_ID % ${#METHODS[@]} ))
SEED_IDX=$(( EXPERIMENT_ID / ${#METHODS[@]} ))

# Get corresponding values
DATASET=${DATASETS[$DATASET_IDX]}
METHOD=${METHODS[$METHOD_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Set config file path
CONFIG_PATH="configs/$DATASET/$METHOD/20.yaml"


# Log system info
echo "====================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPUs Allocated: $CUDA_VISIBLE_DEVICES"
echo "Running experiment: Dataset=$DATASET, Method=$METHOD, Seed=$SEED"
echo "====================================="

srun python tools/train.py --cfg "$CONFIG_PATH" --seed "$SEED"