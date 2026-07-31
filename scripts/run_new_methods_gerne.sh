#!/bin/bash
#SBATCH -c4
#SBATCH --mem=24G
#SBATCH --gres shard:24
#SBATCH --job-name="gerne-nat"
#SBATCH --output=slurm/slurm_%x_%A_%a.out
#SBATCH --error=slurm/slurm_%x_%A_%a.err
#SBATCH --array=0-9   # 8 datasets × 1 methods × 5 seeds
#SBATCH --time=06:00:00
#SBATCH --exclude=iti-54-41

# Environment setup
export PYTHONPATH="/mnt/cephfs/home/gsarridis/projects/vb-mitigator/"
cd /mnt/cephfs/home/gsarridis/projects/vb-mitigator

source /mnt/cephfs/home/gsarridis/anaconda3/etc/profile.d/conda.sh
conda activate dl310_audio

# Define datasets, methods, and seeds
# DATASETS="bias_in_bios jigsaw_toxic_comments speech_accent_archive urbansounds58 chexpert_nih urbancars waterbirds celeba"
DATASETS="urbancars waterbirds"

METHODS="gerne"
SEEDS="1 2 3 4 5"

# Convert to arrays
set -- $DATASETS; DATASET_ARR=("$@")
set -- $METHODS; METHOD_ARR=("$@")
set -- $SEEDS; SEED_ARR=("$@")

NUM_DATASETS=${#DATASET_ARR[@]}
NUM_METHODS=${#METHOD_ARR[@]}
NUM_SEEDS=${#SEED_ARR[@]}

EXPERIMENT_ID=$SLURM_ARRAY_TASK_ID

# Compute indices
# DATASET_IDX=$(( EXPERIMENT_ID / (NUM_METHODS * NUM_SEEDS) ))
# METHOD_IDX=$(( (EXPERIMENT_ID / NUM_SEEDS) % NUM_METHODS ))
# SEED_IDX=$(( EXPERIMENT_ID % NUM_SEEDS ))
SEED_IDX=$(( EXPERIMENT_ID / (NUM_DATASETS * NUM_METHODS) ))
DATASET_IDX=$(( (EXPERIMENT_ID / NUM_METHODS) % NUM_DATASETS ))
METHOD_IDX=$(( EXPERIMENT_ID % NUM_METHODS ))

DATASET=${DATASET_ARR[$DATASET_IDX]}
METHOD=${METHOD_ARR[$METHOD_IDX]}
SEED=${SEED_ARR[$SEED_IDX]}

CONFIG_PATH="configs/$DATASET/$METHOD/dev.yaml"

echo "====================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPUs Allocated: $CUDA_VISIBLE_DEVICES"
echo "Running experiment: Dataset=$DATASET, Method=$METHOD, Seed=$SEED"
echo "Config: $CONFIG_PATH"
echo "====================================="

srun python tools/train.py --cfg "$CONFIG_PATH" --seed "$SEED"
