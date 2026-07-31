#!/bin/bash
#SBATCH -c4
#SBATCH --mem=24G
#SBATCH --gres shard:24
#SBATCH --job-name="new-celeba"
#SBATCH --output=slurm/slurm_%x_%A_%a.out
#SBATCH --error=slurm/slurm_%x_%A_%a.err
#SBATCH --array=0-19   # 1 datasets × 4 methods × 5 seeds
#SBATCH --time=06:00:00
#SBATCH --exclude=iti-54-41


# Environment setup
source /mnt/cephfs/home/gsarridis/anaconda3/etc/profile.d/conda.sh
conda activate dl310_audio

export PYTHONPATH="/mnt/cephfs/home/gsarridis/projects/vb-mitigator/"
cd /mnt/cephfs/home/gsarridis/projects/vb-mitigator

# Define datasets, methods, and seeds
# CelebA uses "blonde.yaml", others use "dev.yaml"
METHODS="bias_ensemble george bpa nsf"
SEEDS="1 2 3 4 5"

set -- $METHODS; METHOD_ARR=("$@")
set -- $SEEDS; SEED_ARR=("$@")

NUM_METHODS=${#METHOD_ARR[@]}
NUM_SEEDS=${#SEED_ARR[@]}

# 3 datasets: celeba, waterbirds, urbancars
NUM_DATASETS=1
EXPERIMENT_ID=$SLURM_ARRAY_TASK_ID

# DATASET_IDX=$(( EXPERIMENT_ID / (NUM_METHODS * NUM_SEEDS) ))
# METHOD_IDX=$(( (EXPERIMENT_ID / NUM_SEEDS) % NUM_METHODS ))
# SEED_IDX=$(( EXPERIMENT_ID % NUM_SEEDS ))
SEED_IDX=$(( EXPERIMENT_ID / (NUM_DATASETS * NUM_METHODS) ))
DATASET_IDX=$(( (EXPERIMENT_ID / NUM_METHODS) % NUM_DATASETS ))
METHOD_IDX=$(( EXPERIMENT_ID % NUM_METHODS ))

METHOD=${METHOD_ARR[$METHOD_IDX]}
SEED=${SEED_ARR[$SEED_IDX]}


DATASET="celeba"
CONFIG_PATH="configs/celeba/$METHOD/blonde.yaml"
echo "====================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPUs Allocated: $CUDA_VISIBLE_DEVICES"
echo "Running experiment: Dataset=$DATASET, Method=$METHOD, Seed=$SEED"
echo "Config: $CONFIG_PATH"
echo "====================================="

srun python tools/train.py --cfg "$CONFIG_PATH" --seed "$SEED"
