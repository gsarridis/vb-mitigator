#!/bin/bash
#SBATCH -c4
#SBATCH --mem=24G
#SBATCH --gres shard:24
#SBATCH --job-name="new-methods-ucf101"
#SBATCH --output=slurm/slurm_%x_%A_%a.out
#SBATCH --error=slurm/slurm_%x_%A_%a.err
#SBATCH --array=0-11   # 4 methods × 3 seeds
#SBATCH --time=10:00:00

# Environment setup
source /mnt/cephfs/home/gsarridis/anaconda3/etc/profile.d/conda.sh
conda activate dl310

export PYTHONPATH="/mnt/cephfs/home/gsarridis/projects/vb-mitigator/"
cd /mnt/cephfs/home/gsarridis/projects/vb-mitigator

# Define methods and seeds
METHODS="bias_ensemble george bpa nsf"
SEEDS="0 1 2"

set -- $METHODS; METHOD_ARR=("$@")
set -- $SEEDS; SEED_ARR=("$@")

NUM_METHODS=${#METHOD_ARR[@]}
NUM_SEEDS=${#SEED_ARR[@]}

EXPERIMENT_ID=$SLURM_ARRAY_TASK_ID

METHOD_IDX=$(( EXPERIMENT_ID % NUM_METHODS ))
SEED_IDX=$(( EXPERIMENT_ID / NUM_METHODS ))

METHOD=${METHOD_ARR[$METHOD_IDX]}
SEED=${SEED_ARR[$SEED_IDX]}

CONFIG_PATH="configs/ucf101/$METHOD/scuba_swin.yaml"

echo "====================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPUs Allocated: $CUDA_VISIBLE_DEVICES"
echo "Running experiment: Dataset=ucf101, Method=$METHOD, Seed=$SEED"
echo "Config: $CONFIG_PATH"
echo "====================================="

srun python tools/train.py --cfg "$CONFIG_PATH" --seed "$SEED"
