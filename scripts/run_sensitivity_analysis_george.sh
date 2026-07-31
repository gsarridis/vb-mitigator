#!/bin/bash
#SBATCH -c4
#SBATCH --mem=18G
#SBATCH --gres shard:24
#SBATCH --job-name="sens-geo"
#SBATCH --output=slurm/slurm_%x_%A_%a.out
#SBATCH --error=slurm/slurm_%x_%A_%a.err
#SBATCH --array=0-5   # 1 datasets × 1 methods × 2 ratios × 3 seeds
#SBATCH --time=06:00:00
#SBATCH --exclude=iti-54-41

# Environment setup
source /mnt/cephfs/home/gsarridis/anaconda3/etc/profile.d/conda.sh
conda activate dl310_audio

export PYTHONPATH="/mnt/cephfs/home/gsarridis/projects/vb-mitigator/"
cd /mnt/cephfs/home/gsarridis/projects/vb-mitigator

# Methods
METHODS="george"
SEEDS="1 2 3"

set -- $METHODS; METHOD_ARR=("$@")
set -- $SEEDS; SEED_ARR=("$@")

NUM_METHODS=${#METHOD_ARR[@]}
NUM_SEEDS=${#SEED_ARR[@]}

NUM_COMBOS=2

EXPERIMENT_ID=$SLURM_ARRAY_TASK_ID

# COMBO_IDX=$(( EXPERIMENT_ID / (NUM_METHODS * NUM_SEEDS) ))
# METHOD_IDX=$(( (EXPERIMENT_ID / NUM_SEEDS) % NUM_METHODS ))
# SEED_IDX=$(( EXPERIMENT_ID % NUM_SEEDS ))
SEED_IDX=$(( EXPERIMENT_ID / (NUM_METHODS * NUM_COMBOS) ))
METHOD_IDX=$(( (EXPERIMENT_ID / NUM_COMBOS) % NUM_METHODS ))
COMBO_IDX=$(( EXPERIMENT_ID % NUM_COMBOS ))

METHOD=${METHOD_ARR[$METHOD_IDX]}
SEED=${SEED_ARR[$SEED_IDX]}

# Map combo index to dataset + ratio
case $COMBO_IDX in
    0) DATASET="chexpert_nih"; CONFIG_PATH="configs/chexpert_nih/$METHOD/sensitivity_0_7.yaml" ;;
    1) DATASET="chexpert_nih"; CONFIG_PATH="configs/chexpert_nih/$METHOD/sensitivity_0_8.yaml" ;;
esac

echo "====================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPUs Allocated: $CUDA_VISIBLE_DEVICES"
echo "Running sensitivity: Dataset=$DATASET, Method=$METHOD, Seed=$SEED"
echo "Config: $CONFIG_PATH"
echo "====================================="

srun python tools/train.py --cfg "$CONFIG_PATH" --seed "$SEED"
