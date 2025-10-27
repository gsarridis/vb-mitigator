#!/bin/bash
#SBATCH --partition=anakin
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --job-name="ucf101"
#SBATCH --output=slurm/slurm_%x_%A_%a.out
#SBATCH --error=slurm/slurm_%x_%A_%a.err
#SBATCH --time=03:00:00

# Usage:
# sbatch run_job.sh METHOD SEED
# Example:
# sbatch run_job.sh erm 1

# Read arguments
METHOD=$1
SEED=$2

# Check arguments
if [ -z "$METHOD" ] || [ -z "$SEED" ]; then
    echo "Usage: sbatch $0 <method> <seed>"
    exit 1
fi

# Environment setup
export PYTHONPATH="/home/isarridis/projects/vb-mitigator/"
cd /home/isarridis/projects/vb-mitigator

source /home/isarridis/anaconda3/etc/profile.d/conda.sh
conda activate dl310

# Fixed dataset
DATASET="ucf101"

# Config path
CONFIG_PATH="configs/$DATASET/$METHOD/scuba_swin.yaml"

# Log system info
echo "====================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs Allocated: $CUDA_VISIBLE_DEVICES"
echo "Running experiment: Dataset=$DATASET, Method=$METHOD, Seed=$SEED"
echo "====================================="

# Run the experiment
srun python tools/train.py --cfg "$CONFIG_PATH" --seed "$SEED"
