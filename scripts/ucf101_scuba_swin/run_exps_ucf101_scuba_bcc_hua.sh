#!/bin/bash
#SBATCH --partition=anakin
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --job-name="ucf101"
#SBATCH --output=slurm/slurm_%x_%A_%a.out
#SBATCH --error=slurm/slurm_%x_%A_%a.err
#SBATCH --array=0-0   # 12 methods × 5 seeds
#SBATCH --time=03:00:00  # 3-hour slots
# Environment setup

# Environment setup
export PYTHONPATH="/home/isarridis/projects/vb-mitigator/"
cd /home/isarridis/projects/vb-mitigator

source /home/isarridis/anaconda3/etc/profile.d/conda.sh
conda activate dl310


srun python tools/train.py --cfg ./configs/ucf101/erm/bcc_scuba_swin.yaml 