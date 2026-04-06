#!/bin/bash
## Resource Request
#SBATCH -c8
#SBATCH --mem=32G
#SBATCH --gres shard:24
#SBATCH --job-name="tag_extract"
#SBATCH --output=slurm/slurm_%x_%j.out
#SBATCH --error=slurm/slurm_%x_%j.err
#SBATCH --time=48:00:00

# ============================================
# Tag Extraction SLURM Script
# ============================================
# 
# Usage:
#   sbatch scripts/run_tag_extraction.sh <dataset> [resume_stage]
#
# Examples:
#   sbatch scripts/run_tag_extraction.sh waterbirds
#   sbatch scripts/run_tag_extraction.sh celeba 2
#   sbatch scripts/run_tag_extraction.sh utkface 3
#
# ============================================

# Environment setup
source /mnt/cephfs/home/gsarridis/anaconda3/etc/profile.d/conda.sh
conda activate dl313

export PYTHONPATH="/mnt/cephfs/home/gsarridis/projects/vb-mitigator/"
cd /mnt/cephfs/home/gsarridis/projects/vb-mitigator

# ============================================
# Configuration
# ============================================

# Config file path
CONFIG_PATH="/mnt/cephfs/home/gsarridis/projects/vb-mitigator/configs/urbancars/tag_extraction/tag_extraction_example.yaml"


srun python tools/train.py --cfg "$CONFIG_PATH"
