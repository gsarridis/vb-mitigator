#!/bin/bash
# ============================================
# Run SAE Analysis on UTKFace
# ============================================
# 
# This script:
# 1. First trains an ERM model (if no checkpoint exists)
# 2. Then runs SAE analysis on the pretrained model
#
# Usage:
#   bash ./scripts/utkface/sae/sae.sh
#
# Or with specific checkpoint:
#   bash ./scripts/utkface/sae/sae.sh /path/to/checkpoint.pth
# ============================================

set -e

echo "============================================"
echo "Running SAE Analysis"
echo "============================================"


# Run SAE analysis
python tools/train.py --cfg configs/utkface/sae/dev.yaml

echo "============================================"
echo "SAE Analysis Complete!"
echo "============================================"
echo "Check outputs/utkface_sae/ for results"

