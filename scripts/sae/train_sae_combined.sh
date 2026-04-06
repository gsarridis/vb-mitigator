#!/bin/bash
# Train SAE on Combined Dataset
#
# This script trains a Sparse Autoencoder on diverse visual features from:
# - CUB-200: Bird species
# - Stanford Cars: Car types
# - Places365: Selected backgrounds
# - LVIS/COCO: Selected objects
#
# Usage:
#   ./run_sae_combined.sh
#
# Make sure to set the dataset paths below!

set -e

# ============================================
# Dataset Paths - UPDATE THESE!
# ============================================
CUB200_ROOT="/mnt/cephfs/home/gsarridis/datasets/CUB_200_2011"
STANFORD_CARS_ROOT="/nas3-2/gsarridis/backups/154/projects/Whac-A-Mole/data/stanford_cars"
PLACES365_ROOT="/nas3-2/gsarridis/backups/154/projects/Whac-A-Mole/data/places"
LVIS_ROOT="/nas3-2/gsarridis/backups/154/projects/Whac-A-Mole/data/lvis"
COCO_IMAGES_ROOT="/nas3-2/gsarridis/backups/154/projects/Whac-A-Mole/data/coco"

# ============================================
# Training Settings
# ============================================
ENCODER_TYPE="openclip"
MODEL_NAME="ViT-B-32"
PRETRAINED="openai"

DICT_SIZE=4096
NUM_STEPS=50000
BATCH_SIZE=256
LR=1e-4

OUTPUT_DIR="./outputs/sae_combined"

# ============================================
# Run Training
# ============================================
echo "=========================================="
echo "SAE Combined Dataset Training"
echo "=========================================="
echo ""
echo "Datasets:"
echo "  CUB-200:       $CUB200_ROOT"
echo "  Stanford Cars: $STANFORD_CARS_ROOT"
echo "  Places365:     $PLACES365_ROOT"
echo "  LVIS:          $LVIS_ROOT"
echo "  COCO:          $COCO_IMAGES_ROOT"
echo ""
echo "Encoder: $ENCODER_TYPE / $MODEL_NAME"
echo "SAE Dict Size: $DICT_SIZE"
echo "Training Steps: $NUM_STEPS"
echo ""

python -c "
from tools.train_sae_combined import SAECombinedTrainer

trainer = SAECombinedTrainer(
    # VLM Encoder
    encoder_type='${ENCODER_TYPE}',
    model_name='${MODEL_NAME}',
    pretrained='${PRETRAINED}',
    
    # Dataset paths
    cub200_root='${CUB200_ROOT}',
    stanford_cars_root='${STANFORD_CARS_ROOT}',
    places365_root='${PLACES365_ROOT}',
    lvis_root='${LVIS_ROOT}',
    coco_images_root='${COCO_IMAGES_ROOT}',
    
    # SAE settings
    dict_size=${DICT_SIZE},
    num_steps=${NUM_STEPS},
    batch_size=${BATCH_SIZE},
    lr=${LR},
    
    # Output
    output_dir='${OUTPUT_DIR}',
)

trainer.run_full_pipeline()
"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Outputs saved to: $OUTPUT_DIR"
echo "  - features.pt"
echo "  - sae_checkpoints/ae.pt"
echo "  - analysis_results.json"