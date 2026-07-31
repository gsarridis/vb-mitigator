#!/bin/bash
# ===========================================================================
# ImageNet-9M mitigation sweep (SLURM array, self-submitting, 3 dependent stages)
#
#   methods      : erm (stage 1) + flac, badd, maviasb, bpa (stage 2)
#   scenarios    : jpeg, resize (single) + multi
#   backbones    : resnet18, resnet50
#   correlations : 0.8 0.9 0.95 0.99 0.999
#   seeds        : 1 2 3
#
# Prerequisites are trained first, then the mitigators depend on them:
#   stage 0  BCCs       : per-backbone bias-capturing classifiers (balanced data)
#   stage 1  ERM        : the erm baselines (also bpa's reference checkpoint)
#   stage 2  mitigators : flac/badd/maviasb (need BCCs) + bpa (needs ERM ckpt)
#
# Usage:
#   bash scripts/run_imagenet9m_sweep.sh          # submits the 3 array jobs
# ===========================================================================
#SBATCH --job-name=in9m-sweep
#SBATCH -c4
#SBATCH --mem=24G
#SBATCH --gres shard:24
#SBATCH --time=06:00:00
#SBATCH --output=slurm/slurm_%x_%A_%a.out
#SBATCH --error=slurm/slurm_%x_%A_%a.err
#SBATCH --exclude=iti-54-41


set -euo pipefail

# ---- sweep axes ----
SCENARIOS=(jpeg resize multi)
MODELS=(resnet18 resnet50)
CORRS=(0.8 0.9 0.95 0.99 0.999)
SEEDS=(1 2 3)
METHODS_MIT=(flac badd maviasb bpa)
BCC_BIASES=(jpeg resize)

nscen=${#SCENARIOS[@]}; nmodel=${#MODELS[@]}; ncorr=${#CORRS[@]}
nseed=${#SEEDS[@]}; nmethod=${#METHODS_MIT[@]}; nbias=${#BCC_BIASES[@]}

N_BCC=$(( nbias * nmodel ))                              # 4
N_ERM=$(( nscen * nmodel * ncorr * nseed ))             # 90
N_MIT=$(( nmethod * nscen * nmodel * ncorr * nseed ))   # 360

# ===========================================================================
# SUBMIT MODE: no SLURM_ARRAY_TASK_ID -> submit the 3 stages with dependencies
# ===========================================================================
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SELF=$(realpath "$0")
    echo "Submitting sweep: BCC=${N_BCC}, ERM=${N_ERM}, MIT=${N_MIT} tasks"
    jid_bcc=$(sbatch --parsable --job-name in9m-bcc --array=0-$((N_BCC-1)) --export=ALL,STAGE=bcc "$SELF")
    jid_erm=$(sbatch --parsable --job-name in9m-erm --array=0-$((N_ERM-1)) --export=ALL,STAGE=erm "$SELF")
    jid_mit=$(sbatch --parsable --job-name in9m-mit \
                     --dependency=afterok:${jid_bcc}:${jid_erm} \
                     --array=0-$((N_MIT-1)) --export=ALL,STAGE=mit "$SELF")
    echo "  stage0 BCC job  : $jid_bcc"
    echo "  stage1 ERM job  : $jid_erm"
    echo "  stage2 MIT job  : $jid_mit  (starts after $jid_bcc and $jid_erm)"
    exit 0
fi

# ===========================================================================
# WORKER MODE: run one experiment identified by STAGE + SLURM_ARRAY_TASK_ID
# ===========================================================================
source /mnt/cephfs/home/gsarridis/anaconda3/etc/profile.d/conda.sh
conda activate dl310_audio
export PYTHONPATH="/mnt/cephfs/home/gsarridis/projects/vb-mitigator/"
cd /mnt/cephfs/home/gsarridis/projects/vb-mitigator

i=$SLURM_ARRAY_TASK_ID

case "$STAGE" in
  bcc)
    bias=${BCC_BIASES[$(( i / nmodel ))]}
    model=${MODELS[$(( i % nmodel ))]}
    scen=$bias
    corr=0.5                                   # BCCs are trained on balanced data
    seed=1
    [ "$bias" = "jpeg" ] && base="configs/imagenet9m/erm/bcc_jpg.yaml" \
                         || base="configs/imagenet9m/erm/bcc_resize.yaml"
    tag="bcc_${bias}_${model}"
    ;;
  erm)
    seed=${SEEDS[$(( i % nseed ))]}
    corr=${CORRS[$(( (i / nseed) % ncorr ))]}
    model=${MODELS[$(( (i / (nseed*ncorr)) % nmodel ))]}
    scen=${SCENARIOS[$(( i / (nseed*ncorr*nmodel) ))]}
    base="configs/imagenet9m/erm/${scen}.yaml"
    tag="${scen}_${model}_c${corr}_s${seed}"
    ;;
  mit)
    seed=${SEEDS[$(( i % nseed ))]}
    corr=${CORRS[$(( (i / nseed) % ncorr ))]}
    model=${MODELS[$(( (i / (nseed*ncorr)) % nmodel ))]}
    scen=${SCENARIOS[$(( (i / (nseed*ncorr*nmodel)) % nscen ))]}
    method=${METHODS_MIT[$(( i / (nseed*ncorr*nmodel*nscen) ))]}
    base="configs/imagenet9m/${method}/${scen}.yaml"
    tag="${scen}_${model}_c${corr}_s${seed}"
    ;;
  *)
    echo "unknown STAGE=$STAGE"; exit 1 ;;
esac

echo "=== STAGE=$STAGE task=$i | base=$base scen=$scen model=$model corr=$corr seed=$seed tag=$tag ==="

TMPCFG=$(mktemp --suffix=.yaml)
trap 'rm -f "$TMPCFG"' EXIT
python scripts/gen_imagenet9m_cfg.py "$base" "$TMPCFG" "$scen" "$model" "$corr" "$tag"
python tools/train.py --cfg "$TMPCFG" --seed "$seed"
