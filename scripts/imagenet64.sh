#!/usr/bin/env bash
# ImageNet 64x64, class-conditional, EDM ADM: FID 2.50 (Table 2).
#
# Initialized from the EDM ImageNet-64 checkpoint. Noise sigma = 0.2 on both interpolants,
# batch 2048, Adam lr 1e-5, betas (0, 0.999), EMA decay 0.999, 8 GPUs with 32 images per GPU
# per accumulation round. The paper run reached its lowest FID-50k after ~14.3 Mimg.
#
# Rerunning the script resumes from the latest training state. The reported number is the
# lowest fid50k_full in runs/imagenet64/metric-fid50k_full.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGENET_TRAIN=${IMAGENET_TRAIN:-downloads/imagenet/ILSVRC/Data/CLS-LOC/train}
DATA=${DATA:-datasets/imagenet-64x64.zip}
OUTDIR=${OUTDIR:-runs/imagenet64}
GPUS=${GPUS:-8}
DURATION=${DURATION:-20}  # Mimg
SEED=${SEED:-565548478}   # seed of the paper run

if [ ! -e "$DATA" ]; then
    python dataset_tool.py --source="$IMAGENET_TRAIN" --dest="$DATA" \
        --resolution=64x64 --transform=center-crop
fi

RESUME=()
if compgen -G "$OUTDIR/training-state-*.pt" > /dev/null; then RESUME=(--resume="$OUTDIR"); fi
torchrun --standalone --nproc_per_node="$GPUS" train.py --preset=imagenet64 \
    --data="$DATA" --outdir="$OUTDIR" \
    --duration="$DURATION" --seed="$SEED" ${RESUME[@]+"${RESUME[@]}"} "$@"
