#!/usr/bin/env bash
# CIFAR-10 32x32, unconditional, EDM DDPM++: FID 2.27 (Table 2).
#
#   1. converts CIFAR-10 to the EDM dataset format (if needed),
#   2. runs the 50K-kimg denoising warmup (Appendix C.3; the paper used 8 GPUs),
#   3. trains MGM for 20 Mimg (78,125 iterations at batch 256; the paper used 2 GPUs):
#      critic-only noise sigma = 0.01, Adam lr 1e-5, betas (0, 0.999), EMA decay 0.999.
#
# Rerunning the script resumes the warmup and the MGM run from their latest states.
# The reported number is the lowest fid50k_full in runs/cifar10/metric-fid50k_full.jsonl
# (the paper run reached it after ~19.5 Mimg).
set -euo pipefail
cd "$(dirname "$0")/.."

CIFAR_TARBALL=${CIFAR_TARBALL:-downloads/cifar10/cifar-10-python.tar.gz}
DATA=${DATA:-datasets/cifar10-32x32.zip}
WARMUP_DIR=${WARMUP_DIR:-runs/cifar10-warmup}
OUTDIR=${OUTDIR:-runs/cifar10}
WARMUP_GPUS=${WARMUP_GPUS:-8}
GPUS=${GPUS:-2}
SEED=${SEED:-565548478}   # seed of the paper run; D.3 also used 67 and 1337

if [ ! -e "$DATA" ]; then
    python dataset_tool.py --source="$CIFAR_TARBALL" --dest="$DATA"
fi

WARMUP_PKL="$WARMUP_DIR/network-snapshot-050000.pkl"
if [ ! -f "$WARMUP_PKL" ]; then
    RESUME=()
    if compgen -G "$WARMUP_DIR/warmup-state-*.pt" > /dev/null; then RESUME=(--resume="$WARMUP_DIR"); fi
    torchrun --standalone --nproc_per_node="$WARMUP_GPUS" warmup_cifar.py \
        --data="$DATA" --outdir="$WARMUP_DIR" --duration-kimg=50000 ${RESUME[@]+"${RESUME[@]}"}
fi

RESUME=()
if compgen -G "$OUTDIR/training-state-*.pt" > /dev/null; then RESUME=(--resume="$OUTDIR"); fi
torchrun --standalone --nproc_per_node="$GPUS" train.py --preset=cifar10 \
    --data="$DATA" --outdir="$OUTDIR" --checkpoint="$WARMUP_PKL" \
    --duration=20 --seed="$SEED" ${RESUME[@]+"${RESUME[@]}"} "$@"
