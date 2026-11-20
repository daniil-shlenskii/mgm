#!/usr/bin/env bash
# ImageNet 512x512, class-conditional, EDM2-S on SD-VAE latents: FID 2.36 (Table 2).
#
# Initialized from the EDM2-S 0.190 posthoc reconstruction. Noise sigma = 0.5 on both
# interpolants, batch 256, Adam lr 1e-5, betas (0, 0.999), EMA half-life 2 Mimg with a 5%
# ramp-up, 8 GPUs with 32 images per GPU. The paper run reached its lowest FID-50k after
# ~30.2 Mimg. Only the best-FID snapshot and the latest training state are kept.
#
# The latent dataset img512-sd.zip is built with NVlabs/edm2's dataset_tool.py:
#   python dataset_tool.py convert --source=<ImageNet train dir> --dest=img512.zip \
#       --resolution=512x512 --transform=center-crop-dhariwal
#   python dataset_tool.py encode --source=img512.zip --dest=img512-sd.zip
#
# Rerunning the script resumes from the latest training state. The reported number is the
# lowest fid50k_full in runs/imagenet512/metric-fid50k_full.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-datasets/img512-sd.zip}
OUTDIR=${OUTDIR:-runs/imagenet512}
GPUS=${GPUS:-8}
DURATION=${DURATION:-40}  # Mimg
SEED=${SEED:-1278978921}  # seed of the paper run

if [ ! -e "$DATA" ]; then
    echo "Missing $DATA: build it with NVlabs/edm2's dataset_tool.py (see the header of this script)." >&2
    exit 1
fi

RESUME=()
if compgen -G "$OUTDIR/training-state-*.pt" > /dev/null; then RESUME=(--resume="$OUTDIR"); fi
torchrun --standalone --nproc_per_node="$GPUS" train.py --preset=imagenet512 \
    --data="$DATA" --outdir="$OUTDIR" \
    --duration="$DURATION" --seed="$SEED" ${RESUME[@]+"${RESUME[@]}"} "$@"
