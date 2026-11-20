<h1 align="center">Midpoint Generative Models</h1>

<p align="center">
  We train a one-step generator using flow matching between its own samples and real data.
</p>

<p align="center">
  Daniil Shlenskii<sup>*</sup>, Nikita Gushchin<sup>*</sup>, Lev Novitskiy, Dmitry V. Dylov, Alexander Korotin<br>
  <sub>* Equal contribution</sub>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.29920">Paper (arXiv)</a> ·
  <a href="https://neurips.cc/virtual/2026/poster/154433">NeurIPS 2026</a>
</p>

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="The generator maps noise to a sample. The critic sees a randomly flipped interpolation and predicts the displacement from the generated sample to the real one.">
</p>

<p align="center">
  At the midpoint, the velocity field vanishes exactly when the generated and real distributions match.
  A random time flip extends this symmetry to other interpolation times.
</p>

## Results

<p align="center">
  <img src="assets/imagenet512-samples.png" width="100%" alt="Twelve generated images arranged in two rows, including dogs, otters, geysers, parrots, valleys and arctic foxes."><br>
  <em>One-step samples on ImageNet 512×512.</em>
</p>

<p align="center">
  <strong>FID-50k ↓</strong> · CIFAR-10 <strong>2.27</strong> · ImageNet 64×64 <strong>2.50</strong> · ImageNet 512×512 <strong>2.36</strong>
</p>

## Install

Requires Python 3.11–3.13 and CUDA GPUs. From this directory:

```bash
uv sync
source .venv/bin/activate
```

## Datasets

CIFAR-10 and ImageNet 64×64 use the EDM dataset format:

```bash
python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip
python dataset_tool.py --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \
    --dest=datasets/imagenet-64x64.zip --resolution=64x64 --transform=center-crop
```

ImageNet 512×512 uses the SD-VAE latent dataset of EDM2 (8×64×64 mean/std latents). Build it with the `dataset_tool.py` of [NVlabs/edm2](https://github.com/NVlabs/edm2):

```bash
python dataset_tool.py convert --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \
    --dest=datasets/img512.zip --resolution=512x512 --transform=center-crop-dhariwal
python dataset_tool.py encode --source=datasets/img512.zip --dest=datasets/img512-sd.zip
```

Pretrained checkpoints, FID reference statistics, the Inception network and the SD-VAE (`stabilityai/sd-vae-ft-mse`) are downloaded on first use.

## CIFAR-10 denoising warmup

To skip the warmup and start MGM training directly, pass the URL of NVIDIA's [pretrained unconditional CIFAR-10 DDPM++ checkpoint](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl) to `train.py` via `--checkpoint`. To reproduce the paper's CIFAR-10 result (FID-50k 2.27), use the 50K-kimg warmup below.

```bash
torchrun --standalone --nproc_per_node=8 warmup_cifar.py \
    --data=datasets/cifar10-32x32.zip --outdir=runs/warmup --duration-kimg=50000
```

This trains DDPM++ with EDM's CIFAR-10 recipe and the folded log-normal noise distribution of Appendix C.3, and saves EMA snapshots `network-snapshot-<kimg>.pkl` at 10K, 20K and 50K kimg (`--duration-kimg=200000` adds 100K and 200K for the warmup ablation). `--resume=runs/warmup` continues from the latest `warmup-state-*.pt`.

## MGM training

```bash
# CIFAR-10 (20 Mimg = 78,125 iterations at batch 256)
torchrun --standalone --nproc_per_node=2 train.py --preset=cifar10 \
    --data=datasets/cifar10-32x32.zip --outdir=runs/cifar10 \
    --checkpoint=runs/warmup/network-snapshot-050000.pkl

# ImageNet 64x64
torchrun --standalone --nproc_per_node=8 train.py --preset=imagenet64 \
    --data=datasets/imagenet-64x64.zip --outdir=runs/imagenet64 --duration=100

# ImageNet 512x512
torchrun --standalone --nproc_per_node=8 train.py --preset=imagenet512 \
    --data=datasets/img512-sd.zip --outdir=runs/imagenet512 --duration=100
```

## Generation

```bash
python generate_onestep.py --network=runs/cifar10/network-snapshot-020000.pkl \
    --outdir=out/cifar10 --seeds=0-49999
```

EDM and EDM2 snapshots are detected automatically; EDM2 latents are decoded with the SD-VAE. For class-conditional networks the class is random unless `--class` is given.

## Training cost

| Preset | GPUs | Training time | Lowest FID reached at |
| --- | --- | --- | --- |
| `cifar10` | 2× H100 | ~2 days | ~19.5 Mimg (of 20) |
| `imagenet64` | 8× H100 | ~3 days | ~14.3 Mimg |
| `imagenet512` | 8× H100 | ~4 days | ~30.2 Mimg |

Times are those of the paper runs (Appendix C.1) and exclude the initialization: the ImageNet presets start from the public EDM/EDM2 checkpoints, and CIFAR-10 first runs the 50K-kimg warmup. The ImageNet 512×512 run used TF32 matmuls, while the release trains in strict fp32 and therefore takes longer. In forward-pass equivalents of the 56M network, the whole CIFAR-10 pipeline costs 350M (150M warmup + 200M MGM training), the lowest complete post-initialization estimate among the methods compared in Appendix F.

## Citation

```bibtex
@inproceedings{shlenskii2026midpoint,
  title     = {Midpoint Generative Models},
  author    = {Shlenskii, Daniil and Gushchin, Nikita and Novitskiy, Lev and Dylov, Dmitry V. and Korotin, Alexander},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  eprint    = {2605.29920},
  archivePrefix = {arXiv}
}
```

## Licenses

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). The root [LICENSE.txt](LICENSE.txt) applies to Apache-2.0 components; the NVIDIA-derived EDM/EDM2 components keep their CC BY-NC-SA 4.0 terms.
