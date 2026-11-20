# Midpoint Generative Models [[arXiv](https://arxiv.org/abs/2605.29920)] [[NeurIPS 2026](https://neurips.cc/virtual/2026/poster/154433)]

> [**Midpoint Generative Models**](https://arxiv.org/abs/2605.29920)<br>
> Daniil Shlenskii\*, Nikita Gushchin\*, Lev Novitskiy, Dmitry V. Dylov, Alexander Korotin<br>
> *NeurIPS 2026* (\* equal contribution)

One-step MGM training and sampling for the EDM and EDM2 backbones used in the paper.

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="MGM overview: the generator maps noise z to x0; x0 and a data sample x1 are joined by a randomly flipped interpolation, and a vector-valued critic regresses x1 - x0 from the flipped point">
</p>

## Configurations

`train.py --preset` selects one of the configurations of the paper:

| Preset | Backbone | Data | Initialization | Global batch | Noise σ | Noisy interpolants | Generator EMA |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| `cifar10` | EDM DDPM++ (56M) | CIFAR-10 32×32, pixels | 50K-kimg denoising warmup (`warmup_cifar.py`) | 256 | 0.01 | critic only | decay 0.999 |
| `imagenet64` | EDM ADM (296M) | ImageNet 64×64, pixels, class-conditional | [EDM ImageNet-64 checkpoint](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl) | 2048 | 0.2 | critic and generator | decay 0.999 |
| `imagenet512` | EDM2-S (280M) | ImageNet 512×512, SD-VAE latents, class-conditional | [EDM2-S checkpoint](https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.190.pkl) | 256 | 0.5 | critic and generator | half-life 2 Mimg, 5% ramp-up |

The presets reproduce the runs behind the reported FIDs (CIFAR-10 2.27, ImageNet 64×64 2.50, ImageNet 512×512 2.36). Both ImageNet runs added the noise to both interpolants, and the ImageNet 512×512 run used an EMA half-life, unlike the critic-only noise and EMA decay 0.999 listed in Table 3. `scripts/` has one launch script per preset.

All presets use Adam with learning rate 1e-5 and betas (0, 0.999) for both networks, one critic update per generator update, t ~ U[0, 1/2], σ_init = 2.5 and fp32. FID-50k against the EDM/EDM2 reference statistics is evaluated every 100 kimg during the first 1 Mimg, then every 500 kimg, and at the end of training. The generator and the critic are initialized from the same pretrained denoiser; the critic is f(x, t) = D(x, 0.002 + t) − x.

### Training cost

| Preset | GPUs | Training time | Lowest FID reached at |
| --- | --- | --- | --- |
| `cifar10` | 2× H100 | ~2 days | ~19.5 Mimg (of 20) |
| `imagenet64` | 8× H100 | ~3 days | ~14.3 Mimg |
| `imagenet512` | 8× H100 | ~4 days | ~30.2 Mimg |

Times are those of the paper runs (Appendix C.1) and exclude the initialization: the ImageNet presets start from the public EDM/EDM2 checkpoints, and CIFAR-10 first runs the 50K-kimg warmup. The ImageNet 512×512 run used TF32 matmuls, while the release trains in strict fp32 and therefore takes longer. In forward-pass equivalents of the 56M network, the whole CIFAR-10 pipeline costs 350M (150M warmup + 200M MGM training), the lowest complete post-initialization estimate among the methods compared in Appendix F.

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

`--duration` is in Mimg; the CIFAR-10 preset defaults to 20. The paper reports the lowest FID-50k of the generator EMA during training (`metric-fid50k_full.jsonl` in the run directory). `--batch-gpu` caps the per-GPU microbatch (gradient accumulation; the ImageNet presets default to 32, as in the paper runs on 8 GPUs) without changing the effective batch. To continue a run, rerun the same command with `--resume=<run dir>`, which loads its latest `training-state-*.pt` (the options come from the command line, not from the saved run). `--dry-run` prints the resolved options.

A run directory contains `training_options.json`, `log.txt`, `stats.jsonl`, `metric-fid50k_full.jsonl`, preview grids `fakes_<kimg>.png`, EMA snapshots `network-snapshot-<kimg>.pkl` and training states `training-state-<kimg>.pt`. For ImageNet 512×512 only the initial snapshot, the best-FID snapshot and the latest training state are kept.

### Ablations

| Paper | Options |
| --- | --- |
| Table 1a, warmup duration | `--noise=0.05 --checkpoint=runs/warmup/network-snapshot-<kimg>.pkl`; `--noise=0.05 --random-init` for 0K |
| Table 1b, noise on both interpolants | `--noise=σ --noise-on-generator=True` |
| Table 1c, critic-only noise | `--noise=σ --noise-on-generator=False` |
| Table 1d, midpoint-only training | `--t-min=0.5 --t-max=0.5 --noise=σ` |
| Appendix D.1, trigonometric interpolant | `--interpolant=trigonometric` |
| Appendix D.2, critic-to-generator update ratio | `--critic-steps-per-g=2` (or 4) |
| Appendix D.3, seeds | `--seed=<int>` (the paper runs used 565548478, 67 and 1337) |
| Appendix D.4, ImageNet 64×64 noise | `--preset=imagenet64 --noise=σ --noise-on-generator=True\|False` |

## Generation

```bash
python generate_onestep.py --network=runs/cifar10/network-snapshot-020000.pkl \
    --outdir=out/cifar10 --seeds=0-49999
```

EDM and EDM2 snapshots are detected automatically; EDM2 latents are decoded with the SD-VAE. For class-conditional networks the class is random unless `--class` is given.

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
