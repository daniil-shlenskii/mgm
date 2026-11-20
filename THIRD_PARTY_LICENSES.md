# File licenses and upstream attribution

This repository contains components with different licenses. `LICENSE.txt` is Apache License 2.0 and does **not** relicense NVIDIA-derived files.

| Files | Origin | License |
| --- | --- | --- |
| `dataset_tool.py`, `dnnlib/**`, `edm/**`, `torch_utils/{__init__,distributed,misc,persistence,training_stats}.py` | [NVlabs/edm](https://github.com/NVlabs/edm) and subsequent MGM adaptations | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) as stated in the file headers |
| `edm2/{__init__,dataset,misc,networks_modified}.py` | [NVlabs/edm2](https://github.com/NVlabs/edm2) and subsequent SiD/MGM adaptations | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) as stated in the file headers |
| `warmup_cifar.py`, `metrics/fid.py` | MGM adaptations of the [NVlabs/edm](https://github.com/NVlabs/edm) training and FID code | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) as stated in the file headers |
| `edm2/{mgm_networks,sida_networks}.py`, `generate_onestep.py`, `training/unified_training_loop.py` | [SiD](https://github.com/mingyuanzhou/SiD) and MGM code built on NVlabs/edm and NVlabs/edm2 code | Apache License 2.0 as stated in the file headers for the SiD/MGM parts; the NVIDIA-derived parts remain under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `training/__init__.py` | Copyright © 2024 Mingyuan Zhou, as credited in the file header | [Apache License 2.0](LICENSE.txt) |
| `train.py`, `scripts/*.sh`, `training/midpoint_loss.py`, `sid_sampler.py`, `metrics/__init__.py`, `edm2/vae_utils.py`, `torch_utils/model_registry.py` | MGM release code | Apache License 2.0 (`LICENSE.txt`) |
| `assets/teaser.png`, `assets/imagenet512-samples.png` | Method overview and generated sample grid from [*Midpoint Generative Models*, Shlenskii et al. (2026)](https://arxiv.org/abs/2605.29920v3) | The paper’s [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license covers the authors’ contributions to these figures. Credit the authors, link the license and indicate any changes. Third-party material retains its own terms. |

The NVIDIA files retain their original copyright and license headers. The StyleGAN-derived metric and CUDA-op files that carried a restrictive proprietary header were excluded. The release uses a FID implementation adapted from the official EDM workflow; no StyleGAN CUDA ops are required by the selected networks.

Model weights, datasets, the SD-VAE, and the downloaded Inception detector are separate artifacts with their own terms and are not included in this repository.
