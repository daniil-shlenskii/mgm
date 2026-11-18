# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# FID evaluation adapted from NVlabs/edm fid.py for MGM; see THIRD_PARTY_LICENSES.md.
"""FID-50k for one-step MGM, using the EDM/EDM2 reference statistics."""

import pickle

import numpy as np
import scipy.linalg
import torch

import dnnlib
from sid_sampler import sid_sampler
from torch_utils import distributed as dist


def _reference_stats(path):
    with dnnlib.util.open_url(path, verbose=(dist.get_rank() == 0)) as stream:
        if path.lower().endswith('.npz'):
            with np.load(stream) as data:
                return np.asarray(data['mu']), np.asarray(data['sigma'])
        stats = pickle.load(stream)['fid']
        return np.asarray(stats['mu']), np.asarray(stats['sigma'])


def calculate_fid(G, dataset_kwargs, init_sigma, device, data_stat, detector_url,
                  vae=None, scale=None, bias=None, batch_size=64, num_samples=50000):
    """Generate exactly num_samples images across ranks and compare Inception moments."""
    rank, world = dist.get_rank(), dist.get_world_size()
    if rank != 0:
        torch.distributed.barrier()
    with dnnlib.util.open_url(detector_url, verbose=(rank == 0)) as stream:
        detector = torch.jit.load(stream, map_location=device).eval()
    if rank == 0:
        torch.distributed.barrier()
    mu_real, cov_real = _reference_stats(data_stat)
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    count = num_samples // world + (rank < num_samples % world)
    feature_sum = torch.zeros(2048, dtype=torch.float64, device=device)
    feature_outer = torch.zeros(2048, 2048, dtype=torch.float64, device=device)
    if vae is not None:
        scale_t = torch.as_tensor(scale, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        bias_t = torch.as_tensor(bias, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == 'cuda' else [])
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(2024 + rank)
        with torch.no_grad():
            for start in range(0, count, batch_size):
                n = min(batch_size, count - start)
                z = torch.randn(n, G.img_channels, G.img_resolution, G.img_resolution, device=device)
                labels = None
                if G.label_dim:
                    indices = torch.randint(len(dataset), (n,)).tolist()
                    labels = torch.as_tensor(np.stack([dataset.get_label(i) for i in indices]), device=device)
                images = sid_sampler(G, latents=z, class_labels=labels, init_sigma=init_sigma)
                if vae is not None:
                    images = vae.decode((images.float() - bias_t) / scale_t).sample
                    images = images.clamp(0, 1).mul(255).to(torch.uint8)
                else:
                    images = (images.float() * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                features = detector(images, return_features=True).to(torch.float64)
                feature_sum += features.sum(dim=0)
                feature_outer += features.T @ features
    torch.distributed.all_reduce(feature_sum)
    torch.distributed.all_reduce(feature_outer)
    if rank != 0:
        return float('nan')
    mu_gen = feature_sum.cpu().numpy() / num_samples
    cov_gen = feature_outer.cpu().numpy() / num_samples - np.outer(mu_gen, mu_gen)
    root = scipy.linalg.sqrtm(cov_gen @ cov_real)
    fid = np.square(mu_gen - mu_real).sum() + np.trace(cov_gen + cov_real - 2 * root)
    return float(np.real(fid))
