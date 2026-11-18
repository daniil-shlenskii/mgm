# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""Generate images with a one-step MGM generator snapshot."""

import os
import pickle
import re

import click
import numpy as np
import PIL.Image
import torch
import tqdm

import dnnlib
from torch_utils import distributed as dist
from torch_utils import model_registry

#----------------------------------------------------------------------------
# Random generator that allows specifying a different seed for each generated sample.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def compress_to_npz(folder_path, file_paths):
    """Stacks the given PNGs into <folder_path>.npz."""
    npz_path = f'{folder_path.rstrip(os.sep)}.npz'
    first = np.asarray(PIL.Image.open(file_paths[0]))
    samples = np.empty([len(file_paths), *first.shape], dtype=np.uint8)
    for i, path in enumerate(tqdm.tqdm(file_paths, desc=f'Compressing images to {npz_path}')):
        samples[i] = np.asarray(PIL.Image.open(path))
    np.savez(npz_path, arr_0=samples)
    print(f'Saved {len(file_paths)} images from {folder_path} to {npz_path}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--num', 'num_fid_samples',  help='Maximum number of images in the .npz, 0 = no .npz', metavar='INT', type=click.IntRange(min=0), default=50000, show_default=True)
@click.option('--sigma_G', 'sigma_G',      help='Generator noise level sigma_init', metavar='FLOAT',                type=float, default=2.5, show_default=True)
@click.option('--vae', 'vae_name',         help='SD-VAE used to decode EDM2 latents', metavar='NAME',               type=str, default='stabilityai/sd-vae-ft-mse', show_default=True)

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, num_fid_samples, sigma_G, vae_name, device=torch.device('cuda')):
    """Generate images with a one-step MGM generator G(z) = D(sigma_G z, sigma_G).

    Saves <outdir>/<seed>.png and stacks them into <outdir>.npz. EDM2 snapshots
    are decoded with the SD-VAE.

    Examples:

    \b
    # Generate 64 images and save them as out/*.png and out.npz
    python generate_onestep.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=<network_path>

    \b
    # Generate 50000 images using 8 GPUs
    torchrun --standalone --nproc_per_node=8 generate_onestep.py --outdir=out --seeds=0-49999 --batch=64 \\
        --network=<network_path>
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    if class_idx is not None and not (0 <= class_idx < net.label_dim):
        raise click.ClickException(f'--class={class_idx} is out of range for a network with label_dim={net.label_dim}')

    # EDM2 networks generate SD-VAE latents; decode them in chunks to bound memory.
    vae = None
    if model_registry.detect_codebase(net) == 'edm2':
        from edm2.vae_utils import load_sd_vae
        from training.unified_training_loop import edm2_vae_decode
        vae = load_sd_vae(vae_name, device=device)

    def decode_samples(images):
        if vae is not None:
            images = torch.cat([edm2_vae_decode(vae, chunk) for chunk in images.split(8)])
        else:
            images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8)
        return images.permute(0, 2, 3, 1).cpu().numpy()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        sigma = sigma_G * torch.ones([batch_size, 1, 1, 1], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
            if class_idx is not None:
                class_labels[:, :] = 0
                class_labels[:, class_idx] = 1

        with torch.no_grad():
            images = net(sigma_G * latents.to(torch.float64), sigma, class_labels).to(torch.float64)

        # Save images.
        images_np = decode_samples(images)
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    torch.distributed.barrier()
    if dist.get_rank() == 0 and num_fid_samples > 0:
        image_path = lambda seed: os.path.join(outdir, f'{seed-seed%1000:06d}' if subdirs else '', f'{seed:06d}.png')
        compress_to_npz(outdir, [image_path(seed) for seed in seeds[:num_fid_samples]])
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
