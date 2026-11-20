# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Adapted from NVlabs/edm training for MGM; see THIRD_PARTY_LICENSES.md.
"""CIFAR-10 EDM denoising warmup with the folded noise distribution of Appendix C.3.

EDM's CIFAR-10 DDPM++ recipe (batch 512, Adam lr 1e-3 with a 10 Mimg ramp-up, EMA
half-life 0.5 Mimg with a 0.05 ramp-up, dropout 0.13, augmentation p=0.12), except
that log sigma ~ N(-1.2, 1.2^2) is reflected around log(2.5).
"""

import copy
import json
import math
import os
import pickle
import re

import click
import numpy as np
import torch

from edm.augment import AugmentPipe
from edm.dataset import ImageFolderDataset
from edm.networks import EDMPrecond
from edm2 import misc as misc2
from torch_utils import distributed as dist, misc

GLOBAL_BATCH = 512
MILESTONES_KIMG = (10000, 20000, 50000, 100000, 200000) # Table 1a


def folded_edm_loss(net, images, labels, augment_pipe):
    """Standard EDM loss, reflecting log sigma above log(2.5)."""
    log_sigma = torch.randn(images.shape[0], 1, 1, 1, device=images.device) * 1.2 - 1.2
    ceiling = math.log(2.5)
    sigma = torch.exp(torch.where(log_sigma > ceiling, 2 * ceiling - log_sigma, log_sigma))
    weight = (sigma.square() + 0.5 ** 2) / (sigma * 0.5).square()
    augmented, augment_labels = augment_pipe(images)
    noisy = augmented + torch.randn_like(augmented) * sigma
    pred = net(noisy, sigma, labels, augment_labels=augment_labels)
    return weight * (pred - augmented).square()


def latest_warmup_state(path):
    if os.path.isfile(path):
        return path
    candidates = [(int(m.group(1)), os.path.join(path, name)) for name in os.listdir(path)
                  if (m := re.fullmatch(r'warmup-state-(\d+)\.pt', name))]
    if not candidates:
        raise click.ClickException(f'No warmup-state-*.pt in {path}')
    return max(candidates)[1]


@click.command()
@click.option('--data',          help='CIFAR-10 dataset (dataset_tool.py output)', metavar='ZIP|DIR', type=click.Path(exists=True), required=True)
@click.option('--outdir',        help='Where to save the snapshots', metavar='DIR', type=click.Path(), required=True)
@click.option('--duration-kimg', help='Warmup duration', metavar='KIMG', type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT', type=click.IntRange(min=1), default=None)
@click.option('--seed',          help='Random seed', metavar='INT', type=int, default=0, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--dump-kimg',     help='How often to save a resumable warmup state', metavar='KIMG', type=click.IntRange(min=1), default=5000, show_default=True)
@click.option('--resume',        help='Resume from a warmup-state-*.pt file or the latest one in a directory', metavar='PT|DIR', type=click.Path(exists=True), default=None)
def main(data, outdir, duration_kimg, batch_gpu, seed, workers, dump_kimg, resume):
    """Denoising warmup for CIFAR-10 MGM (Appendix C.3).

    Saves EMA snapshots network-snapshot-<kimg>.pkl at the Table 1a milestones
    (10K, 20K, 50K, 100K and 200K kimg) up to --duration-kimg, and at the end.
    The CIFAR-10 MGM configuration initializes from network-snapshot-050000.pkl.
    """
    torch.multiprocessing.set_start_method('spawn', force=True)
    dist.init()
    device = torch.device('cuda', dist.get_local_rank())
    world = dist.get_world_size()
    if GLOBAL_BATCH % world:
        raise click.ClickException(f'Number of GPUs must divide batch size {GLOBAL_BATCH}')
    per_rank = GLOBAL_BATCH // world
    batch_gpu = min(batch_gpu or per_rank, per_rank)
    if per_rank % batch_gpu:
        raise click.ClickException('--batch-gpu must divide the batch per GPU')
    rounds = per_rank // batch_gpu
    np.random.seed((seed * world + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    state = None
    processed = 0
    if resume is not None:
        resume = latest_warmup_state(resume)
        state = torch.load(resume, map_location='cpu', weights_only=False)
        processed = state['processed']
        dist.print0(f'Resuming from {resume} at {processed // 1000} kimg')

    dataset = ImageFolderDataset(path=os.path.abspath(data), use_labels=False, xflip=False, cache=True)
    if dataset.image_shape != [3, 32, 32]:
        raise click.ClickException(f'Expected CIFAR-10 RGB 32x32, got {dataset.image_shape}')
    sampler = misc2.InfiniteSampler(dataset=dataset, rank=dist.get_rank(), num_replicas=world, seed=seed, start_idx=processed)
    loader = iter(torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=batch_gpu, pin_memory=True,
        num_workers=workers, prefetch_factor=2,
    ))
    net = EDMPrecond(
        img_resolution=32, img_channels=3, label_dim=0, model_type='SongUNet',
        embedding_type='positional', encoder_type='standard', decoder_type='standard',
        channel_mult_noise=1, resample_filter=[1, 1], model_channels=128,
        channel_mult=[2, 2, 2], dropout=0.13, use_fp16=False, augment_dim=9,
    ).train().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    if state is not None:
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False) # after DDP: rank 0's initial weights on every rank
    if state is not None:
        ema.load_state_dict(state['ema'])
    augment = AugmentPipe(p=0.12, xflip=1e8, yflip=1, scale=1,
                          rotate_frac=1, aniso=1, translate_frac=1)
    del state

    if dist.get_rank() == 0:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'warmup_options.json'), 'w') as stream:
            json.dump(dict(data=os.path.abspath(data), duration_kimg=duration_kimg,
                           global_batch=GLOBAL_BATCH, batch_gpu=batch_gpu, seed=seed,
                           edm_augment=0.12, folded_sigma_max=2.5, resume=resume), stream, indent=2)
    torch.distributed.barrier()

    milestones = {kimg for kimg in MILESTONES_KIMG if kimg <= duration_kimg} | {duration_kimg}
    saved = {kimg for kimg in milestones if os.path.isfile(os.path.join(outdir, f'network-snapshot-{kimg:06d}.pkl'))}
    next_report = (processed // 1000 // 1000 + 1) * 1000
    next_dump = (processed // 1000 // dump_kimg + 1) * dump_kimg
    previous_state = None
    if resume is not None and os.path.dirname(os.path.abspath(resume)) == os.path.abspath(outdir):
        previous_state = os.path.abspath(resume) # pruned by the next state dump
    while processed < duration_kimg * 1000:
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(rounds):
            images, labels = next(loader)
            images = images.to(device).to(torch.float32) / 127.5 - 1
            labels = labels.to(device)
            with misc.ddp_sync(ddp, round_idx == rounds - 1):
                loss = folded_edm_loss(ddp, images, labels, augment)
                loss.sum().mul(1 / per_rank).backward()
        lr_factor = min(processed / (10000 * 1000), 1.0)
        for group in optimizer.param_groups:
            group['lr'] = 1e-3 * lr_factor
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()
        half_life = min(500 * 1000, processed * 0.05)
        ema_beta = 0.5 ** (GLOBAL_BATCH / max(half_life, 1e-8))
        with torch.no_grad():
            for p_ema, p_net in zip(ema.parameters(), net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
        processed += GLOBAL_BATCH
        kimg = processed // 1000
        done = processed >= duration_kimg * 1000
        if kimg >= next_report:
            dist.print0(f'Warmup: {kimg}/{duration_kimg} kimg, loss {loss.mean().item():.5f}')
            next_report += 1000
        if (kimg in milestones or done) and kimg not in saved:
            saved.add(kimg)
            if dist.get_rank() == 0:
                name = os.path.join(outdir, f'network-snapshot-{kimg:06d}.pkl')
                with open(name, 'wb') as stream:
                    pickle.dump(dict(ema=copy.deepcopy(ema).cpu().eval()), stream)
                dist.print0(f'Saved {name}')
            torch.distributed.barrier()
        if kimg >= next_dump or done:
            next_dump = (kimg // dump_kimg + 1) * dump_kimg
            if dist.get_rank() == 0:
                name = os.path.join(outdir, f'warmup-state-{kimg:06d}.pt')
                torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                                processed=processed), name)
                if previous_state is not None and os.path.exists(previous_state):
                    os.remove(previous_state)
                previous_state = name
            torch.distributed.barrier()


if __name__ == '__main__':
    main()
