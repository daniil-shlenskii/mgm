"""Train one-step MGM generators with the configurations and ablations of the paper."""

import json
import os
import pickle
import re

import click
import torch

import dnnlib
from torch_utils import distributed as dist, model_registry
from training.unified_training_loop import training_loop

#----------------------------------------------------------------------------
# Configurations of the runs behind the reported FIDs (Table 2): CIFAR-10 2.27, ImageNet-64 2.50,
# ImageNet-512 2.36. `ema_halflife_kimg=None` means a fixed per-iteration generator EMA decay of
# 0.999; otherwise an EMA half-life with EDM's 5% ramp-up. Both ImageNet runs added the noise to
# both interpolants, and the ImageNet-512 run used an EMA half-life of 2 Mimg and the EDM2-S 0.190
# teacher (Table 3 lists critic-only noise and decay 0.999 for all three).

PRESETS = {
    'cifar10': dict(
        codebase='edm', cond=False, shape=[3, 32, 32], batch=256, batch_gpu=None, noise=0.01, noise_on_generator=False,
        duration=20, checkpoint=None, ema_halflife_kimg=None,
        data_stat='https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz',
    ),
    'imagenet64': dict(
        codebase='edm', cond=True, shape=[3, 64, 64], batch=2048, batch_gpu=32, noise=0.2, noise_on_generator=True,
        duration=None, ema_halflife_kimg=None,
        checkpoint='https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl',
        data_stat='https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz',
    ),
    'imagenet512': dict(
        codebase='edm2', cond=True, shape=[8, 64, 64], batch=256, batch_gpu=32, noise=0.5, noise_on_generator=True,
        duration=None, ema_halflife_kimg=2000,
        checkpoint='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.190.pkl',
        data_stat='https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl',
    ),
}
FID_DETECTOR = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'

#----------------------------------------------------------------------------

def latest_state(path):
    """A training-state-*.pt file, or the latest one directly inside a run directory."""
    if os.path.isfile(path):
        return path
    candidates = [(int(m.group(1)), os.path.join(path, name)) for name in os.listdir(path)
                  if (m := re.fullmatch(r'training-state-(\d+)\.pt', name))]
    if not candidates:
        raise click.ClickException(f'No training-state-*.pt in {path}')
    return max(candidates)[1]

def check_checkpoint_codebase(path, expected):
    """Cross-checks a local checkpoint against the preset (remote URLs are not downloaded twice)."""
    if re.match(r'^[a-z]+://', path):
        return
    if not os.path.isfile(path):
        raise click.ClickException(f'--checkpoint {path} does not exist')
    with open(path, 'rb') as stream:
        data = pickle.load(stream)
    network = data.get('ema') if isinstance(data, dict) else None
    if network is None:
        raise click.ClickException(f'{path} has no "ema" network')
    if model_registry.detect_codebase(network) != expected:
        raise click.ClickException(f'{path} is not an {expected} network')

#----------------------------------------------------------------------------

@click.command()
@click.option('--preset',             help='Paper configuration', type=click.Choice(sorted(PRESETS)), required=True)
@click.option('--data',               help='EDM dataset (CIFAR-10, ImageNet-64) or EDM2 SD-VAE latent dataset (ImageNet-512)', metavar='ZIP|DIR', type=click.Path(exists=True), required=True)
@click.option('--outdir',             help='Run directory', metavar='DIR', type=click.Path(), required=True)
@click.option('--checkpoint',         help='Pretrained denoiser for both networks  [default: preset]', metavar='PKL|URL', type=str, default=None)
@click.option('--random-init',        help='Random initialization instead of a checkpoint (CIFAR-10 0K warmup ablation)', is_flag=True)
@click.option('--resume',             help='Resume from a training-state-*.pt file or the latest one in a run directory', metavar='PT|DIR', type=click.Path(exists=True), default=None)
@click.option('--data-stat',          help='FID reference statistics  [default: preset]', metavar='NPZ|PKL|URL', type=str, default=None)
@click.option('--duration',           help='MGM training duration  [default: preset; required for ImageNet]', metavar='MIMG', type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--batch',              help='Total batch size  [default: preset]', metavar='INT', type=click.IntRange(min=1), default=None)
@click.option('--batch-gpu',          help='Limit batch size per GPU (gradient accumulation)  [default: preset]', metavar='INT', type=click.IntRange(min=1), default=None)
@click.option('--noise',              help='Interpolant noise coefficient sigma, sigma_t^2 = sigma t (1 - t)  [default: preset]', metavar='FLOAT', type=click.FloatRange(min=0), default=None)
@click.option('--noise-on-generator', help='Noise on both interpolants (True) or the critic only (False)  [default: preset]', metavar='BOOL', type=bool, default=None)
@click.option('--t-min',              help='Lower bound of t ~ U[t_min, t_max]', metavar='FLOAT', type=click.FloatRange(min=0, max=0.5), default=0.0, show_default=True)
@click.option('--t-max',              help='Upper bound of t ~ U[t_min, t_max]; t_min = t_max = 0.5 is midpoint-only training', metavar='FLOAT', type=click.FloatRange(min=0, max=0.5), default=0.5, show_default=True)
@click.option('--interpolant',        help='Deterministic interpolant', type=click.Choice(['linear', 'trigonometric']), default='linear', show_default=True)
@click.option('--critic-steps-per-g', help='Critic updates per generator update', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--seed',               help='Random seed', metavar='INT', type=int, default=0, show_default=True)
@click.option('--workers',            help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--tick',               help='How often to print progress', metavar='KIMG', type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--snap',               help='How often to save snapshots and evaluate FID-50k (also at ticks 10-100 of a fresh run and at the end)', metavar='TICKS', type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',               help='How often to save the training state', metavar='TICKS', type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--dry-run',            help='Print training options and exit', is_flag=True)
def main(preset, data, outdir, checkpoint, random_init, resume, data_stat, duration, batch, batch_gpu,
         noise, noise_on_generator, t_min, t_max, interpolant, critic_steps_per_g,
         seed, workers, tick, snap, dump, dry_run):
    """Train a one-step MGM generator on CIFAR-10, ImageNet 64x64 or ImageNet 512x512.

    \b
    # CIFAR-10 from the 50K-kimg denoising warmup (warmup_cifar.py)
    torchrun --standalone --nproc_per_node=2 train.py --preset=cifar10 \\
        --data=datasets/cifar10-32x32.zip --outdir=runs/cifar10 \\
        --checkpoint=runs/warmup/network-snapshot-050000.pkl
    """
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    dist.init()

    paper = PRESETS[preset]
    codebase = paper['codebase']
    batch = paper['batch'] if batch is None else batch
    batch_gpu = paper['batch_gpu'] if batch_gpu is None else batch_gpu
    noise = paper['noise'] if noise is None else noise
    noise_on_generator = paper['noise_on_generator'] if noise_on_generator is None else noise_on_generator
    duration = paper['duration'] if duration is None else duration
    checkpoint_arg = checkpoint
    checkpoint = None if random_init else (checkpoint or paper['checkpoint'])

    # Validate the options.
    if t_min > t_max:
        raise click.ClickException('--t-min must not exceed --t-max')
    if duration is None:
        raise click.ClickException(f'--duration is required for --preset={preset}')
    if random_init and (preset != 'cifar10' or resume is not None or checkpoint_arg is not None):
        raise click.ClickException('--random-init is only for CIFAR-10 training from scratch (no --checkpoint or --resume)')
    if checkpoint is None and resume is None and not random_init:
        raise click.ClickException('CIFAR-10 needs --checkpoint (a warmup_cifar.py snapshot), --random-init or --resume')
    if batch % dist.get_world_size():
        raise click.ClickException(f'--batch={batch} is not divisible by the number of GPUs ({dist.get_world_size()})')
    per_gpu = batch // dist.get_world_size()
    if batch_gpu is not None and batch_gpu < per_gpu and per_gpu % batch_gpu:
        raise click.ClickException(f'--batch-gpu={batch_gpu} must divide the batch per GPU ({per_gpu})')
    if checkpoint is not None and resume is None:
        check_checkpoint_codebase(checkpoint, codebase)

    c = dnnlib.EasyDict()
    c.run_dir = os.path.abspath(outdir)
    c.dataset_kwargs = dnnlib.EasyDict(class_name=f'{codebase}.dataset.ImageFolderDataset', path=os.path.abspath(data),
                                       use_labels=paper['cond'], xflip=False, cache=True)
    dataset = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
    if dataset.image_shape != paper['shape']:
        raise click.ClickException(f'--preset={preset} expects dataset shape {paper["shape"]}, got {dataset.image_shape}')
    if paper['cond'] and not dataset.has_labels:
        raise click.ClickException(f'--preset={preset} requires class labels in dataset.json')
    c.dataset_kwargs.resolution = dataset.resolution
    c.dataset_kwargs.max_size = len(dataset)
    del dataset
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=workers, prefetch_factor=2)

    c.batch_size = batch
    c.batch_gpu = batch_gpu
    c.seed = seed
    c.total_kimg = max(int(duration * 1000), 1)
    c.kimg_per_tick = tick
    c.snapshot_ticks = snap
    c.state_dump_ticks = dump
    c.init_sigma = 2.5
    c.metrics = ['fid50k_full']
    c.data_stat = data_stat or paper['data_stat']
    c.detector_url = FID_DETECTOR
    c.loss_scaling = 1
    c.loss_scaling_G = 100
    c.ema_beta = 0.999
    c.ema_halflife_kimg = paper['ema_halflife_kimg']
    c.critic_steps_per_G = critic_steps_per_g
    c.save_best_and_last = (codebase == 'edm2')
    c.loss_kwargs = dnnlib.EasyDict(
        class_name='training.midpoint_loss.MidpointStochasticLoss',
        t_min=t_min, t_max=t_max, interp_noise_eps=noise,
        noise_on_generator=noise_on_generator, interpolant=interpolant,
    )
    c.critic_optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=1e-5, betas=[0.0, 0.999], eps=1e-8)
    c.g_optimizer_kwargs = dnnlib.EasyDict(c.critic_optimizer_kwargs)

    # Networks: the generator and the critic share the architecture of the pretrained denoiser.
    if codebase == 'edm':
        if preset == 'cifar10': # DDPM++
            arch = dnnlib.EasyDict(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard',
                                   channel_mult_noise=1, resample_filter=[1, 1], model_channels=128, channel_mult=[2, 2, 2])
        else: # ADM
            arch = dnnlib.EasyDict(model_type='DhariwalUNet', model_channels=192, channel_mult=[1, 2, 3, 4])
        arch.update(dropout=0.13, use_fp16=False)
        c.G_network_kwargs = dnnlib.EasyDict(arch, class_name='edm.networks.EDMPrecond')
        c.critic_network_kwargs = dnnlib.EasyDict(arch, class_name='edm.networks.MidpointCriticPrecond')
    else: # EDM2-S
        arch = dnnlib.EasyDict(model_channels=192, dropout=0.0, use_fp16=False)
        c.G_network_kwargs = dnnlib.EasyDict(arch, class_name='edm2.mgm_networks.EDM_2_Precond_Generator')
        c.critic_network_kwargs = dnnlib.EasyDict(arch, class_name='edm2.mgm_networks.EDM2_MidpointCriticPrecond')

    c.critic_init_pkl = checkpoint
    c.G_init_pkl = checkpoint
    if resume is not None:
        c.resume_training = os.path.abspath(latest_state(resume))
        match = re.fullmatch(r'training-state-(\d+)\.pt', os.path.basename(c.resume_training))
        if match is None:
            raise click.ClickException('--resume must be a training-state-<kimg>.pt file or a run directory')
        c.resume_kimg = int(match.group(1))
    elif os.path.isdir(c.run_dir):
        names = os.listdir(c.run_dir)
        if any(re.fullmatch(r'training-state-\d+\.pt', name) for name in names):
            raise click.ClickException(f'{c.run_dir} already contains a run; rerun the same command with --resume={c.run_dir} to continue it, or choose another --outdir')
        if any(re.fullmatch(r'network-snapshot-\d+\.pkl', name) for name in names):
            raise click.ClickException(f'{c.run_dir} holds snapshots of a run that never saved a training state; delete it or choose another --outdir')

    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    if dry_run:
        dist.print0('Dry run; exiting.')
        return
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'w') as stream:
            json.dump(c, stream, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    torch.distributed.barrier()
    training_loop(codebase=codebase, **c)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------
