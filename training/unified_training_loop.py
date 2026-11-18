# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""MGM training loop shared by the EDM (pixel-space) and EDM2 (SD-VAE latent) backbones.

Each iteration takes critic_steps_per_G critic steps and one generator step, each on a
fresh data batch and fresh generator samples (Algorithm 1 of the paper). The only
codebase-specific parts are the VAE encode/decode for EDM2, the generator EMA schedule
(set by the caller) and EDM2's best+last checkpoint pruning.
"""

import contextlib
import copy
import json
import os
import pickle
import threading
import time

import numpy as np
import PIL.Image
import psutil
import torch

import dnnlib
from edm2 import misc as misc2
from edm2.vae_utils import load_sd_vae
from metrics.fid import calculate_fid
from sid_sampler import sid_sampler
from torch_utils import distributed as dist
from torch_utils import misc, training_stats

#----------------------------------------------------------------------------
# Preview grids.

def setup_snapshot_image_grid(training_set, random_seed=0, target_wh=(7680, 4320), grid_range=(7, 32)):
    """Picks a deterministic grid of dataset samples for the reals/fakes preview PNGs.
    The grid has target_wh // resolution tiles per side, clipped to grid_range. With
    labels, each row holds one class, cycling through the classes in a shuffled order."""
    rnd = np.random.RandomState(random_seed)
    target_w, target_h = target_wh
    lo, hi = grid_range
    gw = int(np.clip(target_w // training_set.image_shape[2], lo, hi))
    gh = int(np.clip(target_h // training_set.image_shape[1], lo, hi))

    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
    else:
        # Group dataset indices by label without touching the images (ImageNet is too
        # large to scan sample by sample, and its classes are stored in sorted order).
        raw_labels = training_set._get_raw_labels()[training_set._raw_idx]
        if raw_labels.ndim == 2: # one-hot
            raw_labels = raw_labels.argmax(axis=1)
        label_order = np.unique(raw_labels)
        rnd.shuffle(label_order)
        grid_indices = []
        for y in range(gh):
            indices = np.flatnonzero(raw_labels == label_order[y % len(label_order)])
            rnd.shuffle(indices)
            grid_indices += [int(indices[x % len(indices)]) for x in range(gw)]

    labels = np.stack([training_set.get_label(i) for i in grid_indices])
    return (gw, gh), labels, grid_indices

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def save_image_grid_in_background(img, fname, drange, grid_size):
    """save_image_grid from a background thread, so that a slow write on rank 0 does not
    stall the other ranks at their next collective. Join the returned thread before exiting."""
    thread = threading.Thread(target=save_image_grid, args=(img, fname, drange, grid_size), name=f'save_image_grid:{os.path.basename(fname)}')
    thread.start()
    return thread

def generate_grid_tiles(make_tiles, num_tiles, batch, device):
    """Splits the preview grid's tiles evenly across all ranks, runs make_tiles(idx) -> uint8 images
    on each rank's share in calls of at most `batch` tiles, and all-gathers the result, so every rank
    returns all num_tiles images in grid order. Each rank pads its share by repeating the last tile,
    so all ranks contribute the same shape to the all-gather."""
    world_size, rank = dist.get_world_size(), dist.get_rank()
    per_rank = -(-num_tiles // world_size)
    idx = torch.arange(rank * per_rank, (rank + 1) * per_rank).clamp(max=num_tiles - 1)
    tiles = torch.cat([make_tiles(chunk) for chunk in idx.split(batch)]).contiguous()
    gathered = torch.empty([world_size * per_rank, *tiles.shape[1:]], dtype=tiles.dtype, device=device)
    torch.distributed.all_gather_into_tensor(gathered, tiles)
    return gathered[:num_tiles]

#----------------------------------------------------------------------------
# Checkpoints.

def save_data(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def save_training_state(fname, critic, G, G_ema, critic_optimizer, g_optimizer):
    """Full training state: fp32 state_dicts of critic, G and G_ema plus both optimizer states."""
    torch.save(dict(
        critic_state=critic.state_dict(),
        G_state=G.state_dict(),
        G_ema_state=G_ema.state_dict(),
        critic_optimizer_state=critic_optimizer.state_dict(),
        g_optimizer_state=g_optimizer.state_dict(),
    ), fname)

def load_training_state(fname, critic, G, G_ema, critic_optimizer, g_optimizer):
    """Loads a training state saved by save_training_state, staggering disk reads (rank 0 first,
    then everyone else). Extra keys written by older versions of the code are ignored."""
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    data = torch.load(fname, map_location=torch.device('cpu'), weights_only=False)
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    critic.load_state_dict({k: v.float() for k, v in data['critic_state'].items()})
    G.load_state_dict({k: v.float() for k, v in data['G_state'].items()})
    G_ema.load_state_dict({k: v.float() for k, v in data['G_ema_state'].items()})
    critic_optimizer.load_state_dict(data['critic_optimizer_state'])
    g_optimizer.load_state_dict(data['g_optimizer_state'])
    del data
    torch.distributed.barrier()

def load_module_from_pkl(module, pkl_path, name='module'):
    """Initializes `module` from the 'ema' network of a pickle. Every parameter and buffer of
    `module` must exist in the pickled network with the same shape."""
    dist.print0(f'Loading {name} weights from "{pkl_path}"...')
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    with dnnlib.util.open_url(pkl_path, verbose=(dist.get_rank() == 0)) as f:
        data = pickle.load(f)
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    misc.copy_params_and_buffers(src_module=data['ema'], dst_module=module, require_all=True)
    del data

def append_line(jsonl_line, fname):
    with open(fname, 'at') as f:
        f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------
# EDM2 SD-VAE latents (EDM2's normalization constants for the 8-channel mean/std latents).

vae_raw_mean = np.array([5.81, 3.25, 0.12, -2.15], dtype=np.float32)
vae_raw_std = np.array([4.17, 4.62, 3.71, 3.28], dtype=np.float32)
vae_scale = np.float32(0.5) / vae_raw_std
vae_bias = np.float32(0) - vae_raw_mean * vae_scale

def edm2_vae_encode_latents(x):  # raw latents (mean, std) => final latents
    scale = torch.tensor(vae_scale, dtype=torch.float32, device=x.device)
    bias = torch.tensor(vae_bias, dtype=torch.float32, device=x.device)
    mean, std = x.to(torch.float32).chunk(2, dim=1)
    x = mean + torch.randn_like(mean) * std
    x = x * scale.reshape(1, -1, 1, 1)
    x = x + bias.reshape(1, -1, 1, 1)
    return x

def edm2_vae_decode(vae, x):  # final latents => raw pixels (uint8)
    scale = torch.as_tensor(vae_scale, dtype=torch.float32, device=x.device)
    bias = torch.as_tensor(vae_bias, dtype=torch.float32, device=x.device)
    x = (x.to(torch.float32) - bias.view(1, -1, 1, 1)) / scale.view(1, -1, 1, 1)
    with torch.no_grad():
        x = vae.decode(x).sample
    # EDM2 encodes pixels in [0, 1], so the decoder output is in [0, 1] too.
    return (x.clamp(0, 1) * 255).to(torch.uint8)

#----------------------------------------------------------------------------

@contextlib.contextmanager
def _fork_numpy_and_torch_rng(device):
    """Restores the numpy and torch RNG states on exit, so that preview images do not shift
    the training randomness."""
    numpy_state = np.random.get_state()
    cuda_devices = []
    if device.type == 'cuda':
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield
    finally:
        np.random.set_state(numpy_state)

def _restore_best_fid(run_dir):
    """Best FID and its snapshot from metric-fid50k_full.jsonl, for best+last pruning after a resume."""
    best_fid, best_pkl = float('inf'), None
    fname = os.path.join(run_dir, 'metric-fid50k_full.jsonl')
    if os.path.isfile(fname):
        with open(fname) as f:
            for line in f:
                entry = json.loads(line)
                pkl = os.path.join(run_dir, entry['snapshot_pkl'])
                if entry['fid50k_full'] < best_fid and os.path.isfile(pkl):
                    best_fid, best_pkl = entry['fid50k_full'], pkl
    return best_fid, best_pkl

#----------------------------------------------------------------------------

def training_loop(
    codebase,                           # 'edm' (pixel space) or 'edm2' (SD-VAE latents)
    run_dir,                            # Output directory.
    dataset_kwargs,                     # Options for the training set.
    data_loader_kwargs,                 # Options for torch.utils.data.DataLoader.
    G_network_kwargs,                   # Options for the generator network.
    critic_network_kwargs,              # Options for the critic network.
    loss_kwargs,                        # Options for training.midpoint_loss.MidpointStochasticLoss.
    critic_optimizer_kwargs,            # Options for the critic optimizer.
    g_optimizer_kwargs,                 # Options for the generator optimizer.
    seed                = 0,            # Global random seed.
    batch_size          = 512,          # Total batch size for one training iteration.
    batch_gpu           = None,         # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,       # Training duration, measured in thousands of training images.
    ema_beta            = 0.999,        # Fixed per-iteration generator EMA decay (ema_halflife_kimg=None).
    ema_halflife_kimg   = None,         # Generator EMA half-life instead of a fixed decay, None = fixed.
    ema_rampup_ratio    = 0.05,         # EMA ramp-up coefficient for ema_halflife_kimg, None = no ramp-up.
    loss_scaling        = 1,            # Loss scaling factor of the critic loss.
    loss_scaling_G      = 100,          # Loss scaling factor of the generator loss.
    critic_steps_per_G  = 1,            # Critic updates per generator update.
    init_sigma          = 2.5,          # Noise level of the one-step generator G(z) = D(init_sigma z, init_sigma).
    kimg_per_tick       = 10,           # Interval of progress prints.
    snapshot_ticks      = 50,           # How often to save network snapshots and evaluate FID, None = disable.
    state_dump_ticks    = 500,          # How often to dump the training state, None = disable.
    critic_init_pkl     = None,         # Initialize the critic from this pickle, None = random init.
    G_init_pkl          = None,         # Initialize the generator from this pickle, None = random init.
    resume_training     = None,         # Resume from this training-state-*.pt file.
    resume_kimg         = 0,            # kimg of resume_training.
    metrics             = ('fid50k_full',), # Metrics to evaluate at snapshots ('fid50k_full' or none).
    data_stat           = None,         # FID reference statistics (.npz or .pkl, path or URL).
    detector_url        = None,         # Inception network for FID.
    vae_name            = 'stabilityai/sd-vae-ft-mse', # [edm2] SD-VAE used to decode latents.
    force_normalization = True,         # [edm2] EDM2 forced weight normalization.
    use_gan             = True,         # [edm2] Apply the forced weight normalization through a pre-hook.
    save_best_and_last  = False,        # Keep only the best-FID snapshot and the latest training state.
    cudnn_benchmark     = True,         # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    assert codebase in ('edm', 'edm2'), f'Unknown codebase "{codebase}"'
    is_edm2 = (codebase == 'edm2')
    metrics = list(metrics or [])
    assert set(metrics) <= {'fid50k_full'}, f'Unsupported metrics: {metrics}'
    assert metrics or not save_best_and_last, 'save_best_and_last selects snapshots by FID'

    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    cur_nimg = resume_kimg * 1000

    # Load dataset. Every iteration draws critic_steps_per_G + 1 batches (critic steps plus
    # the generator step), so a resumed run continues the sampler where it stopped.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataset_sampler = misc2.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=cur_nimg * (critic_steps_per_G + 1))
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Data samples -> network space: SD-VAE latents for EDM2, [-1, 1] pixels for EDM.
    if is_edm2:
        vae = load_sd_vae(vae_name, device=device)
        encode_fn = edm2_vae_encode_latents
        decode_fn = lambda x: edm2_vae_decode(vae, x)
    else:
        vae = None
        encode_fn = lambda x: x.to(torch.float32) / 127.5 - 1
        decode_fn = lambda x: (x.to(torch.float32) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    # Construct networks. The interface follows an actually encoded sample, so that it is
    # correct for VAE latents as well as for pixels.
    dist.print0('Constructing networks...')
    ref_image, ref_label = dataset_obj[0]
    ref_image = encode_fn(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
    if is_edm2:
        interface_kwargs.update(force_normalization=force_normalization, use_gan=use_gan)
    del ref_image, ref_label

    critic = dnnlib.util.construct_class_by_name(**critic_network_kwargs, **interface_kwargs) # f_psi
    critic.train().requires_grad_(True).to(device)
    G = dnnlib.util.construct_class_by_name(**G_network_kwargs, **interface_kwargs) # G_theta
    G.train().requires_grad_(True).to(device)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, G.img_channels, G.img_resolution, G.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, G.label_dim], device=device)
            misc.print_module_summary(critic, [images, sigma, labels], max_nesting=2)
            misc.print_module_summary(G, [images, sigma, labels], max_nesting=2)

    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    critic_optimizer = dnnlib.util.construct_class_by_name(params=critic.parameters(), **critic_optimizer_kwargs)
    g_optimizer = dnnlib.util.construct_class_by_name(params=G.parameters(), **g_optimizer_kwargs)

    # Initialize both networks from a pretrained denoiser, unless resuming a training state.
    if resume_training is None:
        if critic_init_pkl is not None:
            load_module_from_pkl(critic, critic_init_pkl, name='critic')
        if G_init_pkl is not None:
            load_module_from_pkl(G, G_init_pkl, name='generator')
        if critic_init_pkl is None and G_init_pkl is None:
            # Random init (0K warmup ablation): both networks start from the same draw, as
            # psi <- eta and theta <- eta with a random eta.
            misc.copy_params_and_buffers(src_module=G, dst_module=critic, require_all=True)
    else:
        dist.print0(f'Loading training state from "{resume_training}"...')
        load_training_state(resume_training, critic, G, G_ema, critic_optimizer, g_optimizer)

    # Wrap with DDP. This broadcasts rank 0's critic and G to all ranks.
    dist.print0('Setting up DDP...')
    critic_ddp = torch.nn.parallel.DistributedDataParallel(critic, device_ids=[device], broadcast_buffers=False, find_unused_parameters=False)
    G_ddp = torch.nn.parallel.DistributedDataParallel(G, device_ids=[device], broadcast_buffers=False, find_unused_parameters=False)
    critic_ddp.eval().requires_grad_(False)
    G_ddp.eval().requires_grad_(False)
    if resume_training is None:
        misc.copy_params_and_buffers(src_module=G, dst_module=G_ema, require_all=True)

    # Preview grid with fixed latents, so that snapshots are comparable. Rank 0 lays out the grid,
    # every rank generates its share of the tiles, and rank 0 writes the PNGs in the background.
    grid_gen_batch = 1 if is_edm2 else batch_gpu # EDM2 decodes 512x512 images, one at a time.
    grid_z = None
    grid_meta = [None, None, None] # grid_size, grid_indices, grid_labels
    if dist.get_rank() == 0:
        original_seed = torch.initial_seed()
        torch.manual_seed(2024)
        grid_wh, grid_range = ((512, 512), (2, 8)) if is_edm2 else ((7680, 4320), (7, 32))
        grid_size, grid_labels, grid_indices = setup_snapshot_image_grid(training_set=dataset_obj, target_wh=grid_wh, grid_range=grid_range)
        grid_z = torch.randn([grid_labels.shape[0], G_ema.img_channels, G_ema.img_resolution, G_ema.img_resolution], device=device)
        torch.manual_seed(original_seed)
        grid_meta = [grid_size, grid_indices, grid_labels]
    torch.distributed.broadcast_object_list(grid_meta, src=0)
    grid_size, grid_indices, grid_labels = grid_meta
    if grid_z is None:
        grid_z = torch.empty([grid_labels.shape[0], G_ema.img_channels, G_ema.img_resolution, G_ema.img_resolution], device=device)
    torch.distributed.broadcast(grid_z, src=0)
    grid_c = torch.from_numpy(grid_labels).to(device)
    grid_save_threads = []

    def generate_grid_fakes():
        make_tiles = lambda idx: decode_fn(sid_sampler(G_ema, latents=grid_z[idx], class_labels=grid_c[idx], init_sigma=init_sigma))
        with torch.no_grad(), _fork_numpy_and_torch_rng(device):
            return generate_grid_tiles(make_tiles, num_tiles=len(grid_indices), batch=grid_gen_batch, device=device)

    if resume_training is None:
        dist.print0('Exporting sample images...')
        make_tiles = lambda idx: decode_fn(encode_fn(torch.stack([torch.as_tensor(dataset_obj[grid_indices[i]][0]) for i in idx.tolist()]).to(device)))
        with torch.no_grad(), _fork_numpy_and_torch_rng(device):
            reals = generate_grid_tiles(make_tiles, num_tiles=len(grid_indices), batch=grid_gen_batch, device=device)
        if dist.get_rank() == 0:
            grid_save_threads.append(save_image_grid_in_background(reals.cpu().numpy(), os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size))
        del reals
        fakes = generate_grid_fakes()
        if dist.get_rank() == 0:
            grid_save_threads.append(save_image_grid_in_background(fakes.cpu().numpy(), os.path.join(run_dir, 'fakes_init.png'), drange=[0, 255], grid_size=grid_size))
        del fakes

    torch.distributed.barrier()

    # Save an initial snapshot.
    data = dict(ema=copy.deepcopy(G_ema).eval().requires_grad_(False).cpu())
    if dist.get_rank() == 0:
        save_data(data=data, fname=os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'))
    del data

    # Best+last pruning state (rank 0).
    best_fid, previous_best_pkl = _restore_best_fid(run_dir) if resume_training is not None else (float('inf'), None)
    previous_pt = None
    if resume_training is not None and os.path.dirname(os.path.abspath(resume_training)) == os.path.abspath(run_dir):
        previous_pt = os.path.abspath(resume_training) # pruned by the next state dump with save_best_and_last

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time

    while True:
        # ------------------------------------------------------------------
        # Update the critic f_psi on samples from the frozen generator.
        for _critic_step in range(critic_steps_per_G):
            critic_ddp.train().requires_grad_(True)
            G_ddp.eval().requires_grad_(False)
            critic_optimizer.zero_grad(set_to_none=True)

            for round_idx in range(num_accumulation_rounds):
                real_images, labels = next(dataset_iterator)
                with torch.no_grad():
                    y_pos = encode_fn(real_images.to(device))
                labels = labels.to(device)
                z = torch.randn_like(y_pos)

                with misc.ddp_sync(G_ddp, False):
                    with torch.no_grad():
                        y_neg = sid_sampler(G_ddp, latents=z, class_labels=labels, init_sigma=init_sigma)

                with misc.ddp_sync(critic_ddp, (round_idx == num_accumulation_rounds - 1)):
                    loss, info = loss_fn(critic=critic_ddp, images_pos=y_pos, images_neg=y_neg, labels=labels)
                    loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                training_stats.report('Loss/critic', info['critic_loss'].mean().item())

            critic_ddp.eval().requires_grad_(False)
            for param in critic.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            critic_optimizer.step()

        # ------------------------------------------------------------------
        # Update the generator G_theta through the frozen critic.
        G_ddp.train().requires_grad_(True)
        critic_ddp.eval().requires_grad_(False)
        g_optimizer.zero_grad(set_to_none=True)

        for round_idx in range(num_accumulation_rounds):
            real_images, labels = next(dataset_iterator)
            with torch.no_grad():
                y_pos = encode_fn(real_images.to(device))
            labels = labels.to(device)
            z = torch.randn_like(y_pos)

            with misc.ddp_sync(G_ddp, (round_idx == num_accumulation_rounds - 1)):
                y_neg = sid_sampler(G_ddp, latents=z, class_labels=labels, init_sigma=init_sigma)
                with misc.ddp_sync(critic_ddp, False):
                    loss, info = loss_fn.generator_loss(critic=critic_ddp, images_pos=y_pos, images_neg=y_neg, labels=labels)
                loss.sum().mul(loss_scaling_G / batch_gpu_total).backward()

            term1 = info['term1'].mean().item()
            term2 = info['term2'].mean().item()
            training_stats.report('Loss/generator', term1 - term2)
            training_stats.report('Loss/generator_term1', term1)
            training_stats.report('Loss/generator_term2', term2)

        G_ddp.eval().requires_grad_(False)
        for param in G.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        g_optimizer.step()

        # ------------------------------------------------------------------
        # Update the generator EMA.
        with torch.no_grad():
            if ema_halflife_kimg is None:
                for p_ema, p_G in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p_G.detach().lerp(p_ema, ema_beta))
            else:
                ema_halflife_nimg = ema_halflife_kimg * 1000
                if ema_rampup_ratio is not None:
                    ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
                beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
                for p_ema, p_G in zip(G_ema.parameters(), G.parameters()):
                    p_ema.lerp_(p_G.detach(), 1 - beta)
            for b_ema, b_G in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b_G.detach())

        # ------------------------------------------------------------------
        # Maintenance once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Snapshot: preview grid, FID-50k and the EMA network pickle. Besides every
        # snapshot_ticks ticks and the end of training, a fresh run also takes snapshots
        # at ticks 10, 20, ..., 100.
        early_tick = (resume_training is None and cur_tick in range(10, 101, 10))
        if (snapshot_ticks is not None) and (cur_tick != 0) and (done or cur_tick % snapshot_ticks == 0 or early_tick):
            dist.print0('Exporting sample images...')
            fakes = generate_grid_fakes()
            if dist.get_rank() == 0:
                grid_save_threads = [thread for thread in grid_save_threads if thread.is_alive()]
                grid_save_threads.append(save_image_grid_in_background(fakes.cpu().numpy(), os.path.join(run_dir, f'fakes_{cur_nimg//1000:06d}.png'), drange=[0, 255], grid_size=grid_size))
            del fakes

            snapshot_pkl = f'network-snapshot-{cur_nimg//1000:06d}.pkl'
            current_fid = float('inf')
            if 'fid50k_full' in metrics:
                dist.print0('Evaluating fid50k_full...')
                fid = calculate_fid(
                    G=G_ema, dataset_kwargs=dataset_kwargs, init_sigma=init_sigma,
                    device=device, data_stat=data_stat, detector_url=detector_url,
                    vae=vae, scale=vae_scale, bias=vae_bias, batch_size=1 if is_edm2 else 64,
                )
                if dist.get_rank() == 0:
                    current_fid = fid
                    dist.print0(f'fid50k_full {fid:g}')
                    append_line(json.dumps(dict(fid50k_full=fid, kimg=cur_nimg / 1000, snapshot_pkl=snapshot_pkl,
                                                timestamp=time.time())),
                                os.path.join(run_dir, 'metric-fid50k_full.jsonl'))

            data = dict(ema=copy.deepcopy(G_ema).eval().requires_grad_(False).cpu())
            if dist.get_rank() == 0:
                if not save_best_and_last:
                    save_data(data=data, fname=os.path.join(run_dir, snapshot_pkl))
                elif current_fid < best_fid:
                    best_fid = current_fid
                    new_best = os.path.join(run_dir, snapshot_pkl)
                    save_data(data=data, fname=new_best)
                    if previous_best_pkl is not None and previous_best_pkl != new_best and os.path.exists(previous_best_pkl):
                        os.remove(previous_best_pkl)
                    previous_best_pkl = new_best
            del data

        # Full training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            new_pt = os.path.abspath(os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
            dist.print0(f'Saving {os.path.basename(new_pt)}')
            save_training_state(fname=new_pt, critic=critic, G=G, G_ema=G_ema, critic_optimizer=critic_optimizer, g_optimizer=g_optimizer)
            if save_best_and_last and previous_pt is not None and previous_pt != new_pt and os.path.exists(previous_pt):
                os.remove(previous_pt)
            previous_pt = new_pt

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            append_line(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())), os.path.join(run_dir, 'stats.jsonl'))

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    for thread in grid_save_threads:
        thread.join()
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
