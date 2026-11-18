# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""EDM2 U-Net and EDM preconditioning for the MGM generator, adapted from SiD/SiDA
("Adversarial Score Identity Distillation: Rapidly Surpassing the Teacher in One Step").

The network is EDM2's magnitude-preserving U-Net (parameter names unchanged, so NVIDIA
EDM2 checkpoints load directly), with EDM2's forced weight normalization applied by a
forward pre-hook of every MPConv in training mode (force_normalization, use_gan).
"""

import numpy as np
import torch
from torch_utils import persistence

from edm2.networks_modified import mp_silu, mp_sum, mp_cat, MPFourier, Block, MPConv

#----------------------------------------------------------------------------

@persistence.persistent_class
class UNet_EncoderDecoder(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        force_normalization,                # EDM2 forced weight normalization.
        use_gan,                            # Apply it through the MPConv forward pre-hook.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [16,8],       # List of resolutions with self-attention.
        label_balance       = 0.5,          # Balance between noise embedding (0) and class embedding (1).
        concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[], force_normalization=force_normalization, use_gan=use_gan)
        self.emb_label = MPConv(label_dim, cemb, kernel=[], force_normalization=force_normalization, use_gan=use_gan) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3], force_normalization=force_normalization, use_gan=use_gan)
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, force_normalization, use_gan, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, force_normalization, use_gan, flavor='enc', attention=(res in attn_resolutions), **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, force_normalization, use_gan, flavor='dec', attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, force_normalization, use_gan, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, force_normalization, use_gan, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, force_normalization, use_gan, flavor='dec', attention=(res in attn_resolutions), **block_kwargs)
        self.out_conv = MPConv(cout, img_channels, kernel=[3,3], force_normalization=force_normalization, use_gan=use_gan)

    def forward(self, x, noise_labels, class_labels):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x

#----------------------------------------------------------------------------
# EDM preconditioning, D(x, sigma) = c_skip x + c_out F(c_in x, c_noise).

@persistence.persistent_class
class EDM_2_Precond_Generator(torch.nn.Module):
    def __init__(self,
        img_resolution,         # Image resolution.
        img_channels,           # Image channels.
        label_dim,              # Class label dimensionality. 0 = unconditional.
        force_normalization,    # EDM2 forced weight normalization.
        use_gan,                # Apply it through the MPConv forward pre-hook.
        use_fp16        = False, # Run the model at FP16 precision?
        sigma_data      = 0.5,  # Expected standard deviation of the training data.
        **unet_kwargs,          # Keyword arguments for UNet_EncoderDecoder.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.unet = UNet_EncoderDecoder(img_resolution=img_resolution, img_channels=img_channels, label_dim=label_dim,
                                        force_normalization=force_normalization, use_gan=use_gan, **unet_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Run the model.
        F_x = self.unet((c_in * x).to(dtype), c_noise, class_labels)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

#----------------------------------------------------------------------------
