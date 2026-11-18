# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""Network architectures for Midpoint Generative Models (MGM) on EDM2.

The generator reuses the magnitude-preserving EDM2 network. The midpoint
critic predicts the displacement field along the interpolation path, using
sigma = 0.002 + t and the residual D(x, sigma) - x.
"""

import torch
from torch_utils import persistence

from edm2.sida_networks import (
    UNet_EncoderDecoder,
    EDM_2_Precond_Generator,
)

# Re-export so the rest of the MGM-EDM2 package has a single import surface.
__all__ = [
    'UNet_EncoderDecoder',
    'EDM_2_Precond_Generator',
    'EDM2_MidpointCriticPrecond',
]


@persistence.persistent_class
class EDM2_MidpointCriticPrecond(torch.nn.Module):
    """EDM2-S critic with the paper's shifted-time residual output."""

    def __init__(self, img_resolution, img_channels, label_dim,
                 force_normalization, use_gan, use_fp16=False, sigma_data=0.5,
                 **unet_kwargs):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.force_normalization = force_normalization
        self.use_gan = use_gan
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.unet = UNet_EncoderDecoder(
            img_resolution=img_resolution, img_channels=img_channels,
            label_dim=label_dim, force_normalization=force_normalization,
            use_gan=use_gan, **unet_kwargs,
        )

    def forward(self, x, t, class_labels=None, force_fp32=False):
        x = x.to(torch.float32)
        sigma = (0.002 + t).to(torch.float32).reshape(-1, 1, 1, 1)
        if self.label_dim:
            if class_labels is None:
                class_labels = torch.zeros([1, self.label_dim], device=x.device)
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)
        else:
            class_labels = None
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4
        f_x = self.unet((c_in * x).to(dtype), c_noise, class_labels)
        return c_skip * x + c_out * f_x.to(torch.float32) - x
