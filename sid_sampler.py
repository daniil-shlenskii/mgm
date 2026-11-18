"""One-step sampling for EDM and EDM2 MGM generators."""

import torch


def sid_sampler(net, *, latents, class_labels=None, init_sigma=2.5, **kwargs):
    sigma = torch.full((latents.shape[0], 1, 1, 1), init_sigma, device=latents.device)
    return net(latents * init_sigma, sigma, class_labels, **kwargs)
