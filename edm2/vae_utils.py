"""Load the frozen Stable Diffusion VAE used by EDM2-S MGM."""

from diffusers import AutoencoderKL


def load_sd_vae(pretrained_vae_model_name_or_path, device):
    vae = AutoencoderKL.from_pretrained(pretrained_vae_model_name_or_path)
    return vae.eval().requires_grad_(False).to(device)
