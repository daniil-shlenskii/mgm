"""Midpoint Generative Models objective shared by EDM and EDM2."""

import math

import torch

from torch_utils import persistence


@persistence.persistent_class
class MidpointStochasticLoss:
    """Critic and generator losses of MGM (Algorithm 1 / 2 of the paper).

    images_pos is a data batch x_1 and images_neg a generator batch x_0. With
    t ~ U[t_min, t_max] and b ~ Bernoulli(1/2), the critic sees the flipped
    interpolant x~ = b x_t + (1 - b) x_{1-t}, where
    x_t = I_t(x_0, x_1) + sqrt(interp_noise_eps * t (1 - t)) * eps.
    The critic regresses x_1 - x_0; the generator minimizes
    2 <f(x~, t), x_1 - x_0> - |f(x~, t)|^2.
    The critic interpolant always carries the noise; the generator interpolant
    carries it only with noise_on_generator=True (critic-only noise otherwise).
    """

    def __init__(self, t_min=0.0, t_max=0.5, interp_noise_eps=0.0,
                 noise_on_generator=False, interpolant='linear'):
        if not 0 <= t_min <= t_max <= 0.5:
            raise ValueError('Expected 0 <= t_min <= t_max <= 0.5')
        if interp_noise_eps < 0:
            raise ValueError('Interpolation noise must be nonnegative')
        if interpolant not in ('linear', 'trigonometric'):
            raise ValueError(f'Unknown interpolant: {interpolant}')
        self.t_min = t_min
        self.t_max = t_max
        self.interp_noise_eps = interp_noise_eps
        self.noise_on_generator = noise_on_generator
        self.interpolant = interpolant

    def sample_xtb(self, y_pos, y_neg, noisy):
        """Returns t, the flipped interpolant x~ and the target y_pos - y_neg."""
        t = self.t_min + torch.rand(
            y_pos.shape[0], 1, 1, 1, device=y_pos.device, dtype=y_pos.dtype
        ) * (self.t_max - self.t_min)
        if self.interpolant == 'linear':
            w_pos = t
            w_neg = 1 - t
        else:  # trigonometric, Appendix D.1
            w_pos = torch.sin(math.pi * t / 2)
            w_neg = torch.cos(math.pi * t / 2)
        x_t = w_neg * y_neg + w_pos * y_pos
        x_1mt = w_pos * y_neg + w_neg * y_pos
        if noisy and self.interp_noise_eps > 0:
            # sigma_t = sigma_{1-t}, and only one of the two branches is kept per
            # sample, so a single noise draw serves both (Algorithm 2).
            noise = torch.randn_like(x_t) * (self.interp_noise_eps * t * (1 - t)).sqrt()
            x_t = x_t + noise
            x_1mt = x_1mt + noise
        b = (torch.rand(y_pos.shape[0], 1, 1, 1, device=y_pos.device) < 0.5).to(y_pos.dtype)
        return t, b * x_t + (1 - b) * x_1mt, y_pos - y_neg

    def __call__(self, critic, images_pos, images_neg, labels=None):
        """Critic loss; images_neg is detached."""
        t, x_tb, diff = self.sample_xtb(images_pos, images_neg.detach(), noisy=True)
        pred = critic(x_tb, t, labels)
        loss = (pred - diff.detach()).square()
        return loss, {'critic_loss': loss}

    def generator_loss(self, critic, images_pos, images_neg, labels=None):
        t, x_tb, diff = self.sample_xtb(images_pos, images_neg, noisy=self.noise_on_generator)
        pred = critic(x_tb, t, labels)
        term1 = 2 * diff * pred
        term2 = pred.square()
        return term1 - term2, {'term1': term1, 'term2': term2}
