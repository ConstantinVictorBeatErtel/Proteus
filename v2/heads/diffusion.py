"""DDPM-style Diffusion-Policy head configured for IDM (h=1, obs=2).

This is an in-tree minimal implementation that mirrors the
configuration that ``lerobot.policies.diffusion`` would produce when
asked for ``horizon=1, n_obs_steps=2, n_action_steps=1``. We keep it
self-contained so the ``v2`` package runs even when the optional
``lerobot`` install is unavailable.

The model conditions a tiny noise predictor on ``concat(obs_t,
obs_next)`` and a sinusoidal embedding of the diffusion timestep, then
denoises a single 7-D action under a cosine-schedule DDPM. At inference
we run ``n_inference_steps`` reverse steps; for training the loss is
the standard noise-prediction MSE.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_betas(num_steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, num_steps, num_steps + 1) / num_steps
    a_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    a_bar = a_bar / a_bar[0]
    betas = 1.0 - (a_bar[1:] / a_bar[:-1])
    return betas.clamp(1e-5, 0.999)


class _SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class DiffusionPolicyIDM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 7,
        n_obs_steps: int = 2,
        hidden_dim: int = 256,
        time_emb_dim: int = 128,
        n_train_timesteps: int = 100,
        n_inference_steps: int = 16,
        clip_action_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_obs_steps = int(n_obs_steps)
        self.n_train_timesteps = int(n_train_timesteps)
        self.n_inference_steps = int(n_inference_steps)
        self.clip_action_range = clip_action_range

        self.time_emb = _SinusoidalTimeEmb(time_emb_dim)
        cond_dim = self.obs_dim * self.n_obs_steps + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(self.action_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

        betas = _cosine_betas(self.n_train_timesteps)
        alphas = 1.0 - betas
        a_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bar", a_bar, persistent=False)
        self.register_buffer("sqrt_ab", torch.sqrt(a_bar), persistent=False)
        self.register_buffer("sqrt_one_minus_ab", torch.sqrt(1.0 - a_bar), persistent=False)

    def _cond(self, obs_t: torch.Tensor, obs_next: torch.Tensor) -> torch.Tensor:
        if self.n_obs_steps == 2:
            return torch.cat([obs_t, obs_next], dim=-1)
        if self.n_obs_steps == 1:
            return obs_t
        raise ValueError(f"unsupported n_obs_steps={self.n_obs_steps}")

    def _eps(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.time_emb(t)
        h = torch.cat([x, cond, emb], dim=-1)
        return self.net(h)

    def loss(self, obs_t: torch.Tensor, obs_next: torch.Tensor, a_target: torch.Tensor) -> torch.Tensor:
        B = a_target.shape[0]
        t = torch.randint(0, self.n_train_timesteps, (B,), device=a_target.device)
        noise = torch.randn_like(a_target)
        sab = self.sqrt_ab[t].unsqueeze(-1)
        somab = self.sqrt_one_minus_ab[t].unsqueeze(-1)
        x_t = sab * a_target + somab * noise
        cond = self._cond(obs_t, obs_next)
        eps = self._eps(x_t, t, cond)
        return F.mse_loss(eps, noise)

    @torch.no_grad()
    def sample(self, obs_t: torch.Tensor, obs_next: torch.Tensor) -> torch.Tensor:
        B = obs_t.shape[0]
        device = obs_t.device
        cond = self._cond(obs_t, obs_next)
        x = torch.randn(B, self.action_dim, device=device)
        steps = torch.linspace(self.n_train_timesteps - 1, 0, self.n_inference_steps, device=device).long()
        for i, t in enumerate(steps):
            tt = t.expand(B)
            eps = self._eps(x, tt, cond)
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[steps[i + 1]] if i + 1 < len(steps) else torch.tensor(1.0, device=device)
            x0_pred = (x - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab).clamp(min=1e-6)
            if self.clip_action_range is not None:
                lo, hi = self.clip_action_range
                x0_pred = x0_pred.clamp(lo, hi)
            sigma = torch.sqrt(((1 - ab_prev) / (1 - ab)).clamp(min=0.0)) * torch.sqrt((1 - ab / ab_prev).clamp(min=0.0))
            mean = torch.sqrt(ab_prev) * x0_pred + torch.sqrt((1 - ab_prev - sigma**2).clamp(min=0.0)) * eps
            if i + 1 < len(steps):
                noise = torch.randn_like(x)
                x = mean + sigma * noise
            else:
                x = mean
        if self.clip_action_range is not None:
            lo, hi = self.clip_action_range
            return x.clamp(lo, hi)
        return x

    def forward(self, obs_t: torch.Tensor, obs_next: torch.Tensor) -> torch.Tensor:
        """Sample a single action chunk (h=1)."""
        return self.sample(obs_t, obs_next)
