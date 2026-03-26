import math

import torch
from torch import nn


def get_1d_sincos_encode(steps: torch.Tensor, emb_dim: int, max_period: int=10000) -> torch.Tensor:
    assert steps.dim() == 1, f"Parameter `steps` must be a 1D tensor, but got {steps.dim()}D."

    half_dim = emb_dim // 2
    emb = torch.exp(- math.log(max_period) *\
        torch.arange(0, half_dim, device=steps.device).float() / half_dim)
    emb = steps[:, None].float() * emb[None, :]

    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

    if emb_dim % 2 == 1: emb = nn.functional.pad(emb, (0, 1))
    assert emb.shape == (steps.shape[0], emb_dim)

    return emb


class Timestep(nn.Module):
    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_1d_sincos_encode(timesteps, self.time_emb_dim)


class TimestepEmbed(nn.Module):
    def __init__(self, in_dim: int, time_emb_dim: int, act_fn_name: str="GELU"):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, time_emb_dim),
            getattr(nn, act_fn_name)(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.mlp(sample)
