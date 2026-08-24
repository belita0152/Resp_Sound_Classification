# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor,
               snr_db: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p_c = clean.pow(2).mean(dim=(1, 2), keepdim=True) + eps
    p_n = noise.pow(2).mean(dim=(1, 2), keepdim=True) + eps
    scale = torch.sqrt(p_c / (p_n * torch.pow(10.0, snr_db.view(-1, 1, 1) / 10.0)))
    return clean + scale * noise


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = est - est.mean(dim=(1, 2), keepdim=True)
    ref = ref - ref.mean(dim=(1, 2), keepdim=True)
    alpha = ((est * ref).sum(dim=(1, 2), keepdim=True)
             / (ref.pow(2).sum(dim=(1, 2), keepdim=True) + eps))
    target = alpha * ref
    return 10 * torch.log10((target.pow(2).sum(dim=(1, 2)) + eps)
                            / ((est - target).pow(2).sum(dim=(1, 2)) + eps))


@torch.no_grad()
def energy_ratio_db(model: nn.Module, x: torch.Tensor, eps: float = 1e-10) -> float:
    """10·log10( E[x^2] / E[f(x)^2] )
    x=잡음만  → 클수록 좋음 (noise suppression)
    x=원신호  → 0 에 가까울수록 좋음 (signal retention, 부호 반전해 사용)
    """
    out = model(x)
    p_in = x.pow(2).mean(dim=(1, 2)) + eps
    p_out = out.pow(2).mean(dim=(1, 2)) + eps
    return (10 * torch.log10(p_in / p_out)).mean().item()
