# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoising_metric import si_sdr


def multi_res_stft_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    n_ffts: Tuple[int, ...] = (256, 512, 1024),
    eps: float = 1e-4,
) -> torch.Tensor:
    """다해상도 magnitude STFT 손실을 float32로 계산한다."""
    estimate = estimate.squeeze(1).float()
    target = target.squeeze(1).float()
    total = estimate.new_zeros(())

    # 작은 spectral magnitude에서 fp16 log gradient가 overflow하지 않게 한다.
    with torch.cuda.amp.autocast(enabled=False):
        for n_fft in n_ffts:
            window = torch.hann_window(
                n_fft, device=estimate.device, dtype=estimate.dtype,
            )
            estimate_stft = torch.stft(
                estimate, n_fft=n_fft, hop_length=n_fft // 4,
                win_length=n_fft, window=window, return_complex=True,
            ).abs()
            target_stft = torch.stft(
                target, n_fft=n_fft, hop_length=n_fft // 4,
                win_length=n_fft, window=window, return_complex=True,
            ).abs()
            total = total + F.l1_loss(estimate_stft, target_stft)
            total = total + F.l1_loss(
                torch.log(estimate_stft + eps), torch.log(target_stft + eps),
            )
    return total / len(n_ffts)


class ReconLoss(nn.Module):
    """Waveform L1, multi-resolution STFT, SI-SDR 항을 결합한다."""

    def __init__(self, w_l1: float = 1.0, w_stft: float = 1.0,
                 w_sisdr: float = 0.0):
        super().__init__()
        self.w_l1 = float(w_l1)
        self.w_stft = float(w_stft)
        self.w_sisdr = float(w_sisdr)

    def forward(self, estimate: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loss = estimate.new_zeros(())
        if self.w_l1 > 0:
            loss = loss + self.w_l1 * (estimate - target).abs().mean()
        if self.w_stft > 0:
            loss = loss + self.w_stft * multi_res_stft_loss(estimate, target)
        if self.w_sisdr > 0:
            loss = loss - self.w_sisdr * si_sdr(estimate, target).mean() / 10.0
        return loss


class TrainingLoss(nn.Module):
    """Classification과 선택적 denoising recon loss를 계산한다."""

    def __init__(
        self,
        class_weights: torch.Tensor | None,
        use_dnet: bool,
        w_rec: float = 1.0,
        w_cls: float = 1.0,
        w_l1: float = 1.0,
        w_stft: float = 1.0,
        w_sisdr: float = 0.0,
    ):
        super().__init__()
        self.use_dnet = bool(use_dnet)
        self.w_rec = float(w_rec)
        self.w_cls = float(w_cls)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.recon_loss = ReconLoss(w_l1, w_stft, w_sisdr)

    def forward(
        self,
        estimate: torch.Tensor | None,
        target_waveform: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        loss = self.criterion(logits.float(), labels)
        if not self.use_dnet:
            recon_loss = loss.new_zeros(())
            return self.w_cls * loss, recon_loss, loss

        recon_loss = self.recon_loss(estimate.float(), target_waveform)
        total = self.w_rec * recon_loss + self.w_cls * loss
        return total, recon_loss, loss
