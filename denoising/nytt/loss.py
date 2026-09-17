# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from denoising.utils.denoising_metric import si_sdr


SR = 16_000
DEFAULT_STFT_FFTS = (256, 512, 1024)
BAND_LIMITED_STFT_FFTS = (256, 1024, 2048)


def multi_res_stft_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    n_ffts: Tuple[int, ...] = DEFAULT_STFT_FFTS,
    eps: float = 1e-4,
    sr: int = SR,
    fmax_hz: float | None = None,
) -> torch.Tensor:
    """Magnitude STFT 손실을 여러 해상도에서 float32로 계산한다."""
    estimate = estimate.squeeze(1).float()
    target = target.squeeze(1).float()
    total = estimate.new_zeros(())

    # 작은 spectral magnitude에서 fp16 log gradient가 overflow하지 않게 한다.
    with torch.amp.autocast(device_type=estimate.device.type, enabled=False):
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

            # 폐음 분류기가 보지 않는 고주파 대역은 손실에서 제외한다.
            if fmax_hz is not None:
                n_bins = n_fft // 2 + 1
                keep = math.ceil(float(fmax_hz) / (sr / 2) * n_bins)
                keep = max(1, min(keep, n_bins))
                estimate_stft = estimate_stft[:, :keep]
                target_stft = target_stft[:, :keep]

            total = total + F.l1_loss(estimate_stft, target_stft)
            total = total + F.l1_loss(
                torch.log(estimate_stft + eps), torch.log(target_stft + eps),
            )
    return total / len(n_ffts)


class ReconLoss(nn.Module):
    """Waveform L1, multi-resolution STFT, SI-SDR 항을 결합한다."""

    def __init__(
        self,
        w_l1: float = 1.0,
        w_stft: float = 1.0,
        w_sisdr: float = 0.0,
        *,
        sr: int = SR,
        stft_ffts: Tuple[int, ...] = DEFAULT_STFT_FFTS,
        stft_fmax_hz: float | None = 2000.0,
    ):
        super().__init__()
        self.w_l1 = float(w_l1)
        self.w_stft = float(w_stft)
        self.w_sisdr = float(w_sisdr)
        self.sr = int(sr)
        self.stft_ffts = tuple(int(n_fft) for n_fft in stft_ffts)
        self.stft_fmax_hz = stft_fmax_hz

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        loss = estimate.new_zeros(())
        if self.w_l1 > 0:
            loss = loss + self.w_l1 * (estimate - target).abs().mean()
        if self.w_stft > 0:
            loss = loss + self.w_stft * multi_res_stft_loss(
                estimate,
                target,
                n_ffts=self.stft_ffts,
                sr=self.sr,
                fmax_hz=self.stft_fmax_hz,
            )
        if self.w_sisdr > 0:
            loss = loss - self.w_sisdr * si_sdr(estimate, target).mean() / 10.0
        return loss


class TrainingLoss(nn.Module):
    """Classification과 선택적 denoising reconstruction loss를 계산한다."""

    def __init__(
        self,
        class_weights: torch.Tensor | None,
        use_dnet: bool,
        w_rec: float = 1.0,
        w_cls: float = 1.0,
        w_l1: float = 1.0,
        w_stft: float = 1.0,
        w_sisdr: float = 0.0,
        *,
        sr: int = SR,
        stft_ffts: Tuple[int, ...] | None = None,
        stft_fmax_hz: float | None = 2000.0,
        aux_ce_weight: float = 0.2,
        aux_use_class_weight: bool = False,
    ):
        super().__init__()
        self.use_dnet = bool(use_dnet)
        self.w_rec = float(w_rec)
        self.w_cls = float(w_cls)
        self.aux_ce_weight = float(aux_ce_weight)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.aux_criterion = nn.CrossEntropyLoss(
            weight=class_weights if aux_use_class_weight else None,
        )

        if stft_ffts is None:
            stft_ffts = (
                BAND_LIMITED_STFT_FFTS
                if stft_fmax_hz is not None
                else DEFAULT_STFT_FFTS
            )
        self.recon_loss = ReconLoss(
            w_l1,
            w_stft,
            w_sisdr,
            sr=sr,
            stft_ffts=stft_ffts,
            stft_fmax_hz=stft_fmax_hz,
        )

    def forward(
        self,
        estimate: torch.Tensor | None,
        target_waveform: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.ndim == 3:
            loss = self.criterion(logits[:, 0].float(), labels)
            if logits.shape[1] > 1 and self.aux_ce_weight > 0:
                aux_loss = torch.stack([
                    self.aux_criterion(logits[:, index].float(), labels)
                    for index in range(1, logits.shape[1])
                ]).mean()
                loss = loss + self.aux_ce_weight * aux_loss
        else:
            loss = self.criterion(logits.float(), labels)

        if not self.use_dnet:
            recon_loss = loss.new_zeros(())
            return self.w_cls * loss, recon_loss, loss

        recon_loss = self.recon_loss(estimate.float(), target_waveform)
        total = self.w_rec * recon_loss + self.w_cls * loss
        return total, recon_loss, loss
