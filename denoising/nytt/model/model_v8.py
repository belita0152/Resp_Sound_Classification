from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from denoising.nytt.model.model_v4 import (
    DUNet,
    LogMelExtractor,
    parse_view_freq_keep,
    parse_view_nfft,
    safe_n_mels,
)


VIEW_POOLS = {
    "tonal": ((1, 2),) * 4,
    "transient": ((2, 1), (2, 1), (1, 1), (1, 1)),
    "square": ((2, 2), (2, 2), (1, 2), (1, 2)),
}
VIEW_FREQ_KEEP = {"tonal": 32, "transient": 16, "square": 16}
VIEW_REDUCED_DIM = {"tonal": 256, "transient": 128, "square": 128}
VIEW_NFFT = {"tonal": 2048, "transient": 512, "square": 1024}
TOPK_RATIO = 0.125


def parse_view_pools(spec):
    pools = dict(VIEW_POOLS)
    if not spec:
        return pools
    for chunk in str(spec).split(";"):
        name, values = chunk.split("=", 1)
        steps = [tuple(map(int, token.split("x", 1)))
                 for token in values.split(",") if token.strip()]
        if len(steps) == 1:
            steps *= 4
        if len(steps) != 4:
            raise ValueError(f"{name}: pooling 4개가 필요합니다: {steps}")
        pools[name.strip()] = tuple(steps)
    return pools


class TemporalStatsPool(nn.Module):
    def __init__(self, dim, topk_ratio=TOPK_RATIO):
        super().__init__()
        self.topk_ratio = topk_ratio
        self.projection = nn.Sequential(
            nn.Linear(dim * 3, dim, bias=False),
            nn.LayerNorm(dim),
            nn.SiLU(inplace=True),
        )
        self.out_dim = dim

    def forward(self, x):
        k = max(1, math.ceil(x.shape[-1] * self.topk_ratio))
        mean = x.mean(dim=-1)
        topk = x.topk(k, dim=-1, sorted=False).values.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        return self.projection(torch.cat([mean, topk, std], dim=-1))


class SequenceViewBranch(nn.Module):
    def __init__(self, base, dim, freq_keep, pools, reduced_dim):
        super().__init__()
        channels = [base, base * 2, base * 4, base * 4]
        blocks, previous = [], 1
        for out_channels, pool in zip(channels, pools):
            layers = [
                nn.Conv2d(previous, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            ]
            if pool != (1, 1):
                layers.append(nn.MaxPool2d(pool, pool))
            blocks.append(nn.Sequential(*layers))
            previous = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.freq_keep = freq_keep
        self.proj = nn.Sequential(
            nn.Conv2d(previous, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.frequency_reduce = nn.Sequential(
            nn.Conv1d(dim * freq_keep, reduced_dim, 1, bias=False),
            nn.BatchNorm1d(reduced_dim),
            nn.SiLU(inplace=True),
        )
        self.pool = TemporalStatsPool(reduced_dim)
        self.out_dim = reduced_dim

    def forward(self, mel):
        h = mel
        for block in self.blocks:
            h = block(h)
        h = self.proj(h)
        h = F.adaptive_avg_pool2d(h, (self.freq_keep, h.shape[-1]))
        return self.frequency_reduce(h.flatten(1, 2))


class MultiViewMelClassifier(nn.Module):
    def __init__(self, num_classes=5, base=32, dropout=0.1,
                 views: Tuple[str, ...] = ("tonal", "transient", "square"),
                 dim=32, freq_keep=None, pools=None, reduced_dim=None):
        super().__init__()
        self.views = tuple(views)
        pools = pools or VIEW_POOLS

        def resolve(value, view, default):
            return int(value.get(view, default) if isinstance(value, dict)
                       else default if value is None else value)

        keeps = [resolve(freq_keep, view, VIEW_FREQ_KEEP[view])
                 for view in self.views]
        dims = [resolve(dim, view, 32) for view in self.views]
        reduced_dims = [resolve(reduced_dim, view, VIEW_REDUCED_DIM[view])
                        for view in self.views]
        self.branches = nn.ModuleList([
            SequenceViewBranch(base, d, keep, pools[view], out_dim)
            for view, d, keep, out_dim
            in zip(self.views, dims, keeps, reduced_dims)
        ])
        self.head_input_dim = sum(branch.out_dim for branch in self.branches)
        hidden = max(128, self.head_input_dim // 2)
        self.fc = nn.Sequential(
            nn.LayerNorm(self.head_input_dim),
            nn.Linear(self.head_input_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        mels = list(x) if isinstance(x, (list, tuple)) else [x] * len(self.branches)
        pooled = [branch.pool(branch(mel))
                  for branch, mel in zip(self.branches, mels)]
        return self.fc(torch.cat(pooled, dim=-1))


class NyTTClassifier(nn.Module):
    def __init__(self, dunet, num_classes=5, base=32, dropout=0.1,
                 mel_normalize="minmax", top_db=80.0, cls_dim=32,
                 cls_pool="stats",
                 cls_views: Tuple[str, ...] = ("tonal", "transient", "square"),
                 cls_view_pools=None, cls_view_freq_keep=None,
                 cls_view_nfft=None, cls_view_nmels=None,
                 cls_view_reduced_dim=None, cls_view_dim=None,
                 n_mels=64, mel_fmax=2000.0):
        super().__init__()
        self.dunet = dunet
        self.use_denoiser = dunet is not None
        self.frontend = LogMelExtractor(
            normalize=mel_normalize, top_db=top_db,
            n_mels=n_mels, fmax=mel_fmax,
        )
        self.view_frontends = None
        if cls_view_nfft:
            fronts = []
            for view in cls_views:
                n_fft = cls_view_nfft.get(view, VIEW_NFFT[view])
                requested = (cls_view_nmels.get(view, n_mels)
                             if isinstance(cls_view_nmels, dict)
                             else cls_view_nmels or n_mels)
                fronts.append(LogMelExtractor(
                    n_fft=n_fft,
                    n_mels=safe_n_mels(n_fft, requested, fmax=mel_fmax),
                    fmax=mel_fmax,
                    normalize=mel_normalize,
                    top_db=top_db,
                ))
            self.view_frontends = nn.ModuleList(fronts)
        self.classifier = MultiViewMelClassifier(
            num_classes, base, dropout, views=cls_views,
            dim=cls_view_dim if cls_view_dim is not None else cls_dim,
            freq_keep=cls_view_freq_keep,
            pools=cls_view_pools,
            reduced_dim=cls_view_reduced_dim,
        )

    def extract_mels(self, x):
        if self.view_frontends is None:
            return self.frontend(x)
        return [frontend(x) for frontend in self.view_frontends]

    def forward(self, x):
        if self.dunet is None:
            return None, self.classifier(self.extract_mels(x))
        x_hat = self.dunet(x)
        rms_in = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
        rms_hat = x_hat.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
        x_norm = x_hat * (rms_in / rms_hat)
        return x_hat, self.classifier(self.extract_mels(x_norm))

