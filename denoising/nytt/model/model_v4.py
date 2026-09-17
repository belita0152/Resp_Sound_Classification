"""
NyTT v4 model architecture.
- multi-view pooling with view-specific mel transforms

Input [B, 1, T]
        │
        ├─ No Denoising      ─────────────────────┐
        │                                         │
        └─ DUNet                                  │
             Down × 4                             │
             → Bottleneck                         │
             → Up × 4 + skip connection           │
             → denoised waveform                  │
             → RMS norm                           │
                                                  ▼
                                      Mel spectrogram transform
                                                  │
                 ┌────────────────────────────────┼──────────────────────────┐
                 ▼                                ▼                          ▼
              tonal view                    transient view              square view
            Frequency                           Time                     General 2D
                 │                                │                          │
          SequenceViewBranch               SequenceViewBranch         SequenceViewBranch
                 │                                │                          │
             TemporalPool                    TemporalPool                TemporalPool
                 └────────────────────────────────┼──────────────────────────┘
                                                  ▼
                                               concat
                                                  ▼
                                            Linear classifier
                                                  ▼
                                             logits [B, C]

"""


from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# -----------------------------------------------------------------------------
# [Step 1] Denoiser for reducing background noise
# 1. Denoising UNet (DUNet)
# Reference: https://arxiv.org/pdf/1505.04597
# -----------------------------------------------------------------------------

SR = 16000
SEC = 10.0
N_SAMPLES = int(SEC * SR)


class DoubleConv1D(nn.Module):
    """(Conv1d + BN + ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 dilation: int = 1):
        super().__init__()
        dilation = max(1, int(dilation))
        pad = dilation * (kernel // 2)
        self.dilation = dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation,
                      bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation,
                      bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2,
                 dilation: int = 1):
        super().__init__()
        self.conv = DoubleConv1D(in_ch, out_ch, kernel, dilation=dilation)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock1D(nn.Module):
    """ConvTranspose1d → skip concat → DoubleConv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 kernel: int = 3, pool: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch // 2, kernel_size=pool, stride=pool)
        self.conv = DoubleConv1D(in_ch // 2 + skip_ch, out_ch, kernel)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="linear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DUNet(nn.Module):
    """1D U-Net with BatchNorm that predicts either the clean waveform directly"""
    def __init__(self, in_channels: int = 1,
                 channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
                 kernel: int = 3, stride: int = 2,
                 use_skip: bool = True, predict: str = "signal",
                 dilations: Tuple[int, ...] = (1, 1, 1, 1)):
        super().__init__()

        if len(channels) != 5:
            raise ValueError(
                "channels 는 [enc1,enc2,enc3,enc4,bottleneck] 5개여야 함: "
                f"{channels}")

        dilations = tuple(max(1, int(d)) for d in dilations)
        if len(dilations) != 4:
            raise ValueError(f"dilations 는 down block 4개분이어야 함: {dilations}")
        self.use_skip, self.predict = use_skip, predict
        self.dilations = dilations
        enc_channels, bottleneck_ch = tuple(channels[:4]), channels[4]
        down, prev = [], in_channels
        for ch, d in zip(enc_channels, dilations):
            down.append(DownBlock1D(prev, ch, kernel, stride, dilation=d))
            prev = ch
        self.down_blocks = nn.ModuleList(down)
        self.bottleneck = DoubleConv1D(prev, bottleneck_ch, kernel,
                                       dilation=dilations[-1])
        rev_enc = list(enc_channels[::-1])
        up, prev = [], bottleneck_ch
        for i, skip_ch in enumerate(rev_enc):
            out_ch = rev_enc[i + 1] if i + 1 < len(rev_enc) else enc_channels[0]
            up.append(UpBlock1D(prev, skip_ch if use_skip else 0, out_ch,
                                kernel, stride))
            prev = out_ch
        self.up_blocks = nn.ModuleList(up)
        self.final_conv = nn.Conv1d(enc_channels[0], in_channels, kernel_size=1)
        self.latent_channels = bottleneck_ch   # 분류기가 참조할 병목 채널 수

    def encode(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")
        skips, h = [], x
        for block in self.down_blocks:
            h, skip = block(h)
            skips.append(skip)
        h = self.bottleneck(h)
        return h, skips

    def decode(self, h: torch.Tensor, skips, x_in: torch.Tensor):
        for i, block in enumerate(self.up_blocks):
            skip = skips[-(i + 1)] if self.use_skip else None
            h = block(h, skip)
        out = self.final_conv(h)
        if out.shape[-1] != x_in.shape[-1]:
            out = F.interpolate(out, size=x_in.shape[-1], mode="linear",
                                align_corners=False)
        out = 2.0 * torch.sigmoid(out) - 1.0   # 논문 sigmoid → 부호 있는 파형용 재조정
        return x_in - out if self.predict == "residual" else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, skips = self.encode(x)
        return self.decode(h, skips, x)


# =============================================================================
# [Step 2 - Classification]
# 2. Log mel transforms
# =============================================================================

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank(sr=SR, n_fft=512, n_mels=64, fmin=0.0, fmax=8000.0) -> np.ndarray:
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for k in range(1, n_mels + 1):
        lo, c, hi = edges[k - 1], edges[k], edges[k + 1]
        up = (freqs >= lo) & (freqs <= c)
        dn = (freqs > c) & (freqs <= hi)
        if c > lo:
            fb[k - 1, up] = (freqs[up] - lo) / (c - lo)
        if hi > c:
            fb[k - 1, dn] = (hi - freqs[dn]) / (hi - c)
    return fb


class LogMelExtractor(nn.Module):
    """Convert waveforms into normalized log-mel spectrograms."""
    def __init__(self, sr=SR, n_fft=1024, hop=160, n_mels=64,
                 fmin=0.0, fmax=2000.0, top_db: float = 80.0,
                 normalize: str = "minmax", eps: float = 1e-10):
        super().__init__()
        self.n_fft, self.hop, self.eps = n_fft, hop, eps
        self.top_db, self.normalize = top_db, normalize
        self.register_buffer("window", torch.hamming_window(n_fft))
        self.register_buffer("fb", torch.from_numpy(
            mel_filterbank(sr, n_fft, n_mels, fmin, fmax)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop,
                          win_length=self.n_fft, window=self.window,
                          center=True, return_complex=True).abs().pow(2)
        mel = torch.matmul(self.fb, spec)                       # [B, n_mels, T]
        # librosa.power_to_db 와 동일: 10·log10(P), 최댓값 기준 top_db 로 하한 절단
        mel_db = 10.0 * torch.log10(mel + self.eps)
        peak = mel_db.amax(dim=(1, 2), keepdim=True)
        mel_db = torch.maximum(mel_db, peak - self.top_db)
        if self.normalize == "minmax":
            lo = mel_db.amin(dim=(1, 2), keepdim=True)
            mel_db = (mel_db - lo) / (peak - lo).clamp_min(1e-8)
        return mel_db.unsqueeze(1)


# =============================================================================
# 3. Multi-view head
# =============================================================================

class TemporalPool(nn.Module):
    def __init__(self, dim: int, kind: str = "attn"):
        super().__init__()
        if kind not in ("avg", "attn"):
            raise ValueError(f"temporal pooling은 avg 또는 attn만 지원합니다: {kind}")
        self.kind = kind
        if kind == "attn":
            self.proj = nn.Linear(dim, dim)
            self.score = nn.Linear(dim, 1)
        self.out_dim = dim

    def forward(self, x: torch.Tensor):
        if self.kind == "avg":
            return x.mean(dim=-1)
        h = x.transpose(1, 2)                                            # [B,T,C]
        a = torch.softmax(self.score(torch.tanh(self.proj(h))), dim=1)   # [B,T,1]
        return (h * a).sum(dim=1)


VIEW_POOLS = {
    "tonal": ((1, 2),) * 4,
    "transient": ((2, 1),) * 4,
    "square": ((2, 2),) * 4,
}
VIEW_FREQ_KEEP = {"tonal": 16, "transient": 4, "square": 4}
VIEW_NFFT = {"tonal": 2048, "transient": 512, "square": 1024}
VIEW_REDUCED_DIM = 128


def safe_n_mels(n_fft: int, requested: int = 64, sr: int = SR,
                fmax: float = 2000.0) -> int:
    usable = int(np.floor(fmax / (sr / 2) * (n_fft // 2 + 1)))
    return max(4, min(int(requested), max(1, usable)))


def parse_view_nfft(spec, views=None):
    text = "" if spec is None else str(spec).strip()
    if not text or text.lower() in ("none", "off", "shared"):
        return None
    if text.lower() == "auto":
        return dict(VIEW_NFFT)
    if "=" not in text:
        value = int(text)
        return {v: value for v in (views or VIEW_NFFT)}
    out = dict(VIEW_NFFT)
    out.update({
        name.strip(): int(value)
        for chunk in text.split(",") if chunk.strip()
        for name, value in [chunk.split("=", 1)]
    })
    return out


def parse_view_pools(spec: str | None):
    pools = dict(VIEW_POOLS)
    if not spec:
        return pools
    for chunk in str(spec).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"--cls_view_pools 형식 오류(=이 없음): {chunk}")
        name, values = chunk.split("=", 1)
        steps = []
        for token in values.split(","):
            token = token.strip().lower()
            if not token:
                continue
            if "x" not in token:
                raise ValueError(f"pooling 은 '<freq>x<time>' 형식이어야 함: {token}")
            fp, tp = token.split("x", 1)
            steps.append((max(1, int(fp)), max(1, int(tp))))
        if len(steps) == 1:
            steps = steps * 4
        if len(steps) != 4:
            raise ValueError(f"{name}: 블록 4개분 pooling 이 필요함 (받은 {len(steps)})")
        pools[name.strip()] = tuple(steps)
    return pools


def parse_view_freq_keep(spec):
    text = "" if spec is None else str(spec).strip()
    if not text:
        return None
    if "=" not in text:
        return int(text)
    return {
        name.strip(): int(value)
        for chunk in text.split(",") if chunk.strip()
        for name, value in [chunk.split("=", 1)]
    }


class SequenceViewBranch(nn.Module):
    def __init__(self, base: int, dim: int, freq_keep: int, pools,
                 pool: str = "avg", reduced_dim: int | None = None):
        super().__init__()
        channels = [base, base * 2, base * 4, base * 4]
        self.blocks = nn.ModuleList()
        previous = 1
        for channels_out, (freq_pool, time_pool) in zip(channels, pools):
            layers = [
                nn.Conv2d(previous, channels_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.SiLU(inplace=True),
            ]
            if freq_pool > 1 or time_pool > 1:
                layers.append(nn.MaxPool2d((freq_pool, time_pool), (freq_pool, time_pool)))
            self.blocks.append(nn.Sequential(*layers))
            previous = channels_out

        self.freq_keep = int(freq_keep)
        self.proj = nn.Sequential(
            nn.Conv2d(previous, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        reduced_dim = int(reduced_dim or VIEW_REDUCED_DIM)
        self.frequency_reduce = nn.Sequential(
            nn.Conv1d(
                dim * self.freq_keep, reduced_dim,
                kernel_size=1, bias=False),
            nn.BatchNorm1d(reduced_dim),
            nn.SiLU(inplace=True),
        )
        self.pool = TemporalPool(reduced_dim, pool)
        self.out_dim = self.pool.out_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        h = mel
        for block in self.blocks:
            h = block(h)
        h = self.proj(h)                                      # [B,D,F,T]
        h = F.adaptive_avg_pool2d(h, (self.freq_keep, h.shape[-1]))
        h = h.flatten(1, 2)                                  # [B,D*Fk,T]
        return self.frequency_reduce(h)                       # [B,128,T]


class MultiViewMelClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, base: int = 32,
                 dropout: float = 0.1,
                 views: Tuple[str, ...] = ("tonal", "transient", "square"),
                 dim=32, freq_keep=None, pools=None,
                 pool: str = "avg", reduced_dim: int | None = None):
        super().__init__()
        self.views = tuple(views)
        pools = pools or VIEW_POOLS
        unknown = [view for view in self.views if view not in pools]
        if unknown:
            raise ValueError(f"모르는 view: {unknown}")
        if not self.views:
            raise ValueError("view가 하나 이상 필요합니다.")

        def resolve(value, view, default):
            return int(value.get(view, default) if isinstance(value, dict)
                       else default if value is None else value)

        keeps = [resolve(freq_keep, view, VIEW_FREQ_KEEP.get(view, 4))
                 for view in self.views]
        dims = [resolve(dim, view, 32) for view in self.views]
        reduced_dim = int(reduced_dim or VIEW_REDUCED_DIM)

        self.branches = nn.ModuleList([
            SequenceViewBranch(base, d, keep, pools[view], pool,
                               reduced_dim=reduced_dim)
            for view, d, keep in zip(self.views, dims, keeps)])

        self.view_dim = self.branches[0].out_dim
        self.head_input_dim = self.view_dim * len(self.branches)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.head_input_dim, num_classes)

    def forward(self, x) -> torch.Tensor:
        mels = list(x) if isinstance(x, (list, tuple)) else [x] * len(self.branches)
        if len(mels) != len(self.branches):
            raise ValueError(
                f"view {len(self.branches)}개인데 mel이 {len(mels)}개 들어왔습니다.")

        pooled = []
        for branch, mel in zip(self.branches, mels):
            sequence = branch(mel)                            # [B,128,T_view]
            pooled.append(branch.pool(sequence))               # [B,D_view]

        return self.fc(self.dropout(torch.cat(pooled, dim=-1)))  # [B,C]


class NyTTClassifier(nn.Module):
    def __init__(self, dae: DUNet | None, num_classes: int = 5,
                 base: int = 32, dropout: float = 0.1,
                 mel_normalize: str = "minmax", top_db: float = 80.0,
                 cls_dim: int = 32,
                 cls_pool: str = "avg",
                 cls_views: Tuple[str, ...] = ("tonal", "transient", "square"),
                 cls_view_pools=None, cls_view_freq_keep=None,
                 cls_view_nfft=None,
                 cls_view_nmels=None, cls_view_reduced_dim=None,
                 cls_view_dim=None,
                 n_mels: int = 64, mel_fmax: float = 2000.0):
        super().__init__()
        self.dae = dae
        self.use_denoiser = dae is not None
        self.frontend = LogMelExtractor(normalize=mel_normalize, top_db=top_db,
                                         n_mels=n_mels, fmax=mel_fmax)
        self.view_frontends = None
        if cls_view_nfft:
            fronts = []
            for view in cls_views:
                n_fft = int(cls_view_nfft.get(view, VIEW_NFFT.get(view, 1024)))
                if isinstance(cls_view_nmels, dict):
                    requested = int(cls_view_nmels.get(view, n_mels))
                elif cls_view_nmels is not None:
                    requested = int(cls_view_nmels)
                else:
                    requested = n_mels
                mels = safe_n_mels(n_fft, requested, fmax=mel_fmax)
                fronts.append(LogMelExtractor(n_fft=n_fft, n_mels=mels, fmax=mel_fmax,
                                               normalize=mel_normalize, top_db=top_db))
            self.view_frontends = nn.ModuleList(fronts)
        self.classifier = MultiViewMelClassifier(
            num_classes, base, dropout, views=cls_views,
            dim=(cls_view_dim if cls_view_dim is not None else cls_dim),
            reduced_dim=cls_view_reduced_dim,
            freq_keep=cls_view_freq_keep, pools=cls_view_pools,
            pool=cls_pool)

    def extract_mels(self, x: torch.Tensor):
        if self.view_frontends is None:
            return self.frontend(x)
        return [frontend(x) for frontend in self.view_frontends]

    def forward(self, x: torch.Tensor):
        if self.dae is None:
            return None, self.classifier(self.extract_mels(x))

        x_hat = self.dae(x)
        rms_in = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
        rms_hat = x_hat.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
        x_norm = x_hat * (rms_in / rms_hat)
        return x_hat, self.classifier(self.extract_mels(x_norm))
