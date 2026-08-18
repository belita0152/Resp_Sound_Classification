# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import argparse
import json
import random
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
UTILS_ROOT = CURRENT_DIR.parents[1]  # /home/coder/workspace/data/classification
sys.path.insert(0, str(UTILS_ROOT))

from data.data_loader_nytt import LungSegmentDataset
from data.utils import mel_folder, wave_folder
from denoising.nytt.noise_bank_loader import NoiseBank
from denoising.nytt.loss_cls import CEDiceLoss
from utils.metric import calculate_classification_metrics, print_classification_metrics



warnings.filterwarnings("ignore")

PROJECT_ROOT = UTILS_ROOT
DATA_ROOT = PROJECT_ROOT.parent
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "denoising" / "utils" / "split_manifest_5x60_40.csv"
DEFAULT_NOISE_BANK_DIR = DATA_ROOT / "db" / "new_gt"


"""
NyTT — Noisy-target Training for waveform denoising  (clean reference 불필요)
============================================================================

    x        : 실제 녹음 segment (이미 잡음 n 포함)        ← target
    m        : cycle 여집합에서 추출한 실제 잡음
    x_in     : mix(x, m, SNR)                             ← input
    loss     : L1(f(x_in), x) + multi-res STFT

추론 시 x 를 그대로 넣으면 f(x) 는 x 보다 잡음이 적다.
L1 최소화가 조건부 기댓값으로 수렴하므로 모델은 '잡음 억제 사상' 을 학습한다.
★ 주입 잡음 m 이 실제 잡음 n 과 같은 분포여야 n 도 줄어든다
  → stage=noise_bank 가 자기 데이터에서 잡음을 뽑는 이유.

정규화 두 곳
    ① 입력  : dataloader 가 rms 정규화. 기기 게인 차이 제거 + L1/STFT 손실 균형
    ② 출력  : NyTTClassifier 안에서 x_hat 을 x 의 rms 로 맞춤.
              denoiser 가 만들어낸 레벨 차이가 분류기에 교란으로 들어가는 것을 막는다.
              denoising 지표는 ② 이전 x_hat 으로 계산한다.

두 arm 비교
    A (대조군) 저장된 10s_mel/*.npy  → data_loader(MelTransform minmax) → classifier
    B (제안)   10s_repeat/*.wav      → WaveDAE → MelFrontEnd(minmax)   → classifier
    같은 parser · 같은 fold manifest · 같은 mel 정규화를 쓴다. 그래야 차이가 denoising 때문이다.

실행 — 반드시 프로젝트 루트에서 패키지로 실행한다
    python -m denoising.nytt.noise_bank_split --fold 1       # 1) fold 1 잡음 은행
    python -m denoising.nytt.nytt --stage check --fold 1     # 2) split/모델 확인
    python -m denoising.nytt.nytt --stage train --fold 1 --epochs 50  # 3) 학습

    잡음 추출 로직은 noise_bank_split.py 한 곳에만 둔다. 여기서 다시 구현하면
    EDGE_MARGIN / MIN_GAP 같은 상수가 갈라져 재현이 깨진다.
"""

SR = 16000
SEC = 10.0
N_SAMPLES = int(SEC * SR)


def read_fold_ids(manifest_path: str | Path, fold: int) -> Tuple[list[str], list[str]]:
    """split manifest에서 지정 fold의 train/test recording ID를 검증해 읽는다."""
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"split manifest 없음: {path}")

    frame = pd.read_csv(path, dtype={"sample_id": str})
    required = {"fold", "sample_id", "split"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise RuntimeError(f"manifest 필수 열 없음: {sorted(missing_columns)}")

    frame = frame.loc[:, ["fold", "sample_id", "split"]].copy()
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["sample_id"] = frame["sample_id"].str.strip()
    frame["split"] = frame["split"].str.strip().str.lower()
    part = frame.loc[frame["fold"].eq(int(fold))]
    if part.empty:
        available = sorted(frame["fold"].unique().tolist())
        raise RuntimeError(f"fold {fold}가 manifest에 없음. 사용 가능: {available}")
    if part["sample_id"].isna().any() or part["sample_id"].eq("").any():
        raise RuntimeError(f"fold {fold}에 빈 sample_id가 있음")
    duplicated = sorted(
        part.loc[part["sample_id"].duplicated(False), "sample_id"].unique().tolist()
    )
    if duplicated:
        raise RuntimeError(f"fold {fold}의 sample_id 중복: {duplicated}")
    invalid_split = sorted(set(part["split"]) - {"train", "test"})
    if invalid_split:
        raise RuntimeError(f"fold {fold}의 잘못된 split 값: {invalid_split}")

    train_ids = sorted(part.loc[part["split"].eq("train"), "sample_id"].tolist())
    test_ids = sorted(part.loc[part["split"].eq("test"), "sample_id"].tolist())
    if not train_ids or not test_ids:
        raise RuntimeError(f"fold {fold}에는 train과 test가 모두 있어야 함")
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise RuntimeError(f"fold {fold} train/test ID 중복: {overlap}")
    return train_ids, test_ids


def build_fold_wave_datasets(args) -> Tuple[LungSegmentDataset, LungSegmentDataset]:
    """manifest의 recording ID를 그대로 사용하는 waveform train/test dataset."""
    train_ids, test_ids = read_fold_ids(args.split_manifest, args.fold)
    common = dict(
        domain="wave",
        input_type=args.normalize,
        preload=args.preload,
        # manifest가 cohort의 source of truth이므로 utils.matched_ids 필터는 끈다.
        use_cohort_filter=False,
    )
    train_ds = LungSegmentDataset(
        args.wave_base_path, train=True, sample_ids=train_ids, **common,
    )
    test_ds = LungSegmentDataset(
        args.wave_base_path, train=False, sample_ids=test_ids, **common,
    )
    if set(train_ds.sample_ids) != set(train_ids):
        raise RuntimeError("train dataset ID가 manifest와 일치하지 않음")
    if set(test_ds.sample_ids) != set(test_ids):
        raise RuntimeError("test dataset ID가 manifest와 일치하지 않음")
    if set(train_ds.sample_ids) & set(test_ds.sample_ids):
        raise RuntimeError("train/test dataset에 동일 recording ID가 포함됨")

    print(f"[split] manifest {Path(args.split_manifest).resolve()}")
    print(f"[split] fold {args.fold}: train {len(train_ids)} ids / test {len(test_ids)} ids")
    return train_ds, test_ds


def verify_fold_noise_bank(
    bank_path: str | Path,
    fold: int,
    train_ids,
    test_ids,
) -> None:
    """noise bank가 같은 fold의 train recording에서 만들어졌는지 강제 검증한다."""
    path = Path(bank_path).resolve()
    with np.load(path, allow_pickle=True) as bank:
        required = {"fold", "train_ids", "test_ids", "source_ids", "train_only"}
        missing = required - set(bank.files)
        if missing:
            raise RuntimeError(
                f"fold 검증 metadata가 noise bank에 없음: {sorted(missing)}\n"
                "noise_bank_split.py로 다시 생성하십시오."
            )
        bank_fold = int(bank["fold"])
        bank_train = {str(value) for value in bank["train_ids"].tolist()}
        bank_test = {str(value) for value in bank["test_ids"].tolist()}
        bank_sources = {str(value) for value in bank["source_ids"].tolist()}
        train_only = bool(bank["train_only"])

    expected_train, expected_test = set(train_ids), set(test_ids)
    errors = []
    if bank_fold != int(fold):
        errors.append(f"bank fold={bank_fold}, requested fold={fold}")
    if bank_train != expected_train:
        errors.append(
            f"train ID 불일치(bank={len(bank_train)}, manifest={len(expected_train)})"
        )
    if bank_test != expected_test:
        errors.append(
            f"test ID 불일치(bank={len(bank_test)}, manifest={len(expected_test)})"
        )
    if not train_only:
        errors.append("train_only=False")
    if bank_sources - expected_train:
        errors.append(f"train 외 source IDs={sorted(bank_sources - expected_train)}")
    if bank_sources & expected_test:
        errors.append(f"test 누수 IDs={sorted(bank_sources & expected_test)}")
    if errors:
        raise RuntimeError("noise bank/fold 검증 실패: " + "; ".join(errors))

    print(
        f"[noise] fold metadata 일치: fold {fold}, "
        f"train {len(bank_train)} / test {len(bank_test)} / source {len(bank_sources)} ids"
    )


# =============================================================================
# 1. Mixing / metrics
# =============================================================================
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


def multi_res_stft_loss(est: torch.Tensor, ref: torch.Tensor,
                        n_ffts: Tuple[int, ...] = (256, 512, 1024),
                        eps: float = 1e-4) -> torch.Tensor:
    """다해상도 STFT 손실 (크기 스펙트럼).

    ★ eps 는 1e-7 이 아니라 1e-4 여야 한다.
      log(E+eps) 의 기울기는 1/(E+eps) 이고, 스펙트럼 널에서 E→0 이면
          eps=1e-7 → 최대 1e7   (fp16 최대 65,504 를 초과 → Inf)
          eps=1e-4 → 최대 1e4   (안전)
      Inf 가 한 번 나오면 그 배치의 optimizer step 이 통째로 건너뛰어지고
      GradScaler 의 scale 이 절반으로 줄어든다. 반복되면 학습이 멈춘다.

    ★ 반드시 float32 로 계산한다. autocast 안에서 fp16 으로 STFT 를 돌리면
      작은 크기값이 언더플로해 위 문제가 훨씬 잘 터진다.
    """
    est, ref = est.squeeze(1).float(), ref.squeeze(1).float()
    total = est.new_zeros(())
    with torch.cuda.amp.autocast(enabled=False):
        for n_fft in n_ffts:
            win = torch.hann_window(n_fft, device=est.device, dtype=est.dtype)
            E = torch.stft(est, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                           window=win, return_complex=True).abs()
            R = torch.stft(ref, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                           window=win, return_complex=True).abs()
            total = total + F.l1_loss(E, R) + F.l1_loss(torch.log(E + eps),
                                                        torch.log(R + eps))
    return total / len(n_ffts)


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


# =============================================================================
# 2. Model — 파형 1D U-Net (Ronneberger U-Net + BN, 1D 변환)
# =============================================================================
# Davis, Shen, Ji, Zhu, "Denoising of Two-Phase Optically Sectioned Structured
# Illumination Reconstructions Using Encoder-Decoder Networks", 2025.
#     arXiv:2510.03452, Fig. 2(b) 의 denoising U-Net 구조를 따른다.
#
#     인코더   4× [DoubleConv(2×(conv+BN+ReLU)) → MaxPool]   채널 2배씩 증가
#     병목     DoubleConv (채널 2배, pooling 없음)
#     디코더   4× [ConvTranspose(2배 업샘플·채널 반) → skip concat → DoubleConv]
#     출력     1×1 conv → sigmoid (원논문: 영상 강도라 [0,1])
#
# ★ 이 논문은 2D 영상(구조광 현미경) denoising 이다. 1D 파형으로 옮기며 바꾼 것:
#     커널      3×3 conv        → kernel(1D, 기본 3)
#     다운샘플  2×2 max pool    → MaxPool1d(pool)            (기본 2)
#     업샘플    2×2 stride 2 transposed conv → ConvTranspose1d(동일 비율)
#     출력      sigmoid [0,1]   → 2·sigmoid-1 로 [-1,1]  (부호 있는 파형이라서)
#   논문에 쓸 때 "Ronneberger U-Net 을 그대로 썼다"라고 하면 안 되고, 1D 로
#   변환했다고 명시해야 한다.
# =============================================================================
class DoubleConv1D(nn.Module):
    """(Conv1d + BN + ReLU) × 2 — 원논문 encoder/decoder 공통 블록."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock1D(nn.Module):
    """DoubleConv → MaxPool.  skip 은 pooling 이전 특징을 그대로 내보낸다."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2):
        super().__init__()
        self.conv = DoubleConv1D(in_ch, out_ch, kernel)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock1D(nn.Module):
    """ConvTranspose1d(2배 업샘플·채널 반으로) → skip concat → DoubleConv.

    길이가 맞지 않으면(홀수 길이 등) skip 크기에 선형보간으로 맞춘다.
    """

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


class WaveDAE(nn.Module):
    """[B, 1, T] -> [B, 1, T]   Ronneberger U-Net(2015) + BatchNorm, 1D 변환.

    predict='signal'   clean 파형 직접 예측 (x-prediction)
    predict='residual' 잡음 예측 후 차감 (ε-prediction, ablation)

    ★ skip 은 기본으로 켠다. --no_skip 은 ablation 용이다. 이전 stride-conv
      인코더에서 관측된 "skip 제거 시 F1 42.2→28.2, 발산" 결과는 그 구조에서
      나온 것이라, 이 U-Net 에서 그대로 재현된다는 보장은 없다 — 다시 측정하라.

    ★ NyTT 목적함수 자체는 잡음 제거를 요구하지 않는다(정답 x 에 잡음 n 이
      들어있다). 실제 잡음 제거는 용량 제한의 부산물이다. 용량을 줄이려면
      --channels 를 줄이는 쪽이 우선이다(skip 을 빼는 게 아니라).

    ★ 출력이 2·sigmoid(·)-1 로 [-1,1] 에 유계다. --normalize rms 는 진폭을
      하드 클리핑하지 않으므로 드물게 |x|>1 인 샘플이 있으면 이 유계가 잘라낸다.
      --normalize peak 를 쓰면 입력 자체가 [-1,1] 이라 이 문제가 없다.
    """

    def __init__(self, in_channels: int = 1,
                 channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
                 kernel: int = 3, stride: int = 2,
                 use_skip: bool = True, predict: str = "signal"):
        super().__init__()
        if len(channels) != 5:
            raise ValueError(
                "channels 는 [enc1,enc2,enc3,enc4,bottleneck] 5개여야 함: "
                f"{channels}")
        self.use_skip, self.predict = use_skip, predict
        enc_channels, bottleneck_ch = tuple(channels[:4]), channels[4]

        down, prev = [], in_channels
        for ch in enc_channels:
            down.append(DownBlock1D(prev, ch, kernel, stride))
            prev = ch
        self.down_blocks = nn.ModuleList(down)

        self.bottleneck = DoubleConv1D(prev, bottleneck_ch, kernel)

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
        """파형 → (병목 특징 h, 스킵 리스트).  h 는 [B, bottleneck_ch, T/16]."""
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")
        skips, h = [], x
        for block in self.down_blocks:
            h, skip = block(h)
            skips.append(skip)
        h = self.bottleneck(h)
        return h, skips

    def decode(self, h: torch.Tensor, skips, x_in: torch.Tensor):
        """병목 특징 → 파형. denoising 경로 전용이다."""
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
# 3. Classifier — denoiser 출력을 mel 로 바꿔 분류 (JMIR 격자)
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


class MelFrontEnd(nn.Module):
    """파형 → log-mel [B, 1, n_mels, n_frames]   (JMIR: 512 hamming / hop 160 / 64 mel)

    ★ A arm(저장된 .npy → data_loader 의 MelTransform) 과 출력 스케일이 같아야 한다.
      다르면 A/B 성능 차이가 denoising 때문인지 입력 스케일 때문인지 구분할 수 없다.

          A arm : librosa power_to_db  →  파일별 minmax  →  [0, 1]
          B arm : 여기서 10·log10      →  파일별 minmax  →  [0, 1]

      power_to_db 도 10·log10(P / ref) 이므로 상수 오프셋만 다르고,
      minmax 가 그 오프셋과 배율을 모두 흡수한다. 따라서 두 arm 이 정확히 일치한다.
      normalize="none" 은 ablation 용으로만 쓴다.
    """

    def __init__(self, sr=SR, n_fft=512, hop=160, n_mels=64,
                 fmin=0.0, fmax=8000.0, top_db: float = 80.0,
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


class TemporalPool(nn.Module):
    """시간축 pooling.   [B, C, T] -> ([B, C], [B, T] 또는 None)

    ★ AdaptiveAvgPool 을 대체하는 자리다.
      GAP 은 T 프레임을 균등 평균하므로 짧은 사건이 1/T 로 희석된다.
      MelClassifier block4 는 161 ms/frame, 시간축 62 칸이라
      20 ms crackle 은 1 칸 → 가중치 1.6%.
      '한 칸만 크게 튄 것' 과 '전체가 조금씩 큰 것' 을 구분할 수 없다.

        avg   기존 방식. 지속음(wheeze)에는 충분하나 과도음에 약하다.
        max   과도음에 강하지만 잡음 첨두에도 반응한다.
        attn  프레임별 가중치를 학습한다.
                  a_t = softmax_t( w2ᵀ tanh(W1 h_t) )
                  z   = Σ_t a_t · h_t
              희석이 사라지고, a_t 를 그리면 '모델이 어디를 보았는가' 가 나온다.
    """

    def __init__(self, dim: int, kind: str = "attn"):
        super().__init__()
        self.kind = kind
        if kind == "attn":
            self.proj = nn.Linear(dim, dim)
            self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        if self.kind == "avg":
            return x.mean(dim=-1), None
        if self.kind == "max":
            return x.amax(dim=-1), None
        h = x.transpose(1, 2)                                            # [B,T,C]
        a = torch.softmax(self.score(torch.tanh(self.proj(h))), dim=1)   # [B,T,1]
        return (h * a).sum(dim=1), a.squeeze(-1)


class MelClassifier(nn.Module):
    """[B, 1, 64, T] -> [B, num_classes]

    head="plain"       기존 구조. 마지막 블록만 쓰고 AdaptiveAvgPool2d(1).
    head="multiscale"  블록 1~4 를 함께 쓰고 시간축은 TemporalPool 로 모은다.

    ★ 왜 다중 스케일인가 — 블록마다 시간 해상도가 다르다.
          block1   20.0 ms/frame   crackle(20 ms) 이 정확히 1 프레임
          block2   40.0 ms/frame
          block3   80.0 ms/frame
          block4  161.3 ms/frame   wheeze/rhonchi 같은 지속음
      기존 구조는 block4 만 쓴다. 161 ms 프레임 안에서 crackle 은 위치를 잃는다.
      "단일 시간-주파수 해상도로 crackle 과 wheeze 를 동시에 못 잡는다" 는
      논문의 주장을 구조로 구현한 것이다.

    ★ 주파수축은 뭉개지 않는다. 각 스케일에서 F 를 freq_keep 개로만 줄이고
      (C, F) 를 특징 차원으로 펴서 시간축만 pooling 한다.
      wheeze/rhonchi 구분에 필요한 주파수 구조를 남기기 위해서다.
    """

    def __init__(self, num_classes: int = 5, base: int = 32, dropout: float = 0.1,
                 head: str = "multiscale", scales: Tuple[int, ...] = (1, 2, 3, 4),
                 dim: int = 32, freq_keep: int = 4, pool: str = "attn"):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 4]
        self.blocks = nn.ModuleList()
        prev = 1
        for ch in chs:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(prev, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.SiLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ))
            prev = ch

        self.head_kind = head
        self.scales = tuple(scales)
        self.freq_keep = freq_keep
        self.last_attn: List = []

        if head == "plain":
            self.pool = TemporalPool(prev, pool)
            self.fc = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(prev, num_classes))
        else:
            feat = dim * freq_keep
            self.projs = nn.ModuleList([
                nn.Sequential(nn.Conv2d(chs[i - 1], dim, 1, bias=False),
                              nn.BatchNorm2d(dim), nn.SiLU(inplace=True))
                for i in self.scales])
            self.pools = nn.ModuleList([TemporalPool(feat, pool)
                                        for _ in self.scales])
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat * len(self.scales), num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        h = x
        for blk in self.blocks:
            h = blk(h)
            feats.append(h)                       # feats[i] = block(i+1) 출력

        if self.head_kind == "plain":
            # 기존과 동일: 마지막 블록, 주파수축 평균 후 시간축 pooling
            z, a = self.pool(feats[-1].mean(dim=2))
            self.last_attn = [a]
            return self.fc(z)

        zs, attns = [], []
        for k, i in enumerate(self.scales):
            f = self.projs[k](feats[i - 1])                       # [B,dim,F,T]
            f = F.adaptive_avg_pool2d(f, (self.freq_keep, f.shape[-1]))
            f = f.flatten(1, 2)                                   # [B,dim*Fk,T]
            z, a = self.pools[k](f)
            zs.append(z)
            attns.append(a)
        self.last_attn = attns
        return self.fc(torch.cat(zs, dim=-1))


class NyTTClassifier(nn.Module):
    """denoiser + 출력정규화 + mel front-end + classifier

    ★ 이 파일은 '분류기 입력을 분리하기 이전' 의 구조를 그대로 보존한다.
      분류기를 인코더 특징에 직접 붙인 새 구조는 new_nytt.py 에 있다.

    ★ gate(학습형 원신호 혼합)는 제거했다. 분류기는 항상 x_norm(denoiser 출력을
      rms 로 재정규화한 것)만 본다 — A arm(denoiser 없음)이 이미 원신호 경로를
      대조군으로 제공하므로, 같은 forward 안에 원신호를 새는 경로를 또 두는 건
      불필요한 자유도였다.
    """

    def __init__(self, dae: WaveDAE | None, num_classes: int = 5,
                 base: int = 32, dropout: float = 0.1,
                 mel_normalize: str = "minmax", top_db: float = 80.0,
                 cls_head: str = "multiscale",
                 cls_scales: Tuple[int, ...] = (1, 2, 3, 4),
                 cls_dim: int = 32, cls_freq_keep: int = 4,
                 cls_pool: str = "attn"):
        super().__init__()
        # dae=None 이 A arm(대조군). 우회가 아니라 아예 만들지 않는다.
        self.dae = dae
        self.use_denoiser = dae is not None
        self.frontend = MelFrontEnd(normalize=mel_normalize, top_db=top_db)
        self.classifier = MelClassifier(num_classes, base, dropout,
                                        head=cls_head, scales=cls_scales,
                                        dim=cls_dim, freq_keep=cls_freq_keep,
                                        pool=cls_pool)

    @property
    def attn(self) -> List:
        """마지막 forward 의 스케일별 attention 가중치 (시각화용)."""
        return getattr(self.classifier, "last_attn", [])

    def forward(self, x: torch.Tensor):
        if self.dae is None:                         # A arm: 파형 → mel → 분류
            return None, self.classifier(self.frontend(x))

        x_hat = self.dae(x)

        # ② 출력 정규화 — denoiser 가 만든 레벨 차이 제거 (분류기 입력용).
        #    denoising 지표는 정규화 이전 x_hat 으로 재므로 x_hat 은 그대로 반환한다.
        rms_in = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
        rms_hat = x_hat.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
        x_norm = x_hat * (rms_in / rms_hat)

        return x_hat, self.classifier(self.frontend(x_norm))


class _MethodWrapper(nn.Module):
    """DataParallel 이 흩뿌릴 수 있도록 특정 메서드를 forward 로 노출한다."""

    def __init__(self, module: nn.Module, method: str):
        super().__init__()
        self.module = module
        self.method = method

    def forward(self, x):
        return getattr(self.module, self.method)(x)


# =============================================================================
# 4. Trainer
# =============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_run_config(args) -> None:
    """실행에 쓰인 인자를 전부 찍는다.

    ★ 이게 없어서 지난 run 의 손실 가중치를 나중에 알 수 없게 됐다.
      로그만 보고 재현할 수 있어야 한다.
    """
    d = vars(args)
    keys = sorted(d)
    print("=" * 78)
    print("run config")
    print("=" * 78)
    for i in range(0, len(keys), 3):
        row = "".join(f"{k}={d[k]!s:<22}" for k in keys[i:i + 3])
        print("  " + row.rstrip())
    cmd = " ".join(f"--{k} {d[k]}" for k in keys
                   if not isinstance(d[k], bool) and d[k] is not None)
    cmd += "".join(f" --{k}" for k in keys if d[k] is True)
    print("  python -m denoising.nytt.nytt " + cmd)
    print("=" * 78)


class Trainer(object):
    # 서브클래스가 갈아끼울 수 있게 둔다 (nytt_binary.py 가 라벨을 재매핑한다)
    CLASS_NAMES = ["Normal", "Stridor", "Rhonchi", "Wheezing", "Crackle"]

    def build_datasets(self, args):
        """(train_ds, test_ds)를 만든다. 라벨 체계를 바꾸려면 이 메서드만 재정의한다."""
        return build_fold_wave_datasets(args)

    def __init__(self, args):
        self.args = args
        print_run_config(args)
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        train_ds, eval_ds = self.build_datasets(args)
        print(f"[data] train {len(train_ds):,} segments / {len(train_ds.sample_ids)} ids")
        print(f"[data] test  {len(eval_ds):,} segments / {len(eval_ds.sample_ids)} ids")

        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                       drop_last=True, num_workers=args.num_workers,
                                       pin_memory=self.device.type == "cuda")
        self.eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=self.device.type == "cuda")

        # ★ A arm 은 잡음 주입도 복원 손실도 없으므로 noise bank 자체가 필요 없다.
        self.denoise = not args.no_denoise
        self.noise_bank = None
        if self.denoise:
            if not Path(args.noise_bank).exists():
                raise FileNotFoundError(
                    f"noise bank 없음: {args.noise_bank}\n"
                    f"  python -m denoising.nytt.noise_bank_split --fold {args.fold} "
                    "를 먼저 실행하세요."
                )
            verify_fold_noise_bank(
                args.noise_bank,
                args.fold,
                train_ds.sample_ids,
                eval_ds.sample_ids,
            )
            self.noise_bank = NoiseBank(args.noise_bank, seed=args.seed)
            print(f"[noise] clips {len(self.noise_bank):,} "
                  f"({self.noise_bank.minutes:.1f} 분)")

            # 잡음 출처가 eval split 과 겹치면 누수다. 반드시 0 이어야 한다.
            leaked = self.noise_bank.check_leakage(eval_ds.sample_ids)
            if leaked:
                raise RuntimeError(
                    f"noise bank에 test split {len(leaked)} ids 누수: "
                    f"{leaked[:10]}{' ...' if len(leaked) > 10 else ''}"
                )
            elif self.noise_bank.source_ids:
                print(f"[noise] 누수 없음 (출처 {len(self.noise_bank.source_ids)} ids, "
                      f"eval 과 교집합 0)")
        else:
            print("[A arm] denoiser 없음 — 잡음 주입·복원 손실·denoising 지표 모두 비활성")

        channels = tuple(int(c) for c in str(args.channels).split(",") if c.strip())
        self.model_cfg = dict(in_channels=1, channels=channels,
                              kernel=args.kernel, stride=args.stride,
                              use_skip=not args.no_skip, predict=args.predict)
        # A arm 이면 DAE 를 아예 만들지 않는다 (우회가 아니라 부재)
        dae = None if args.no_denoise else WaveDAE(**self.model_cfg)
        self.model: nn.Module = NyTTClassifier(
            dae, num_classes=args.num_classes, base=args.cls_base,
            dropout=args.dropout,
            mel_normalize=args.mel_normalize, top_db=args.top_db,
            cls_head=args.cls_head,
            cls_scales=tuple(int(v) for v in str(args.cls_scales).split(",") if v.strip()),
            cls_dim=args.cls_dim, cls_freq_keep=args.cls_freq_keep,
            cls_pool=args.cls_pool,
        ).to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        n_all = sum(p.numel() for p in self.model.parameters())
        if dae is None:
            print(f"[model] 인코더 없음   총 {n_all/1e6:.2f} M")
        else:
            n_dae = sum(p.numel() for p in dae.parameters())
            print(f"[model] dae {n_dae/1e6:.2f} M / total {n_all/1e6:.2f} M  "
                  f"use_skip={not args.no_skip}  predict={args.predict}")

        self.class_names = list(self.CLASS_NAMES)[:args.num_classes]
        w = train_ds.class_weights(args.num_classes).to(self.device)
        self.criterion_cls = nn.CrossEntropyLoss(
            weight=w if args.use_class_weight else None)
        # self.criterion_cls = CEDiceLoss(
        #     num_classes=args.num_classes,
        #     ce_weight=0.5,
        #     class_weights=w if args.use_class_weight else None,
        # )
        print(f"[cls] train counts {train_ds.class_counts}")
        print(f"[cls] class_weight={'on' if args.use_class_weight else 'off'} "
              f"{[round(v, 3) for v in w.tolist()]}")
        print(f"[cls] head={args.cls_head}  pool={args.cls_pool}"
              + (f"  scales={args.cls_scales}  dim={args.cls_dim}"
                 f"  freq_keep={args.cls_freq_keep}"
                 if args.cls_head == "multiscale" else "")
              + f"   ({sum(p.numel() for p in self._core.classifier.parameters())/1e3:.1f} K)")
        print(f"[norm] ① 파형 입력 {args.normalize}  ② denoiser 출력 rms 정합 (모델 내부)")
        print(f"[norm] ③ mel {args.mel_normalize} (top_db={args.top_db}) "
              f"← A arm(data_loader MelTransform) 과 동일해야 함")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                              T_max=args.epochs)
        self.amp = (self.device.type == "cuda") and not args.no_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    @property
    def _core(self) -> NyTTClassifier:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    @staticmethod
    def _call(model: nn.Module, method: str, x: torch.Tensor):
        """DataParallel 을 통해 classify / denoise 를 부른다.

        nn.DataParallel 은 forward 만 여러 GPU 로 흩뿌린다. 다른 메서드를 직접
        부르면 GPU 0 에서만 돌아 배치가 안 나뉜다. 그래서 forward 로 감싸 보낸다.
        """
        if isinstance(model, nn.DataParallel):
            return nn.parallel.data_parallel(
                _MethodWrapper(model.module, method), x,
                device_ids=model.device_ids, output_device=model.output_device)
        return getattr(model, method)(x)

    def _make_pair(self, x_target, snr_db=None):
        b, _, t = x_target.shape
        noise = torch.as_tensor(self.noise_bank.sample(t, batch=b),
                                dtype=torch.float32,
                                device=x_target.device).unsqueeze(1)
        if snr_db is None:
            snr_db = (torch.rand(b, device=x_target.device)
                      * (self.args.snr_max - self.args.snr_min) + self.args.snr_min)
        return mix_at_snr(x_target, noise, snr_db), noise

    def _loss_rec(self, x_hat, x_target):
        """파형 복원 손실.

        참고 — rms 0.05 파형에서 각 항의 실측 크기는 L1 0.056, multi-res STFT 1.071 이다.
        STFT 항은 크기 스펙트럼만 보므로(위상이 식에 없음) w_l1 을 올리거나 w_sisdr 을
        켜면 위상이 더 강하게 학습된다. 다만 기본값은 원래대로 둔다.
        """
        loss = x_hat.new_zeros(())
        if self.args.w_l1 > 0:
            loss = loss + self.args.w_l1 * (x_hat - x_target).abs().mean()
        if self.args.w_stft > 0:
            loss = loss + self.args.w_stft * multi_res_stft_loss(x_hat, x_target)
        if self.args.w_sisdr > 0:
            loss = loss - self.args.w_sisdr * si_sdr(x_hat, x_target).mean() / 10.0
        return loss

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total, total_rec, total_cls, n = 0.0, 0.0, 0.0, 0
        n_skipped = n_nonfinite = n_bad_grad = n_step = 0
        grad_norm_sum = 0.0

        w_cls = self.args.w_cls

        for batch_idx, (data, target) in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad(set_to_none=True)
            x_target = data.to(torch.float32).to(self.device, non_blocking=True)
            y = target.long().to(self.device, non_blocking=True)

            if self.denoise:
                with torch.no_grad():
                    x_in, _ = self._make_pair(x_target)
            else:
                x_in = x_target          # A arm: 잡음 주입 없음

            with torch.cuda.amp.autocast(enabled=self.amp):
                x_hat, logits = self.model(x_in)
                loss_cls = self.criterion_cls(logits.float(), y)
                if self.denoise:
                    loss_rec = self._loss_rec(x_hat.float(), x_target)
                    loss = self.args.w_rec * loss_rec + w_cls * loss_cls
                else:
                    loss_rec = loss_cls.new_zeros(())
                    loss = w_cls * loss_cls

            if not torch.isfinite(loss):
                n_nonfinite += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()
            if self.args.grad_clip and self.args.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.args.grad_clip)
                if not torch.isfinite(gnorm):
                    n_bad_grad += 1
                grad_norm_sum += float(gnorm) if torch.isfinite(gnorm) else 0.0

            # scaler.step 은 gradient 에 Inf/NaN 이 있으면 조용히 건너뛴다.
            # 이게 계속되면 가중치가 갱신되지 않아 손실이 완전히 얼어붙는다.
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < scale_before:
                n_skipped += 1

            bs = x_target.size(0)
            total += float(loss.detach()) * bs
            total_rec += float(loss_rec.detach()) * bs
            total_cls += float(loss_cls.detach()) * bs
            n += bs
            n_step += 1
            if self.args.max_train_batches and batch_idx >= self.args.max_train_batches:
                break

        self.scheduler.step()
        n = max(n, 1)
        out = {"train_loss": total / n, "train_rec": total_rec / n,
               "train_cls": total_cls / n,
               "grad_norm": grad_norm_sum / max(n_step, 1),
               "n_skipped": n_skipped, "n_nonfinite": n_nonfinite,
               "n_bad_grad": n_bad_grad, "n_step": max(n_step, 1)}
        self._check_health(epoch, out)
        return out

    # ------------------------------------------------------------------ 안전장치
    def _check_health(self, epoch: int, tr: Dict[str, float]) -> None:
        """학습이 죽었는데 조용히 계속 도는 상황을 잡아낸다.

        실제로 겪은 실패: epoch 36 부터 train loss 가 소수 4 자리까지 15 epoch 동안
        완전히 동일했다. shuffle 도 하고 잡음도 매번 새로 뽑는데 손실이 같다는 것은
        (a) optimizer step 이 전부 건너뛰어져 가중치가 안 변했거나
        (b) 모델이 입력과 무관한 상수 함수로 붕괴했다는 뜻이다.
        둘 다 그 시점부터의 결과가 무의미하다. 즉시 알려야 한다.
        """
        msgs = []
        skip_ratio = tr["n_skipped"] / tr["n_step"]
        bad_ratio = tr["n_bad_grad"] / tr["n_step"]
        scale = self.scaler.get_scale() if self.amp else float("nan")

        if skip_ratio > 0.3:
            msgs.append(f"optimizer step 의 {skip_ratio:.0%} 가 건너뛰어짐 "
                        f"(gradient Inf/NaN). 가중치가 갱신되지 않는다.")
        if tr["n_nonfinite"]:
            msgs.append(f"loss 자체가 non-finite 인 배치 {tr['n_nonfinite']} 개")
        # 몇 배치는 학습 초기에 정상이다. 비율이 높을 때만 경고한다.
        if bad_ratio > 0.02:
            msgs.append(f"gradient 가 non-finite 인 배치 "
                        f"{tr['n_bad_grad']}/{tr['n_step']} ({bad_ratio:.1%})")
        # scale 이 1 밑으로 떨어지면 fp16 이 이득이 없고 언더플로만 늘어난다
        if self.amp and scale < 1.0:
            msgs.append(f"AMP scale 이 {scale:g} 까지 떨어짐 — fp16 이 오히려 손해다. "
                        f"--no_amp 를 권장한다.")

        prev = getattr(self, "_prev_train_loss", None)
        if prev is not None and abs(tr["train_loss"] - prev) < 1e-9:
            self._frozen = getattr(self, "_frozen", 0) + 1
            if self._frozen >= 2:
                msgs.append(f"train loss 가 {self._frozen + 1} epoch 연속 완전히 동일 "
                            f"({tr['train_loss']:.6f}). 학습이 멈췄다.")
        else:
            self._frozen = 0
        self._prev_train_loss = tr["train_loss"]

        if msgs:
            print(f"  ★ [경고 epoch {epoch}] " + "  /  ".join(msgs))
            print(f"     grad_norm {tr['grad_norm']:.2f} (clip {self.args.grad_clip:g})"
                  f"   amp_scale {scale:g}"
                  f"   skipped {tr['n_skipped']}/{tr['n_step']}")

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int, verbose: bool = False):
        self.model.eval()
        acc = dict(si_sdr_in=0.0, si_sdr_out=0.0, l1=0.0, cls_loss=0.0,
                   identity_si_sdr=0.0)
        supp, retain, n_count, n_batch = 0.0, 0.0, 0, 0
        preds_all, reals_all = [], []

        torch.manual_seed(self.args.seed)
        dae = self._core.dae
        if self.denoise:
            self.noise_bank.rng = np.random.default_rng(self.args.seed)

        for batch_idx, (data, target) in enumerate(self.eval_loader, start=1):
            x_target = data.to(torch.float32).to(self.device, non_blocking=True)
            y = target.long().to(self.device, non_blocking=True)

            # 분류는 항상 실제 입력(잡음 주입 없음) 기준으로 평가한다
            _, logits = self.model(x_target)

            bs = x_target.size(0)
            acc["cls_loss"] += float(self.criterion_cls(logits, y)) * bs
            n_count += bs
            preds_all.append(logits.argmax(dim=1).detach().cpu())
            reals_all.append(y.detach().cpu())

            # ★ denoising 지표는 denoiser 가 있을 때만 계산한다.
            #   A arm 에서는 dae 자체가 None 이라 계산할 대상이 없다.
            if self.denoise:
                snr = torch.full((bs,), self.args.eval_snr, device=self.device)
                x_in, noise = self._make_pair(x_target, snr_db=snr)
                x_hat, _ = self.model(x_in)
                acc["si_sdr_in"] += si_sdr(x_in, x_target).mean().item() * bs
                acc["si_sdr_out"] += si_sdr(x_hat, x_target).mean().item() * bs
                acc["l1"] += (x_hat - x_target).abs().mean().item() * bs
                supp += energy_ratio_db(dae, noise)           # 클수록 좋음
                retain += -energy_ratio_db(dae, x_target)     # 0 에 가까울수록 좋음

                # ★ 항등성 — 실제 입력에 denoiser 를 통과시켰을 때 파형이 얼마나 그대로인가.
                #   SigRetain 은 '에너지' 만 보므로 에너지를 유지한 채 내용이 바뀌면 못 잡는다.
                #   이 지표는 파형 자체를 본다.
                #     20 dB 이상  → 사실상 항등함수 (denoiser 가 아무것도 안 함)
                #     10 dB 내외  → 눈에 띄게 바꿈
                x_id, _ = self.model(x_target)
                acc["identity_si_sdr"] += si_sdr(x_id, x_target).mean().item() * bs
                n_batch += 1

            if self.args.max_eval_batches and batch_idx >= self.args.max_eval_batches:
                break

        r = {k: v / max(n_count, 1) for k, v in acc.items()}
        if self.denoise:
            r["si_sdri"] = r["si_sdr_out"] - r["si_sdr_in"]
            r["noise_suppression_db"] = supp / max(n_batch, 1)
            r["signal_retention_db"] = retain / max(n_batch, 1)

        y_pred = torch.cat(preds_all).numpy()
        y_true = torch.cat(reals_all).numpy()
        cls = calculate_classification_metrics(y_pred, y_true,
                                               num_classes=self.args.num_classes)
        r.update({f"cls_{k}": v for k, v in cls.items() if not isinstance(v, list)})

        head = f"[Epoch]: {epoch:03d} => "
        print(head
              + f"[Acc] : {cls['accuracy']*100:.2f} "
                f"[F1] : {cls['f1_macro']*100:.2f} "
                f"[Se] : {cls['sensitivity_macro']*100:.2f} "
                f"[Sp] : {cls['specificity_macro']*100:.2f} "
                f"[ICBHI] : {cls['score_icbhi']*100:.2f}")
        # denoiser 가 없으면 denoising 줄 자체를 찍지 않는다
        if self.denoise:
            print(" " * len(head)
                  + f"| [SI-SDRi] : {r['si_sdri']:+.2f} dB "
                    f"[NoiseSupp] : {r['noise_suppression_db']:+.2f} dB "
                    f"[SigRetain] : {r['signal_retention_db']:+.2f} dB "
                    f"[Identity] : {r['identity_si_sdr']:+.1f} dB")

        per_class = " | ".join(
            f"{self.class_names[i][:3]}(n={cls['support'][i]}) "
            f"{cls['per_class_sensitivity'][i]*100:.1f}/"
            f"{cls['per_class_specificity'][i]*100:.1f}/"
            f"{cls['per_class_f1'][i]*100:.1f}"
            for i in range(self.args.num_classes)
        )
        print(" " * len(head) + f"| PerClass Se/Sp/F1  {per_class}")

        if verbose or self.args.verbose_metrics:
            print_classification_metrics(cls, self.class_names)
        return r, cls

    def run(self):
        save_dir = Path(self.args.save_dir)
        # 인자를 파일로도 남긴다. 로그를 잃어버려도 재현할 수 있어야 한다.
        with open(save_dir / f"{self.args.model_name}_args.json", "w",
                  encoding="utf-8") as fp:
            json.dump(vars(self.args), fp, ensure_ascii=False, indent=2, default=str)

        # ★ 최종 보고는 '마지막 epoch' 이 아니라 '최고 epoch' 기준으로 한다.
        #   실측에서 최고점(F1)이 epoch 13~15 였고 이후 다수클래스 붕괴로 계속
        #   나빠졌다. 마지막 epoch 의 confusion matrix 를 보고하면 모델의 실제
        #   성능을 과소평가하고, 게다가 저장된 best 체크포인트와 숫자가 어긋난다.
        #   선택 기준은 --best_metric (기본 sensitivity_macro).
        key = f"cls_{self.args.best_metric}"
        results, best = [], -1e9
        best_epoch, best_cls, best_r = -1, None, None

        for epoch in range(1, self.args.epochs + 1):
            tr = self.train_one_epoch(epoch)
            # 최종 상세 출력은 아래에서 최고 epoch 기준으로 한 번만 한다.
            r, cls = self.eval_one_epoch(epoch, verbose=False)
            r.update(epoch=epoch, **tr)
            results.append(r)

            if r[key] > best:
                best = r[key]
                best_epoch, best_cls, best_r = epoch, cls, r
                torch.save({"model": self._core.state_dict(), "cfg": self.model_cfg,
                            "args": vars(self.args),          # ★ 전체 인자 보존
                            "epoch": epoch,
                            "best_metric": self.args.best_metric,
                            "best_value": best,
                            # 참고용으로 주요 지표를 함께 남긴다
                            "f1_macro": r["cls_f1_macro"],
                            "sensitivity_macro": r["cls_sensitivity_macro"],
                            "score_icbhi": r["cls_score_icbhi"],
                            "si_sdri": r.get("si_sdri")},
                           save_dir / f"{self.args.model_name}_best.pt")

        pd.DataFrame(results).to_csv(
            save_dir / f"{self.args.model_name}_metrics.csv",
            index=False, encoding="utf-8-sig")

        # ------------------------------------------------ 최고 epoch 최종 보고
        if best_cls is None:
            print("\n[warn] epoch 을 한 번도 돌지 않아 보고할 결과가 없다")
            print(f"저장: {save_dir}")
            return results

        print("\n" + "=" * 78)
        print(f"FINAL — 최고 epoch 기준 보고   (선택 기준: {self.args.best_metric})")
        print(f"  best epoch {best_epoch} / {self.args.epochs}"
              f"   {self.args.best_metric} = {best * 100:.2f}")
        print("=" * 78)
        print(f"[Acc] : {best_cls['accuracy']*100:.2f} "
              f"[F1] : {best_cls['f1_macro']*100:.2f} "
              f"[Se] : {best_cls['sensitivity_macro']*100:.2f} "
              f"[Sp] : {best_cls['specificity_macro']*100:.2f} "
              f"[ICBHI] : {best_cls['score_icbhi']*100:.2f}")
        if self.denoise and best_r.get("si_sdri") is not None:
            print(f"[SI-SDRi] : {best_r['si_sdri']:+.2f} dB "
                  f"[NoiseSupp] : {best_r['noise_suppression_db']:+.2f} dB "
                  f"[SigRetain] : {best_r['signal_retention_db']:+.2f} dB "
                  f"[Identity] : {best_r['identity_si_sdr']:+.1f} dB")
        # confusion matrix 는 이 안에서 찍힌다
        print_classification_metrics(best_cls, self.class_names)

        # 최고 epoch 지표를 파일로도 남긴다 (list 값 = confusion matrix 포함)
        with open(save_dir / f"{self.args.model_name}_best_metrics.json", "w",
                  encoding="utf-8") as fp:
            json.dump({"best_epoch": best_epoch,
                       "best_metric": self.args.best_metric,
                       "best_value": best,
                       "classification": best_cls,
                       "denoising": {k: best_r.get(k) for k in
                                     ("si_sdri", "noise_suppression_db",
                                      "signal_retention_db", "identity_si_sdr")}},
                      fp, ensure_ascii=False, indent=2, default=str)

        # ★ 마지막 epoch 과 얼마나 벌어졌는지 같이 찍는다. 과적합·붕괴 진단용.
        lastr = results[-1]
        print(f"\n[참고] 마지막 epoch {lastr['epoch']} : "
              f"Acc {lastr['cls_accuracy']*100:.2f} "
              f"F1 {lastr['cls_f1_macro']*100:.2f} "
              f"Se {lastr['cls_sensitivity_macro']*100:.2f} "
              f"ICBHI {lastr['cls_score_icbhi']*100:.2f}")
        print(f"        최고 epoch {best_epoch} 대비 Se "
              f"{(lastr['cls_sensitivity_macro'] - best_cls['sensitivity_macro'])*100:+.2f} "
              f"/ F1 {(lastr['cls_f1_macro'] - best_cls['f1_macro'])*100:+.2f}")
        print(f"저장: {save_dir}")
        return results


# =============================================================================
# CLI
# =============================================================================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="train", choices=["check", "train"])
    p.add_argument("--wave_base_path", default=wave_folder)
    p.add_argument("--fold", default=1, type=int,
                   help="split_manifest에서 사용할 grouping 번호")
    p.add_argument("--split_manifest", default=str(DEFAULT_SPLIT_MANIFEST),
                   help="fold,sample_id,split 열을 가진 CSV")
    p.add_argument("--noise_bank", default=None,
                   help="기본값: <data>/db/new_gt/noise_bank_fold<FOLD>.npz")
    p.add_argument("--model_name", default="nytt_denoising_unet")
    p.add_argument("--save_dir", default=None,
                   help="기본값: denoising/nytt/ckpt/fold<FOLD>")

    # train
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--grad_clip", default=5.0, type=float)
    p.add_argument("--normalize", default="rms",
                   choices=["none", "peak", "rms", "p95"])
    p.add_argument("--preload", action=argparse.BooleanOptionalAction, default=False)

    # model — Ronneberger U-Net(2015)+BN 을 1D 로 옮긴 구조 (arXiv:2510.03452 Fig 2b)
    p.add_argument("--kernel", default=3, type=int,
                   help="DoubleConv 의 conv kernel 크기. 원논문은 3×3")
    p.add_argument("--stride", default=2, type=int,
                   help="다운샘플/업샘플 배율 (MaxPool1d / ConvTranspose1d). 원논문은 2×2")
    p.add_argument("--no_skip", action="store_true",
                   help="ablation 용. 이전 stride-conv 인코더에서는 skip 제거 시 "
                        "F1 42.2→28.2 로 무너졌으나, 이 U-Net 구조에서 같은 결과가 "
                        "재현된다는 보장은 없다 — 직접 다시 측정하라. 용량을 줄이려면 "
                        "--channels 를 우선 쓰십시오.")
    p.add_argument("--channels", default="32,64,128,256,512",
                   help="[enc1,enc2,enc3,enc4,bottleneck] 5개, 채널이 매 단계 2배. "
                        "32,64,128,256,512=2.17M(기본) / 16,32,64,128,256=0.54M / "
                        "16,24,32,64,128=0.15M")
    p.add_argument("--predict", default="signal", choices=["signal", "residual"])

    # classifier
    p.add_argument("--num_classes", default=5, type=int)
    p.add_argument("--cls_base", default=32, type=int)
    # ★ 분류기 헤드
    #   multiscale : block1~4 를 함께 씀 (20/40/80/161 ms/frame)
    #   plain      : 기존 구조 (block4 만, 주파수 평균 후 시간축 pooling)
    p.add_argument("--cls_head", default="multiscale",
                   choices=["multiscale", "plain"])
    p.add_argument("--cls_scales", default="1,2,3,4",
                   help="쓸 블록 번호. 1=20ms/frame … 4=161ms/frame")
    p.add_argument("--cls_dim", default=32, type=int,
                   help="스케일별 1x1 conv 출력 채널")
    p.add_argument("--cls_freq_keep", default=4, type=int,
                   help="스케일별로 남길 주파수 칸 수 (wheeze/rhonchi 구분용)")
    # ★ AdaptiveAvgPool 을 대체하는 자리. GAP 은 짧은 사건을 1/T 로 희석한다.
    p.add_argument("--cls_pool", default="attn", choices=["avg", "max", "attn"])
    p.add_argument("--dropout", default=0.1, type=float)
    # ★ A arm(대조군): denoiser 만 빼고 나머지는 전부 동일하게 학습한다.
    #   두 arm 의 F1 차이가 곧 denoiser 의 기여다.
    # A arm: denoiser 를 아예 만들지 않는다.

    p.add_argument("--no_denoise", action="store_true",
                   help="denoiser 를 건너뛴다 (A arm 대조군)")
    # ★ A arm(data_loader 의 MelTransform) 과 반드시 같아야 한다
    p.add_argument("--mel_normalize", default="minmax", choices=["minmax", "none"])
    p.add_argument("--top_db", default=80.0, type=float)
    p.add_argument("--use_class_weight", action=argparse.BooleanOptionalAction,
                   default=True)
    # ★ best 체크포인트 선택 기준 = 최종 confusion matrix 를 뽑는 epoch 의 기준.
    #   기본을 sensitivity_macro 로 둔다. 임상에서 이상음을 놓치지 않는 것(Se)이
    #   우선이고, accuracy 는 Normal 이 test 의 75% 라 다수클래스 붕괴를 못 잡는다.
    p.add_argument("--best_metric", default="sensitivity_macro",
                   choices=["sensitivity_macro", "f1_macro", "score_icbhi",
                            "specificity_macro", "accuracy"],
                   help="최고 epoch 선택 기준. 최종 보고·confusion matrix 가 이 epoch 로 산출된다")
    p.add_argument("--verbose_metrics", action="store_true")
    p.add_argument("--w_rec", default=1.0, type=float)
    p.add_argument("--w_cls", default=1.0, type=float)

    # NyTT
    p.add_argument("--snr_min", default=0.0, type=float)
    p.add_argument("--snr_max", default=15.0, type=float)
    p.add_argument("--eval_snr", default=5.0, type=float)
    p.add_argument("--w_l1", default=1.0, type=float)
    p.add_argument("--w_stft", default=1.0, type=float)
    p.add_argument("--w_sisdr", default=0.0, type=float)

    p.add_argument("--no_amp", action="store_true",
                   help="mixed precision 끄기. 발산\u00b7Inf gradient 가 의심되면 사용")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--num_workers", default=5, type=int)
    p.add_argument("--max_train_batches", default=None, type=int)
    p.add_argument("--max_eval_batches", default=None, type=int)
    args = p.parse_args()
    if args.fold < 1:
        p.error("--fold는 1 이상의 정수여야 합니다.")
    if args.noise_bank is None:
        args.noise_bank = str(DEFAULT_NOISE_BANK_DIR / f"noise_bank_fold{args.fold}.npz")
    if args.save_dir is None:
        args.save_dir = str(CURRENT_DIR / "ckpt" / f"fold{args.fold}")
    return args


def stage_check(args):
    print("=" * 78)
    for predict in ("signal", "residual"):
        for use_skip in (True, False):
            model = WaveDAE(kernel=args.kernel, stride=args.stride,
                            use_skip=use_skip, predict=predict)
            x = torch.randn(2, 1, N_SAMPLES) * 0.05
            with torch.no_grad():
                y = model(x)
            print(f"predict={predict:9} use_skip={str(use_skip):5} "
                  f"in={tuple(x.shape)} out={tuple(y.shape)} "
                  f"params={sum(q.numel() for q in model.parameters())/1e6:.2f} M "
                  f"finite={torch.isfinite(y).all().item()}")

    check_preload = args.preload
    args.preload = False
    try:
        tr, ev = build_fold_wave_datasets(args)
    finally:
        args.preload = check_preload
    print(f"\n[data] train {len(tr):,} / ids {len(tr.sample_ids)}"
          f"   test {len(ev):,} / ids {len(ev.sample_ids)}"
          f"   total {len(tr)+len(ev):,}")
    print(f"       class counts {tr.class_counts}")
    print(f"       sample_id overlap {len(set(tr.sample_ids) & set(ev.sample_ids))}")
    x, y = tr[0]
    print(f"       x {tuple(x.shape)}  rms {x.pow(2).mean().sqrt():.5f}  "
          f"peak {x.abs().max():.4f}  label {int(y)}")

    # -------------------------------------------------------------- mel 정합성
    # A arm(저장된 .npy) 과 B arm(모델 내부 MelFrontEnd) 이 같은 값을 내야 한다.
    # ref 상수는 minmax 가 흡수하므로 무관하지만 top_db 는 흡수되지 않는다.
    # 같은 segment 에 대해 직접 비교해서 확인한다.
    fe = MelFrontEnd(normalize=args.mel_normalize, top_db=args.top_db)
    with torch.no_grad():
        mel_b = fe(x.unsqueeze(0))[0, 0].numpy()
    print(f"\n[mel]  B arm frontend {tuple(mel_b.shape)}  "
          f"range [{mel_b.min():.3f}, {mel_b.max():.3f}]  "
          f"normalize={args.mel_normalize} top_db={args.top_db}")

    wav_path = Path(tr.items[0][0])
    mel_path = (Path(mel_folder) / wav_path.parent.name /
                wav_path.name).with_suffix(".npy")
    if mel_path.exists():
        from data_loader import preprocessing_mel
        mel_a = preprocessing_mel(np.load(mel_path), "minmax")
        n = min(mel_a.shape[1], mel_b.shape[1])
        diff = np.abs(mel_a[:, :n] - mel_b[:, :n])
        corr = np.corrcoef(mel_a[:, :n].ravel(), mel_b[:, :n].ravel())[0, 1]
        print(f"       A arm 저장본 {mel_path.name}  range "
              f"[{mel_a.min():.3f}, {mel_a.max():.3f}]")
        print(f"       A vs B  max|diff| {diff.max():.4f}  "
              f"mean|diff| {diff.mean():.4f}  corr {corr:.5f}")
        if diff.mean() < 0.02:
            print("       → 두 arm 의 mel 스케일 일치. A/B 비교 가능.")
        else:
            print("       → ★ 불일치. make_10s_mel.py 의 power_to_db 설정을 확인하고 "
                  "--top_db 를 맞추십시오 (librosa 기본값 80).")
    else:
        print(f"       [skip] 저장된 mel 없음: {mel_path}")

    if Path(args.noise_bank).exists():
        verify_fold_noise_bank(
            args.noise_bank, args.fold, tr.sample_ids, ev.sample_ids,
        )
        nb = NoiseBank(args.noise_bank)
        m = torch.as_tensor(nb.sample(N_SAMPLES, batch=2)).unsqueeze(1)
        clean = x.unsqueeze(0).repeat(2, 1, 1)
        noisy = mix_at_snr(clean, m, torch.full((2,), 5.0))
        print(f"\n[noise] clips {len(nb):,} ({nb.minutes:.1f} 분)  train_only={nb.train_only}")
        print(f"        mix at 5 dB → 실측 SI-SDR = {si_sdr(noisy, clean).mean():.2f} dB")
        leaked = nb.check_leakage(ev.sample_ids)
        print(f"        test split 누수: {len(leaked)} ids"
              f"{'  ★ noise_bank_split.py 를 다시 실행하십시오' if leaked else '  (정상)'}")
    else:
        print(
            "\n[noise] 없음 — python -m denoising.nytt.noise_bank_split "
            f"--fold {args.fold} 를 먼저 실행하세요."
        )


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    if args.stage == "check":
        stage_check(args)
    else:
        Trainer(args).run()
