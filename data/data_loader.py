from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from data.parser import SegmentItem, SegmentMelParser


"""
DataLoader for segment(cycle)-level classification

    domain="mel"   <ROOT>/db/new_gt/10s_mel/<YYMMDD>/<sid>_c000_<label>.npy
                       |  MelTransform (minmax 정규화 + 채널 추가)
                       +--> x [1, 64, 1001]     ← A arm: denoising 없는 대조군
                       +--> y  scalar (long)

    domain="wave"  <ROOT>/db/new_gt/10s_repeat/<YYMMDD>/<sid>_c000_<label>.wav
                       |  WaveTransform (rms 정규화 + 채널 추가)
                       +--> x [1, 160000]       ← B arm: NyTT denoiser 입력
                       +--> y  scalar (long)

"""

Domain = Literal["mel", "wave"]
MelNormalizeKind = Literal["none", "minmax", "instance", "global"]
WaveNormalizeKind = Literal["none", "peak", "rms", "p95"]

SR = 16000
SEC = 10.0
N_SAMPLES = int(SEC * SR)          # 160,000
N_MELS, N_FRAMES = 64, 1001

DOMAIN_CFG = {
    "mel": dict(suffix=".npy", normalize="minmax"),
    "wave": dict(suffix=".wav", normalize="rms"),
}


# -----------------------------------------------------------------------------
# Helpers  (data_loader.preprocessing / sliding_window_1d 에 대응)
# -----------------------------------------------------------------------------
def preprocessing_mel(mel: np.ndarray, kind: MelNormalizeKind = "minmax",
                      stats: Tuple[float, float] | None = None,
                      eps: float = 1e-8) -> np.ndarray:
    """mel(dB) 정규화.

    make_10s_mel.py 는 ref=1.0 기준 dB 로 저장하므로 파일별 절대 레벨이 다르다.
        minmax   : 파일별 [0, 1] 로 스케일
        instance : 파일별 zero-mean unit-var
        global   : 학습 split 에서 구한 (mean, std) 로 전체 통일  ← 파일간 레벨 차이 보존
        none     : 저장값 그대로
    """
    mel = np.asarray(mel, dtype=np.float32)

    if kind == "none":
        return mel
    if kind == "minmax":
        mel = mel - mel.min()
        return mel / max(float(mel.max()), eps)
    if kind == "instance":
        return (mel - mel.mean()) / max(float(mel.std()), eps)
    if kind == "global":
        if stats is None:
            raise ValueError("kind='global' 은 stats=(mean, std) 가 필요합니다.")
        mean, std = stats
        return (mel - mean) / max(float(std), eps)
    raise ValueError(f"Unknown normalize kind: {kind}")


def preprocessing_wave(y: np.ndarray, kind: WaveNormalizeKind = "rms",
                       target_rms: float = 0.05, eps: float = 1e-8) -> np.ndarray:
    """파형 입력 정규화.

    peak 정규화는 접촉 아티팩트에 지배된다(실측 18~29 dB). 그래서 rms 를 기본으로 둔다.
    rms 정규화는 전체에 상수를 곱하는 것이라 SNR·스펙트럼 모양·시간 구조는 그대로 남고
    기기 게인과 접촉 압력에 의한 크기 차이만 사라진다.
    """
    y = np.asarray(y, dtype=np.float32)
    if kind == "none":
        return y
    if kind == "peak":
        return y / max(float(np.abs(y).max()), eps)
    if kind == "p95":
        return y / max(float(np.percentile(np.abs(y), 95)), eps)
    if kind == "rms":
        return y * (target_rms / max(float(np.sqrt((y ** 2).mean())), eps))
    raise ValueError(f"Unknown wave normalize kind: {kind}")


def fix_frames(mel: np.ndarray, n_frames: int) -> np.ndarray:
    """프레임 수를 n_frames 로 강제. 부족하면 0 패딩, 넘치면 잘라낸다."""
    if mel.shape[1] == n_frames:
        return mel
    if mel.shape[1] > n_frames:
        return mel[:, :n_frames]
    pad = np.zeros((mel.shape[0], n_frames - mel.shape[1]), dtype=mel.dtype)
    return np.concatenate([mel, pad], axis=1)


def fix_length(y: np.ndarray, n_samples: int) -> np.ndarray:
    """샘플 수를 n_samples 로 강제. 10s repeat padding 이 되어 있으므로 보정은 거의 없다."""
    if len(y) == n_samples:
        return y
    if len(y) > n_samples:
        return y[:n_samples]
    return np.pad(y, (0, n_samples - len(y)))


@dataclass
class MelTransform:
    """이미 mel 인 입력에 적용하는 변환 (transform.WindowTransform 과 같은 역할)"""
    normalize: MelNormalizeKind = "minmax"
    stats: Tuple[float, float] | None = None
    output_channels: int = 1

    def __call__(self, mel, label) -> Tuple[torch.Tensor, torch.Tensor]:
        mel = preprocessing_mel(np.asarray(mel, dtype=np.float32),
                                self.normalize, self.stats)
        image = torch.as_tensor(mel, dtype=torch.float32).unsqueeze(0)
        if self.output_channels > 1:
            image = image.repeat(self.output_channels, 1, 1)
        return image, torch.as_tensor(label, dtype=torch.long)


@dataclass
class WaveTransform:
    """파형 입력에 적용하는 변환. MelTransform 과 같은 자리에 놓인다."""
    normalize: WaveNormalizeKind = "rms"
    target_rms: float = 0.05

    def __call__(self, y, label) -> Tuple[torch.Tensor, torch.Tensor]:
        y = preprocessing_wave(np.asarray(y, dtype=np.float32),
                               self.normalize, self.target_rms)
        wave = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0)   # [1, T]
        return wave, torch.as_tensor(label, dtype=torch.long)


# -----------------------------------------------------------------------------
# Base Dataset
# -----------------------------------------------------------------------------
class SegmentMelDataset(Dataset):
    """domain="mel" 이면 .npy 를, domain="wave" 이면 .wav 를 읽는다."""

    def __init__(
        self,
        base_path: str,
        *,
        domain: Domain = "mel",
        train: bool = True,
        train_ratio: float = 0.6,          # ICBHI 2017 challenge 표준 6:4
        sample_ids: Sequence[str] | None = None,
        n_mels: int = N_MELS,
        n_frames: int = N_FRAMES,
        n_samples: int = N_SAMPLES,
        label_map: Dict[str, int] | None = None,
        exclude_label: int = -1,
        use_cohort_filter: bool = True,
        input_type: str | MelTransform | WaveTransform | None = None,
        target_rms: float = 0.05,
        preload: bool | None = None,       # None 이면 mel=True, wave=False
    ):
        super().__init__()
        if domain not in DOMAIN_CFG:
            raise ValueError(f"domain 은 'mel' 또는 'wave' 여야 합니다: {domain}")
        cfg = DOMAIN_CFG[domain]

        self.base_path = base_path
        self.mel_base_path = base_path            # 이전 이름 호환
        self.domain = domain
        self.train = train
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.preload = (domain == "mel") if preload is None else preload

        self.parser = SegmentMelParser(
            base_path,
            suffix=cfg["suffix"],
            label_map=label_map,
            exclude_label=exclude_label,
            use_cohort_filter=use_cohort_filter,
        )

        sample_items = self.parser.build_sample_items()
        if len(sample_items) == 0:
            raise FileNotFoundError(
                f"No usable segment files ({cfg['suffix']}) under: {base_path}\n"
                f"  resolved={Path(base_path).expanduser().resolve()} "
                f"exists={Path(base_path).expanduser().exists()}\n"
                f"  scanned={self.parser.n_files} "
                f"unparsed={len(self.parser.unparsed)} "
                f"no_label={len(self.parser.no_label_file)} "
                f"out_of_range={len(self.parser.out_of_range)} "
                f"dropped={sum(self.parser.dropped.values())}"
            )

        if sample_ids is None:
            # 이전 실행과의 호환용 정렬 기반 분할. 새 실험은 manifest ID를 명시한다.
            split = int(len(sample_items) * float(train_ratio))
            target_sample_items = sample_items[:split] if train else sample_items[split:]
        else:
            # recording-level manifest가 지정한 ID만 사용한다. manifest 순서를 보존하고,
            # 없는 ID를 조용히 버리지 않는다(잘못된 split로 학습되는 것을 방지).
            requested_ids = [str(sample_id).strip() for sample_id in sample_ids]
            duplicated = sorted({
                sample_id for sample_id in requested_ids
                if requested_ids.count(sample_id) > 1
            })
            if duplicated:
                raise ValueError(f"명시한 sample_ids가 중복됩니다: {duplicated}")

            items_by_id = dict(sample_items)
            missing = [sample_id for sample_id in requested_ids
                       if sample_id not in items_by_id]
            if missing:
                raise FileNotFoundError(
                    "manifest ID에 대응하는 usable segment가 없습니다: "
                    f"{missing}\nbase_path={base_path}"
                )
            target_sample_items = [
                (sample_id, items_by_id[sample_id])
                for sample_id in requested_ids
            ]
        self.sample_ids = [sample_id for sample_id, _ in target_sample_items]
        self.items: List[SegmentItem] = [
            item
            for _, segment_items in target_sample_items
            for item in segment_items
        ]
        if len(self.items) == 0:
            raise FileNotFoundError("No segments remained after train/eval split.")

        self.label_arr = np.array([label_id for _, label_id, _, _ in self.items],
                                  dtype=np.int64)

        if input_type is None:
            input_type = cfg["normalize"]
        if isinstance(input_type, str):
            if domain == "mel":
                stats = self._compute_global_stats() if input_type == "global" else None
                self.input_transform = MelTransform(normalize=input_type, stats=stats)
            else:
                self.input_transform = WaveTransform(normalize=input_type,
                                                     target_rms=target_rms)
        else:
            self.input_transform = input_type

        self.data_arr = self._load_all(self.items) if self.preload else None

    # ------------------------------------------------------------------ load
    def _load_one(self, path: str | Path) -> np.ndarray:
        if self.domain == "mel":
            mel = np.load(path).astype(np.float32, copy=False)
            if mel.shape[0] != self.n_mels:
                raise ValueError(
                    f"n_mels mismatch: expected {self.n_mels}, got {mel.shape[0]} ({path})"
                )
            return fix_frames(mel, self.n_frames)

        y, sr = sf.read(str(path), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = np.asarray(y, dtype=np.float32)
        if int(sr) != SR:
            import librosa
            y = librosa.resample(y, orig_sr=int(sr), target_sr=SR).astype(np.float32)
        return fix_length(y, self.n_samples)

    def _load_all(self, items: Sequence[SegmentItem]) -> np.ndarray:
        shape = ((len(items), self.n_mels, self.n_frames) if self.domain == "mel"
                 else (len(items), self.n_samples))
        data_arr = np.empty(shape, dtype=np.float32)
        for i, (path, _, _, _) in enumerate(items):
            data_arr[i] = self._load_one(path)
        return data_arr

    def _compute_global_stats(self, max_files: int = 500) -> Tuple[float, float]:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(self.items), size=min(max_files, len(self.items)),
                         replace=False)
        acc = np.concatenate([self._load_one(self.items[i][0]).ravel() for i in idx])
        return float(acc.mean()), float(acc.std())

    # ------------------------------------------------------------------ imbalance
    @property
    def class_counts(self) -> Dict[int, int]:
        values, counts = np.unique(self.label_arr, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def class_weights(self, num_classes: int | None = None) -> torch.Tensor:
        """CrossEntropyLoss(weight=...) 용. n_total / (n_class * count)

        num_classes 를 주면 그 길이로 맞춘다. train split 에 한 번도 안 나온 클래스가
        있어도 텐서 길이가 줄지 않게 하려는 것이다 (weight 1.0 으로 남음).
        """
        counts = self.class_counts
        n = num_classes or (max(counts) + 1)
        weights = torch.ones(n, dtype=torch.float32)
        total = sum(counts.values())
        for label_id, count in counts.items():
            if 0 <= label_id < n:
                weights[label_id] = total / (len(counts) * count)
        return weights

    def sampler(self) -> WeightedRandomSampler:
        """DataLoader(sampler=...) 용. 클래스 균등 샘플링."""
        counts = self.class_counts
        sample_weights = np.array([1.0 / counts[int(y)] for y in self.label_arr],
                                  dtype=np.float64)
        return WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    # ------------------------------------------------------------------ dataset api
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (self.data_arr[idx] if self.data_arr is not None
             else self._load_one(self.items[idx][0]))
        return self.input_transform(x, self.label_arr[idx])


# -----------------------------------------------------------------------------
# Concrete Datasets
# -----------------------------------------------------------------------------
class LungSegmentDataset(SegmentMelDataset):
    """SNUCH Child Lung Sound — segment(cycle) level classification
        LungSegmentDataset(mel_folder,  domain="mel")   → x [1, 64, 1001]
        LungSegmentDataset(wave_folder, domain="wave")  → x [1, 160000]
    """

    def __init__(
        self,
        base_path: str,
        *,
        domain: Domain = "mel",
        train: bool = True,
        train_ratio: float = 0.6,          # ICBHI 2017 challenge 표준 6:4
        sample_ids: Sequence[str] | None = None,
        use_cohort_filter: bool = True,
        input_type: str | MelTransform | WaveTransform | None = None,
        target_rms: float = 0.05,
        preload: bool | None = None,
    ):
        super().__init__(
            base_path,
            domain=domain,
            train=train,
            train_ratio=train_ratio,
            sample_ids=sample_ids,
            use_cohort_filter=use_cohort_filter,
            input_type=input_type,
            target_rms=target_rms,
            preload=preload,
            n_mels=N_MELS,
            n_frames=N_FRAMES,
            n_samples=N_SAMPLES,
        )


if __name__ == "__main__":
    from data.utils import mel_folder, wave_folder

    for domain, folder in (("mel", mel_folder), ("wave", wave_folder)):
        if not os.path.isdir(folder):
            print(f"[skip] {folder} 없음\n")
            continue

        train_dataset = LungSegmentDataset(folder, domain=domain, train=True,
                                           train_ratio=0.6, preload=False)
        eval_dataset = LungSegmentDataset(folder, domain=domain, train=False,
                                          train_ratio=0.6, preload=False)
        inv_map = {v: k for k, v in train_dataset.parser.label_map.items()}

        n_total = len(train_dataset) + len(eval_dataset)
        print(f"===== domain = {domain}  ({folder}) =====")
        print(f"usable segments (total) : {n_total:,}"
              f"   ids: {len(train_dataset.sample_ids) + len(eval_dataset.sample_ids)}")
        print(f"  train : {len(train_dataset):,}"
              f"  ({len(train_dataset)/n_total:.1%})   ids: {len(train_dataset.sample_ids)}")
        print(f"  eval  : {len(eval_dataset):,}"
              f"  ({len(eval_dataset)/n_total:.1%})   ids: {len(eval_dataset.sample_ids)}")
        print(f"sample_id overlap: "
              f"{len(set(train_dataset.sample_ids) & set(eval_dataset.sample_ids))}")
        print(f"normalize: {train_dataset.input_transform.normalize}")

        print("class counts (train)")
        for label_id, count in sorted(train_dataset.class_counts.items()):
            print(f"  {label_id} {inv_map.get(label_id, '?'):10} {count:6,}"
                  f"  {count/len(train_dataset):7.2%}")
        print(f"class weights: "
              f"{[round(w, 4) for w in train_dataset.class_weights(5).tolist()]}")

        x, y = train_dataset[0]
        print(f"sample  x {tuple(x.shape)}  dtype {x.dtype}")
        if domain == "mel":
            print(f"        range [{x.min():.3f}, {x.max():.3f}]")
        else:
            print(f"        rms {x.pow(2).mean().sqrt():.5f}  peak {x.abs().max():.4f}")
        print(f"        label {int(y)} ({inv_map.get(int(y), '?')})\n")
