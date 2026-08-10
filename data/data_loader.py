from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from parser import SegmentItem, SegmentMelParser


"""
segment(cycle) 단위 분류용 DataLoader

    <ROOT>/new_gt/10s_mel/<YYMMDD>/<sample_id>_c000_<label>.npy   shape (64, 1001)
            |
            v
    MelTransform (정규화 + 채널 추가)
            |
            +--> x  [1, 64, 1001]
            +--> y  scalar (long)
"""

MelNormalizeKind = Literal["none", "minmax", "instance", "global"]


# -----------------------------------------------------------------------------
# Helpers  (data_loader.preprocessing / sliding_window_1d 에 대응)
# -----------------------------------------------------------------------------
def preprocessing_mel(mel: np.ndarray, kind: MelNormalizeKind = "minmax",
                      stats: Tuple[float, float] | None = None,
                      eps: float = 1e-8) -> np.ndarray:
    """mel(dB) 정규화.

    make_10s_mel.py 는 ref=1.0 기준 dB 로 저장하므로 파일별 절대 레벨이 다르다.
        minmax   : 파일별 [0, 1] 로 스케일  (transform.WindowTransform._log_normalize 와 동일 사상)
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


def fix_frames(mel: np.ndarray, n_frames: int) -> np.ndarray:
    """프레임 수를 n_frames 로 강제. 부족하면 0 패딩, 넘치면 잘라낸다."""
    if mel.shape[1] == n_frames:
        return mel
    if mel.shape[1] > n_frames:
        return mel[:, :n_frames]
    pad = np.zeros((mel.shape[0], n_frames - mel.shape[1]), dtype=mel.dtype)
    return np.concatenate([mel, pad], axis=1)


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


# -----------------------------------------------------------------------------
# Base Dataset  (data_loader.SlidingWindowDataset 에 대응)
# -----------------------------------------------------------------------------
class SegmentMelDataset(Dataset):
    def __init__(
        self,
        mel_base_path: str,
        *,
        train: bool = True,
        train_ratio: float = 0.6,          # ICBHI 2017 challenge 표준 6:4
        n_mels: int = 64,
        n_frames: int = 1001,
        label_map: Dict[str, int] | None = None,
        exclude_label: int = -1,
        use_cohort_filter: bool = True,
        input_type: MelNormalizeKind | MelTransform = "minmax",
        preload: bool = True,
    ):
        super().__init__()
        self.mel_base_path = mel_base_path
        self.train = train
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.preload = preload

        self.parser = SegmentMelParser(
            mel_base_path,
            label_map=label_map,
            exclude_label=exclude_label,
            use_cohort_filter=use_cohort_filter,
        )

        sample_items = self.parser.build_sample_items()
        if len(sample_items) == 0:
            raise FileNotFoundError(
                f"No matched segment mel files under: {mel_base_path}"
            )

        # sample_id(=녹음) 단위 분할. data_loader 와 동일하게 정렬 순서로 자른다.
        split = int(len(sample_items) * float(train_ratio))
        target_sample_items = sample_items[:split] if train else sample_items[split:]
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

        if isinstance(input_type, str):
            stats = self._compute_global_stats() if input_type == "global" else None
            self.input_transform = MelTransform(normalize=input_type, stats=stats)
        else:
            self.input_transform = input_type

        # eager load  (data_loader 의 self.data_arr 와 동일한 역할)
        self.data_arr = self._load_all(self.items) if preload else None

    # ------------------------------------------------------------------ load
    def _load_one(self, mel_path: str | Path) -> np.ndarray:
        mel = np.load(mel_path).astype(np.float32, copy=False)
        if mel.shape[0] != self.n_mels:
            raise ValueError(
                f"n_mels mismatch: expected {self.n_mels}, got {mel.shape[0]} ({mel_path})"
            )
        return fix_frames(mel, self.n_frames)

    def _load_all(self, items: Sequence[SegmentItem]) -> np.ndarray:
        data_arr = np.empty((len(items), self.n_mels, self.n_frames), dtype=np.float32)
        for i, (mel_path, _, _, _) in enumerate(items):
            data_arr[i] = self._load_one(mel_path)
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

    def class_weights(self) -> torch.Tensor:
        """CrossEntropyLoss(weight=...) 용. n_total / (n_class * count)"""
        counts = self.class_counts
        weights = torch.ones(max(counts) + 1, dtype=torch.float32)
        total = sum(counts.values())
        for label_id, count in counts.items():
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
        mel = (self.data_arr[idx] if self.data_arr is not None
               else self._load_one(self.items[idx][0]))
        return self.input_transform(mel, self.label_arr[idx])


# -----------------------------------------------------------------------------
# Concrete Datasets
# -----------------------------------------------------------------------------
class LungSegmentDataset(SegmentMelDataset):
    """SNUCH Child Lung Sound — segment(cycle) level classification"""

    def __init__(
        self,
        mel_base_path: str,
        *,
        train: bool = True,
        train_ratio: float = 0.6,          # ICBHI 2017 challenge 표준 6:4
        input_type: MelNormalizeKind | MelTransform = "minmax",
        preload: bool = True,
    ):
        super().__init__(
            mel_base_path,
            train=train,
            train_ratio=train_ratio,
            input_type=input_type,
            preload=preload,
            n_mels=64,
            n_frames=1001,
        )


if __name__ == "__main__":
    from utils import ROOT

    mel_folder = os.path.join(ROOT, "db", "new_gt", "10s_mel")

    train_dataset = LungSegmentDataset(
        mel_folder,
        train=True,
        train_ratio=0.6,
        input_type="minmax",
        preload=True,
    )
    eval_dataset = LungSegmentDataset(
        mel_folder,
        train=False,
        train_ratio=0.6,
        input_type="minmax",
        preload=True,
    )

    inv_map = {v: k for k, v in train_dataset.parser.label_map.items()}

    n_total = len(train_dataset) + len(eval_dataset)
    print(f"usable segments (total) : {n_total:,}"
          f"   ids: {len(train_dataset.sample_ids) + len(eval_dataset.sample_ids)}")
    print(f"  train : {len(train_dataset):,}"
          f"  ({len(train_dataset)/n_total:.1%})   ids: {len(train_dataset.sample_ids)}")
    print(f"  eval  : {len(eval_dataset):,}"
          f"  ({len(eval_dataset)/n_total:.1%})   ids: {len(eval_dataset.sample_ids)}")
    print(f"train data shape: {train_dataset.data_arr.shape}")
    print(f"train label shape: {train_dataset.label_arr.shape}")
    print(f"sample_id overlap: {len(set(train_dataset.sample_ids) & set(eval_dataset.sample_ids))}")

    print("\nclass counts (train)")
    for label_id, count in sorted(train_dataset.class_counts.items()):
        print(f"  {label_id} {inv_map.get(label_id, '?'):10} {count:6,}"
              f"  {count/len(train_dataset):7.2%}")

    print(f"\nclass weights: {[round(w, 4) for w in train_dataset.class_weights().tolist()]}")

    if len(train_dataset) > 0:
        input_tensor, label = train_dataset[0]
        print("\nsample")
        print(f"input.shape: {tuple(input_tensor.shape)}")
        print(f"input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        print(f"label: {int(label)} ({inv_map.get(int(label), '?')})")