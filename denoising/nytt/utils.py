# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import LungSegmentDataset
from data.utils import mel_folder, wave_folder
from denoising.nytt.noise_bank import NoiseBank

DATA_ROOT = PROJECT_ROOT.parent
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "split_manifest_5x60_40.csv"
DEFAULT_NOISE_BANK_DIR = DATA_ROOT / "db" / "new_gt"


def resolve_arm(no_denoise: bool, no_noise_augmentation: bool):
    """A/A-prime/B arm과 모델·augmentation 활성 상태를 결정한다."""
    use_dnet = not bool(no_denoise)
    noise_augmentation = use_dnet and not bool(no_noise_augmentation)
    arm = ("A" if not use_dnet
           else "A-prime" if not noise_augmentation
           else "B")
    return arm, use_dnet, noise_augmentation


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


def load_noise_bank(args, train_ids, test_ids):
    """현재 arm에 필요한 noise bank를 읽고 split 누수를 검증한다."""
    _, use_dnet, noise_augmentation = resolve_arm(
        args.no_denoise, args.no_noise_augmentation,
    )
    if not use_dnet:
        print("[noise] A arm은 noise bank를 사용하지 않습니다.")
        return None

    path = Path(args.noise_bank)
    if not path.is_file():
        if noise_augmentation:
            raise FileNotFoundError(
                f"B arm noise bank 없음: {path}\n"
                f"  python -m denoising.nytt.noise_bank.noise_bank_split --fold {args.fold} "
                "를 먼저 실행하세요."
            )
        print("[noise] A-prime 학습에는 noise bank가 필요하지 않습니다. "
              "최종 denoising 지표는 건너뜁니다.")
        return None

    verify_fold_noise_bank(path, args.fold, train_ids, test_ids)
    bank = NoiseBank(path, seed=args.seed)
    leaked = bank.check_leakage(test_ids)
    if leaked:
        raise RuntimeError(
            f"noise bank에 test split {len(leaked)} ids 누수: "
            f"{leaked[:10]}{' ...' if len(leaked) > 10 else ''}"
        )
    print(f"[noise] clips {len(bank):,} ({bank.minutes:.1f} 분), test 누수 없음")
    return bank



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
    print("  python -m denoising.nytt.nytt_dnet " + cmd)
    print("=" * 78)
