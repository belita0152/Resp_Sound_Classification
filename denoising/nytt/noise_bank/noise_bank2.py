"""Wheezing 오염을 제외한 train recording noise bank를 생성한다.

    python -m denoising.nytt.noise_bank.noise_bank2 --fold 1 \
        --wheeze-mode file --wheeze-labels Wheezing Rhonchi Stridor
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from denoising.nytt.noise_bank.noise_bank_split import (
    DEFAULT_LABEL_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WAV_DIR,
    END_COLUMNS,
    SHEET_CANDIDATES,
    START_COLUMNS,
    TARGET_SR,
    _candidate_starts,
    _is_under,
    _load_wav,
    _pick_column,
    _resample,
    background_intervals,
    index_label_files,
    index_wav_files,
    read_split_manifest,
)

# data/parser.py의 SegmentMelParser와 동일한 라벨 컬럼 후보
LABEL_COLUMNS = ("labels", "label", "class", "class_name")
# 라벨 문자열 구분자 — 'Wheezing, Crackle', 'Rhonchi/Stridor' 같은 복합 라벨 대응
LABEL_SEPARATORS = ("\n", "/", ";", "+", "&")
DEFAULT_WHEEZE_LABELS = ("Wheezing",)
WHEEZE_MODES = ("off", "file", "cycle", "both")
CONFIG_FIELDS = (
    "clip_sec", "hop_sec", "max_per_id", "edge_margin_sec", "min_gap_sec",
    "rms_low_pct", "rms_high_pct", "max_peak", "max_crest_db", "seed",
    "wheeze_mode", "wheeze_guard_sec", "wheeze_min_count", "wheeze_max_ratio",
)


# --------------------------------------------------------------------- 라벨
def label_tokens(raw_label) -> list[str]:
    """라벨 셀 하나를 소문자 토큰 리스트로 만든다. 복합 라벨도 모두 분해한다."""
    if raw_label is None or (isinstance(raw_label, float) and pd.isna(raw_label)):
        return []
    text = str(raw_label).strip()
    if not text:
        return []
    for separator in LABEL_SEPARATORS:
        text = text.replace(separator, ",")
    return [token.strip().lower() for token in text.split(",") if token.strip()]


def read_cycle_table(xlsx_path: Path) -> pd.DataFrame:
    """cycle의 (start_sec, end_sec, label)을 엑셀 행 순서 그대로 읽는다."""
    workbook = pd.ExcelFile(xlsx_path)
    order = [name for name in SHEET_CANDIDATES if name in workbook.sheet_names]
    order.extend(name for name in workbook.sheet_names if name not in order)

    for sheet_name in order:
        try:
            frame = workbook.parse(sheet_name)
        except Exception:
            continue
        start_col = _pick_column(frame.columns, START_COLUMNS)
        end_col = _pick_column(frame.columns, END_COLUMNS)
        if start_col is None or end_col is None:
            continue
        label_col = _pick_column(frame.columns, LABEL_COLUMNS)

        rows = []
        labels = (frame[label_col].tolist() if label_col is not None
                  else [""] * len(frame))
        for (start, end), raw_label in zip(
            frame[[start_col, end_col]].itertuples(index=False, name=None), labels
        ):
            if pd.isna(start) or pd.isna(end):
                continue
            start, end = float(start), float(end)
            if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
                continue
            rows.append({"start_sec": start, "end_sec": end, "raw_label": raw_label})

        table = pd.DataFrame(rows, columns=["start_sec", "end_sec", "raw_label"])
        table = table.sort_values("start_sec", kind="mergesort").reset_index(drop=True)
        table.attrs["label_column"] = label_col
        return table

    raise RuntimeError(f"cycle 시간 컬럼을 찾을 수 없습니다: {xlsx_path}")


def wheeze_mask(table: pd.DataFrame, wheeze_labels: set[str]) -> np.ndarray:
    """각 cycle이 연속음(기본 Wheezing) 라벨을 포함하는지."""
    if table.empty:
        return np.zeros(0, dtype=bool)
    return np.asarray([
        any(token in wheeze_labels for token in label_tokens(raw_label))
        for raw_label in table["raw_label"]
    ], dtype=bool)


# ----------------------------------------------------------------- 구간 연산
def subtract_intervals(
    base: Sequence[tuple[float, float]],
    blocked: Sequence[tuple[float, float]],
    min_length_sec: float,
) -> list[tuple[float, float]]:
    """base 구간에서 blocked 구간을 뺀다. min_length_sec 미만 조각은 버린다."""
    if not blocked:
        return [(s, e) for s, e in base if e - s >= min_length_sec]

    merged: list[list[float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in blocked):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    result: list[tuple[float, float]] = []
    for start, end in base:
        cursor = float(start)
        for block_start, block_end in merged:
            if block_end <= cursor or block_start >= end:
                continue
            if block_start > cursor:
                piece_end = min(block_start, end)
                if piece_end - cursor >= min_length_sec:
                    result.append((cursor, piece_end))
            cursor = max(cursor, block_end)
            if cursor >= end:
                break
        if end - cursor >= min_length_sec:
            result.append((cursor, float(end)))
    return result


def _print_progress(index: int, total: int, sample_id: str, n_clips: int,
                    status: str = "") -> None:
    if index % 25 == 0 or index == total:
        prefix = f"{status} — " if status else ""
        print(f"  [{index:>3}/{total}] {sample_id}  "
              f"{prefix}누적 background clips: {n_clips:,}")


# ------------------------------------------------------------------- 추출
def extract_noise_bank(
    *,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    wav_paths: dict[str, Path],
    label_paths: dict[str, Path],
    raw_wav_root: Path,
    clip_sec: float,
    hop_sec: float,
    max_per_id: int,
    edge_margin_sec: float,
    min_gap_sec: float,
    seed: int,
    wheeze_mode: str,
    wheeze_labels: set[str],
    wheeze_guard_sec: float,
    wheeze_min_count: int,
    wheeze_max_ratio: float,
) -> tuple[np.ndarray, pd.DataFrame, list[str], pd.DataFrame]:
    """train recording의 cycle 여집합에서 고정 길이 background clip을 만든다.

    wheeze_mode에 따라 wheezing 오염 recording/구간을 잡음 후보에서 제외한다.
    반환: (clips, metadata, zero_clip_ids, recording_report)
    """
    clip_samples = int(round(clip_sec * TARGET_SR))
    hop_samples = int(round(hop_sec * TARGET_SR))
    rng = np.random.default_rng(seed)
    clips: list[np.ndarray] = []
    rows: list[dict] = []
    report_rows: list[dict] = []
    zero_clip_ids: list[str] = []

    skip_file = wheeze_mode in ("file", "both")
    apply_guard = wheeze_mode in ("cycle", "both")

    for index, sample_id in enumerate(train_ids, start=1):
        wav_path = Path(wav_paths[sample_id])
        label_path = Path(label_paths[sample_id])
        if not _is_under(wav_path, raw_wav_root):
            raise RuntimeError(
                f"원본 WAV root 밖의 파일을 읽으려 했습니다: {sample_id} -> {wav_path}"
            )

        table = read_cycle_table(label_path)
        mask = wheeze_mask(table, wheeze_labels)
        n_cycles = int(len(table))
        n_wheeze = int(mask.sum())
        wheeze_ratio = (n_wheeze / n_cycles) if n_cycles else 0.0
        has_label_col = table.attrs.get("label_column") is not None
        report_row = {
            "sample_id": sample_id,
            "n_cycles": n_cycles,
            "n_wheeze_cycles": n_wheeze,
            "wheeze_ratio": round(wheeze_ratio, 4),
            "has_label_column": has_label_col,
        }

        excluded = (
            skip_file
            and n_wheeze >= wheeze_min_count
            and wheeze_ratio > wheeze_max_ratio
        )

        if excluded:
            report_rows.append({
                **report_row,
                "excluded": True,
                "exclude_reason": (
                    f"wheeze cycle {n_wheeze}/{n_cycles} ({wheeze_ratio:.1%})"
                ),
                "n_clips": 0,
            })
            _print_progress(index, len(train_ids), sample_id, len(clips), "제외(wheeze)")
            continue

        waveform, original_sr = _load_wav(wav_path)
        waveform = _resample(waveform, original_sr)
        total_sec = len(waveform) / TARGET_SR

        cycles = list(table[["start_sec", "end_sec"]].itertuples(index=False, name=None))
        gaps = background_intervals(
            cycles,
            total_sec=total_sec,
            edge_margin_sec=edge_margin_sec,
            min_gap_sec=max(min_gap_sec, clip_sec),
        )
        guarded_cycles = 0
        if apply_guard and n_wheeze:
            blocked = [
                (max(0.0, start - wheeze_guard_sec),
                 min(total_sec, end + wheeze_guard_sec))
                for start, end in table.loc[
                    mask, ["start_sec", "end_sec"]
                ].itertuples(index=False, name=None)
            ]
            guarded_cycles = len(blocked)
            gaps = subtract_intervals(gaps, blocked, max(min_gap_sec, clip_sec))

        starts = rng.permutation(
            _candidate_starts(gaps, clip_samples, hop_samples)
        ).tolist()

        accepted = 0
        for start in starts:
            if accepted >= max_per_id:
                break
            clip = waveform[start:start + clip_samples]
            if len(clip) != clip_samples:
                continue
            rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))
            if rms < 1e-6:
                continue
            peak = float(np.max(np.abs(clip)))
            clips.append(clip.astype(np.float32, copy=False))
            rows.append({
                "sample_id": sample_id,
                "wav_path": str(wav_path.resolve()),
                "start_sec": round(start / TARGET_SR, 6),
                "end_sec": round((start + clip_samples) / TARGET_SR, 6),
                "rms": rms,
                "peak": peak,
                "crest_db": float(20.0 * np.log10(peak / rms)),
                "n_cycles": n_cycles,
                "n_wheeze_cycles": n_wheeze,
                "recording_sec": round(total_sec, 6),
                "original_sr": original_sr,
                "source_kind": "full_recording_wav",
            })
            accepted += 1

        if accepted == 0:
            zero_clip_ids.append(sample_id)
        report_rows.append({
            **report_row,
            "excluded": False,
            "exclude_reason": (
                f"guard {guarded_cycles} wheeze cycles" if guarded_cycles else ""
            ),
            "n_clips": accepted,
        })
        _print_progress(index, len(train_ids), sample_id, len(clips))

    if not clips:
        raise RuntimeError(
            "background clip을 하나도 추출하지 못했습니다. --wheeze-mode 를 완화하거나 "
            "--clip-sec/--edge-margin-sec 설정을 확인하십시오."
        )

    metadata = pd.DataFrame(rows)
    report = pd.DataFrame(report_rows)
    source_ids = set(metadata["sample_id"])
    train_set, test_set = set(train_ids), set(test_ids)
    leaked = sorted(source_ids & test_set)
    if leaked:
        raise RuntimeError(f"test recording이 noise bank에 섞였습니다: {leaked}")
    outside_train = sorted(source_ids - train_set)
    if outside_train:
        raise RuntimeError(f"train 외 recording이 noise bank에 섞였습니다: {outside_train}")

    # wheeze-mode=file/both 이면 제외 대상 ID가 실제로 하나도 안 들어갔는지 검증
    if skip_file:
        excluded_ids = set(report.loc[report["excluded"], "sample_id"])
        contaminated = sorted(source_ids & excluded_ids)
        if contaminated:
            raise RuntimeError(
                f"wheeze 제외 대상 recording이 noise bank에 남았습니다: {contaminated}"
            )
    return np.stack(clips), metadata, zero_clip_ids, report


def filter_clips(
    clips: np.ndarray,
    metadata: pd.DataFrame,
    *,
    rms_percentiles: tuple[float, float],
    max_peak: float,
    max_crest_db: float | None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    """RMS, peak, crest factor 순서로 clip을 필터링한다."""
    low, high = np.percentile(metadata["rms"], rms_percentiles)
    keep = metadata["rms"].between(low, high, inclusive="both")
    removed = {"rms": int((~keep).sum()), "peak": 0, "crest": 0}

    rejected = metadata["peak"].ge(max_peak)
    removed["peak"] = int((rejected & keep).sum())
    keep &= ~rejected

    if max_crest_db is not None:
        rejected = metadata["crest_db"].gt(max_crest_db)
        removed["crest"] = int((rejected & keep).sum())
        keep &= ~rejected

    filtered_metadata = metadata.loc[keep].reset_index(drop=True)
    if filtered_metadata.empty:
        raise RuntimeError("필터를 통과한 background clip이 없습니다.")
    return clips[keep.to_numpy()], filtered_metadata, removed


def save_noise_bank(
    output_path: Path,
    clips: np.ndarray,
    metadata: pd.DataFrame,
    report: pd.DataFrame,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    args: argparse.Namespace,
    paths: dict[str, Path],
    wheeze_labels: set[str],
) -> tuple[list[str], list[str]]:
    """NoiseBank 호환 NPZ를 임시 파일에 쓴 뒤 원자적으로 교체한다."""
    source_ids = sorted(metadata["sample_id"].unique().tolist())
    excluded_ids = sorted(report.loc[report["excluded"], "sample_id"].tolist())
    config = {name: getattr(args, name) for name in CONFIG_FIELDS}
    config.update({
        "fold": args.fold,
        "manifest": str(paths["manifest"]),
        "raw_wav_root": str(paths["wav"]),
        "label_root": str(paths["label"]),
        "source_kind": "full_recording_wav",
        "target_sr": TARGET_SR,
        "wheeze_labels": sorted(wheeze_labels),
        "wheeze_excluded_ids": excluded_ids,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp.npz")
    np.savez_compressed(
        temporary_path,
        clips=clips.astype(np.float32, copy=False),
        sr=np.int64(TARGET_SR),
        meta=metadata.to_records(index=False),
        source_ids=np.asarray(source_ids, dtype=object),
        train_ids=np.asarray(train_ids, dtype=object),
        test_ids=np.asarray(test_ids, dtype=object),
        train_only=np.bool_(True),
        fold=np.int64(args.fold),
        manifest_path=np.asarray(str(paths["manifest"]), dtype=object),
        source_kind=np.asarray("full_recording_wav", dtype=object),
        wheeze_mode=np.asarray(args.wheeze_mode, dtype=object),
        wheeze_labels=np.asarray(sorted(wheeze_labels), dtype=object),
        wheeze_excluded_ids=np.asarray(excluded_ids, dtype=object),
        recording_report=report.to_records(index=False),
        config_json=np.asarray(json.dumps(config, ensure_ascii=False), dtype=object),
    )
    temporary_path.replace(output_path)
    return source_ids, excluded_ids


# -------------------------------------------------------------------- CLI
def _validate_arguments(args: argparse.Namespace) -> None:
    if args.fold < 1:
        raise ValueError("--fold는 1 이상의 정수여야 합니다.")
    if args.clip_sec <= 0 or args.hop_sec <= 0:
        raise ValueError("--clip-sec와 --hop-sec는 0보다 커야 합니다.")
    if args.max_per_id < 1:
        raise ValueError("--max-per-id는 1 이상이어야 합니다.")
    if args.edge_margin_sec < 0 or args.min_gap_sec < 0:
        raise ValueError("margin과 gap 길이는 음수일 수 없습니다.")
    if not 0 <= args.rms_low_pct < args.rms_high_pct <= 100:
        raise ValueError("RMS percentile은 0 <= low < high <= 100이어야 합니다.")
    if args.max_peak <= 0:
        raise ValueError("--max-peak는 0보다 커야 합니다.")
    if args.wheeze_mode not in WHEEZE_MODES:
        raise ValueError(f"--wheeze-mode는 {WHEEZE_MODES} 중 하나여야 합니다.")
    if args.wheeze_guard_sec < 0:
        raise ValueError("--wheeze-guard-sec는 음수일 수 없습니다.")
    if args.wheeze_min_count < 1:
        raise ValueError("--wheeze-min-count는 1 이상이어야 합니다.")
    if not 0 <= args.wheeze_max_ratio < 1:
        raise ValueError("--wheeze-max-ratio는 0 이상 1 미만이어야 합니다.")
    if not args.wheeze_labels:
        raise ValueError("--wheeze-labels가 비어 있습니다.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="fold별 train WAV에서 NyTT noise bank를 추출하되 "
                    "wheezing 오염 recording/구간을 제외합니다."
    )
    parser.add_argument("--fold", type=int, required=True, help="사용할 grouping 번호")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--wav-dir", type=Path, default=DEFAULT_WAV_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument(
        "--out", type=Path, default=None,
        help=f"기본값: {DEFAULT_OUTPUT_DIR}/noise_bank2_fold<FOLD>[_<mode>].npz",
    )
    parser.add_argument("--clip-sec", type=float, default=2.0)
    parser.add_argument("--hop-sec", type=float, default=None,
                        help="background 후보 간격. 기본값은 clip-sec(겹치지 않음)")
    parser.add_argument("--max-per-id", type=int, default=8)
    parser.add_argument("--edge-margin-sec", type=float, default=0.1)
    parser.add_argument("--min-gap-sec", type=float, default=0.5)
    parser.add_argument("--rms-low-pct", type=float, default=1.0)
    parser.add_argument("--rms-high-pct", type=float, default=99.0)
    parser.add_argument("--max-peak", type=float, default=0.99)
    parser.add_argument("--max-crest-db", type=float, default=None,
                        help="미지정 시 crest factor로 제거하지 않음")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--overwrite", action="store_true")

    group = parser.add_argument_group("wheezing 오염 제어")
    group.add_argument(
        "--wheeze-mode", choices=WHEEZE_MODES, default="file",
        help="off: 제외 안 함(noise_bank_split과 동일) / "
             "file: wheezing이 있는 recording 전체 제외(기본) / "
             "cycle: wheezing cycle 주변만 제외 / both: 둘 다",
    )
    group.add_argument(
        "--wheeze-labels", nargs="+", default=list(DEFAULT_WHEEZE_LABELS),
        help="연속음으로 간주해 제외할 라벨 이름 (기본: Wheezing). "
             "예: --wheeze-labels Wheezing Rhonchi Stridor",
    )
    group.add_argument(
        "--wheeze-guard-sec", type=float, default=1.0,
        help="cycle/both 모드에서 wheezing cycle 앞뒤로 추가로 잘라낼 초 (기본 1.0)",
    )
    group.add_argument(
        "--wheeze-min-count", type=int, default=1,
        help="file/both 모드에서 이 개수 이상의 wheezing cycle이 있으면 recording 제외 (기본 1)",
    )
    group.add_argument(
        "--wheeze-max-ratio", type=float, default=0.0,
        help="file/both 모드에서 wheezing cycle 비율이 이 값을 초과할 때만 제외 (기본 0.0 = 하나라도 있으면 제외)",
    )
    group.add_argument(
        "--report-csv", type=Path, default=None,
        help="recording별 wheezing/제외 내역을 CSV로 저장할 경로",
    )

    args = parser.parse_args()
    if args.hop_sec is None:
        args.hop_sec = args.clip_sec
    _validate_arguments(args)
    return args


def main() -> None:
    args = parse_args()

    manifest_path = args.manifest.resolve()
    train_ids, test_ids = read_split_manifest(manifest_path, args.fold)
    raw_wav_root = args.wav_dir.resolve()
    label_root = args.label_dir.resolve()
    wav_ids = index_wav_files(raw_wav_root)
    label_ids = index_label_files(label_root)

    missing_wav = sorted(sid for sid in train_ids if sid not in wav_ids)
    missing_label = sorted(sid for sid in train_ids if sid not in label_ids)
    if missing_wav or missing_label:
        raise RuntimeError(
            f"train source 누락 — 원본 WAV: {missing_wav}, 라벨 Excel: {missing_label}"
        )

    wheeze_labels = {name.strip().lower() for name in args.wheeze_labels if name.strip()}
    if not wheeze_labels:
        raise ValueError("--wheeze-labels에 빈 문자열만 주어졌습니다.")
    suffix = "" if args.wheeze_mode == "off" else f"_{args.wheeze_mode}"
    output_path = (
        args.out if args.out is not None
        else DEFAULT_OUTPUT_DIR / f"noise_bank2_fold{args.fold}{suffix}.npz"
    ).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"출력 파일이 이미 있습니다: {output_path}\n"
            "덮어쓰려면 --overwrite를 명시하십시오."
        )

    print(
        f"[noise_bank2] fold={args.fold} train/test={len(train_ids)}/{len(test_ids)} "
        f"wav={len(wav_ids):,}\n"
        f"  mode={args.wheeze_mode} labels={sorted(wheeze_labels)} "
        f"guard={args.wheeze_guard_sec:g}s\n"
        f"  clip/hop={args.clip_sec:g}/{args.hop_sec:g}s "
        f"max_per_id={args.max_per_id} output={output_path}"
    )

    clips, metadata, zero_clip_ids, report = extract_noise_bank(
        train_ids=train_ids,
        test_ids=test_ids,
        wav_paths=wav_ids,
        label_paths=label_ids,
        raw_wav_root=raw_wav_root,
        clip_sec=args.clip_sec,
        hop_sec=args.hop_sec,
        max_per_id=args.max_per_id,
        edge_margin_sec=args.edge_margin_sec,
        min_gap_sec=args.min_gap_sec,
        seed=args.seed,
        wheeze_mode=args.wheeze_mode,
        wheeze_labels=wheeze_labels,
        wheeze_guard_sec=args.wheeze_guard_sec,
        wheeze_min_count=args.wheeze_min_count,
        wheeze_max_ratio=args.wheeze_max_ratio,
    )
    extracted_count = len(metadata)
    clips, metadata, removed = filter_clips(
        clips,
        metadata,
        rms_percentiles=(args.rms_low_pct, args.rms_high_pct),
        max_peak=args.max_peak,
        max_crest_db=args.max_crest_db,
    )
    source_ids, wheeze_excluded_ids = save_noise_bank(
        output_path,
        clips,
        metadata,
        report,
        train_ids,
        test_ids,
        args,
        {"manifest": manifest_path, "wav": raw_wav_root, "label": label_root},
        wheeze_labels,
    )

    if args.report_csv is not None:
        report_path = args.report_csv.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False, encoding="utf-8-sig")

    no_label_count = int((~report["has_label_column"]).sum())
    print(
        f"\n[noise_bank2] 완료: clips={extracted_count:,}->{len(metadata):,} "
        f"(RMS/peak/crest 제외={removed['rms']}/{removed['peak']}/{removed['crest']}), "
        f"IDs={len(source_ids)}/{len(train_ids)}, {len(metadata) * args.clip_sec / 60:.1f}분\n"
        f"  wheeze ID={int((report['n_wheeze_cycles'] > 0).sum()):,}, "
        f"제외 ID={len(wheeze_excluded_ids):,}, test 누수=0"
    )
    if no_label_count:
        print(f"★ 라벨 컬럼을 못 찾은 recording {no_label_count}개 — "
              f"wheezing 판정이 불가능해 그대로 사용했습니다. 라벨 엑셀을 확인하십시오.")
    if zero_clip_ids:
        print(f"추출 clip 0개 ID: {len(zero_clip_ids)}개 "
              f"(충분히 긴 background가 없음) — {zero_clip_ids}")
    if args.report_csv is not None:
        print(f"recording report: {args.report_csv.resolve()}")
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
