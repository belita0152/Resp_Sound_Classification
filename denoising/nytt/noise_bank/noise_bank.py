"""
[1단계] noise_bank.py   원본 wav + 라벨 엑셀  →  noise_bank.npz  (잡음 clip 모음)

[2단계] nytt.py         10s_repeat/*.wav + noise_bank.npz  →  학습된 모델
                        매 배치마다 잡음을 꺼내 섞어서 denoiser를 학습.


# Noise bank — clean reference 없이 '잡음만 있는 구간' 을 자기 데이터에서 추출
# Reference: https://arxiv.org/pdf/2211.01198

원리
    라벨 엑셀의 cycle 구간 = 판독자가 호흡 주기로 표시한 곳
    그 여집합(cycle 사이의 빈 구간) = 호흡음이 표시되지 않은 곳
        → 배경 잡음, 접촉/마찰음, 대화음, 기기음, 심음이 남아 있는 구간
        → NyTT 의 '주입할 잡음' 원천으로 쓴다

이렇게 하면
    (1) clean 데이터가 필요 없다
    (2) 주입 잡음이 실제 잡음과 같은 도메인이다  ← NyTT 가 실제 잡음을 줄이는 조건
    (3) 추가 녹음이 필요 없다

출력
    <ROOT>/new_gt/noise_bank.npz    clips [N, L] float32 (16 kHz), meta
"""


# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

CURRENT_DIR = Path(__file__).resolve().parent
UTILS_ROOT = CURRENT_DIR.parents[1]  # /home/coder/workspace/data/classification

sys.path.insert(0, str(UTILS_ROOT))

from data.parser import SegmentMelParser
from data.utils import data_folder, label_ids, matched_ids, mel_folder, noise_bank_path, wave_folder, wav_ids

SR = 16000
CLIP_SEC = 2.0
MIN_GAP_SEC = 0.5          # 이보다 짧은 gap 은 버림
EDGE_MARGIN_SEC = 0.1      # cycle 경계에서 이만큼 떨어진 곳만 사용
MAX_CLIPS_PER_ID = 8

SHEET_CANDIDATES = ["사이클 정보", "사이클정보", "cycle", "Cycle"]
COL_START = ["cycle_start_time", "start_time", "cycle_start", "start"]
COL_END = ["cycle_end_time", "end_time", "cycle_end", "end"]


def pick(columns, candidates):
    norm = {str(c).strip().lower(): c for c in columns}
    for c in candidates:
        if c.lower() in norm:
            return norm[c.lower()]
    return None


def read_cycle_intervals(xlsx_path) -> List[Tuple[float, float]]:
    xl = pd.ExcelFile(xlsx_path)
    order = ([s for s in SHEET_CANDIDATES if s in xl.sheet_names]
             + [s for s in xl.sheet_names if s not in SHEET_CANDIDATES])
    for sh in order:
        try:
            df = xl.parse(sh)
        except Exception:
            continue
        cs, ce = pick(df.columns, COL_START), pick(df.columns, COL_END)
        if cs and ce:
            iv = []
            for _, r in df.iterrows():
                st, et = r.get(cs), r.get(ce)
                if pd.isna(st) or pd.isna(et):
                    continue
                st, et = float(st), float(et)
                if et > st:
                    iv.append((st, et))
            return sorted(iv)
    raise RuntimeError(f"cycle 시간 컬럼 없음: {xlsx_path}")


def complement(intervals: List[Tuple[float, float]], total: float,
               margin: float) -> List[Tuple[float, float]]:
    """cycle 구간의 여집합 (경계에서 margin 만큼 안쪽으로 축소)"""
    gaps, prev = [], 0.0
    for st, et in intervals:
        if st - prev > 0:
            gaps.append((prev, st))
        prev = max(prev, et)
    if total - prev > 0:
        gaps.append((prev, total))

    out = []
    for a, b in gaps:
        a, b = a + margin, b - margin
        if b - a >= MIN_GAP_SEC:
            out.append((a, b))
    return out


def load_wav(path) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.asarray(y, dtype=np.float32), int(sr)


def resample(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return y
    import librosa
    return librosa.resample(y, orig_sr=sr, target_sr=target).astype(np.float32)


def train_split_ids(train_ratio: float = 0.6) -> List[str]:
    """data_loader 의 train split 에 속한 sample_id 만 돌려준다.

    ★ 잡음을 eval split 녹음에서 뽑으면 그 환자의 방음·기기·접촉 특성이
      학습에 새어 들어간다. 같은 병원 같은 기기라 유사도가 높아 무시 못 할 누수다.
      data_loader 와 동일한 규칙(정렬 후 앞 60%)을 그대로 재현한다.
    """
    if os.path.isdir(mel_folder):
        base, suffix = mel_folder, ".npy"
    elif os.path.isdir(wave_folder):
        base, suffix = wave_folder, ".wav"
    else:
        raise FileNotFoundError(
            "segment 폴더가 없어 train split 을 알 수 없습니다.\n"
            f"  {mel_folder}\n  {wave_folder}\n"
            "  둘 중 하나가 있어야 합니다. 전체에서 뽑으려면 --all_ids 를 쓰십시오 "
            "(단 eval 누수 발생)."
        )

    items = SegmentMelParser(base, suffix=suffix).build_sample_items()
    if not items:
        raise FileNotFoundError(
            f"{base} 에서 매칭된 segment 가 없습니다. parser 설정을 확인하십시오.")

    split = int(len(items) * float(train_ratio))
    ids = [sample_id for sample_id, _ in items[:split]]
    print(f"[split] {base} 기준 전체 {len(items)} ids → train {len(ids)} ids "
          f"(ratio {train_ratio})")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=noise_bank_path)
    ap.add_argument("--clip_sec", type=float, default=CLIP_SEC)
    ap.add_argument("--max_per_id", type=int, default=MAX_CLIPS_PER_ID)
    ap.add_argument("--limit", type=int, default=0, help="sample_id 개수 제한")
    ap.add_argument("--train_ratio", type=float, default=0.6,
                    help="train split 비율. 여기에 속한 녹음에서만 잡음을 뽑는다.")
    ap.add_argument("--all_ids", action="store_true",
                    help="train/eval 구분 없이 전체에서 추출 (누수 발생, 비교용)")
    # ★ 충격성 클립 제어 — 아래 crest factor 설명 참조
    ap.add_argument("--max_peak", type=float, default=0.99,
                    help="이 값 이상이면 클리핑된 클립으로 보고 버린다")
    ap.add_argument("--max_crest_db", type=float, default=None,
                    help="20·log10(peak/rms) 상한. 충격성 클립을 걸러낸다. "
                         "미지정이면 분포만 출력하고 거르지 않는다 (권장 20)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    clip_len = int(round(a.clip_sec * SR))
    ids = train_split_ids(a.train_ratio) if not a.all_ids else list(matched_ids)
    if a.limit:
        ids = ids[:a.limit]

    print("=" * 78)
    print(f"data_folder {data_folder}")
    print(f"out         {a.out}")
    print(f"clip        {a.clip_sec:.1f} s ({clip_len:,} samples @ {SR} Hz)")
    print(f"sample_ids  {len(ids)}"
          f"{'  (전체 — 누수 주의)' if a.all_ids else f'  (train split, ratio={a.train_ratio})'}")
    print("=" * 78)

    clips, meta, errors = [], [], []
    for i, sid in enumerate(ids, 1):
        if sid not in wav_ids or sid not in label_ids:
            continue
        try:
            y, sr = load_wav(wav_ids[sid])
            y = resample(y, sr, SR)
            intervals = read_cycle_intervals(label_ids[sid])
        except Exception as ex:
            errors.append(dict(sample_id=sid, err=str(ex)))
            continue

        total = len(y) / SR
        gaps = complement(intervals, total, EDGE_MARGIN_SEC)

        got = 0
        for (a_sec, b_sec) in gaps:
            if got >= a.max_per_id:
                break
            n_fit = int((b_sec - a_sec) * SR) // clip_len
            for k in range(min(n_fit, a.max_per_id - got)):
                s = int(a_sec * SR) + k * clip_len
                clip = y[s:s + clip_len]
                if len(clip) < clip_len:
                    break
                rms = float(np.sqrt((clip ** 2).mean()))
                if rms < 1e-6:                      # 완전 무음(디지털 0)은 제외
                    continue
                peak = float(np.abs(clip).max())
                clips.append(clip.astype(np.float32))
                meta.append(dict(sample_id=sid, start_sec=round(s / SR, 3),
                                 rms=rms, peak=peak,
                                 crest_db=20.0 * np.log10(peak / rms),
                                 n_cycles=len(intervals), total_sec=round(total, 2)))
                got += 1

        if i % 50 == 0 or i == len(ids):
            print(f"  [{i}/{len(ids)}] {sid}  누적 clip {len(clips):,}")

    if not clips:
        sys.exit("잡음 clip 을 추출하지 못했습니다. MIN_GAP_SEC / clip_sec 을 낮춰보세요.")

    arr = np.stack(clips, axis=0)
    df = pd.DataFrame(meta)
    n_raw = len(df)

    # ---------------------------------------------------------------- 필터링
    # ① RMS 상하위 1% — 사실상 무음 / 비정상적으로 큰 구간
    lo, hi = np.percentile(df["rms"], [1, 99])
    keep = (df["rms"] >= lo) & (df["rms"] <= hi)
    n_rms = int((~keep).sum())

    # ② 클리핑 — peak 가 full scale 을 넘으면 파형이 이미 깨져 있다
    clipped = df["peak"] >= a.max_peak
    keep &= ~clipped

    # ③ crest factor = 20·log10(peak/rms)
    #    ★ 이 값이 크다는 것은 클립이 '정상 배경음' 이 아니라
    #      충격성 아티팩트(청진기 마찰, 두드림, 몸부림) 라는 뜻이다.
    #          백색잡음     ~12.5 dB
    #          정상 배경음  12–16 dB
    #          접촉 아티팩트 18–29 dB
    #      충격성 잡음만 주입하면 denoiser 는 '충격음을 지우는 함수' 가 되고,
    #      crackle 도 충격성 신호이므로 같이 지워진다. crackle 은 가장 드물고
    #      임상적으로 가장 중요한 클래스라 이 실패는 치명적이다.
    n_crest = 0
    if a.max_crest_db is not None:
        too_impulsive = df["crest_db"] > a.max_crest_db
        n_crest = int((too_impulsive & keep).sum())
        keep &= ~too_impulsive

    arr, df = arr[keep.values], df[keep].reset_index(drop=True)
    if len(df) == 0:
        sys.exit("필터를 통과한 clip 이 없습니다. --max_crest_db 를 완화하십시오.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, clips=arr, sr=SR,
                        meta=df.to_records(index=False),
                        source_ids=np.array(sorted(df["sample_id"].unique()), dtype=object),
                        train_only=not a.all_ids)

    print("\n" + "=" * 78)
    print("요약")
    print("=" * 78)
    print(f"추출         {n_raw:,} → 채택 {arr.shape[0]:,}")
    print(f"  RMS 1%/99% 제외   {n_rms:,}")
    print(f"  클리핑 제외        {int(clipped.sum()):,}  (peak ≥ {a.max_peak})")
    print(f"  crest 제외         {n_crest:,}"
          + (f"  (> {a.max_crest_db} dB)" if a.max_crest_db is not None
             else "  (--max_crest_db 미지정, 거르지 않음)"))
    print(f"clips        {arr.shape[0]:,} × {arr.shape[1]:,} samples ({a.clip_sec:.1f} s)")
    print(f"sample_ids   {df['sample_id'].nunique():,}")
    print(f"총 길이      {arr.shape[0] * a.clip_sec / 60:.1f} 분")
    print(f"RMS          중앙값 {df['rms'].median():.5f}  "
          f"IQR {df['rms'].quantile(.25):.5f}–{df['rms'].quantile(.75):.5f}")
    print(f"peak         중앙값 {df['peak'].median():.4f}  최대 {df['peak'].max():.4f}")
    print(f"용량         {arr.nbytes/1024**2:.1f} MB (float32)")

    # ------------------------------------------------------------ crest 진단
    c = df["crest_db"]
    print("\ncrest factor = 20·log10(peak/rms)   — 클립이 충격성인지 판정")
    print(f"  중앙값 {c.median():.1f} dB   "
          f"IQR {c.quantile(.25):.1f}–{c.quantile(.75):.1f}   최대 {c.max():.1f}")
    print(f"  {'구간':>14} {'clip':>8} {'비율':>8}   해석")
    for lo_db, hi_db, tag in [(0, 16, "정상 배경음"),
                              (16, 20, "경계"),
                              (20, 25, "충격성 (접촉 아티팩트 가능)"),
                              (25, 1e9, "강한 충격성 — crackle 과 혼동 위험")]:
        n = int(((c >= lo_db) & (c < hi_db)).sum())
        rng_txt = f"{lo_db}–{hi_db} dB" if hi_db < 1e9 else f"{lo_db}+ dB"
        print(f"  {rng_txt:>14} {n:8,} {n/len(c):8.1%}   {tag}")
    if a.max_crest_db is None and (c > 20).mean() > 0.3:
        print("  ★ 충격성 클립이 30% 를 넘습니다. denoiser 가 crackle 을 지울 위험이 있습니다.")
        print("    --max_crest_db 20 으로 다시 만들어 비교해 보십시오.")

    print(f"\n저장: {a.out}")
    if errors:
        print(f"오류 {len(errors)}건: {[e['sample_id'] for e in errors[:5]]}")


class NoiseBank:
    """학습 중 잡음 clip 을 뽑아 쓰는 헬퍼"""

    def __init__(self, path: str | Path, seed: int = 0):
        z = np.load(str(path), allow_pickle=True)
        self.clips = z["clips"]
        self.sr = int(z["sr"])
        # clip 별 출처 (sample_id, start_sec, rms, peak, crest_db ...)
        self.meta = (pd.DataFrame.from_records(z["meta"])
                     if "meta" in z.files else None)
        self.source_ids = (set(z["source_ids"].tolist())
                           if "source_ids" in z.files else set())
        self.train_only = bool(z["train_only"]) if "train_only" in z.files else None
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.clips)

    @property
    def minutes(self) -> float:
        return self.clips.shape[0] * self.clips.shape[1] / self.sr / 60

    def check_leakage(self, eval_sample_ids) -> List[str]:
        """noise bank 출처와 eval split 이 겹치는지 확인. 비어 있어야 정상."""
        return sorted(self.source_ids & set(eval_sample_ids))

    def sample(self, n_samples: int, batch: int = 1,
               fade_ms: float = 10.0) -> np.ndarray:
        """[batch, n_samples] — clip 을 교차 페이드로 이어 붙이고 무작위 위치에서 자른다.

        ★ 그냥 이어 붙이면 clip 경계마다 레벨이 튄다. RMS 가 4배까지 차이나므로
          2 초마다 계단 모양 불연속이 생기고, 그것 자체가 충격성 신호다.
          denoiser 가 그 인공 클릭을 지우도록 학습되면 crackle 도 같이 지운다.

        서로 무관한 잡음이므로 등전력(equal-power) 페이드를 쓴다.
            w_in = √t,  w_out = √(1-t)   →   w_in² + w_out² = 1  (전력 보존)
        선형 페이드는 중점에서 3 dB 가 꺼진다.
        """
        clip_len = self.clips.shape[1]
        fade = min(max(1, int(self.sr * fade_ms / 1000)), clip_len // 4)
        hop = clip_len - fade
        n_rep = int(np.ceil((n_samples + 2 * clip_len) / hop)) + 1

        t = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        w_in, w_out = np.sqrt(t), np.sqrt(1.0 - t)

        out = np.empty((batch, n_samples), dtype=np.float32)
        for b in range(batch):
            idx = self.rng.integers(0, len(self.clips), size=n_rep)
            buf = np.zeros(n_rep * hop + fade, dtype=np.float32)
            for k, i in enumerate(idx):
                clip = self.clips[i].astype(np.float32).copy()
                clip[:fade] *= w_in
                clip[-fade:] *= w_out
                s = k * hop
                buf[s:s + clip_len] += clip
            # 앞뒤 fade 구간은 반쪽만 채워져 있으므로 피해서 자른다
            hi = len(buf) - n_samples - fade
            off = fade + int(self.rng.integers(0, max(1, hi - fade)))
            out[b] = buf[off:off + n_samples]
        return out


if __name__ == "__main__":
    main()
