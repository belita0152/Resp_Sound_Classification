from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import label_ids, matched_ids as cohort_matched_ids

# (path, label_id, sample_id, cycle_idx) — cycle_data_loader.CycleItem 과 동일한 순서
SegmentItem = Tuple[Path, int, str, int]

# 파일명에서는 sample_id 와 cycle_idx 만 가져온다. 라벨은 엑셀에서 읽는다.
#   <sample_id>_c<cycle_idx>_<label>
NAME_RE = re.compile(r"^(?P<sid>.+?)_c(?P<cyc>\d+)_(?P<label>.+)$")

SHEET_CANDIDATES = ["사이클 정보", "사이클정보", "cycle", "Cycle"]
COL_LABEL = ["labels", "label", "class", "class_name"]


def sanitize(s) -> str:
    """parse_cycles.py 가 파일명을 만들 때 쓴 것과 동일한 규칙 (검증용)"""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "NoLabel"
    s = str(s).strip()
    return re.sub(r'[\\/:*?"<>|\s]+', "_", s) if s else "NoLabel"


class SegmentMelParser:
    """잘려 저장된 segment 파일과 라벨 엑셀을 매칭한다.

        <root>/<YYMMDD>/<sample_id>_c<cycle_idx>_<label>.npy

    ★ 라벨은 파일명이 아니라 라벨 엑셀에서 읽는다.
      파일명의 라벨은 sanitize() 를 거친 가공 문자열이라 원본이 아니다.
      파일명에서는 sample_id 와 cycle_idx 만 가져오고,
      utils.label_ids[sample_id] 엑셀의 cycle_idx 번째 행에서 라벨을 읽는다.

    하는 일
        1. 파일명 → (sample_id, cycle_idx)
        2. 엑셀 cycle_idx 행 → 라벨 원본 → 정수
           (Unknown / NaN / 복합 라벨은 exclude_label 로 판정해 데이터셋에서 제외)
        3. utils.matched_ids 코호트 필터 + sample_id 단위 묶기 (환자 단위 분할용)
        4. 엑셀 라벨과 파일명 라벨이 어긋나면 mismatch 로 보고

    엑셀 행은 하나도 건너뛰지 않고 순서 그대로 읽으므로 cycle_idx 가 밀리지 않는다.
    """

    def __init__(
        self,
        mel_base_path: str | Path,
        *,
        suffix: str = ".npy",
        label_col: Optional[List[str]] = None,
        label_map: Optional[Dict[str, int]] = None,
        exclude_label: int = -1,
        use_cohort_filter: bool = True,
    ):
        self.mel_base_path = Path(mel_base_path)
        self.suffix = suffix
        self.label_col = label_col or COL_LABEL
        self.label_map = label_map or {
            "Normal": 0,
            "Stridor": 1,
            "Rhonchi": 2,
            "Wheezing": 3,
            "Crackle": 4,
        }
        self._label_lookup = {k.strip().lower(): v for k, v in self.label_map.items()}
        self.exclude_label = exclude_label
        self.use_cohort_filter = use_cohort_filter
        self.cohort_ids = {str(sample_id) for sample_id in cohort_matched_ids}
        self._cache: Dict[str, List] = {}

    # ------------------------------------------------------------------ 파일명
    @staticmethod
    def parse_segment_name(path: str | Path) -> Tuple[str, int, str] | None:
        """'<sample_id>_c<cycle_idx>_<label>' → (sample_id, cycle_idx, 파일명라벨)"""
        m = NAME_RE.match(Path(path).stem)
        if m is None:
            return None
        return m.group("sid").strip(), int(m.group("cyc")), m.group("label").strip()

    # ------------------------------------------------------------------ 엑셀
    def cycle_labels(self, sample_id: str) -> List:
        """엑셀의 라벨 컬럼을 행 순서 그대로 반환. 어떤 행도 건너뛰지 않는다.

        리스트 인덱스 == cycle_idx == parse_cycles.py 가 자를 때 쓴 행 번호.
        """
        if sample_id in self._cache:
            return self._cache[sample_id]

        path = label_ids.get(sample_id)
        if path is None:
            self._cache[sample_id] = []
            return []

        labels: List = []
        try:
            xl = pd.ExcelFile(path)
            order = ([s for s in SHEET_CANDIDATES if s in xl.sheet_names]
                     + [s for s in xl.sheet_names if s not in SHEET_CANDIDATES])
            for sh in order:
                try:
                    df = xl.parse(sh)
                except Exception:
                    continue
                norm = {str(c).strip().lower(): c for c in df.columns}
                col = next((norm[c.lower()] for c in self.label_col
                            if c.lower() in norm), None)
                if col is not None:
                    labels = df[col].tolist()      # 행 순서 그대로, skip 없음
                    break
        except Exception:
            labels = []

        self._cache[sample_id] = labels
        return labels

    # ------------------------------------------------------------------ 라벨
    def parse_label_value(self, raw_label) -> int:
        """라벨 원본 → 정수. 제외 대상이면 exclude_label(-1).

        제외 대상
            None / NaN / 빈 문자열
            none, unknown, nan       (대소문자 무관)
            복합 라벨                 ('Stridor, Rhonchi', 'Rhonchi/Stridor' 등)
            label_map 에 없는 이름
        """
        if raw_label is None or (isinstance(raw_label, float) and pd.isna(raw_label)):
            return self.exclude_label

        text = str(raw_label).strip()
        if text == "" or text.lower() in {"none", "unknown", "nan", "nolabel"}:
            return self.exclude_label

        parts = (text.replace("\n", ",").replace("/", ",")
                     .replace(";", ",").replace("+", ",")
                     .replace("&", ",").split(","))
        labels = [p.strip() for p in parts if p.strip()]
        if len(labels) == 1:
            return self._label_lookup.get(labels[0].lower(), self.exclude_label)
        return self.exclude_label            # 복합 라벨

    # ------------------------------------------------------------------ 매칭
    def build_sample_items(self) -> List[Tuple[str, List[SegmentItem]]]:
        """[(sample_id, [(path, label_id, sample_id, cycle_idx), ...]), ...]"""
        by_id: Dict[str, List[SegmentItem]] = {}
        self.n_files = 0
        self.n_in_cohort = 0
        self.unparsed: List[str] = []       # 파일명 파싱 실패
        self.no_label_file: List[str] = []  # 엑셀을 못 찾음
        self.out_of_range: List[str] = []   # cycle_idx 가 엑셀 행 수를 넘음
        self.mismatch: List[Tuple[str, str, str]] = []   # 엑셀 라벨 ≠ 파일명 라벨
        self.dropped: Dict[str, int] = {}

        for path in sorted(self.mel_base_path.rglob(f"*{self.suffix}")):
            self.n_files += 1
            parsed = self.parse_segment_name(path)
            if parsed is None:
                self.unparsed.append(path.name)
                continue

            sample_id, cycle_idx, name_label = parsed
            if self.use_cohort_filter and sample_id not in self.cohort_ids:
                continue
            self.n_in_cohort += 1

            labels = self.cycle_labels(sample_id)
            if not labels:
                self.no_label_file.append(path.name)
                continue
            if cycle_idx >= len(labels):
                self.out_of_range.append(f"{path.name} (엑셀 {len(labels)}행)")
                continue

            raw = labels[cycle_idx]
            if sanitize(raw) != name_label:
                self.mismatch.append((path.name, str(raw), name_label))

            label_id = self.parse_label_value(raw)
            if label_id < 0:
                key = "<NaN>" if pd.isna(raw) else str(raw).strip()
                self.dropped[key] = self.dropped.get(key, 0) + 1
                continue

            by_id.setdefault(sample_id, []).append(
                (path, label_id, sample_id, cycle_idx))

        for sample_id in by_id:
            by_id[sample_id].sort(key=lambda item: item[3])       # cycle_idx 순

        return [(sample_id, by_id[sample_id]) for sample_id in sorted(by_id)]

    def summarize(self) -> Dict[str, object]:
        items = self.build_sample_items()
        label_counts: Dict[int, int] = {}
        for _, segs in items:
            for _, label_id, _, _ in segs:
                label_counts[label_id] = label_counts.get(label_id, 0) + 1

        return {
            "n_cohort_ids": len(self.cohort_ids),
            "n_files": self.n_files,
            "n_in_cohort": self.n_in_cohort,
            "n_usable": sum(label_counts.values()),
            "n_dropped": sum(self.dropped.values()),
            "n_usable_ids": len(items),
            "label_counts": label_counts,
            "dropped_labels": dict(self.dropped),
            "unparsed": self.unparsed,
            "no_label_file": self.no_label_file,
            "out_of_range": self.out_of_range,
            "mismatch": self.mismatch,
        }


if __name__ == "__main__":
    from utils import ROOT

    mel_folder = os.path.join(ROOT, "db", "new_gt", "10s_mel")

    parser = SegmentMelParser(mel_folder)
    s = parser.summarize()
    inv_map = {v: k for k, v in parser.label_map.items()}

    print(f"mel folder: {mel_folder}")
    print(f"cohort ids (utils.matched_ids) : {s['n_cohort_ids']}")
    print(f"files on disk      : {s['n_files']:,}")
    print(f"segments in cohort : {s['n_in_cohort']:,}")
    print(f"segments usable    : {s['n_usable']:,}   ids: {s['n_usable_ids']}")
    print(f"segments dropped   : {s['n_dropped']:,}")

    for key, title in [("unparsed", "파일명 파싱 실패"),
                       ("no_label_file", "라벨 엑셀 없음"),
                       ("out_of_range", "cycle_idx 가 엑셀 행 수 초과")]:
        if s[key]:
            print(f"\n★ {title}: {len(s[key]):,}")
            for name in s[key][:10]:
                print(f"    {name}")

    if s["mismatch"]:
        print(f"\n★ 엑셀 라벨 ≠ 파일명 라벨: {len(s['mismatch']):,}")
        for name, excel, fname in s["mismatch"][:10]:
            print(f"    {name}\n      엑셀 {excel!r}  vs  파일명 {fname!r}")

    print("\nlabel distribution (엑셀 기준)")
    total = max(s["n_usable"], 1)
    for label_id in sorted(s["label_counts"]):
        c = s["label_counts"][label_id]
        print(f"  {label_id} {inv_map.get(label_id, '?'):10} {c:7,}  {c/total:7.2%}")
    counts = list(s["label_counts"].values())
    if counts:
        print(f"  imbalance (max/min) = {max(counts)/min(counts):.1f}x")

    print(f"\ndropped ({s['n_dropped']:,})  — 학습/평가에서 완전히 제외")
    for k in sorted(s["dropped_labels"], key=lambda k: -s["dropped_labels"][k]):
        n = s["dropped_labels"][k]
        tag = "multi-label" if any(x in k for x in (",", "/", "+", ";", "&")) else ""
        print(f"  {k[:44]:46} {n:6,}  {tag}")

    items = parser.build_sample_items()
    if items:
        sample_id, segs = items[0]
        path, label_id, _, cycle_idx = segs[0]
        print(f"\n[{sample_id}] segments {len(segs)}")
        print(f"  {path.name}  c{cycle_idx}  label {label_id} ({inv_map.get(label_id, '?')})")
        print(f"  shape {np.load(path).shape}")
