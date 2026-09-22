"""
PSED 2차웨이브 - 차수별 변수 맵 자동 생성
=========================================
1주차 작업의 병목을 자동화합니다.

하는 일
-------
raw/ 안의 모든 차수 파일(.sav/.dta/.sas7bdat)을 열어서
 - 차수별 변수명 목록
 - 변수 라벨(한글 문항 텍스트)
 - 값 라벨(코딩 스킴)
 - 결측률
을 뽑아 하나의 엑셀 파일로 만듭니다. 라벨 텍스트가 비슷한 변수끼리 묶어
"같은 개념인데 차수마다 변수명이 다른" 경우를 후보로 제시합니다.

사람이 할 일은 생성된 엑셀의 `analysis_name` 열을 채우는 것뿐입니다.
그 열이 채워진 파일이 01_build_panel.py의 입력이 됩니다.

사용법
------
  python 00_build_variable_map.py --raw ./raw --out ./variable_map.xlsx

파일명에서 차수를 못 읽어내면 --wave-regex 로 알려주십시오.
"""

import argparse
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import pyreadstat

READERS = {
    ".sav": pyreadstat.read_sav,
    ".zsav": pyreadstat.read_sav,
    ".dta": pyreadstat.read_dta,
    ".sas7bdat": pyreadstat.read_sas7bdat,
}

# 파일명 안의 연도(2016~2035) 또는 "3차"/"w3"/"wave3" 패턴
DEFAULT_WAVE_REGEX = r"(?:(20[1-3]\d)|(?:(?:wave|w|)\s*_?(\d{1,2})\s*차))"


def detect_wave(name: str, regex: str) -> str | None:
    m = re.search(regex, name, flags=re.IGNORECASE)
    if not m:
        return None
    year, ordinal = m.group(1), m.group(2)
    return year if year else f"w{int(ordinal):02d}"


def normalize(text: str) -> str:
    """라벨 비교용 정규화: 공백/괄호/기호 제거, 유니코드 정규화."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"[\s()\[\]{}·.,:;/\\\-_'\"]+", "", text).lower()


def read_meta(path: Path):
    reader = READERS[path.suffix.lower()]
    # metadataonly=True 면 수만 행짜리 파일도 즉시 읽힙니다.
    _, meta = reader(str(path), metadataonly=True)
    df, _ = reader(str(path))
    return df, meta


def describe_file(path: Path, wave: str) -> pd.DataFrame:
    df, meta = read_meta(path)
    labels = meta.column_names_to_labels or {}
    value_labels = meta.variable_value_labels or {}

    rows = []
    for col in meta.column_names:
        series = df[col]
        vl = value_labels.get(col)
        rows.append(
            {
                "wave": wave,
                "source_file": path.name,
                "source_name": col,
                "label": labels.get(col, "") or "",
                "n_obs": int(series.notna().sum()),
                "missing_pct": round(float(series.isna().mean()) * 100, 2),
                "n_unique": int(series.nunique(dropna=True)),
                "dtype": str(series.dtype),
                "value_labels": "; ".join(f"{k}={v}" for k, v in vl.items()) if vl else "",
                "analysis_name": "",  # <- 사람이 채우는 열
                "domain": "",  # <- 사람이 채우는 열 (기본정보/직업능력/...)
                "leakage_risk": "",  # <- 사람이 채우는 열 (y/n)
            }
        )
    return pd.DataFrame(rows)


def build_concept_candidates(long: pd.DataFrame, threshold: float = 0.86) -> pd.DataFrame:
    """
    라벨이 유사한 변수끼리 묶어 '같은 개념 후보'를 제시합니다.
    완전 일치 라벨은 같은 그룹으로, 근사 일치는 따로 표시합니다.
    """
    labelled = long[long["label"].str.len() > 0].copy()
    labelled["label_norm"] = labelled["label"].map(normalize)

    # 1) 정규화 라벨이 완전히 같은 것들
    exact = (
        labelled.groupby("label_norm")
        .agg(
            label_example=("label", "first"),
            waves=("wave", lambda s: ", ".join(sorted(set(s)))),
            n_waves=("wave", lambda s: s.nunique()),
            source_names=("source_name", lambda s: ", ".join(sorted(set(s)))),
            n_distinct_names=("source_name", lambda s: s.nunique()),
        )
        .reset_index()
    )
    exact["match"] = "exact_label"

    # 2) 한 차수에만 있는 라벨들 사이의 근사 일치
    singles = exact[exact["n_waves"] == 1]["label_norm"].tolist()
    fuzzy_rows = []
    for i, a in enumerate(singles):
        for b in singles[i + 1 :]:
            if abs(len(a) - len(b)) > max(6, 0.3 * len(a)):
                continue
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio >= threshold:
                fuzzy_rows.append({"label_norm": a, "near_match": b, "ratio": round(ratio, 3)})
    fuzzy = pd.DataFrame(fuzzy_rows)

    out = exact.sort_values(["n_waves", "label_norm"], ascending=[False, True])
    # 전 차수 일관 변수를 맨 위로 (주분석 후보)
    return out, fuzzy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("./raw"), help="원자료 폴더")
    ap.add_argument("--out", type=Path, default=Path("./variable_map.xlsx"))
    ap.add_argument("--wave-regex", default=DEFAULT_WAVE_REGEX)
    args = ap.parse_args()

    if not args.raw.is_dir():
        print(f"[에러] 폴더가 없습니다: {args.raw}", file=sys.stderr)
        return 1

    files = sorted(p for p in args.raw.rglob("*") if p.suffix.lower() in READERS)
    if not files:
        print(f"[에러] {args.raw} 안에 .sav/.dta/.sas7bdat 파일이 없습니다.", file=sys.stderr)
        print("      EDI 사이트에서 받은 원자료를 이 폴더에 풀어주십시오.", file=sys.stderr)
        return 1

    frames, unknown = [], []
    for path in files:
        wave = detect_wave(path.name, args.wave_regex)
        if wave is None:
            unknown.append(path.name)
            wave = f"UNKNOWN_{path.stem[:20]}"
        print(f"  읽는 중  [{wave}]  {path.name}")
        try:
            frames.append(describe_file(path, wave))
        except Exception as exc:  # 파일 하나가 깨져도 나머지는 계속
            print(f"    !! 실패: {exc}", file=sys.stderr)

    if not frames:
        print("[에러] 읽어낸 파일이 없습니다.", file=sys.stderr)
        return 1

    long = pd.concat(frames, ignore_index=True)
    concepts, fuzzy = build_concept_candidates(long)

    # 차수 x 변수명 존재 여부 격자 — 어느 변수가 전 차수에 있는지 한눈에
    grid = (
        long.assign(present=1)
        .pivot_table(index="source_name", columns="wave", values="present", aggfunc="max", fill_value=0)
        .sort_index()
    )
    grid["n_waves"] = grid.sum(axis=1)
    grid = grid.sort_values("n_waves", ascending=False)

    with pd.ExcelWriter(args.out, engine="openpyxl") as xl:
        long.to_excel(xl, sheet_name="1_변수목록(채울것)", index=False)
        concepts.to_excel(xl, sheet_name="2_개념후보_라벨일치", index=False)
        if not fuzzy.empty:
            fuzzy.to_excel(xl, sheet_name="3_근사일치_확인필요", index=False)
        grid.to_excel(xl, sheet_name="4_차수x변수_격자")

    print()
    print(f"완료 -> {args.out}")
    print(f"  차수 {long['wave'].nunique()}개 · 변수-차수 행 {len(long):,}개")
    print(f"  전 차수 공통 변수명: {int((grid['n_waves'] == long['wave'].nunique()).sum()):,}개")
    if unknown:
        print(f"  [주의] 차수를 못 읽은 파일 {len(unknown)}개 -> --wave-regex 로 지정하십시오:")
        for name in unknown[:10]:
            print(f"         {name}")
    print()
    print("다음 할 일: '1_변수목록(채울것)' 시트의 analysis_name / domain / leakage_risk 열을 채우고")
    print("            python 01_build_panel.py --map variable_map.xlsx 를 실행하십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
