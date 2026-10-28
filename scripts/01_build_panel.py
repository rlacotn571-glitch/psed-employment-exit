"""
PSED - 사람-차수(person-wave) 분석파일 구축
==========================================
00_build_variable_map.py 가 만든 엑셀의 analysis_name 열을 근거로
차수별 파일을 하나의 종단 분석파일로 합칩니다.

핵심 설계 (연구설계 문서와 1:1 대응)
------------------------------------
- 위험집합: t차수 임금근로자만
- 결과변수: t+1차수 미취업 (= 안 B). A/C 안은 --outcome 으로 전환
- 경합사건: t+1차수 은퇴/사망/시설입소는 기본 제외, --competing-as-event 로 포함
- 누출 차단: leakage_risk='y' 변수와 t+1 차수 변수는 예측변수에서 물리적으로 제거
- 파생변수: 재직기간 누적, 전차수 대비 변화량(Δ), 과거 2차수 고용 연속성

사용법
------
  python 01_build_panel.py --map variable_map.xlsx --raw ./raw --out ./analysis.parquet
  python 01_build_panel.py --map variable_map.xlsx --outcome C   # 민감도 분석

먼저 아래 CONFIG 를 변수 맵을 보고 채워주십시오. 여기 적힌 이름은 예시입니다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

# ---------------------------------------------------------------------------
# CONFIG  — 변수 맵을 열어보고 실제 analysis_name 으로 바꿔주십시오
# ---------------------------------------------------------------------------
CONFIG = {
    # --- 2016년(2차웨이브 1차조사) 실제 코드북으로 확정 (2026-09-18) ---
    # 후속 차수에서 변수명이 바뀌면 variable_map 의 analysis_name 으로 여기에 맞춰주십시오.
    "person_id": "pid",                      # 패널 ID
    "empstat": "AB010103",                   # (가공) 경제활동상태 5개 구분
    "empstat_codes": {
        "wage": [1],                         # 임금근로자
        "nonwage": [2, 3],                   # 자영업자, 무급가족종사자
        "unemployed": [4, 5],                # 실업자, 비경제활동인구
    },
    # 주의: 2016년 AB010103 에는 은퇴/사망/시설입소 코드가 없습니다.
    # 비경제활동인구(5) 안에 은퇴가 섞여 있으므로, 경합사건은 별도 문항
    # (비경활 사유 / 패널 탈락사유)으로 식별해 여기에 채워야 합니다.
    # 비워두면 경합사건 제외 단계를 건너뜁니다.
    "competing": {},
    "disability_type": "A010701",            # 주된 장애유형 (15개)
    "severity": "AA010702",                  # (가공) 장애정도 1=중증 2=경증
    "weight": "wt01",                        # 가중치
    "delta_vars": [],                        # 코드북에서 만족도·우울 변수 확정 후 채우기
    "tenure": "",                            # 근속기간 변수 확정 후 채우기
}

OUTCOME_DEFS = {
    # 안 A: 동일 직장 이탈 — 직장 식별자나 재직기간 리셋으로 판별 (별도 컬럼 필요)
    "A": "same_job_exit",
    # 안 B (권고): t+1차수 미취업
    "B": "unemployed_next",
    # 안 C: 임금근로 이탈 (미취업 + 비임금근로 전환)
    "C": "wage_exit_next",
}

READERS = {
    ".sav": pyreadstat.read_sav,
    ".zsav": pyreadstat.read_sav,
    ".dta": pyreadstat.read_dta,
    ".sas7bdat": pyreadstat.read_sas7bdat,
}


def save_table(df: pd.DataFrame, path: Path) -> Path:
    """parquet 엔진이 없는 환경에서도 막히지 않게 csv 로 자동 대체합니다."""
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            path = path.with_suffix(".csv")
            print("  [알림] pyarrow/fastparquet 가 없어 csv 로 저장합니다 -> " + path.name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_map(path: Path) -> pd.DataFrame:
    vm = pd.read_excel(path, sheet_name="1_변수목록(채울것)")
    vm["analysis_name"] = vm["analysis_name"].astype("string").str.strip()
    filled = vm[vm["analysis_name"].notna() & (vm["analysis_name"] != "")].copy()
    if filled.empty:
        raise SystemExit(
            "[에러] analysis_name 열이 비어 있습니다.\n"
            "       variable_map.xlsx 의 '1_변수목록(채울것)' 시트를 먼저 채워주십시오."
        )
    dupes = filled.groupby(["wave", "analysis_name"]).size()
    if (dupes > 1).any():
        bad = dupes[dupes > 1]
        raise SystemExit(f"[에러] 같은 차수에 analysis_name 이 중복됐습니다:\n{bad}")
    return filled


def load_waves(vmap: pd.DataFrame, raw: Path) -> pd.DataFrame:
    """차수별 파일을 analysis_name 으로 리네임해 long 으로 쌓습니다."""
    frames = []
    for (wave, source_file), grp in vmap.groupby(["wave", "source_file"]):
        path = next(raw.rglob(source_file), None)
        if path is None:
            raise SystemExit(f"[에러] 파일을 못 찾았습니다: {source_file} (in {raw})")
        df, _ = READERS[path.suffix.lower()](str(path))
        rename = dict(zip(grp["source_name"], grp["analysis_name"]))
        keep = [c for c in rename if c in df.columns]
        sub = df[keep].rename(columns=rename)
        sub["wave"] = wave
        frames.append(sub)
        print(f"  [{wave}] {source_file}: {len(sub):,}행 · 변수 {len(keep)}개")

    panel = pd.concat(frames, ignore_index=True)
    pid = CONFIG["person_id"]
    if pid not in panel.columns:
        raise SystemExit(f"[에러] 식별자 '{pid}' 가 없습니다. CONFIG['person_id'] 를 고쳐주십시오.")

    # 차수를 순서 있는 정수로
    waves = sorted(panel["wave"].unique())
    panel["wave_idx"] = panel["wave"].map({w: i for i, w in enumerate(waves)})
    return panel.sort_values([pid, "wave_idx"]).reset_index(drop=True)


def classify_status(panel: pd.DataFrame) -> pd.DataFrame:
    col = CONFIG["empstat"]
    if col not in panel.columns:
        raise SystemExit(f"[에러] 경제활동상태 변수 '{col}' 가 없습니다.")
    codes = CONFIG["empstat_codes"]
    panel["is_wage"] = panel[col].isin(codes["wage"])
    panel["is_nonwage"] = panel[col].isin(codes["nonwage"])
    panel["is_unemployed"] = panel[col].isin(codes["unemployed"])

    comp = CONFIG.get("competing") or {}
    all_comp = [v for vals in comp.values() for v in vals]
    panel["is_competing"] = panel[col].isin(all_comp) if all_comp else False
    return panel


def add_derived(panel: pd.DataFrame) -> pd.DataFrame:
    """Δ, 누적 재직, 과거 고용 연속성 — 모두 t차수 이하 정보만 사용."""
    pid = CONFIG["person_id"]
    g = panel.groupby(pid, sort=False)

    for var in CONFIG["delta_vars"]:
        if var in panel.columns:
            panel[f"{var}_delta"] = panel[var] - g[var].shift(1)

    if CONFIG["tenure"] in panel.columns:
        panel["tenure_cummax"] = g[CONFIG["tenure"]].cummax()

    # 과거 2차수 임금근로 이력 (t-1, t-2) — 시점 누출 없음
    panel["wage_lag1"] = g["is_wage"].shift(1)
    panel["wage_lag2"] = g["is_wage"].shift(2)
    panel["wage_hist2"] = panel[["wage_lag1", "wage_lag2"]].sum(axis=1, min_count=1)

    # 관측 순서 (패널 경험)
    panel["n_prior_waves"] = g.cumcount()
    return panel


def _lead_bool(grouped, col: str) -> pd.Series:
    """
    groupby.shift(-1) 은 bool 컬럼을 object(NaN 포함)로 돌려주므로
    nullable boolean 으로 맞춰줍니다. 이걸 안 하면 ~ 연산이 정수 비트반전이 되어
    -1 이 나오고, 그 값이 컬럼 인덱서로 해석돼 KeyError 가 납니다.
    """
    return grouped[col].shift(-1).astype("boolean")


def build_outcome(panel: pd.DataFrame, which: str, competing_as_event: bool) -> pd.DataFrame:
    """t차수 행에 t+1차수 결과를 붙입니다."""
    pid = CONFIG["person_id"]
    g = panel.groupby(pid, sort=False)

    nxt_wave_idx = g["wave_idx"].shift(-1)
    panel["next_unemployed"] = _lead_bool(g, "is_unemployed")
    panel["next_nonwage"] = _lead_bool(g, "is_nonwage")
    panel["next_competing"] = _lead_bool(g, "is_competing")
    panel["next_observed"] = nxt_wave_idx.notna()

    # t+1 이 바로 다음 차수여야 합니다 (한 차수 건너뛴 관측은 제외)
    panel["next_is_adjacent"] = (nxt_wave_idx - panel["wave_idx"]) == 1

    if which == "B":
        y = panel["next_unemployed"]
    elif which == "C":
        y = panel["next_unemployed"] | panel["next_nonwage"]
    elif which == "A":
        col = OUTCOME_DEFS["A"]
        if col not in panel.columns:
            raise SystemExit(
                f"[에러] 안 A 는 '{col}' 컬럼이 필요합니다 (직장 식별자나 재직기간 리셋으로 만드십시오)."
            )
        y = panel[col].astype("boolean")
    else:
        raise SystemExit(f"[에러] 알 수 없는 outcome: {which}")

    if competing_as_event:
        y = y | panel["next_competing"]

    panel["y"] = y.astype("float")
    return panel


def apply_risk_set(panel: pd.DataFrame, competing_as_event: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """포함/제외를 순서대로 적용하고 표본 통제도(flow)를 함께 돌려줍니다."""
    flow = []

    def step(df, label):
        flow.append({"단계": label, "사람-차수": len(df), "사람": df[CONFIG["person_id"]].nunique()})
        return df

    df = step(panel, "전체 사람-차수")
    df = step(df[df["is_wage"]], "t차수 임금근로자")
    df = step(df[df["next_observed"]], "t+1차수 관측됨")
    df = step(df[df["next_is_adjacent"]], "t+1이 인접 차수")
    if not competing_as_event:
        is_comp = df["next_competing"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        df = step(df[~is_comp], "t+1 경합사건 제외")
    df = step(df[df["y"].notna()], "결과변수 확정")

    return df.reset_index(drop=True), pd.DataFrame(flow)


def drop_leakage(df: pd.DataFrame, vmap: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    flagged = (
        vmap.loc[vmap["leakage_risk"].astype("string").str.lower().isin(["y", "yes", "1"]), "analysis_name"]
        .dropna()
        .unique()
        .tolist()
    )
    # 결과변수 구성에 쓴 것들도 예측변수가 될 수 없습니다
    structural = [
        CONFIG["empstat"], "is_wage", "is_nonwage", "is_unemployed", "is_competing",
        "next_unemployed", "next_nonwage", "next_competing", "next_observed", "next_is_adjacent",
    ]
    dropped = sorted(set(flagged) | set(structural))
    present = [c for c in dropped if c in df.columns]
    return df.drop(columns=present), present


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=Path("./raw"))
    ap.add_argument("--out", type=Path, default=Path("./analysis.parquet"))
    ap.add_argument("--outcome", choices=["A", "B", "C"], default="B")
    ap.add_argument("--competing-as-event", action="store_true",
                    help="t+1 은퇴/사망/시설입소를 이탈로 간주 (민감도 분석 3번)")
    args = ap.parse_args()

    vmap = load_map(args.map)
    panel = load_waves(vmap, args.raw)
    panel = classify_status(panel)
    panel = add_derived(panel)
    panel = build_outcome(panel, args.outcome, args.competing_as_event)
    analysis, flow = apply_risk_set(panel, args.competing_as_event)
    analysis, dropped = drop_leakage(analysis, vmap)

    # --- 1주차 핵심 산출물: 웨이브별 사건률 -------------------------------
    rate = (
        analysis.groupby("wave")
        .agg(n=("y", "size"), events=("y", "sum"))
        .assign(event_rate_pct=lambda d: (d["events"] / d["n"] * 100).round(2))
    )

    save_path = save_table(analysis, args.out)
    report = args.out.with_suffix(".report.xlsx")
    with pd.ExcelWriter(report, engine="openpyxl") as xl:
        flow.to_excel(xl, sheet_name="표본통제도", index=False)
        rate.to_excel(xl, sheet_name="웨이브별_사건률")
        pd.DataFrame({"제거된_누출변수": dropped}).to_excel(xl, sheet_name="누출차단", index=False)

    written = save_path
    print()
    print(f"분석파일 -> {written}   ({len(analysis):,}행)")
    print(f"리포트   -> {report}")
    print()
    print(f"결과변수 안 {args.outcome} · 웨이브별 사건률")
    print(rate.to_string())
    overall = analysis["y"].mean() * 100
    print(f"\n전체 사건률: {overall:.2f}%")
    if overall < 5:
        print("  [판단 필요] 5% 미만입니다. 연구설계 문서의 결정지점 — 안 C 로 전환하거나")
        print("              예측 첨단을 2년으로 넓히는 것을 검토하십시오.")
    print(f"누출 차단으로 제거된 변수 {len(dropped)}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
