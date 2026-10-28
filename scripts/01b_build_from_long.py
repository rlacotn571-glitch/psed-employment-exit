"""
PSED 2차웨이브 통합 Long Data -> 분석파일
=========================================
EDI 공표용 누적판에 들어 있는 'PSED_2wave_Long(2016~2022)' 를 직접 씁니다.
차수별 파일을 합치는 00/01 경로는 필요 없습니다 — 이미 person-wave 로 쌓여 있고
변수명도 전 차수 통일돼 있습니다.

실측 (2026-09-18, 22패널 v2 기준)
---------------------------------
- 원자료: 32,039 person-wave · 4,577명 · 1~7차(2016~2022)
- 위험집합: 9,271 person-wave (t 임금근로자 & t+1 인접 관측)
- 결과변수 B 사건: 740건 (7.98%)  -> 안 B 확정
- 결과변수 C 사건: 824건 (8.89%)

사용법
------
  python 01b_build_from_long.py --long "long_2016_2022.sav" --out analysis.csv
  python 01b_build_from_long.py --long ... --outcome C      # 민감도 분석
  python 01b_build_from_long.py --long ... --outcome A      # 동일 일자리 이탈
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

# --- 실제 코드북으로 확정한 키 변수 -----------------------------------------
KEY = {
    "id": "pid",
    "wave": "wave",
    "year": "year",
    "weight_cs": "wt01",        # 횡단면 가중치
    "weight_lt": "wt02",        # 종단면 가중치
    "emp5": "emp05",            # 1 임금 2 자영 3 무급가족 4 실업 5 비경활
    "emp6": "emp06",            # 1 임금기존 2 임금신규 3 비임금기존 4 비임금신규 5 실업 6 비경활
    "jobid": "empjobid",        # 주업 일자리 ID
    "jobcon": "cajobcon",       # 일자리 지속여부
    "distype": "a0710",         # 주된 장애유형 (15개)
    "severity": "grade02",      # 중증 여부
    "tenure": "aca0403",        # 임금근로자 근속기간(개월)
}
WAGE, NONWAGE, JOBLESS = [1], [2, 3], [4, 5]

# 시점 누출을 일으키는 라벨 키워드 — 예측변수에서 제외
LEAK_KEYWORDS = [
    "이직", "구직", "퇴사", "그만", "해고", "실직", "일자리를 잃", "전직",
    "재취업", "권고사직", "취업희망", "일자리 탐색", "구인", "다음 일자리",
    "이직의사", "이직계획",
]

# Δ(전차수 대비 변화량)를 만들 연속형 변수
DELTA_VARS = ["aca0403", "ca2603", "ca2507"]   # 근속기간, 월평균 소득, 실제 근무시간


def build(long_path: Path, predictors: list[str], outcome: str):
    cols = sorted(set(list(KEY.values()) + predictors))
    df, meta = pyreadstat.read_sav(str(long_path), usecols=cols)
    lab = meta.column_names_to_labels

    # --- 특수코드 → 결측: 값 라벨이 '모름/응답거절' 계열이면 NaN ------------------
    # 예: ca2603 월평균소득 999999, aca0403 근속기간 999. 이걸 안 하면 로지스틱이
    # 999999 를 초고소득으로 읽고, SHAP 의존성 그림이 왜곡됩니다.
    SENTINEL_WORDS = ("모름", "응답거절", "무응답")
    vlabels = meta.variable_value_labels or {}
    n_recoded = {}
    for col, labels in vlabels.items():
        if col not in df.columns:
            continue
        bad = [code for code, lab in labels.items() if any(w in str(lab) for w in SENTINEL_WORDS)]
        if bad:
            mask = df[col].isin(bad)
            if mask.any():
                df.loc[mask, col] = np.nan
                n_recoded[col] = int(mask.sum())
    if n_recoded:
        top = sorted(n_recoded.items(), key=lambda kv: -kv[1])[:8]
        print(f"  특수코드 → NaN 처리: {len(n_recoded)}개 변수 · 상위 {top}")

    d = df.sort_values([KEY["id"], KEY["wave"]]).reset_index(drop=True)
    g = d.groupby(KEY["id"], sort=False)

    # --- 파생변수: t차수 이하 정보만 사용 -----------------------------------
    for v in DELTA_VARS:
        if v in d.columns:
            d[f"{v}_delta"] = d[v] - g[v].shift(1)
    d["age"] = d[KEY["year"]] - d["birthy"] if "birthy" in d.columns else np.nan
    d["wage_lag1"] = g[KEY["emp5"]].shift(1).eq(1)
    d["wage_lag2"] = g[KEY["emp5"]].shift(2).eq(1)
    d["wage_hist2"] = d[["wage_lag1", "wage_lag2"]].sum(axis=1)
    d["n_prior_waves"] = g.cumcount()
    d["job_changed_last"] = g[KEY["jobid"]].shift(1).ne(d[KEY["jobid"]]).astype(float)

    # --- 결과변수: t+1 -------------------------------------------------------
    nxt_wave = g[KEY["wave"]].shift(-1)
    nxt_emp = g[KEY["emp5"]].shift(-1)
    nxt_jobid = g[KEY["jobid"]].shift(-1)
    d["_adjacent"] = (nxt_wave - d[KEY["wave"]]) == 1

    if outcome == "B":
        y = nxt_emp.isin(JOBLESS)
    elif outcome == "C":
        y = nxt_emp.isin(NONWAGE + JOBLESS)
    elif outcome == "A":
        y = nxt_emp.isin(JOBLESS) | (nxt_jobid.notna() & d[KEY["jobid"]].notna()
                                     & nxt_jobid.ne(d[KEY["jobid"]]))
    else:
        raise SystemExit(f"알 수 없는 outcome: {outcome}")

    d["y"] = np.where(nxt_emp.notna(), y.astype(float), np.nan)

    # --- 위험집합 -----------------------------------------------------------
    flow = []

    def step(frame, label):
        flow.append({"단계": label, "person_wave": len(frame),
                     "사람": frame[KEY["id"]].nunique()})
        return frame

    rs = step(d, "전체 person-wave")
    rs = step(rs[rs[KEY["emp5"]] == 1], "t차수 임금근로자")
    rs = step(rs[rs["_adjacent"]], "t+1이 인접 차수")
    rs = step(rs[rs["y"].notna()], "결과변수 확정")
    rs = rs.reset_index(drop=True)

    # --- 예측변수에서 구조적 누출 제거 --------------------------------------
    structural = [KEY["emp5"], KEY["emp6"], KEY["jobid"], KEY["jobcon"], "_adjacent"]
    feature_cols = [c for c in rs.columns
                    if c not in structural + [KEY["id"], KEY["wave"], KEY["year"],
                                              KEY["weight_cs"], KEY["weight_lt"], "y"]]
    out = rs[[KEY["id"], KEY["wave"], KEY["year"], KEY["weight_cs"], KEY["weight_lt"],
              KEY["distype"], KEY["severity"], "y"] + feature_cols].copy()
    out = out.loc[:, ~out.columns.duplicated()]
    out.attrs['sentinel_recoded'] = n_recoded
    return out, pd.DataFrame(flow), lab, [c for c in structural if c in rs.columns]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", type=Path, required=True, help="PSED_2wave_Long(...).sav")
    ap.add_argument("--predictors", type=Path, default=Path("predictors_selected.csv"))
    ap.add_argument("--out", type=Path, default=Path("analysis.csv"))
    ap.add_argument("--outcome", choices=["A", "B", "C"], default="B")
    args = ap.parse_args()

    preds = pd.read_csv(args.predictors)["var"].tolist()
    out, flow, lab, dropped = build(args.long, preds, args.outcome)

    rate = (out.groupby("wave")
              .agg(n=("y", "size"), events=("y", "sum"))
              .assign(event_rate_pct=lambda t: (t.events / t.n * 100).round(2)))

    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    report = args.out.with_suffix(".report.xlsx")
    with pd.ExcelWriter(report, engine="openpyxl") as xl:
        flow.to_excel(xl, sheet_name="표본통제도", index=False)
        rate.to_excel(xl, sheet_name="차수별_사건률")
        pd.DataFrame({"변수": [c for c in out.columns],
                      "라벨": [lab.get(c, "") for c in out.columns]}
                     ).to_excel(xl, sheet_name="변수사전", index=False)
        pd.DataFrame({"제거된_구조적누출변수": dropped}).to_excel(xl, sheet_name="누출차단", index=False)
        pd.DataFrame(list(out.attrs.get("sentinel_recoded", {}).items()), columns=["변수", "NaN처리건수"]
                     ).to_excel(xl, sheet_name="특수코드처리", index=False)

    print(flow.to_string(index=False))
    print(f"\n결과변수 안 {args.outcome} · 차수별 사건률")
    print(rate.to_string())
    print(f"\n전체 사건률 {out.y.mean()*100:.2f}%  ({int(out.y.sum()):,}건 / {len(out):,})")
    print(f"예측변수 {len([c for c in out.columns]) - 8}개")
    print(f"\n분석파일 -> {args.out}\n리포트   -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
