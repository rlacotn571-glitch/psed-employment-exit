"""
KoDLD(장애인삶 패널조사) Long CSV → 04_external_validation.py 입력 파일 준비
==========================================================================
입력: KODDI 변수별 다운로드(Long type) CSV — 1행 한글 라벨, 2행 변수명(W1_ 접두사), 3행부터 자료.
출력: kodld_long.csv (PSED 변수명으로 바꾼 person-wave 파일) + kodld_mapping_*.json 두 가지 설계.

  설계 A (with_tenure) : 지수차수 1~3, 근속(C5_07_2m) 포함 — PSED 절제분석 9변수 중 KoDLD에 있는 6개
  설계 B (all_waves)   : 지수차수 1~5, 근속 제외 — 5개

장애유형 코드는 순서가 다르므로 PSED a0710 순서로 재부호화한다 (아래 A0710_MAP).
사용법:  python 05_kodld_prepare.py --csv "<KoDLD long csv>" --outdir kodld_ready
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

# KoDLD BA02 (법정 순서) → PSED a0710 순서
A0710_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 13, 7: 9, 8: 10, 9: 12, 10: 11, 11: 14, 12: 15, 13: 6, 14: 7, 15: 8}
WAVE_YEAR = {1: 2018, 2: 2019, 3: 2020, 4: 2021, 5: 2022, 6: 2023}


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, encoding="cp949", header=1, low_memory=False)
    df.columns = [c[3:] if c.startswith("W1_") else c for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", type=Path, required=True); ap.add_argument("--outdir", type=Path, default=Path("kodld_ready"))
    a = ap.parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)
    df = load(a.csv)
    out = pd.DataFrame({
        "pid": df.PID, "wave": df.WAVE, "year": df.WAVE.map(WAVE_YEAR), "wt01": df.WT, "respond": df.RESPOND,
        # 종사상지위 → PSED emp05 (1 임금 2 자영 3 무급가족 4 무직[실업+비경활])
        "emp05": np.select([df.C5_03A.isin([1, 2, 3]), df.C5_03A.isin([4, 5]), df.C5_03A == 6, df.C5_02A == 2], [1, 2, 3, 4], np.nan),
        "ca2603": df.C5_08C, "aca0403": df.C5_07_2m, "gender": df.BA01, "birthy": df.Q1_06,
        "a0710": df.BA02.map(A0710_MAP), "grade02": df.BA04,
        # 해석용 추가 변수
        "hours_week": df.C5_05_1_3, "fulltime": df.C5_03B, "emp_type": df.C5_03A, "jobsat_overall": df.C5_10_c9,
        "jobsat_security": df.C5_10_c2, "marital": df.BA08, "hh_size": df.Q1_01, "educ": df.BA10 if "BA10" in df else np.nan,
        "region": df.SQ0,
    })
    out["age"] = out.year - out.birthy
    out.to_csv(a.outdir / "kodld_long.csv", index=False)
    base = {"id": "pid", "wave": "wave", "weight": "wt01", "year": "year",
            "emp_status": {"var": "emp05", "wage": [1], "jobless": [4]},
            "predictors": {"ca2603": {"var": "ca2603"}, "gender": {"var": "gender"}, "birthy": {"var": "birthy"},
                           "a0710": {"var": "a0710"}, "grade02": {"var": "grade02"}}}
    withT = json.loads(json.dumps(base)); withT["predictors"]["aca0403"] = {"var": "aca0403"}; withT["index_waves"] = [1, 2, 3]
    allW = json.loads(json.dumps(base)); allW["index_waves"] = [1, 2, 3, 4, 5]
    json.dump(withT, open(a.outdir / "kodld_mapping_with_tenure.json", "w"), indent=1, ensure_ascii=False)
    json.dump(allW, open(a.outdir / "kodld_mapping_all_waves.json", "w"), indent=1, ensure_ascii=False)
    print(out.shape, "\n", out.groupby("wave").emp05.value_counts().unstack().fillna(0).astype(int))
    print("->", a.outdir)


if __name__ == "__main__":
    main()
