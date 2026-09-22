"""
KoDLD 외부검증 — 데이터가 오면 이 스크립트 하나로 끝나도록 미리 만들어 둔 것
=========================================================================
흐름
  1. KoDLD Long 파일(.sav/.csv)을 읽고, mapping.json 으로 변수명을 PSED 이름으로 바꿉니다.
  2. PSED 와 KoDLD 에 공통으로 있는 예측변수만 씁니다 (전체 106개가 다 있을 리 없음).
     - 원고의 절제분석(Table S4)이 9변수(소득·근속·사업장규모·계약·성별·연령·출생연도·장애유형·중증도)로
       전체 모형과 같은 성능을 냈으므로, 이 9개만 매핑돼도 외부검증은 성립합니다.
  3. PSED 전체(1~7차 지수차수)에서 고정 파라미터로 재적합 → KoDLD 에서 단 한 번 평가.
     (원래 모형은 106변수라 그대로 못 옮김. 공통 변수로 재적합한 모형이 '외부검증 대상 모형'이고,
      같은 변수로 PSED 시간적 검증도 다시 보고해 비교 기준을 맞춥니다.)
  4. 보고: AUROC/AUPRC(사람 클러스터 부트스트랩 CI), 보정 기울기·절편, Brier, DCA, 상위 10%,
     성별·중증도별 성능, 절편 재보정(recalibration-in-the-large) 후 보정.

사용법
  python 04_external_validation.py --psed analysis_v3.csv --kodld kodld_long.sav \
      --mapping kodld_mapping.json --params fixed_params.json --outdir results_external

mapping.json 예시 (KoDLD 코드북을 보고 채웁니다 — 값 코딩도 맞춰야 함: 1=예 2=아니오 등)
{
  "id": "pid", "wave": "wave", "weight": "wgt_cs",
  "emp_status": {"var": "econ_act", "wage": [1], "jobless": [4, 5]},
  "predictors": {
    "ca2603": {"var": "income_month", "scale": 1.0},
    "aca0403": {"var": "tenure_months"},
    "ca1203": {"var": "firm_size_cat", "recode": {"1": 1, "2": 2, "3": 3}},
    "ca1401": {"var": "contract_fixed", "recode": {"1": 1, "2": 2}},
    "gender": {"var": "sex"}, "birthy": {"var": "birth_year"},
    "a0710": {"var": "dis_type", "recode": {"1": 1, "2": 2}},
    "grade02": {"var": "severity", "recode": {"1": 1, "2": 2}}
  }
}
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m02", HERE / "02_model.py"); m02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m02)
spec = importlib.util.spec_from_file_location("m03", HERE / "03_sensitivity.py"); m03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m03)


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".sav":
        import pyreadstat
        df, _ = pyreadstat.read_sav(str(path)); return df
    return pd.read_csv(path, low_memory=False)


def build_kodld(raw: pd.DataFrame, mp: dict) -> pd.DataFrame:
    """KoDLD Long → PSED 와 같은 열 이름의 person-wave 위험집합 (t 임금근로자, t+1 인접 관측)."""
    d = pd.DataFrame({"pid": raw[mp["id"]], "wave": raw[mp["wave"]].astype(float), "wt01": raw[mp["weight"]]})
    es = mp["emp_status"]; d["emp05"] = raw[es["var"]]
    for psed_name, spec_ in mp["predictors"].items():
        col = raw[spec_["var"]].astype(float)
        if "recode" in spec_:
            col = col.map({float(k): v for k, v in spec_["recode"].items()})
        if "scale" in spec_:
            col = col * spec_["scale"]
        d[psed_name] = col
    if "birthy" in d and "age" not in d and "year" in mp:
        d["age"] = raw[mp["year"]].astype(float) - d["birthy"]
    d = d.sort_values(["pid", "wave"]).reset_index(drop=True)
    g = d.groupby("pid", sort=False)
    nxt_wave, nxt_emp = g["wave"].shift(-1), g["emp05"].shift(-1)
    d["_adj"] = (nxt_wave - d["wave"]) == 1
    d["y"] = np.where(nxt_emp.notna(), nxt_emp.isin(es["jobless"]).astype(float), np.nan)
    rs = d[d["emp05"].isin(es["wage"]) & d["_adj"] & d["y"].notna()].copy()
    print(f"KoDLD risk set: {len(rs):,} person-waves, {rs.pid.nunique():,} persons, {int(rs.y.sum())} exits ({rs.y.mean()*100:.1f}%)")
    return rs.drop(columns=["_adj", "emp05"])


def evaluate_block(y, p, w, groups, label, n_boot):
    r = m02.evaluate(y, p, w, label)
    r.update({k: v for k, v in m02.bootstrap_ci(y, p, w, n_boot=n_boot, seed=0, groups=groups).items() if k.startswith("au")})
    r.update(m02.top_k_capture(y, p, 0.10, w)); return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psed", type=Path, required=True); ap.add_argument("--kodld", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True); ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--logistic-C", type=float, default=0.0014873521072935117)
    ap.add_argument("--outdir", type=Path, default=Path("results_external")); ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--max-age", type=float, default=None, help="KoDLD 지수시점 연령 상한 (예: 64 → PSED 생산연령 표본과 맞춤)")
    a = ap.parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)

    mp = json.load(open(a.mapping)); params = json.load(open(a.params)); params["logistic"] = {"clf__C": a.logistic_C}
    psed = pd.read_csv(a.psed, low_memory=False); psed = psed[psed.y.notna()]
    ko = build_kodld(read_any(a.kodld), mp)
    if mp.get("index_waves"):
        ko = ko[ko.wave.isin(mp["index_waves"])]; print(f"index waves {mp['index_waves']}: {len(ko):,} person-waves, {int(ko.y.sum())} exits")
    if a.max_age is not None:
        ko = ko[ko.age <= a.max_age]; print(f"age <= {a.max_age}: {len(ko):,} person-waves, {int(ko.y.sum())} exits ({ko.y.mean()*100:.1f}%)")
    common = [c for c in mp["predictors"] if c in psed.columns] + (["age"] if "age" in ko.columns and "age" in psed.columns else [])
    common = list(dict.fromkeys(common))
    print(f"공통 예측변수 {len(common)}개: {common}")
    for c in common:
        psed[c] = psed[c].astype(float); ko[c] = ko[c].astype(float)

    rows, preds = [], {}
    # (a) 비교 기준: 같은 변수로 PSED 시간적 검증 (1~4 → 5~7)
    tr, te = m03.split(psed, [1, 2, 3, 4], [5, 6, 7])
    # (b) 외부검증: PSED 전체로 적합 → KoDLD 평가
    for name in ["logistic", "lightgbm"]:
        m_int = m03.fit_fixed(name, params[name], tr[common], tr.y.to_numpy(float), seed=20260918)
        p_int = m_int.predict_proba(te[common])[:, 1]
        rows.append(evaluate_block(te.y.to_numpy(float), p_int, te.wt01.to_numpy(float), te.pid.to_numpy(), f"{name} | PSED temporal, {len(common)} predictors", a.n_boot))
        m_ext = m03.fit_fixed(name, params[name], psed[common], psed.y.to_numpy(float), seed=20260918)
        p = m_ext.predict_proba(ko[common])[:, 1]; preds[name] = p
        rows.append(evaluate_block(ko.y.to_numpy(float), p, ko.wt01.to_numpy(float), ko.pid.to_numpy(), f"{name} | KoDLD external", a.n_boot))
        # 절편 재보정 (recalibration-in-the-large): 배포 시 필요한 최소 갱신량을 보고
        from sklearn.linear_model import LogisticRegression
        lp = np.log(p / (1 - p)).reshape(-1, 1)
        rc = LogisticRegression(C=1e6).fit(lp, ko.y.to_numpy(float), sample_weight=ko.wt01.to_numpy(float))
        p_rc = rc.predict_proba(lp)[:, 1]
        rr = m02.evaluate(ko.y.to_numpy(float), p_rc, ko.wt01.to_numpy(float), f"{name} | KoDLD after logistic recalibration")
        rr["recal_intercept"], rr["recal_slope"] = float(rc.intercept_[0]), float(rc.coef_[0][0]); rows.append(rr)
        # 성별·중증도
        for col, code, lab in [("gender", 1, "men"), ("gender", 2, "women"), ("grade02", 1, "severe"), ("grade02", 2, "mild")]:
            if col in ko.columns:
                m = (ko[col] == code).to_numpy()
                if m.sum() > 50 and ko.y.to_numpy()[m].sum() >= 10:
                    rows.append(evaluate_block(ko.y.to_numpy(float)[m], p[m], ko.wt01.to_numpy(float)[m], ko.pid.to_numpy()[m], f"{name} | KoDLD {lab}", a.n_boot))
    res = pd.DataFrame(rows)
    dca = m02.dca_table(ko.y.to_numpy(float), preds, weights=ko.wt01.to_numpy(float))
    with pd.ExcelWriter(a.outdir / "external_validation.xlsx", engine="openpyxl") as xw:
        res.to_excel(xw, sheet_name="performance", index=False); dca.to_excel(xw, sheet_name="DCA", index=False)
        pd.DataFrame({"predictor": common}).to_excel(xw, sheet_name="predictors_used", index=False)
    pd.DataFrame(preds).assign(pid=ko.pid.values, wave=ko.wave.values, wt01=ko.wt01.values, y=ko.y.values).to_csv(a.outdir / "predictions_kodld.csv", index=False)
    print(res[["model", "n", "events", "auroc", "auroc_ci_lo", "auroc_ci_hi", "auprc", "cal_slope", "cal_intercept", "sensitivity", "ppv"]].round(3).to_string())
    print(f"\n-> {a.outdir / 'external_validation.xlsx'}")


if __name__ == "__main__":
    main()
