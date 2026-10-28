"""
PSED - 민감도 분석 · 부록 일괄 실행
====================================
연구설계 문서 '남은 분석 (부록·민감도)' 5건을 한 번에 돌립니다.

  S1  대응 ΔAUROC (LightGBM − 로지스틱) 클러스터 부트스트랩 CI   ← 기여 1번의 통계적 근거
  S2  결과변수 C안 / 코로나 기간 제외 / 중증·경증 분리 재추정
  S3  가중치 3방식 비교 (무가중-무가중 · 무가중-가중 · 가중-가중)
  S4  탈락 분석 (탈락자 vs 유지자 기저특성, 탈락 예측 보조모형)
  S5  웨이브별 SHAP 이동 그림

주모형 하이퍼파라미터는 주분석 표본에서 한 번만 튜닝해 고정합니다.
민감도 분석은 데이터·결과변수를 바꾸는 것이지 모형을 다시 고르는 것이 아니기 때문입니다.

사용법
------
  python 03_sensitivity.py --data analysis_v2.csv --data-c analysis_v2_C.csv \
      --long long_2016_2023.sav --outdir ./results_sens --shap-by-wave results_v2/results.xlsx
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

m02 = import_module("02_model")  # 지표·전처리·모델 정의 재사용
warnings.filterwarnings("ignore")

ID, WAVE, W, GROUP, Y = m02.ID_COL, m02.WAVE_COL, m02.WEIGHT_COL, m02.GROUP_COL, m02.TARGET
SEV = "grade02"


# ---------------------------------------------------------------------------
def split(df, train_waves, test_waves):
    ws = df[WAVE].astype(float)
    tr = df[ws.isin(train_waves)].copy()
    te = df[ws.isin(test_waves)].copy()
    return tr, te


def features(df):
    cols = [c for c in df.columns if c not in m02.NON_FEATURES]
    for c in cols:
        if str(df[c].dtype) in ("bool", "boolean"):
            df[c] = df[c].astype(float)
    return cols


def fit_fixed(name, params, Xtr, ytr, seed, sample_weight=None):
    """튜닝된 파라미터를 고정하고 재적합."""
    pipe, _ = m02.candidate_models(Xtr, seed)[name]
    pipe.set_params(**params)
    if sample_weight is not None:
        pipe.fit(Xtr, ytr, clf__sample_weight=sample_weight)
    else:
        pipe.fit(Xtr, ytr)
    return pipe


def paired_delta_auc(y, p_a, p_b, w, groups, n_boot, seed):
    """같은 부트스트랩 표본에서 AUROC(a) − AUROC(b). 사람 단위 클러스터."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups); uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        if y[idx].min() == y[idx].max():
            continue
        deltas.append(roc_auc_score(y[idx], p_a[idx], sample_weight=w[idx])
                      - roc_auc_score(y[idx], p_b[idx], sample_weight=w[idx]))
    d = np.array(deltas)
    return {"delta_mean": float(d.mean()), "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)), "p_delta_le_0": float((d <= 0).mean())}


def run_spec(label, df, train_w, test_w, params, seed, weighted_train=False, subset=None):
    """하나의 민감도 사양을 적합·평가."""
    if subset is not None:
        df = df[subset(df)].copy()
    tr, te = split(df, train_w, test_w)
    cols = features(tr); features(te)
    Xtr, ytr, Xte, yte = tr[cols], tr[Y].to_numpy(), te[cols], te[Y].to_numpy()
    wte = te[W].to_numpy()
    sw = tr[W].to_numpy() if weighted_train else None
    rows = []
    for name in ("logistic", "lightgbm"):
        mdl = fit_fixed(name, params[name], Xtr, ytr, seed, sw)
        p = mdl.predict_proba(Xte)[:, 1]
        r = m02.evaluate(yte, p, wte, f"{name}")
        r.update({"spec": label, "n_train": len(tr), "n_test": len(te),
                  "events_test": int(yte.sum())})
        rows.append(r)
    print(f"  {label:34s} n_te={len(te):5d} ev={int(yte.sum()):3d} | "
          + " ".join(f"{r['model'][:5]} AUROC {r['auroc']:.3f}" for r in rows))
    return rows


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="주분석 파일 (안 B)")
    ap.add_argument("--data-c", type=Path, required=True, help="안 C 파일")
    ap.add_argument("--long", type=Path, required=True, help="Long Data .sav (탈락분석용)")
    ap.add_argument("--shap-by-wave", type=Path, default=None, help="주분석 results.xlsx")
    ap.add_argument("--outdir", type=Path, default=Path("./results_sens"))
    ap.add_argument("--train-waves", nargs="*", type=float, default=[1, 2, 3, 4])
    ap.add_argument("--test-waves", nargs="*", type=float, default=[5, 6, 7])
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--params", type=Path, default=None,
                    help="이전 런의 fixed_params.json — 주면 튜닝을 건너뜁니다")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data); df = df[df[Y].notna()]; df[Y] = df[Y].astype(int)
    dfc = pd.read_csv(args.data_c); dfc = dfc[dfc[Y].notna()]; dfc[Y] = dfc[Y].astype(int)

    # ---- 0. 주분석 사양에서 한 번만 튜닝 → 파라미터 고정 --------------------
    print("[0] 주분석 사양 튜닝 (한 번만)")
    tr, te = split(df, args.train_waves, args.test_waves)
    cols = features(tr); features(te)
    Xtr, ytr, Xte, yte = tr[cols], tr[Y].to_numpy(), te[cols], te[Y].to_numpy()
    wte = te[W].to_numpy()
    fitted, params, preds = {}, {}, {}
    if args.params and args.params.exists():
        params = json.loads(args.params.read_text())
        print(f"  파라미터 고정 (튜닝 생략): {args.params}")
        for name in ("logistic", "lightgbm"):
            fitted[name] = fit_fixed(name, params[name], Xtr, ytr, args.seed)
            preds[name] = fitted[name].predict_proba(Xte)[:, 1]
    else:
        for name in ("logistic", "lightgbm"):
            pipe, grid = m02.candidate_models(Xtr, args.seed)[name]
            fitted[name] = m02.tune(name, pipe, grid, Xtr, ytr, tr[ID], args.seed, args.n_iter, 5)
            params[name] = {k: v for k, v in fitted[name].get_params().items()
                            if k.startswith("clf__") and k in grid}
            preds[name] = fitted[name].predict_proba(Xte)[:, 1]
    (args.outdir / "fixed_params.json").write_text(
        json.dumps({k: {kk: (vv.item() if hasattr(vv, "item") else vv) for kk, vv in v.items()}
                    for k, v in params.items()}, indent=2, ensure_ascii=False))

    # ---- S1. 대응 ΔAUROC ------------------------------------------------------
    print("\n[S1] 대응 ΔAUROC (LightGBM − 로지스틱), 클러스터 부트스트랩")
    s1 = paired_delta_auc(yte, preds["lightgbm"], preds["logistic"], wte, te[ID].to_numpy(),
                          args.n_boot, args.seed)
    print(f"  Δ = {s1['delta_mean']:+.3f}  95% CI [{s1['ci_lo']:+.3f}, {s1['ci_hi']:+.3f}]"
          f"  P(Δ≤0) = {s1['p_delta_le_0']:.3f}")

    # ---- S2. 결과변수 · 기간 · 중증도 -------------------------------------------
    print("\n[S2] 결과변수 C / 코로나 제외 / 중증·경증 분리")
    s2 = []
    s2 += run_spec("주분석 (안 B)", df, args.train_waves, args.test_waves, params, args.seed)
    s2 += run_spec("안 C (비임금 전환 포함)", dfc, args.train_waves, args.test_waves, params, args.seed)
    covid_free_tr = [w for w in args.train_waves if w not in (4,)]
    covid_free_te = [w for w in args.test_waves if w not in (5,)]
    s2 += run_spec("코로나 제외 (4·5차 제거)", df, covid_free_tr, covid_free_te, params, args.seed)
    s2 += run_spec("중증만", df, args.train_waves, args.test_waves, params, args.seed,
                   subset=lambda d: d[SEV] == 1)
    s2 += run_spec("경증만", df, args.train_waves, args.test_waves, params, args.seed,
                   subset=lambda d: d[SEV] == 2)

    # ---- S3. 가중치 3방식 --------------------------------------------------------
    print("\n[S3] 가중치 처리 3방식")
    s3 = []
    for name in ("logistic", "lightgbm"):
        p = preds[name]
        r_uu = m02.evaluate(yte, p, None, name); r_uu["scheme"] = "① 무가중 학습 · 무가중 평가"
        r_uw = m02.evaluate(yte, p, wte, name); r_uw["scheme"] = "② 무가중 학습 · 가중 평가 (주분석)"
        mdl_w = fit_fixed(name, params[name], Xtr, ytr, args.seed, sample_weight=tr[W].to_numpy())
        r_ww = m02.evaluate(yte, mdl_w.predict_proba(Xte)[:, 1], wte, name)
        r_ww["scheme"] = "③ 가중 학습 · 가중 평가"
        s3 += [r_uu, r_uw, r_ww]
        print(f"  {name:10s} ① {r_uu['auroc']:.3f}  ② {r_uw['auroc']:.3f}  ③ {r_ww['auroc']:.3f}"
              f"   (보정기울기 ③ {r_ww['cal_slope']:.2f})")

    # ---- S4. 탈락 분석 ------------------------------------------------------------
    print("\n[S4] 탈락 분석")
    import pyreadstat
    base_cols = [ID, WAVE, "p", W, "emp05", "a0710", "grade02", "gender", "birthy", "aarea", "ca2603", "aca0403"]
    L, _ = pyreadstat.read_sav(str(args.long), usecols=base_cols)
    L = L.sort_values([ID, WAVE])
    g = L.groupby(ID, sort=False)
    L["next_p"] = g["p"].shift(-1)
    L["next_wave"] = g[WAVE].shift(-1)
    at_risk = L[(L["p"] == 1) & L["next_wave"].notna()].copy()
    at_risk["attrited"] = (at_risk["next_p"] != 1).astype(int)
    rate_by_wave = at_risk.groupby(WAVE)["attrited"].agg(n="size", attrited="sum", rate=lambda s: round(s.mean() * 100, 2))
    print(rate_by_wave.to_string())
    # 기저특성 비교 (표준화 평균차)
    def smd(a, b):
        return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)
    comp = []
    for v, lab in [("birthy", "출생연도"), ("gender", "성별(1=남)"), ("grade02", "중증여부(1=중증)"),
                   ("emp05", "경활상태코드"), ("ca2603", "월평균소득"), ("aca0403", "근속기간")]:
        a, b = at_risk.loc[at_risk.attrited == 1, v].dropna(), at_risk.loc[at_risk.attrited == 0, v].dropna()
        comp.append({"변수": lab, "탈락 평균": round(a.mean(), 2), "유지 평균": round(b.mean(), 2), "SMD": round(smd(a, b), 3)})
    comp = pd.DataFrame(comp); print(comp.to_string(index=False))
    # 장애유형별 탈락률
    by_type = at_risk.groupby("a0710")["attrited"].agg(n="size", rate=lambda s: round(s.mean() * 100, 2)).sort_values("rate", ascending=False)
    # 탈락 예측 보조모형 — 주분석 위험집합은 정의상 t+1 관측자뿐이라 탈락자가 없습니다.
    # 따라서 Long Data 에서 t 임금근로자 전체(탈락자 포함)를 다시 읽어 적합합니다.
    pred_cols = [c for c in features(df) if not c.endswith("_delta")
                 and c not in ("age", "wage_lag1", "wage_lag2", "wage_hist2", "n_prior_waves", "job_changed_last")]
    need = sorted(set([ID, WAVE, "p", W, "emp05"] + pred_cols))
    Lw, meta_w = pyreadstat.read_sav(str(args.long), usecols=need)
    for col, labels in (meta_w.variable_value_labels or {}).items():
        if col in Lw.columns:
            bad = [k for k, v in labels.items() if any(w_ in str(v) for w_ in ("모름", "응답거절", "무응답"))]
            if bad:
                Lw.loc[Lw[col].isin(bad), col] = np.nan
    Lw = Lw.merge(at_risk[[ID, WAVE, "attrited"]], on=[ID, WAVE], how="inner")
    Lw = Lw[Lw["emp05"] == 1].copy()
    tr_a, te_a = split(Lw, args.train_waves, args.test_waves)
    cols_a = [c for c in pred_cols if c in tr_a.columns]
    mdl_a = fit_fixed("lightgbm", params["lightgbm"], tr_a[cols_a], tr_a["attrited"].to_numpy(), args.seed)
    r_a = m02.evaluate(te_a["attrited"].to_numpy(), mdl_a.predict_proba(te_a[cols_a])[:, 1],
                       te_a[W].to_numpy(), "탈락 예측 (LightGBM)")
    print(f"  탈락 예측 보조모형: 임금근로자 검증 {len(te_a):,}행 · 탈락 {int(te_a.attrited.sum())}건 · AUROC {r_a['auroc']:.3f}")

    # ---- S5. 웨이브별 SHAP 이동 그림 ---------------------------------------------
    fig_path = None
    if args.shap_by_wave and args.shap_by_wave.exists():
        print("\n[S5] 웨이브별 SHAP 이동 그림")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sw = pd.read_excel(args.shap_by_wave, sheet_name="SHAP_웨이브별")
        top = sw[sw["rank"] <= 8].groupby("feature")["mean_abs_shap"].mean().nlargest(8).index
        piv = sw[sw.feature.isin(top)].pivot_table(index="wave", columns="feature", values="rank")
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for f in piv.columns:
            ax.plot(piv.index.astype(str), piv[f], marker="o", lw=1.6, label=f.replace("num__", ""))
        ax.invert_yaxis(); ax.set_ylabel("SHAP 순위 (1=최상)"); ax.set_xlabel("검증 차수")
        ax.set_title("검증 차수별 상위 예측변수 순위 이동"); ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=2, frameon=False)
        fig.tight_layout(); fig_path = args.outdir / "fig_shap_by_wave.png"; fig.savefig(fig_path, dpi=200)
        print(f"  -> {fig_path}")

    # ---- 저장 -----------------------------------------------------------------
    out = args.outdir / "sensitivity.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        pd.DataFrame([s1]).to_excel(xl, sheet_name="S1_대응ΔAUROC", index=False)
        pd.DataFrame(s2).to_excel(xl, sheet_name="S2_결과변수_기간_중증도", index=False)
        pd.DataFrame(s3).to_excel(xl, sheet_name="S3_가중치3방식", index=False)
        rate_by_wave.to_excel(xl, sheet_name="S4_차수별탈락률")
        comp.to_excel(xl, sheet_name="S4_기저특성비교", index=False)
        by_type.to_excel(xl, sheet_name="S4_장애유형별탈락률")
        pd.DataFrame([r_a]).to_excel(xl, sheet_name="S4_탈락예측모형", index=False)
    print(f"\n완료 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
