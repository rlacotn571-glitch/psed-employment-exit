"""
PSED - 예측모형 학습 · 시간적 검증 · 보정 · DCA · SHAP
======================================================
연구설계 문서의 '분석 파이프라인' 절을 그대로 구현합니다.

지켜지는 규칙
-------------
1. 하이퍼파라미터 튜닝은 초기 차수 안에서만, 사람 단위 GroupKFold 로.
2. 후기 차수(검증셋)는 단 한 번만 평가. 모델 선택에 쓰지 않습니다.
3. 주분석 = 무가중 학습 + 가중 평가. 가중치 3방식 비교표는 부록용으로 함께 산출.
4. AUROC 뿐 아니라 AUPRC · 보정(slope/intercept/Brier) · DCA net benefit 전부 보고.
5. SHAP 은 전역 / 장애유형 하위군 / 웨이브별 세 장.

사용법
------
  python 02_model.py --data analysis.parquet --outdir ./results
  python 02_model.py --data analysis.parquet --train-waves 2016 2017 2018 2019 \
                     --test-waves 2020 2021 2022

--train-waves 를 안 주면 차수를 시간순 6:4 로 자동 분할합니다.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ID_COL = "pid"
WAVE_COL = "wave"
WEIGHT_COL = "wt01"            # 횡단면 가중치
GROUP_COL = "a0710"            # 주된 장애유형 (하위군 분석 축)
TARGET = "y"
NON_FEATURES = {ID_COL, WAVE_COL, "wave_idx", "year", WEIGHT_COL, "wt02", TARGET}


def read_table(path: Path) -> pd.DataFrame:
    """01 단계가 parquet 또는 csv 중 무엇으로 떨어졌든 읽습니다."""
    if path.suffix == ".parquet":
        if path.exists():
            return pd.read_parquet(path)
        alt = path.with_suffix(".csv")
        if alt.exists():
            print(f"  [알림] {path.name} 이 없어 {alt.name} 을 읽습니다.")
            return pd.read_csv(alt)
        raise SystemExit(f"[에러] 파일이 없습니다: {path} / {alt}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def calibration_slope_intercept(y, p, w=None):
    """로짓 스케일에서 보정 기울기/절편. 기울기 1, 절편 0 이 완벽."""
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y, sample_weight=w)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def net_benefit(y, p, threshold, w=None):
    """
    결정곡선분석의 net benefit.
    threshold = 개입 판단 임계확률. 가중치를 주면 모집단 기준 net benefit.
    """
    w = np.ones_like(y, dtype=float) if w is None else np.asarray(w, dtype=float)
    n = w.sum()
    flagged = p >= threshold
    tp = w[(flagged) & (y == 1)].sum()
    fp = w[(flagged) & (y == 0)].sum()
    odds = threshold / (1 - threshold)
    return (tp / n) - (fp / n) * odds


def dca_table(y, preds: dict, weights=None, thresholds=None) -> pd.DataFrame:
    thresholds = np.arange(0.02, 0.51, 0.02) if thresholds is None else thresholds
    w = np.ones_like(y, dtype=float) if weights is None else np.asarray(weights, float)
    prev = np.average(y, weights=w)
    rows = []
    for t in thresholds:
        row = {"threshold": round(float(t), 3)}
        row["treat_none"] = 0.0
        row["treat_all"] = prev - (1 - prev) * (t / (1 - t))
        for name, p in preds.items():
            row[name] = net_benefit(y, p, t, w)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(y, p, w=None, label="") -> dict:
    slope, intercept = calibration_slope_intercept(y, p, w)
    return {
        "model": label,
        "n": int(len(y)),
        "events": int(np.sum(y)),
        "event_rate": float(np.average(y, weights=w)),
        "auroc": float(roc_auc_score(y, p, sample_weight=w)),
        "auprc": float(average_precision_score(y, p, sample_weight=w)),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }


def bootstrap_ci(y, p, w=None, n_boot=500, seed=0, groups=None) -> dict:
    """
    AUROC·AUPRC 의 95% 부트스트랩 신뢰구간.
    groups(사람 ID)를 주면 사람 단위 클러스터 부트스트랩 — person-wave 의 상관을 존중합니다.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    w = np.ones_like(y, dtype=float) if w is None else np.asarray(w, float)
    aucs, aps = [], []
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([idx_by_g[g] for g in pick])
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx], sample_weight=w[idx]))
        aps.append(average_precision_score(y[idx], p[idx], sample_weight=w[idx]))
    return {
        "auroc_ci_lo": float(np.percentile(aucs, 2.5)), "auroc_ci_hi": float(np.percentile(aucs, 97.5)),
        "auprc_ci_lo": float(np.percentile(aps, 2.5)), "auprc_ci_hi": float(np.percentile(aps, 97.5)),
        "n_boot": len(aucs),
    }


def top_k_capture(y, p, k_frac, w=None) -> dict:
    """상위 k% 를 선별하면 사건의 몇 %를 잡는가 — 리뷰어 질문 2번 대비."""
    w = np.ones_like(y, dtype=float) if w is None else np.asarray(w, float)
    order = np.argsort(-p)
    cutoff = int(np.ceil(len(p) * k_frac))
    sel = order[:cutoff]
    captured = w[sel][y[sel] == 1].sum()
    total = w[y == 1].sum()
    return {
        "top_k_pct": round(k_frac * 100, 1),
        "n_selected": int(cutoff),
        "sensitivity": float(captured / total) if total else np.nan,
        "ppv": float(captured / w[sel].sum()) if w[sel].sum() else np.nan,
    }


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def make_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    # bool 은 SimpleImputer 가 거부하므로 수치로 취급합니다.
    num = X.select_dtypes(include=[np.number, "bool", "boolean"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def candidate_models(X: pd.DataFrame, seed: int) -> dict:
    """로지스틱을 동등한 조건에서 성실하게 튜닝한 baseline 으로 세웁니다."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier

    return {
        "logistic": (
            Pipeline([("prep", make_preprocessor(X, scale=True)),
                      ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000))]),  # lbfgs: 절편 무벌점
            {"clf__C": np.logspace(-3, 2, 30)},
        ),
        "random_forest": (
            Pipeline([("prep", make_preprocessor(X, scale=False)),
                      ("clf", RandomForestClassifier(random_state=seed, n_jobs=-1))]),
            {
                "clf__n_estimators": [300, 600],
                "clf__max_depth": [None, 6, 10, 16],
                "clf__min_samples_leaf": [1, 5, 20, 50],
                "clf__max_features": ["sqrt", 0.3, 0.6],
            },
        ),
        "lightgbm": (
            Pipeline([("prep", make_preprocessor(X, scale=False)),
                      ("clf", LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1))]),
            {
                "clf__n_estimators": [200, 400, 800],
                "clf__learning_rate": [0.01, 0.03, 0.1],
                "clf__num_leaves": [15, 31, 63],
                "clf__min_child_samples": [10, 30, 80],
                "clf__subsample": [0.7, 1.0],
                "clf__colsample_bytree": [0.6, 0.8, 1.0],
                "clf__reg_lambda": [0, 1, 10],
            },
        ),
    }


def tune(name, pipe, grid, X, y, groups, seed, n_iter, folds) -> Pipeline:
    cv = GroupKFold(n_splits=folds)
    search = RandomizedSearchCV(
        pipe, grid, n_iter=n_iter, scoring="average_precision",
        cv=cv.split(X, y, groups=groups), random_state=seed, n_jobs=-1, refit=True,
    )
    search.fit(X, y)
    print(f"    {name}: 내적 AUPRC {search.best_score_:.4f}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def shap_frames(model: Pipeline, X: pd.DataFrame, subgroup: pd.Series | None, waves: pd.Series):
    import shap

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    Xt = prep.transform(X)
    names = list(prep.get_feature_names_out())
    Xt = pd.DataFrame(Xt, columns=names, index=X.index)

    explainer = shap.TreeExplainer(clf)
    vals = explainer.shap_values(Xt)
    if isinstance(vals, list):           # 이진분류에서 리스트로 오는 버전 대응
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    sv = pd.DataFrame(vals, columns=names, index=X.index)

    glob = (
        sv.abs().mean().sort_values(ascending=False)
        .rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
    )

    by_group = pd.DataFrame()
    if subgroup is not None:
        parts = []
        for key, idx in subgroup.groupby(subgroup).groups.items():
            s = sv.loc[idx].abs().mean().sort_values(ascending=False)
            parts.append(
                pd.DataFrame({"subgroup": key, "feature": s.index, "mean_abs_shap": s.values,
                              "rank": np.arange(1, len(s) + 1), "n": len(idx)})
            )
        by_group = pd.concat(parts, ignore_index=True)

    by_wave = pd.DataFrame()
    parts = []
    for key, idx in waves.groupby(waves).groups.items():
        s = sv.loc[idx].abs().mean().sort_values(ascending=False)
        parts.append(
            pd.DataFrame({"wave": key, "feature": s.index, "mean_abs_shap": s.values,
                          "rank": np.arange(1, len(s) + 1), "n": len(idx)})
        )
    if parts:
        by_wave = pd.concat(parts, ignore_index=True)

    return glob, by_group, by_wave, sv


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("./results"))
    ap.add_argument("--train-waves", nargs="*", default=None)
    ap.add_argument("--test-waves", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--primary", choices=["logistic", "random_forest", "lightgbm"], default=None,
                    help="주모형 사전지정. 안 주면 검증셋 가중 AUPRC 최대 모형을 쓰지만, "
                         "사전지정이 보고 기준상 더 안전합니다.")
    ap.add_argument("--n-boot", type=int, default=500, help="부트스트랩 반복 (0이면 생략)")
    ap.add_argument("--person-disjoint", action="store_true",
                    help="부록용: 훈련 차수에 등장한 사람을 검증에서 제외한 완전분리 검증도 실행")
    ap.add_argument("--min-subgroup-n", type=int, default=100,
                    help="이 미만 하위군은 SHAP 표에 n 과 함께 탐색적으로만 표시")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.data)
    for col in (ID_COL, WAVE_COL, TARGET):
        if col not in df.columns:
            raise SystemExit(f"[에러] 필수 컬럼 '{col}' 이 없습니다.")
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)

    waves = sorted(df[WAVE_COL].astype(str).unique())
    if args.train_waves:
        tr_w, te_w = [str(w) for w in args.train_waves], [str(w) for w in args.test_waves or []]
        if not te_w:
            te_w = [w for w in waves if w not in tr_w]
    else:
        cut = max(1, int(len(waves) * 0.6))
        tr_w, te_w = waves[:cut], waves[cut:]
    print(f"차수 분할  훈련={tr_w}  검증(시간적)={te_w}")

    ws = df[WAVE_COL].astype(str)
    train, test = df[ws.isin(tr_w)].copy(), df[ws.isin(te_w)].copy()
    if train.empty or test.empty:
        raise SystemExit("[에러] 훈련 또는 검증 표본이 비었습니다. --train-waves 를 확인하십시오.")

    # 사람 누출 방지: 훈련에 등장한 사람은 검증에서 제외할지 여부.
    # 시간적 검증에서는 같은 사람의 후기 관측을 쓰는 것이 설계상 정당하므로 유지하되,
    # 부록에 '사람 완전분리' 결과를 함께 보고합니다.
    overlap = len(set(train[ID_COL]) & set(test[ID_COL]))
    print(f"  훈련·검증 공통 인물 {overlap:,}명 (시간적 검증에서는 정상. 부록에 완전분리 결과 병기)")

    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    # bool/boolean -> float (결측 보존). 전처리기가 수치로 다룰 수 있게 합니다.
    for c in feature_cols:
        if str(df[c].dtype) in ("bool", "boolean"):
            train[c] = train[c].astype("float")
            test[c] = test[c].astype("float")

    Xtr, ytr = train[feature_cols], train[TARGET].to_numpy()
    Xte, yte = test[feature_cols], test[TARGET].to_numpy()
    wte = test[WEIGHT_COL].to_numpy() if WEIGHT_COL in test.columns else None
    print(f"  훈련 {len(Xtr):,}행 · 검증 {len(Xte):,}행 · 예측변수 {len(feature_cols)}개")
    print(f"  사건률  훈련 {ytr.mean()*100:.2f}%  검증 {yte.mean()*100:.2f}%")

    print("\n[1/4] 초기 차수 안에서 튜닝 (사람 단위 GroupKFold)")
    fitted, preds = {}, {}
    for name, (pipe, grid) in candidate_models(Xtr, args.seed).items():
        fitted[name] = tune(name, pipe, grid, Xtr, ytr, train[ID_COL], args.seed, args.n_iter, args.folds)
        preds[name] = fitted[name].predict_proba(Xte)[:, 1]

    # 튜닝된 하이퍼파라미터를 모형별로 저장 — 민감도·재현 런에서 그대로 고정해 씁니다.
    best = {name: {k: (v.item() if hasattr(v, "item") else v)
                   for k, v in fitted[name].get_params().items() if k in grid}
            for name, (_, grid) in candidate_models(Xtr, args.seed).items()}
    (args.outdir / "fixed_params.json").write_text(json.dumps(best, indent=2, ensure_ascii=False))
    pd.DataFrame(preds).assign(**{ID_COL: test[ID_COL].values, WAVE_COL: test[WAVE_COL].values,
                                  WEIGHT_COL: wte, "y": yte}).to_csv(args.outdir / "predictions_val.csv", index=False)

    print("\n[2/4] 시간적 검증 — 단 한 번 평가")
    metrics_w = [evaluate(yte, p, wte, f"{n} (가중평가)") for n, p in preds.items()]
    metrics_u = [evaluate(yte, p, None, f"{n} (무가중)") for n, p in preds.items()]
    metrics = pd.DataFrame(metrics_w + metrics_u)
    print(metrics[["model", "auroc", "auprc", "brier", "cal_slope", "cal_intercept"]].to_string(index=False))

    if args.primary:
        primary = args.primary
        print(f"\n  주모형(사전지정): {primary}")
    else:
        primary = max(preds, key=lambda n: average_precision_score(yte, preds[n], sample_weight=wte))
        print(f"\n  주모형(규칙 = 가중 AUPRC 최대): {primary}")
        print("  [권고] TRIPOD+AI 기준으로는 주모형을 검증셋 성적으로 고르지 않는 편이 안전합니다.")
        print("         --primary lightgbm 처럼 사전지정하고, 나머지는 비교모형으로 보고하십시오.")

    ci_rows = []
    if args.n_boot > 0:
        print(f"\n  사람 단위 클러스터 부트스트랩 {args.n_boot}회 (95% CI)")
        for n, p in preds.items():
            ci = bootstrap_ci(yte, p, wte, n_boot=args.n_boot, seed=args.seed,
                              groups=test[ID_COL].to_numpy())
            ci_rows.append({"model": n, **ci})
            print(f"    {n:14s} AUROC [{ci['auroc_ci_lo']:.3f}, {ci['auroc_ci_hi']:.3f}]"
                  f"  AUPRC [{ci['auprc_ci_lo']:.3f}, {ci['auprc_ci_hi']:.3f}]")
    ci_df = pd.DataFrame(ci_rows)

    disjoint_df = pd.DataFrame()
    if args.person_disjoint:
        seen = set(train[ID_COL])
        mask = ~test[ID_COL].isin(seen).to_numpy()
        n_dis = int(mask.sum()); ev_dis = int(yte[mask].sum())
        print(f"\n  [부록] 사람 완전분리 검증: 검증셋 중 신규 인물 {n_dis:,}행 · 사건 {ev_dis}")
        if ev_dis >= 10:
            rows = []
            for n, p in preds.items():
                r = evaluate(yte[mask], p[mask], None if wte is None else wte[mask], f"{n} (완전분리·가중)")
                rows.append(r)
                print(f"    {n:14s} AUROC {r['auroc']:.3f}  AUPRC {r['auprc']:.3f}")
            disjoint_df = pd.DataFrame(rows)
        else:
            print("    사건이 10건 미만이라 생략합니다.")

    print("\n[3/4] 보정 · DCA · 상위 k% 선별")
    cal_rows = []
    for name, p in preds.items():
        frac_pos, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        cal_rows.append(pd.DataFrame({"model": name, "mean_predicted": mean_pred, "observed": frac_pos}))
    calib = pd.concat(cal_rows, ignore_index=True)
    dca = dca_table(yte, preds, wte)
    capture = pd.DataFrame(
        [{"model": n, **top_k_capture(yte, p, k, wte)}
         for n, p in preds.items() for k in (0.05, 0.10, 0.20, 0.30)]
    )

    print("\n[4/4] SHAP — 전역 / 장애유형 하위군 / 웨이브별")
    sub = test[GROUP_COL] if GROUP_COL in test.columns else None
    if primary == "logistic":
        print("  주모형이 로지스틱이라 TreeExplainer 를 건너뜁니다. 계수표로 대체하십시오.")
        glob = by_group = by_wave = pd.DataFrame()
    else:
        glob, by_group, by_wave, sv = shap_frames(fitted[primary], Xte, sub, test[WAVE_COL].astype(str))
        sv.to_csv(args.outdir / "shap_values.csv", index=False, encoding="utf-8-sig")
        if not by_group.empty:
            small = by_group.groupby("subgroup")["n"].first()
            flag = small[small < args.min_subgroup_n].index.tolist()
            by_group["exploratory_only"] = by_group["subgroup"].isin(flag)
            if flag:
                print(f"  n<{args.min_subgroup_n} 하위군 {len(flag)}개는 탐색적으로 표시: {flag}")
        print(glob.head(15).to_string(index=False))

    out = args.outdir / "results.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        metrics.to_excel(xl, sheet_name="성능_시간적검증", index=False)
        if not ci_df.empty:
            ci_df.to_excel(xl, sheet_name="부트스트랩_CI", index=False)
        if not disjoint_df.empty:
            disjoint_df.to_excel(xl, sheet_name="부록_사람완전분리", index=False)
        calib.to_excel(xl, sheet_name="보정곡선", index=False)
        dca.to_excel(xl, sheet_name="DCA", index=False)
        capture.to_excel(xl, sheet_name="상위k선별", index=False)
        if not glob.empty:
            glob.to_excel(xl, sheet_name="SHAP_전역", index=False)
        if not by_group.empty:
            by_group.to_excel(xl, sheet_name="SHAP_장애유형", index=False)
        if not by_wave.empty:
            by_wave.to_excel(xl, sheet_name="SHAP_웨이브별", index=False)

    (args.outdir / "run_config.json").write_text(
        json.dumps({"train_waves": tr_w, "test_waves": te_w, "primary_model": primary,
                    "seed": args.seed, "n_features": len(feature_cols),
                    "n_train": len(Xtr), "n_test": len(Xte)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n완료 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
