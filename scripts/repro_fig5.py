"""Reproduce the PSED temporal-validation predictions of the reduced (7-predictor) models used in Fig. 5a,
exactly as fig5_external.py / 04_external_validation.py fit them; accept only if the metrics equal those stored in
external_validation_kodld.xlsx (sheet A_tenure_w1-3). KoDLD predictions are read from
$PSED_RESULTS/predictions_kodld_A_tenure.csv (verified against the same sheet)."""
import os
RES = os.environ.get("PSED_RESULTS", "results")        # result tables and saved predictions
CFG = os.environ.get("PSED_CONFIG", "config")          # fixed hyperparameters, KoDLD mapping
SCR = os.environ.get("PSED_SCRIPTS", "scripts")        # pipeline scripts (02_model.py)
WORK = os.environ.get("PSED_WORK", "work")             # individual-level derived files (not distributed)
FIGOUT = os.environ.get("PSED_FIGURES", "figures_out") # figure output folder
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd
PIPE = Path(SCR); R = Path(RES)
OUT = Path(os.path.join(RES, "repro")); PRIV = Path(WORK)
def load(n):
    s = importlib.util.spec_from_file_location(n, PIPE / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m02, m03 = load("02_model"), load("03_sensitivity")
params = json.load(open(Path(CFG) / "fixed_params.json")); params["logistic"] = {"clf__C": 0.0014873521072935117}
mp = json.load(open(Path(CFG) / "kodld_mapping_with_tenure.json"))
psed = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "analysis_v3.csv", low_memory=False)
psed = psed[psed.y.notna()]
common = list(mp["predictors"]) + ["age"]
for c in common: psed[c] = psed[c].astype(float)
tr, te = m03.split(psed, [1, 2, 3, 4], [5, 6, 7])
A = pd.read_excel(R / "external_validation_kodld.xlsx", "A_tenure_w1-3")
out = te[["pid", "wave", "wt01", "y"]].copy(); chk = {}
for name in ["logistic", "lightgbm"]:
    p = m03.fit_fixed(name, params[name], tr[common], tr.y.to_numpy(float), seed=20260918).predict_proba(te[common])[:, 1]
    out[name] = p
    r = m02.evaluate(te.y.to_numpy(float), p, te.wt01.to_numpy(float))
    ref = A[A.model.str.startswith(f"{name} | PSED temporal")].iloc[0]
    chk[name] = {k: (float(r[k]), float(ref[k]), float(abs(r[k] - ref[k]))) for k in ["auroc", "auprc", "brier", "cal_slope", "cal_intercept"]}
    print(name, {k: f"{v[0]:.6f} vs {v[1]:.6f}" for k, v in chk[name].items()})
k = pd.read_csv(R / "predictions_kodld_A_tenure.csv")
for name in ["logistic", "lightgbm"]:
    r = m02.evaluate(k.y.to_numpy(float), k[name].to_numpy(float), k.wt01.to_numpy(float))
    ref = A[A.model == f"{name} | KoDLD external"].iloc[0]
    chk[f"kodld_{name}"] = {kk: (float(r[kk]), float(ref[kk]), float(abs(r[kk] - ref[kk]))) for kk in ["auroc", "auprc", "brier", "cal_slope", "cal_intercept"]}
    print("KoDLD", name, {kk: f"{v[0]:.6f} vs {v[1]:.6f}" for kk, v in chk[f"kodld_{name}"].items()})
worst = max(v[2] for d in chk.values() for v in d.values()); print("worst |diff| vs stored metrics:", worst)
json.dump(chk, open(OUT / "repro_check_fig5.json", "w"), indent=2)
if worst > 1e-6: raise SystemExit("not reproduced")
out.to_csv(PRIV / "predictions_psed_reduced7.csv", index=False)
