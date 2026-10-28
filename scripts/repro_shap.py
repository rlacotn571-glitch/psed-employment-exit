"""Rebuild the individual-level inputs of Figs 3, 4 and S1 (not distributed because they derive from the microdata):
shap_val.csv and X_val_transformed.csv (TreeSHAP of the frozen LightGBM on the temporal validation set).
Exactly mirrors refit_predict.py (same data file content, split, feature list, frozen params, seed); the result is
accepted only if the refitted validation predictions equal predictions_val.csv."""
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
OUT = Path(os.path.join(RES, "repro")); OUT.mkdir(parents=True, exist_ok=True)
PRIV = Path(WORK); PRIV.mkdir(parents=True, exist_ok=True)  # individual-level files stay out of the shared outputs
def load(n):
    s = importlib.util.spec_from_file_location(n, PIPE / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m02, m03 = load("02_model"), load("03_sensitivity")
params = json.load(open(Path(CFG) / "fixed_params.json"))
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "analysis_v3.csv", low_memory=False)
tr, te = m03.split(df, [1, 2, 3, 4], [5, 6, 7]); cols = m03.features(tr); m03.features(te)
ytr = tr.y.to_numpy(float)
pv = pd.read_csv(R / "predictions_val.csv")
res = {}
for name in ["lightgbm", "logistic"]:
    p_ = dict(params[name])
    m = m03.fit_fixed(name, p_, tr[cols], ytr, seed=20260918)
    p = m.predict_proba(te[cols])[:, 1]
    d = np.abs(p - pv[name].to_numpy())
    res[name] = float(d.max()); print(f"{name}: max|refit - predictions_val| = {d.max():.3e}", flush=True)
    if name == "lightgbm": lgb = m
json.dump(res, open(OUT / "repro_check_predictions.json", "w"), indent=2)
if res["lightgbm"] > 1e-9:
    raise SystemExit("LightGBM refit does not reproduce predictions_val.csv — SHAP not written")
import shap
prep, clf = lgb.named_steps["prep"], lgb.named_steps["clf"]
Xt = prep.transform(te[cols]); names = list(prep.get_feature_names_out())
sv = shap.TreeExplainer(clf).shap_values(Xt)
if isinstance(sv, list): sv = sv[1]
sv = np.asarray(sv)
if sv.ndim == 3: sv = sv[:, :, 1]
pd.DataFrame(sv, columns=names).to_csv(PRIV / "shap_val.csv", index=False)
pd.DataFrame(Xt, columns=names).to_csv(PRIV / "X_val_transformed.csv", index=False)
print("shap", sv.shape)
