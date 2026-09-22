"""Refit the three frozen models on waves 1-4, predict waves 5-7, save predictions + SHAP for figures."""
import os
import importlib.util, json, time
from pathlib import Path
import numpy as np, pandas as pd
PIPE = Path(os.environ.get("PSED_PIPELINE", Path(__file__).resolve().parent)); OUTD = Path(os.environ.get("PSED_RESULTS", "results"))
spec = importlib.util.spec_from_file_location("m02", PIPE / "02_model.py"); m02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m02)
spec = importlib.util.spec_from_file_location("m03", PIPE / "03_sensitivity.py"); m03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m03)
params = json.load(open(OUTD / "fixed_params.json"))
params["logistic"] = {"clf__C": json.load(open("logistic_lbfgs.json"))["C"]}
df = pd.read_csv("analysis_v3.csv", low_memory=False)
tr, te = m03.split(df, [1,2,3,4], [5,6,7]); cols = m03.features(tr); m03.features(te)
ytr, yte, wte = tr.y.to_numpy(float), te.y.to_numpy(float), te.wt01.to_numpy(float)
pred = te[["pid","wave","wt01","a0710","grade02","y"]].copy()
models = {}
for name in ["logistic", "random_forest", "lightgbm"]:
    t = time.time()
    if name == "random_forest":
        pipe, grid = m02.candidate_models(tr[cols], 20260918)[name]
        m = m02.tune(name, pipe, grid, tr[cols], ytr, tr["pid"], 20260918, 12, 5)
        params[name] = {k: (v.item() if hasattr(v, "item") else v) for k, v in m.get_params().items() if k in grid}
        print("RF params", params[name], flush=True)
    else:
        m = m03.fit_fixed(name, params[name], tr[cols], ytr, seed=20260918)
    p = m.predict_proba(te[cols])[:, 1]; pred[name] = p; models[name] = m
    r = m02.evaluate(yte, p, wte, name); print(name, {k: round(v,3) for k,v in r.items() if isinstance(v,float)}, f"{time.time()-t:.0f}s", flush=True)
pred.to_csv("predictions_val.csv", index=False)
json.dump(params, open(OUTD / "fixed_params.json", "w"), indent=2)
# SHAP for LightGBM on validation set
import shap
lgb = models["lightgbm"]; prep = lgb.named_steps["prep"]; clf = lgb.named_steps["clf"]
Xt = prep.transform(te[cols]); names = list(prep.get_feature_names_out())
ex = shap.TreeExplainer(clf); sv = ex.shap_values(Xt)
if isinstance(sv, list): sv = sv[1]
pd.DataFrame(sv, columns=names).to_csv("shap_val.csv", index=False)
pd.DataFrame(Xt, columns=names).to_csv("X_val_transformed.csv", index=False)
print("shap done", sv.shape, flush=True)
