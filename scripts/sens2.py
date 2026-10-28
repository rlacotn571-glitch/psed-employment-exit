"""Extra robustness: (S6) fairness by sex / age group / severity on the frozen model predictions,
(S7) multiple imputation (5 imputations, IterativeImputer) instead of median imputation,
(S8) excluding index-wave age >= 60 (approximate removal of regular retirement as competing event).
Logistic = L-BFGS refit; LightGBM = frozen params."""
import os
import importlib.util, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline

PIPE = Path(os.environ.get("PSED_PIPELINE", Path(__file__).resolve().parent)); OUTD = Path(os.environ.get("PSED_RESULTS", "results"))
spec = importlib.util.spec_from_file_location("m02", PIPE / "02_model.py"); m02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m02)
spec = importlib.util.spec_from_file_location("m03", PIPE / "03_sensitivity.py"); m03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m03)
params = json.load(open(OUTD / "fixed_params.json"))
params["logistic"] = {"clf__C": json.load(open("logistic_lbfgs.json"))["C"]}
SEED = 20260918
df = pd.read_csv("analysis_v3.csv", low_memory=False)
pred = pd.read_csv("predictions_val.csv")
out = {}

# ---------------- S6 fairness on frozen predictions ---------------------------
te_full = df[df.wave.isin([5, 6, 7])].reset_index(drop=True)
assert len(te_full) == len(pred) and (te_full.pid.values == pred.pid.values).all()
g = pd.DataFrame({"sex": np.where(te_full.gender == 2, "Women", "Men"),
                  "age": np.where(te_full.age < 45, "Age <45", "Age 45+"),
                  "sev": np.where(te_full.grade02 == 1, "Severe", "Mild/other")})
rows = []
for col in ["sex", "age", "sev"]:
    for lvl in sorted(g[col].unique()):
        m = (g[col] == lvl).to_numpy()
        for model in ["lightgbm", "logistic"]:
            r = m02.evaluate(pred.y.to_numpy(float)[m], pred[model].to_numpy()[m], pred.wt01.to_numpy(float)[m], model)
            ci = m02.bootstrap_ci(pred.y.to_numpy(float)[m], pred[model].to_numpy()[m], pred.wt01.to_numpy(float)[m],
                                  n_boot=500, seed=SEED, groups=pred.pid.to_numpy()[m])
            r.update(subgroup=lvl, **{k: v for k, v in ci.items() if k.startswith("auroc")})
            rows.append(r)
s6 = pd.DataFrame(rows); print(s6[["subgroup", "model", "n", "events", "auroc", "auroc_ci_lo", "auroc_ci_hi", "cal_slope", "cal_intercept"]].round(3).to_string())

# ---------------- helpers ---------------------------------------------------
def run(label, d, trw, tew, imputer=None, n_imp=1):
    tr, te = m03.split(d, trw, tew); cols = m03.features(tr); m03.features(te)
    ytr, yte, wte = tr.y.to_numpy(float), te.y.to_numpy(float), te.wt01.to_numpy(float)
    res = []
    sets = []
    for k in range(n_imp):
        if imputer is None:
            sets.append((tr[cols], te[cols]))
        else:
            t0 = time.time()
            imp = IterativeImputer(max_iter=5, n_nearest_features=30, random_state=SEED + k, sample_posterior=True)
            sets.append((pd.DataFrame(imp.fit_transform(tr[cols]), columns=cols), pd.DataFrame(imp.transform(te[cols]), columns=cols)))
            print(f"  imputation {k+1} done {time.time()-t0:.0f}s", flush=True)
    for name in ["logistic", "lightgbm"]:
        ps = []
        for Xtr_i, Xte_i in sets:
            m = m03.fit_fixed(name, params[name], Xtr_i, ytr, seed=SEED)
            ps.append(m.predict_proba(Xte_i)[:, 1])
        p = np.mean(ps, axis=0)
        r = m02.evaluate(yte, p, wte, name); r.update(spec=label, n_train=len(tr)); res.append(r)
        print(f"  {label} {name}: AUROC {r['auroc']:.3f} AUPRC {r['auprc']:.3f} slope {r['cal_slope']:.2f}", flush=True)
    return res

# ---------------- S8 age < 60 at index ---------------------------------------
t = time.time()
s8 = run("Index age <60 (n_val, events below)", df[df.age < 60], [1,2,3,4], [5,6,7])
print("S8 done", f"{time.time()-t:.0f}s")
# ---------------- S7 multiple imputation --------------------------------------
t = time.time()
miss = df[m03.features(df.copy())].isna().mean()
print("missingness: mean %.3f, max %.3f (%s), n>5%%: %d" % (miss.mean(), miss.max(), miss.idxmax(), (miss > 0.05).sum()))
s7 = run("Multiple imputation (5 imputations, pooled predictions)", df, [1,2,3,4], [5,6,7], imputer=True, n_imp=5)
print("S7 done", f"{time.time()-t:.0f}s")

with pd.ExcelWriter(OUTD / "sensitivity.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
    s6.to_excel(xw, sheet_name="S6_fairness_subgroups", index=False)
    pd.DataFrame(s7).to_excel(xw, sheet_name="S7_multiple_imputation", index=False)
    pd.DataFrame(s8).to_excel(xw, sheet_name="S8_age_under60", index=False)
    pd.DataFrame({"feature": miss.index, "missing_frac": miss.values}).to_excel(xw, sheet_name="missingness", index=False)
json.dump({"s6": s6.to_dict(orient="records"), "s7": s7, "s8": s8}, open("sens2.json", "w"), indent=1, default=float)
print("written")
