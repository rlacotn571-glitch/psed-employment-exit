"""Manuscript tables: Table 1 (weighted+unweighted), Table 2 (performance, lbfgs logistic), Table 3 (top-k)."""
import os
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(os.environ.get("PSED_WORK", "."))  # folder holding analysis_v3.csv and the model outputs
PIPE = Path(os.environ.get("PSED_PIPELINE", Path(__file__).resolve().parent))
OUTD = Path(os.environ.get("PSED_RESULTS", "results"))

spec = importlib.util.spec_from_file_location("m02", PIPE / "02_model.py"); m02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m02)
spec = importlib.util.spec_from_file_location("m03", PIPE / "03_sensitivity.py"); m03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m03)

df = pd.read_csv(ROOT / "analysis_v3.csv", low_memory=False)
tr, te = m03.split(df, [1, 2, 3, 4], [5, 6, 7])
cols = m03.features(tr); m03.features(te)
ytr, yte = tr["y"].to_numpy(float), te["y"].to_numpy(float)
wte = te["wt01"].to_numpy(float)

# ---- refit lbfgs logistic with frozen C, get predictions --------------------
C = json.load(open(ROOT / "logistic_lbfgs.json"))["C"]
lr = m03.fit_fixed("logistic", {"clf__C": C}, tr[cols], ytr, seed=0)
p_lr = lr.predict_proba(te[cols])[:, 1]
chk = m02.evaluate(yte, p_lr, wte, "logistic")
print("logistic weighted check:", {k: round(v, 3) for k, v in chk.items() if isinstance(v, float)})

# person-disjoint subset
disj = ~te["pid"].isin(tr["pid"]).to_numpy()
lr_disj = m02.evaluate(yte[disj], p_lr[disj], wte[disj], "logistic")
lr_topk = [m02.top_k_capture(yte, p_lr, k / 100, wte) | {"model": "logistic"} for k in (5, 10, 20, 30)]

# ---- Table 2 ---------------------------------------------------------------
xl = pd.ExcelFile(OUTD / "results_main.xlsx")
perf = pd.read_excel(xl, "성능_시간적검증"); ci = pd.read_excel(xl, "부트스트랩_CI"); disjs = pd.read_excel(xl, "부록_사람완전분리")
lj = json.load(open(ROOT / "logistic_lbfgs.json"))
def row(model, label, block):
    r = perf[perf.model == f"{model} ({block})"].iloc[0].to_dict()
    return r
def fmt(x, d=3): return f"{x:.{d}f}"
rows = []
names = {"logistic": "Logistic regression (L2, L-BFGS)", "random_forest": "Random forest", "lightgbm": "LightGBM (primary)"}
for m in ["logistic", "random_forest", "lightgbm"]:
    if m == "logistic":
        w, u, c = lj["weighted"], lj["unweighted"], lj["ci"]; d = lr_disj
    else:
        w = row(m, names[m], "가중평가"); u = row(m, names[m], "무가중"); c = ci[ci.model == m].iloc[0].to_dict()
        d = disjs[disjs.model.str.startswith(m)].iloc[0].to_dict()
    rows.append({"Model": names[m],
                 "AUROC (95% CI)": f"{fmt(w['auroc'])} ({fmt(c['auroc_ci_lo'])}–{fmt(c['auroc_ci_hi'])})",
                 "AUPRC (95% CI)": f"{fmt(w['auprc'])} ({fmt(c['auprc_ci_lo'])}–{fmt(c['auprc_ci_hi'])})",
                 "Brier": fmt(w["brier"]), "Calibration slope": fmt(w["cal_slope"], 2), "Calibration intercept": fmt(w["cal_intercept"], 2),
                 "AUROC, unweighted": fmt(u["auroc"]), "AUROC, person-disjoint (n=335, 54 events)": fmt(d["auroc"]),
                 "AUPRC, person-disjoint": fmt(d["auprc"])})
t2 = pd.DataFrame(rows)
t2_note = ("Temporal validation set: index waves 5–7 (2020–2022), 4,530 person-waves, 1,783 persons, 326 exits (7.2% unweighted; 8.9% weighted). "
           "Weighted metrics use the cross-sectional survey weight of the index wave; 95% CIs from 500 person-clustered bootstrap resamples. "
           f"Paired ΔAUROC LightGBM − logistic = +{lj['delta']['delta_mean']:.3f} (95% CI +{lj['delta']['ci_lo']:.3f} to +{lj['delta']['ci_hi']:.3f}).")

# ---- Table 3 ---------------------------------------------------------------
topk = pd.read_excel(xl, "상위k선별")
topk = topk[topk.model != "logistic"]
topk = pd.concat([pd.DataFrame(lr_topk)[topk.columns], topk], ignore_index=True)
t3 = topk.pivot(index="top_k_pct", columns="model", values=["sensitivity", "ppv"])
t3 = pd.DataFrame({
    "Top-k flagged (%)": t3.index.astype(int),
    "n flagged": topk.groupby("top_k_pct").n_selected.first().values,
    "Sensitivity, logistic (%)": (t3["sensitivity"]["logistic"] * 100).round(1).values,
    "PPV, logistic (%)": (t3["ppv"]["logistic"] * 100).round(1).values,
    "Sensitivity, RF (%)": (t3["sensitivity"]["random_forest"] * 100).round(1).values,
    "PPV, RF (%)": (t3["ppv"]["random_forest"] * 100).round(1).values,
    "Sensitivity, LightGBM (%)": (t3["sensitivity"]["lightgbm"] * 100).round(1).values,
    "PPV, LightGBM (%)": (t3["ppv"]["lightgbm"] * 100).round(1).values,
})

# ---- Table 1 ---------------------------------------------------------------
d = df.copy()
w = d["wt01"].to_numpy(float)
def wpct(mask, sub):
    m = mask & sub; return 100 * w[m].sum() / w[sub].sum()
def upct(mask, sub):
    m = mask & sub; return 100 * m.sum() / sub.sum()
def wmean(x, sub):
    v = d.loc[sub, x].to_numpy(float); ww = w[sub]; ok = ~np.isnan(v); return np.average(v[ok], weights=ww[ok])
def umean(x, sub): return d.loc[sub, x].mean()
exit_, stay = d["y"].eq(1).to_numpy(), d["y"].eq(0).to_numpy()
allm = np.ones(len(d), bool)
def line(label, kind, mask=None, var=None, scale=1, dec=1):
    out = {"Characteristic": label}
    for nm, sub in [("Exit (n=856)", exit_), ("Remained (n=9,924)", stay), ("All (n=10,780)", allm)]:
        if kind == "pct":
            out[f"{nm}, unweighted"] = f"{upct(mask, sub):.{dec}f}"; out[f"{nm}, weighted"] = f"{wpct(mask, sub):.{dec}f}"
        else:
            out[f"{nm}, unweighted"] = f"{umean(var, sub)/scale:.{dec}f}"; out[f"{nm}, weighted"] = f"{wmean(var, sub)/scale:.{dec}f}"
    return out
# check available columns
print([c for c in ["gender", "a0101", "sex", "grade02", "a0710", "ca2603", "aca0403", "ca1401", "ca1203", "age", "ca1301"] if c in d.columns])
sexcol = next(c for c in ["gender", "a0101", "sex"] if c in d.columns)
print("sex values", d[sexcol].value_counts().to_dict())
t1 = [
    line("Age, years (mean)", "mean", var="age", dec=1),
    line("Women (%)", "pct", mask=d[sexcol].eq(2).to_numpy()),
    line("Severe disability (%)", "pct", mask=d["grade02"].eq(1).to_numpy()),
    line("Physical disability (%)", "pct", mask=d["a0710"].eq(1).to_numpy()),
    line("Brain lesion (%)", "pct", mask=d["a0710"].eq(2).to_numpy()),
    line("Visual (%)", "pct", mask=d["a0710"].eq(3).to_numpy()),
    line("Hearing (%)", "pct", mask=d["a0710"].eq(4).to_numpy()),
    line("Intellectual (%)", "pct", mask=d["a0710"].eq(6).to_numpy()),
    line("Other disability types (%)", "pct", mask=~d["a0710"].isin([1, 2, 3, 4, 6]).to_numpy()),
    line("Monthly income, KRW million (mean)", "mean", var="ca2603", scale=100, dec=2),
    line("Tenure in current job, months (mean)", "mean", var="aca0403", dec=0),
    line("Fixed-term contract (%)", "pct", mask=d["ca1401"].eq(1).to_numpy()),
    line("Job designated for people with disabilities (%)", "pct", mask=d["ca1301"].eq(1).to_numpy()),
]
t1 = pd.DataFrame(t1)
print(t1.to_string())
t1_note = ("Person-waves in the analytic sample (index waves 1–7). Weighted columns use the cross-sectional survey weight of the index wave. "
           "Disability-type codes follow the PSED classification; 'other' pools the ten remaining statutory types. "
           "Verify the code→label mapping for sex, contract and designated-job items against the codebook before submission.")

with pd.ExcelWriter(OUTD / "manuscript_tables.xlsx", engine="openpyxl") as xw:
    t1.to_excel(xw, sheet_name="Table1", index=False); pd.DataFrame({"note": [t1_note]}).to_excel(xw, sheet_name="Table1_note", index=False)
    t2.to_excel(xw, sheet_name="Table2", index=False); pd.DataFrame({"note": [t2_note]}).to_excel(xw, sheet_name="Table2_note", index=False)
    t3.to_excel(xw, sheet_name="Table3", index=False)
    pd.DataFrame([lr_disj]).to_excel(xw, sheet_name="logistic_lbfgs_disjoint", index=False)
    pd.DataFrame(lr_topk).to_excel(xw, sheet_name="logistic_lbfgs_topk", index=False)
print(t2.to_string()); print(t3.to_string()); print(lr_disj)
json.dump({"logistic_lbfgs_disjoint": lr_disj, "logistic_lbfgs_topk": lr_topk}, open(ROOT / "logistic_lbfgs_extra.json", "w"), indent=2)
