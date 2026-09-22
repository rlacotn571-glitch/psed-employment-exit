"""S9 ablation: how much of the model's value is carried by the economic core / simple rules."""
import os
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
PIPE = Path(os.environ.get("PSED_PIPELINE", Path(__file__).resolve().parent)); OUTD = Path(os.environ.get("PSED_RESULTS", "results"))
spec = importlib.util.spec_from_file_location("m02", PIPE / "02_model.py"); m02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m02)
spec = importlib.util.spec_from_file_location("m03", PIPE / "03_sensitivity.py"); m03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m03)
params = json.load(open(OUTD / "fixed_params.json")); params["logistic"] = {"clf__C": json.load(open("logistic_lbfgs.json"))["C"]}
SEED = 20260918
df = pd.read_csv("analysis_v3.csv", low_memory=False)
tr, te = m03.split(df, [1,2,3,4], [5,6,7]); allcols = m03.features(tr); m03.features(te)
ytr, yte, wte = tr.y.to_numpy(float), te.y.to_numpy(float), te.wt01.to_numpy(float)
groups = te.pid.to_numpy()
psych = [c for c in allcols if c.startswith("f06") or c.startswith("f07")]
health = [c for c in allcols if c.startswith("g0") or c == "ca2508"]
discr = [c for c in allcols if c.startswith("ca41") or c == "ca4001"]
econ4 = ["ca2603", "aca0403", "ca1203", "ca1401"]
econ_dom = [c for c in allcols if c.startswith("ca2") or c.startswith("ca1") or c in ("aca0403", "emptype02", "emppme", "jobtype", "jobctotal")]
econ_dom = [c for c in econ_dom if c not in health]
demo_dis = ["gender", "birthy", "age", "area", "aarea", "a0710", "grade02", "type04", "type06"]
specs = {
    "All 106 predictors": allcols,
    "Economic core only (income, tenure, workplace size, fixed-term)": econ4,
    "Economic core + sex, age, disability type, severity": econ4 + ["gender", "age", "birthy", "a0710", "grade02"],
    "Job/employment domain only (%d)" % len(econ_dom): econ_dom,
    "All minus psychological scales (22)": [c for c in allcols if c not in psych],
    "All minus psychological, health and discrimination items": [c for c in allcols if c not in psych + health + discr],
    "Demographics and disability only": demo_dis,
}
# simple rule: rank by -income then -tenure (no fitting)
rows = []
for label, cols in specs.items():
    for name in ["logistic", "lightgbm"]:
        m = m03.fit_fixed(name, params[name], tr[cols], ytr, seed=SEED)
        p = m.predict_proba(te[cols])[:, 1]
        r = m02.evaluate(yte, p, wte, name)
        ci = m02.bootstrap_ci(yte, p, wte, n_boot=300, seed=SEED, groups=groups)
        r.update(spec=label, n_predictors=len(cols), auroc_ci_lo=ci["auroc_ci_lo"], auroc_ci_hi=ci["auroc_ci_hi"])
        r.update(m02.top_k_capture(yte, p, 0.10, wte)); rows.append(r)
        print(f"{label[:60]:60s} {name:9s} AUROC {r['auroc']:.3f} ({ci['auroc_ci_lo']:.3f}-{ci['auroc_ci_hi']:.3f}) AUPRC {r['auprc']:.3f} top10 sens {r['sensitivity']:.3f} ppv {r['ppv']:.3f}", flush=True)
# rule-based: income below median AND tenure below median -> flagged; and a rank score
inc, ten = te.ca2603.to_numpy(float), te.aca0403.to_numpy(float)
score = -(pd.Series(inc).rank(pct=True).fillna(0.5).to_numpy() + pd.Series(ten).rank(pct=True).fillna(0.5).to_numpy())
r = m02.evaluate(yte, (score - score.min()) / (score.max() - score.min()), wte, "rule: income+tenure rank")
ci = m02.bootstrap_ci(yte, (score - score.min()) / (score.max() - score.min()), wte, n_boot=300, seed=SEED, groups=groups)
r.update(spec="Rule: rank sum of income and tenure (no model)", n_predictors=2, auroc_ci_lo=ci["auroc_ci_lo"], auroc_ci_hi=ci["auroc_ci_hi"])
r.update(m02.top_k_capture(yte, (score - score.min()) / (score.max() - score.min()), 0.10, wte)); rows.append(r)
print("rule AUROC %.3f (%.3f-%.3f) top10 sens %.3f ppv %.3f" % (r["auroc"], ci["auroc_ci_lo"], ci["auroc_ci_hi"], r["sensitivity"], r["ppv"]))
out = pd.DataFrame(rows)
with pd.ExcelWriter(OUTD / "sensitivity.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
    out.to_excel(xw, sheet_name="S9_ablation", index=False)
out.to_json("ablation.json", orient="records", indent=1)
