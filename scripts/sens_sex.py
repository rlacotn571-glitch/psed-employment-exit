import os
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
PIPE = Path(os.environ.get("PSED_PIPELINE", Path(__file__).resolve().parent)); OUTD = Path(os.environ.get("PSED_RESULTS", "results"))
spec = importlib.util.spec_from_file_location("m02", PIPE / "02_model.py"); m02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m02)
spec = importlib.util.spec_from_file_location("m03", PIPE / "03_sensitivity.py"); m03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m03)
params = json.load(open(OUTD / "fixed_params.json")); params["logistic"] = {"clf__C": json.load(open("logistic_lbfgs.json"))["C"]}
df = pd.read_csv("analysis_v3.csv", low_memory=False)
rows = []
for sex, lab in [(1, "Men only"), (2, "Women only")]:
    d = df[df.gender == sex]
    tr, te = m03.split(d, [1,2,3,4], [5,6,7]); cols = m03.features(tr); m03.features(te)
    for name in ["logistic", "lightgbm"]:
        m = m03.fit_fixed(name, params[name], tr[cols], tr.y.to_numpy(float), seed=20260918)
        p = m.predict_proba(te[cols])[:, 1]
        r = m02.evaluate(te.y.to_numpy(float), p, te.wt01.to_numpy(float), name); r.update(spec=f"Re-fitted {lab}", n_train=len(tr), events_train=int(tr.y.sum())); rows.append(r)
        print(lab, name, round(r["auroc"],3), round(r["auprc"],3), round(r["cal_slope"],2), len(tr), int(tr.y.sum()), len(te), int(te.y.sum()))
# descriptive: women vs men in validation, weighted
te = df[df.wave.isin([5,6,7])]
for sex in [1,2]:
    s = te[te.gender==sex]; w = s.wt01
    print("sex", sex, "n", len(s), "events", int(s.y.sum()), "wrate %.3f" % np.average(s.y, weights=w),
          "income mean %.0f sd %.0f" % (s.ca2603.mean(), s.ca2603.std()), "tenure mean %.0f sd %.0f" % (s.aca0403.mean(), s.aca0403.std()),
          "fixed-term %.2f" % (s.ca1401.eq(1).mean()), "designated %.2f" % (s.ca1301.eq(1).mean()), "severe %.2f" % s.grade02.eq(1).mean(),
          "age %.1f" % s.age.mean())
pd.DataFrame(rows).to_json("sens_sex.json", orient="records", indent=1)
with pd.ExcelWriter(OUTD / "sensitivity.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
    pd.DataFrame(rows).to_excel(xw, sheet_name="S6b_refit_by_sex", index=False)
