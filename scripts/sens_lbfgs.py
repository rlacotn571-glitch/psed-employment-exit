import os
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
PIPE = Path(os.environ.get("PSED_PIPELINE", Path(__file__).resolve().parent)); OUTD = Path(os.environ.get("PSED_RESULTS", "results"))
spec = importlib.util.spec_from_file_location("m02", PIPE / "02_model.py"); m02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m02)
spec = importlib.util.spec_from_file_location("m03", PIPE / "03_sensitivity.py"); m03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m03)
C = json.load(open("logistic_lbfgs.json"))["C"]
df = pd.read_csv("analysis_v3.csv", low_memory=False); dfc = pd.read_csv("analysis_v3_C.csv", low_memory=False)
def run(label, d, trw, tew, sev=None):
    tr, te = m03.split(d, trw, tew)
    if sev is not None:
        tr, te = tr[tr.grade02 == sev].copy(), te[te.grade02 == sev].copy()
    cols = m03.features(tr); m03.features(te)
    lr = m03.fit_fixed("logistic", {"clf__C": C}, tr[cols], tr.y.to_numpy(float), seed=20260918)
    p = lr.predict_proba(te[cols])[:, 1]
    r = m02.evaluate(te.y.to_numpy(float), p, te.wt01.to_numpy(float), "logistic (lbfgs)")
    r.update(spec=label, n_train=len(tr), n_test=len(te)); return r
rows = [run("주분석 (안 B)", df, [1,2,3,4], [5,6,7]),
        run("안 C (비임금 전환 포함)", dfc, [1,2,3,4], [5,6,7]),
        run("코로나 제외 (4·5차 제거)", df, [1,2,3], [6,7]),
        run("중증만", df, [1,2,3,4], [5,6,7], sev=1),
        run("경증만", df, [1,2,3,4], [5,6,7], sev=2)]
out = pd.DataFrame(rows); print(out[["spec","n_test","events","auroc","auprc","cal_slope","cal_intercept"]].round(3).to_string())
with pd.ExcelWriter(OUTD / "sensitivity.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
    out.to_excel(xw, sheet_name="S2_logistic_lbfgs", index=False)
out.to_json("sens_lbfgs.json", orient="records", indent=1, force_ascii=False)
