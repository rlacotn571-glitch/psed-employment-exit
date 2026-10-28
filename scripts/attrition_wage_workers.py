"""Panel attrition among wage workers (manuscript Results, 'Sensitivity analyses'; Additional file 1, Table S5).

Attrition = a wage worker at index wave t (waves 1-7) who was not interviewed at wave t+1.
Compares index-wave characteristics of those lost vs retained (standardised mean differences)
after recoding "don't know / refused" codes (income 999999, tenure 999) to missing.

usage: python scripts/attrition_wage_workers.py --long "PSED_2wave_8th_03_Long Data.sav" --out results/attrition_wage_workers.xlsx
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/attrition_wage_workers.xlsx"))
    a = ap.parse_args()
    cols = ["pid", "wave", "year", "p", "emp05", "birthy", "gender", "grade02", "ca2603", "aca0403"]
    d, _ = pyreadstat.read_sav(str(a.long), usecols=cols)
    d.loc[d.ca2603 == 999999, "ca2603"] = np.nan  # don't know / refused
    d.loc[d.aca0403 == 999, "aca0403"] = np.nan
    d = d.sort_values(["pid", "wave"]).reset_index(drop=True)
    g = d.groupby("pid", sort=False)
    d["next_emp"] = g["emp05"].shift(-1)
    e = d[(d.emp05 == 1) & (d.wave <= 7)].copy()          # wage workers with a following wave
    e["lost"] = e.next_emp.isna().astype(int)             # not interviewed (status unknown) at t+1
    byw = e.groupby("wave").agg(n=("lost", "size"), lost=("lost", "sum"))
    byw["attrition_pct"] = (byw.lost / byw.n * 100).round(1)
    byw.loc["all"] = [byw.n.sum(), byw.lost.sum(), round(byw.lost.sum() / byw.n.sum() * 100, 1)]
    e["age"] = e.year - e.birthy
    e["female"] = (e.gender == 2).astype(float)
    e["severe"] = (e.grade02 == 1).astype(float)
    rows = []
    for v, lab in [("age", "Age at index wave"), ("birthy", "Birth year"), ("female", "Women (proportion)"),
                   ("severe", "Severe disability (proportion)"), ("ca2603", "Monthly income, KRW 10,000"),
                   ("aca0403", "Tenure, months")]:
        x1, x0 = e.loc[e.lost == 1, v].dropna(), e.loc[e.lost == 0, v].dropna()
        smd = (x1.mean() - x0.mean()) / np.sqrt((x1.var() + x0.var()) / 2)
        rows.append({"characteristic": lab, "lost_mean": x1.mean(), "retained_mean": x0.mean(), "smd": smd,
                     "n_lost": len(x1), "n_retained": len(x0)})
    comp = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(a.out) as xw:
        byw.to_excel(xw, sheet_name="attrition_by_wave")
        comp.to_excel(xw, sheet_name="lost_vs_retained", index=False)
    print(byw.to_string()); print(comp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
