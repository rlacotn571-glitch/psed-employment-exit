"""Fig 5 — External validation in KoDLD. (a) ROC, (b) calibration (deciles, person-clustered CI), (c) AUROC forest.
Run from a folder containing analysis_v3.csv and kodld_ready/kodld_long.csv (see 05_kodld_prepare.py)."""
import os
import json, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve
import figstyle as F
from figstyle import plt, OK, MODEL_COL, MODEL_LAB, MODEL_MK, DOUBLE, MM, save, panel_label, ref_line, INK2, MUTED, SHADE

HERE = Path(os.environ.get("PSED_PIPELINE", Path(__file__).resolve().parent.parent / "scripts"))
def _load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m02, m03 = _load("02_model"), _load("03_sensitivity")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
R = Path(os.environ.get("PSED_RESULTS", "results"))
params = json.load(open(R / "fixed_params.json")); params["logistic"] = {"clf__C": 0.0014873521072935117}
rng = np.random.default_rng(20260918)

def calib_deciles(y, p, w, groups, nb=10, n_boot=300):
    edges = np.quantile(p, np.linspace(0, 1, nb + 1)); edges[-1] += 1e-9
    b = np.clip(np.digitize(p, edges[1:-1]), 0, nb - 1)
    mp = np.array([np.average(p[b == k], weights=w[b == k]) for k in range(nb)])
    ob = np.array([np.average(y[b == k], weights=w[b == k]) for k in range(nb)])
    uniq = np.unique(groups); idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = np.full((n_boot, nb), np.nan)
    for i in range(n_boot):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        for k in range(nb):
            m = idx[b[idx] == k]
            if len(m): boots[i, k] = np.average(y[m], weights=w[m])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0); return mp, ob, lo, hi

# ---- data
psed = pd.read_csv("analysis_v3.csv", low_memory=False); psed = psed[psed.y.notna()]
mp = json.load(open("kodld_ready/kodld_mapping_with_tenure.json"))
ko = m03 and __import__("importlib").util  # placeholder to keep linter quiet
ext = _load("04_external_validation")
ko = ext.build_kodld(ext.read_any(Path("kodld_ready/kodld_long.csv")), mp); ko = ko[ko.wave.isin(mp["index_waves"])]
common = list(mp["predictors"]) + ["age"]
for c in common: psed[c] = psed[c].astype(float); ko[c] = ko[c].astype(float)
tr, te = m03.split(psed, [1, 2, 3, 4], [5, 6, 7])
P = {}
for name in ["logistic", "lightgbm"]:
    P[("psed", name)] = m03.fit_fixed(name, params[name], tr[common], tr.y.to_numpy(float), seed=20260918).predict_proba(te[common])[:, 1]
    P[("kodld", name)] = m03.fit_fixed(name, params[name], psed[common], psed.y.to_numpy(float), seed=20260918).predict_proba(ko[common])[:, 1]
Y = {"psed": (te.y.to_numpy(float), te.wt01.to_numpy(float), te.pid.to_numpy()), "kodld": (ko.y.to_numpy(float), ko.wt01.to_numpy(float), ko.pid.to_numpy())}

fig = plt.figure(figsize=(DOUBLE, 180 * MM)); gs = fig.add_gridspec(2, 12, height_ratios=[1, 1.0], hspace=0.5, wspace=0.0)
# top panels span the full width; the forest panel starts at column 3 so its long row labels sit in the space under panel a
ax_roc, ax_cal, ax_for = fig.add_subplot(gs[0, 0:5]), fig.add_subplot(gs[0, 7:12]), fig.add_subplot(gs[1, 3:12])
# (a) ROC
for coh, ls, lab in [("psed", "--", "PSED temporal (waves 5–7)"), ("kodld", "-", "KoDLD external (waves 1–3)")]:
    y, w, _ = Y[coh]
    for name in ["lightgbm", "logistic"]:
        fpr, tpr, _ = roc_curve(y, P[(coh, name)], sample_weight=w)
        auc = m02.evaluate(y, P[(coh, name)], w)["auroc"]
        ax_roc.plot(fpr, tpr, ls, color=MODEL_COL[name], label=f"{'LightGBM' if name=='lightgbm' else 'Logistic'}, {'PSED' if coh=='psed' else 'KoDLD'} {auc:.3f}")
ref_line(ax_roc, [0, 1], [0, 1])
ax_roc.set_xlabel("1 − specificity"); ax_roc.set_ylabel("Sensitivity"); ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
ax_roc.legend(loc="lower right", fontsize=9, title="AUROC", title_fontsize=9, handlelength=1.7, labelspacing=0.35); panel_label(ax_roc, "a")
# (b) calibration in KoDLD
y, w, g = Y["kodld"]
for name in ["lightgbm", "logistic"]:
    mpv, ob, lo, hi = calib_deciles(y, P[("kodld", name)], w, g)
    ax_cal.errorbar(mpv, ob, yerr=[ob - lo, hi - ob], fmt=MODEL_MK[name] + "-", color=MODEL_COL[name], ms=6, lw=1.8, capsize=2.5, elinewidth=1.0, mec="white", mew=0.9, label=MODEL_LAB[name], zorder=3)
ref_line(ax_cal, [0, 0.5], [0, 0.5])
ax_cal.axhline(np.average(y, weights=w), color=MUTED, lw=1.1, ls=(0, (1.5, 2)))
ax_cal.text(0.49, np.average(y, weights=w) + 0.012, "Observed rate", fontsize=9, color=INK2, ha="right")
ax_cal.set_xlabel("Predicted 1-year exit probability"); ax_cal.set_ylabel("Observed proportion (weighted)")
ax_cal.set_xlim(0, 0.5); ax_cal.set_ylim(0, 0.5); ax_cal.legend(loc="lower right", fontsize=9.5); panel_label(ax_cal, "b")
# (c) forest of AUROC (from external_validation_kodld.xlsx)
xl = pd.ExcelFile(R / "external_validation_kodld.xlsx"); S = pd.read_excel(xl, "summary"); ceil = pd.read_excel(xl, "kodld_internal_refit_ceiling")
def row(sheet, pat): 
    r = S[(S.design == sheet) & S.model.str.contains(pat, regex=False)].iloc[0]; return r.auroc, r.auroc_ci_lo, r.auroc_ci_hi
A = pd.read_excel(xl, "A_tenure_w1-3")
rows = []
for name in ["lightgbm", "logistic"]:
    r = A[A.model.str.contains(f"{name} | PSED", regex=False)].iloc[0]; rows.append((f"PSED temporal (7 predictors)", name, r.auroc, r.auroc_ci_lo, r.auroc_ci_hi))
    rows.append(("KoDLD waves 1–3, with tenure", name) + row("A_tenure_w1-3", f"{name} | KoDLD external"))
    rows.append(("KoDLD waves 1–3, age ≤64", name) + row("A_tenure_age64", f"{name} | KoDLD external"))
    rows.append(("KoDLD waves 1–5, no tenure", name) + row("B_allwaves_w1-5", f"{name} | KoDLD external"))
    rows.append(("KoDLD men", name) + row("A_tenure_w1-3", f"{name} | KoDLD men"))
    rows.append(("KoDLD women", name) + row("A_tenure_w1-3", f"{name} | KoDLD women"))
    rows.append(("KoDLD severe", name) + row("A_tenure_w1-3", f"{name} | KoDLD severe"))
    rows.append(("KoDLD mild", name) + row("A_tenure_w1-3", f"{name} | KoDLD mild"))
    c = ceil[(ceil.model == name) & ceil.design.str.startswith("with tenure")].iloc[0]; rows.append(("Refit within KoDLD", name, c.auroc, c.ci_lo, c.ci_hi))
labels = list(dict.fromkeys(r[0] for r in rows)); ypos = {l: len(labels) - 1 - i for i, l in enumerate(labels)}
for lab, name, a, lo, hi in rows:
    yy = ypos[lab] + (0.17 if name == "lightgbm" else -0.17)
    ax_for.errorbar(a, yy, xerr=[[a - lo], [hi - a]], fmt=MODEL_MK[name], color=MODEL_COL[name], ms=6.5, capsize=3, lw=1.5, mec="white", mew=0.9, zorder=3)
ax_for.grid(False); ax_for.axvline(0.5, color=MUTED, lw=1.1, ls=(0, (1.5, 2)), zorder=1); ax_for.set_yticks(list(ypos.values())); ax_for.set_yticklabels(list(ypos.keys())); ax_for.tick_params(axis="y", length=0)
ax_for.set_xlabel("AUROC (95% CI)"); ax_for.set_xlim(0.5, 0.78); ax_for.set_xticks([0.5, 0.6, 0.7])
for i in range(len(labels)):
    if i % 2: ax_for.axhspan(i - 0.5, i + 0.5, color=SHADE, zorder=0, lw=0)
ax_for.axhline(ypos["KoDLD waves 1–3, with tenure"] + 0.5, color=INK2, lw=0.8); ax_for.axhline(ypos["KoDLD mild"] - 0.5, color=INK2, lw=0.8)
from matplotlib.lines import Line2D
ax_for.legend([Line2D([], [], marker="o", color=MODEL_COL["lightgbm"], ls="", ms=7, mec="white"), Line2D([], [], marker="s", color=MODEL_COL["logistic"], ls="", ms=7, mec="white")],
              ["LightGBM", "Logistic regression"], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True,
              handletextpad=0.5, columnspacing=1.2)
panel_label(ax_for, "c", x=-0.42, y=1.06)
save(fig, "fig5_external_validation", OUT, tiff=True)
for name in ["lightgbm", "logistic"]:
    print(name, "KoDLD mean predicted %.3f observed %.3f" % (np.average(P[("kodld", name)], weights=w), np.average(y, weights=w)))
