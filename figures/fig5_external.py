"""Fig. 5 - external validation in KoDLD: (a) ROC, (b) KoDLD calibration by decile, (c) AUROC forest.
Data: $PSED_WORK/predictions_psed_reduced7.csv (PSED temporal predictions of the reduced models, reproduced by
repro_fig5.py; metrics equal external_validation_kodld.xlsx to 1e-15), $PSED_RESULTS/predictions_kodld_A_tenure.csv
(KoDLD predictions) and external_validation_kodld.xlsx (forest values). Bootstrap seed and call order are fixed."""
import os
RES = os.environ.get("PSED_RESULTS", "results")        # result tables and saved predictions
CFG = os.environ.get("PSED_CONFIG", "config")          # fixed hyperparameters, KoDLD mapping
SCR = os.environ.get("PSED_SCRIPTS", "scripts")        # pipeline scripts (02_model.py)
WORK = os.environ.get("PSED_WORK", "work")             # individual-level derived files (not distributed)
FIGOUT = os.environ.get("PSED_FIGURES", "figures_out") # figure output folder
import importlib.util, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve
from matplotlib.lines import Line2D
import pubstyle as S
from pubstyle import plt, MODEL_COL, MODEL_LAB, MODEL_MK, MODEL_MS, INK, INK2, MUTED, SHADE

OUT = sys.argv[1] if len(sys.argv) > 1 else FIGOUT
R = RES
spec = importlib.util.spec_from_file_location("m02", os.path.join(SCR, "02_model.py"))
m02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m02)
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


ps = pd.read_csv(os.path.join(WORK, "predictions_psed_reduced7.csv"))
ko = pd.read_csv(f"{R}/predictions_kodld_A_tenure.csv")
Y = {"psed": (ps.y.to_numpy(float), ps.wt01.to_numpy(float), ps.pid.to_numpy()),
     "kodld": (ko.y.to_numpy(float), ko.wt01.to_numpy(float), ko.pid.to_numpy())}
P = {(c, m): (ps if c == "psed" else ko)[m].to_numpy(float) for c in ("psed", "kodld") for m in ("lightgbm", "logistic")}

W, H = 170.0, 150.0
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
TOPY, AH = H - 6.5 - 56.0, 56.0
ax_roc = fig.add_axes([16.0 / W, TOPY / H, 60.0 / W, AH / H])
ax_cal = fig.add_axes([103.0 / W, TOPY / H, 60.0 / W, AH / H])
ax_for = fig.add_axes([56.0 / W, 11.0 / H, 107.0 / W, 59.0 / H])
printed = {}

# (a) ROC ---------------------------------------------------------------------------------------------
S.ref_line(ax_roc, [0, 1], [0, 1])
hs, ls_ = [], []
for coh, ls, cname in [("psed", (0, (4, 1.8)), "PSED"), ("kodld", "-", "KoDLD")]:
    y, w, _ = Y[coh]
    for name in ["lightgbm", "logistic"]:
        fpr, tpr, _ = roc_curve(y, P[(coh, name)], sample_weight=w)
        auc = m02.evaluate(y, P[(coh, name)], w)["auroc"]; printed[f"auroc_{coh}_{name}"] = round(auc, 3)
        h, = ax_roc.plot(fpr, tpr, ls=ls, color=MODEL_COL[name], lw=S.LW_DATA)
        hs.append(h); ls_.append(f"{'LightGBM' if name == 'lightgbm' else 'Logistic'}, {cname}   {auc:.3f}")
ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
t = np.arange(0, 1.01, 0.2)
ax_roc.set_xticks(t); ax_roc.set_yticks(t); ax_roc.set_xticklabels([f"{v:.1f}" for v in t]); ax_roc.set_yticklabels([f"{v:.1f}" for v in t])
ax_roc.set_xlabel("1 − specificity"); ax_roc.set_ylabel("Sensitivity")
ax_roc.legend(hs, ls_, loc="lower right", title="AUROC", alignment="left", handlelength=2.4)

# (b) calibration in KoDLD -----------------------------------------------------------------------------
y, w, g = Y["kodld"]
cal = {name: calib_deciles(y, P[("kodld", name)], w, g) for name in ["lightgbm", "logistic"]}   # same order as submitted
obs = np.average(y, weights=w); printed["kodld_observed_rate"] = round(obs, 3)
xlim, ylim = 0.35, 0.60
assert max(np.nanmax(c[3]) for c in cal.values()) < ylim and max(np.nanmax(c[0]) for c in cal.values()) < xlim
S.ref_line(ax_cal, [0, xlim], [0, xlim])
ax_cal.axhline(obs, color=MUTED, lw=S.LW_REF, ls=(0, (4, 2.5)), zorder=1)
ax_cal.text(xlim * 0.985, obs - 0.010, "Observed KoDLD\nexit rate", ha="right", va="top", color=INK2, linespacing=1.15)
hh = []
for name in ["lightgbm", "logistic"]:
    mp, ob, lo, hi = cal[name]
    ax_cal.errorbar(mp, ob, yerr=[ob - lo, hi - ob], fmt="none", ecolor=MODEL_COL[name], elinewidth=0.6, capsize=1.3,
                    capthick=0.6, alpha=0.8, zorder=2)
    ax_cal.plot(mp, ob, "-", color=MODEL_COL[name], lw=0.9, zorder=3)
    ax_cal.plot(mp, ob, MODEL_MK[name], color=MODEL_COL[name], ms=MODEL_MS[name], mec="white", mew=0.5, zorder=4)
    hh.append(Line2D([], [], color=MODEL_COL[name], lw=0.9, marker=MODEL_MK[name], ms=MODEL_MS[name], mec="white", mew=0.5))
ax_cal.set_xlim(0, xlim); ax_cal.set_ylim(0, ylim)
ax_cal.set_xticks(np.arange(0, xlim + 1e-9, 0.1)); ax_cal.set_xticklabels([f"{v:.1f}" for v in np.arange(0, xlim + 1e-9, 0.1)])
ax_cal.set_yticks(np.arange(0, ylim + 1e-9, 0.1)); ax_cal.set_yticklabels([f"{v:.1f}" for v in np.arange(0, ylim + 1e-9, 0.1)])
ax_cal.set_xlabel("Predicted 1-year exit probability (decile mean)"); ax_cal.set_ylabel("Observed proportion exiting (weighted)")
ax_cal.legend(hh, [MODEL_LAB["lightgbm"], MODEL_LAB["logistic"]], loc="upper left", handlelength=2.2)

# (c) AUROC forest ------------------------------------------------------------------------------------
xl = pd.ExcelFile(f"{R}/external_validation_kodld.xlsx")
Ssum = pd.read_excel(xl, "summary"); ceil = pd.read_excel(xl, "kodld_internal_refit_ceiling"); A = pd.read_excel(xl, "A_tenure_w1-3")
def row(design, pat):
    r = Ssum[(Ssum.design == design) & Ssum.model.str.contains(pat, regex=False)].iloc[0]; return r.auroc, r.auroc_ci_lo, r.auroc_ci_hi
rows = []
for name in ["lightgbm", "logistic"]:
    r = A[A.model.str.contains(f"{name} | PSED", regex=False)].iloc[0]
    rows.append(("PSED temporal (7 predictors)", name, r.auroc, r.auroc_ci_lo, r.auroc_ci_hi))
    rows.append(("KoDLD waves 1–3, with tenure", name) + row("A_tenure_w1-3", f"{name} | KoDLD external"))
    rows.append(("KoDLD waves 1–3, age ≤64", name) + row("A_tenure_age64", f"{name} | KoDLD external"))
    rows.append(("KoDLD waves 1–5, no tenure", name) + row("B_allwaves_w1-5", f"{name} | KoDLD external"))
    rows.append(("KoDLD men", name) + row("A_tenure_w1-3", f"{name} | KoDLD men"))
    rows.append(("KoDLD women", name) + row("A_tenure_w1-3", f"{name} | KoDLD women"))
    rows.append(("KoDLD severe", name) + row("A_tenure_w1-3", f"{name} | KoDLD severe"))
    rows.append(("KoDLD mild", name) + row("A_tenure_w1-3", f"{name} | KoDLD mild"))
    c = ceil[(ceil.model == name) & ceil.design.str.startswith("with tenure")].iloc[0]
    rows.append(("Refit within KoDLD", name, c.auroc, c.ci_lo, c.ci_hi))
labels = list(dict.fromkeys(r_[0] for r_ in rows)); ypos = {l: len(labels) - 1 - i for i, l in enumerate(labels)}
for i in range(len(labels)):
    if i % 2: ax_for.axhspan(i - 0.5, i + 0.5, color=SHADE, zorder=0, lw=0)
for lab, name, a, lo, hi in rows:
    yy = ypos[lab] + (0.17 if name == "lightgbm" else -0.17)
    ax_for.plot([lo, hi], [yy, yy], color=MODEL_COL[name], lw=0.9, solid_capstyle="butt", zorder=2)
    ax_for.plot(a, yy, MODEL_MK[name], color=MODEL_COL[name], ms=MODEL_MS[name] + 0.4, mec="white", mew=0.5, zorder=3)
    printed[f"forest_{lab}_{name}"] = f"{a:.3f} ({lo:.3f}–{hi:.3f})"
ax_for.axhline(ypos["KoDLD waves 1–3, with tenure"] + 0.5, color=INK2, lw=0.5)
ax_for.axhline(ypos["KoDLD mild"] - 0.5, color=INK2, lw=0.5)
ax_for.set_yticks(list(ypos.values())); ax_for.set_yticklabels(list(ypos.keys())); ax_for.tick_params(axis="y", length=0, pad=3)
ax_for.spines["left"].set_visible(False)
ax_for.set_ylim(-0.5, len(labels) - 0.5)
ax_for.set_xlim(0.5, 0.78); ax_for.set_xticks([0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
ax_for.set_xticklabels(["0.50", "0.55", "0.60", "0.65", "0.70", "0.75"])
ax_for.set_xlabel("AUROC (95% CI)")
ax_for.legend([Line2D([], [], marker=MODEL_MK[m], color=MODEL_COL[m], ls="-", lw=0.9, ms=MODEL_MS[m] + 0.4, mec="white", mew=0.5)
               for m in ("lightgbm", "logistic")], [MODEL_LAB["lightgbm"], MODEL_LAB["logistic"]],
              loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2, handlelength=2.0, borderaxespad=0)

for a_, s_, dx in [(ax_roc, "a", 13.0), (ax_cal, "b", 13.0)]:
    S.panel_label(a_, s_, xy_fig=((a_.get_position().x0 * W - dx) / W, (a_.get_position().y1 * H + 5.0) / H))
S.panel_label(ax_for, "c", xy_fig=(3.0 / W, (ax_for.get_position().y1 * H + 6.5) / H))
for k, v in printed.items(): print(k, v)
for name in ["lightgbm", "logistic"]:
    print(name, "KoDLD mean predicted %.3f observed %.3f" % (np.average(P[("kodld", name)], weights=w), obs))
S.save(fig, "fig5_external_validation", OUT)
