"""Fig. 2 - performance in temporal validation (calibration, decision curves, ROC, precision-recall).
Data: $PSED_RESULTS/predictions_val.csv. Decile edges, weights, bootstrap seed and call order are fixed,
so every point and interval reproduces the published figure."""
import os
RES = os.environ.get("PSED_RESULTS", "results")        # result tables and saved predictions
CFG = os.environ.get("PSED_CONFIG", "config")          # fixed hyperparameters, KoDLD mapping
SCR = os.environ.get("PSED_SCRIPTS", "scripts")        # pipeline scripts (02_model.py)
WORK = os.environ.get("PSED_WORK", "work")             # individual-level derived files (not distributed)
FIGOUT = os.environ.get("PSED_FIGURES", "figures_out") # figure output folder
import importlib.util, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from matplotlib.lines import Line2D
import pubstyle as S
from pubstyle import plt, MODEL_COL, MODEL_LAB, MODEL_MK, MODEL_MS, INK, INK2, MUTED

OUT = sys.argv[1] if len(sys.argv) > 1 else FIGOUT
pred = pd.read_csv(os.path.join(RES, "predictions_val.csv"))
y, w = pred.y.to_numpy(float), pred.wt01.to_numpy(float)
MODELS = ["lightgbm", "random_forest", "logistic"]
rng = np.random.default_rng(0)                      # same seed as the submitted figure
spec = importlib.util.spec_from_file_location("m02", os.path.join(SCR, "02_model.py"))
m02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m02)


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
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return mp, ob, lo, hi


def net_benefit(y, p, t, w):
    pos = p >= t
    tp = w[pos & (y == 1)].sum() / w.sum(); fp = w[pos & (y == 0)].sum() / w.sum()
    return tp - fp * t / (1 - t)


# ---------------------------------------------------------------- layout (mm)
W, H = 170.0, 146.0
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
AW, AH = 66.0, 54.0
COLS, ROWS = (16.0, 98.0), (H - 6.5 - AH, H - 6.5 - AH - 18.0 - AH)
def axes_at(c, r):
    return fig.add_axes([COLS[c] / W, ROWS[r] / H, AW / W, AH / H])
ax_cal, ax_dca, ax_roc, ax_pr = axes_at(0, 0), axes_at(1, 0), axes_at(0, 1), axes_at(1, 1)
for a_, s_ in zip([ax_cal, ax_dca, ax_roc, ax_pr], "abcd"):
    S.panel_label(a_, s_, xy_fig=((a_.get_position().x0 * W - 13.0) / W, (a_.get_position().y1 * H + 5.0) / H))

printed = {}
# (a) calibration -------------------------------------------------------------------------------
cal = {}
for m in MODELS:
    cal[m] = calib_deciles(y, pred[m].to_numpy(), w, pred.pid.to_numpy())
lim = 0.30
assert max(np.nanmax(c[3]) for c in cal.values()) < lim and max(np.nanmax(c[0]) for c in cal.values()) < lim
S.ref_line(ax_cal, [0, lim], [0, lim])
handles, labels, rows = [], [], []
for m in MODELS:
    mp, ob, lo, hi = cal[m]
    slope, icpt = m02.calibration_slope_intercept(y, pred[m].to_numpy(), w)
    printed[f"cal_{m}"] = (round(slope, 2), round(icpt, 2))
    ax_cal.errorbar(mp, ob, yerr=[ob - lo, hi - ob], fmt="none", ecolor=MODEL_COL[m], elinewidth=0.6, capsize=1.3,
                    capthick=0.6, alpha=0.75, zorder=2)
    ax_cal.plot(mp, ob, "-", color=MODEL_COL[m], lw=0.9, zorder=3)
    ax_cal.plot(mp, ob, MODEL_MK[m], color=MODEL_COL[m], ms=MODEL_MS[m], mec="white", mew=0.5, zorder=4)
    handles.append(Line2D([], [], color=MODEL_COL[m], lw=0.9, marker=MODEL_MK[m], ms=MODEL_MS[m], mec="white", mew=0.5))
    labels.append(MODEL_LAB[m])
    rows.append((MODEL_LAB[m], f"{slope:.2f}".replace("-", "\u2212"), f"{icpt:.2f}".replace("-", "\u2212")))
ax_cal.set_xlim(0, lim); ax_cal.set_ylim(0, lim)
ticks = np.arange(0, lim + 1e-9, 0.1)
ax_cal.set_xticks(ticks); ax_cal.set_yticks(ticks)
ax_cal.set_xticklabels([f"{t:.1f}" for t in ticks]); ax_cal.set_yticklabels([f"{t:.1f}" for t in ticks])
ax_cal.set_xlabel("Predicted probability of exit (decile mean)")
ax_cal.set_ylabel("Observed proportion exiting (weighted)")
ax_cal.legend(handles, labels, loc="upper left", handlelength=2.2)
# calibration slope / intercept as a right-aligned mini-table in the empty lower-right corner
tx = {"name": 0.655, "slope": 0.785, "icpt": 0.985}
y0, dy = 0.045, 0.062
ax_cal.text(tx["slope"], y0 + 3 * dy, "Slope", transform=ax_cal.transAxes, ha="right", va="bottom", color=INK2)
ax_cal.text(tx["icpt"], y0 + 3 * dy, "Intercept", transform=ax_cal.transAxes, ha="right", va="bottom", color=INK2)
for i, (nm, sl, ic) in enumerate(rows):
    yy = y0 + (2 - i) * dy
    ax_cal.text(tx["name"], yy, nm, transform=ax_cal.transAxes, ha="right", va="bottom")
    ax_cal.text(tx["slope"], yy, sl, transform=ax_cal.transAxes, ha="right", va="bottom")
    ax_cal.text(tx["icpt"], yy, ic, transform=ax_cal.transAxes, ha="right", va="bottom")

# (b) decision curves -----------------------------------------------------------------------------
ths = np.arange(0.02, 0.501, 0.01)
prev = np.average(y, weights=w)
ax_dca.plot(ths, [prev - (1 - prev) * t / (1 - t) for t in ths], color=MUTED, lw=S.LW_REF, ls=(0, (4, 2.5)), label="Treat all")
ax_dca.axhline(0, color=MUTED, lw=S.LW_REF, ls=(0, (1, 1.6)), label="Treat none")
for m in MODELS:
    ax_dca.plot(ths, [net_benefit(y, pred[m].to_numpy(), t, w) for t in ths], color=MODEL_COL[m], lw=S.LW_DATA,
                label=MODEL_LAB[m])
ax_dca.set_xlim(0.02, 0.5); ax_dca.set_ylim(-0.02, 0.08)
ax_dca.set_xticks([0.02, 0.1, 0.2, 0.3, 0.4, 0.5]); ax_dca.set_xticklabels(["2%", "10%", "20%", "30%", "40%", "50%"])
ax_dca.set_yticks(np.arange(-0.02, 0.081, 0.02)); ax_dca.set_yticklabels([f"{t:.2f}".replace("-", "−") for t in np.arange(-0.02, 0.081, 0.02)])
ax_dca.set_xlabel("Threshold probability"); ax_dca.set_ylabel("Net benefit")
h_, l_ = ax_dca.get_legend_handles_labels()
order = [2, 3, 4, 0, 1]
ax_dca.legend([h_[i] for i in order], [l_[i] for i in order], loc="upper right")

# (c) ROC -----------------------------------------------------------------------------------------
S.ref_line(ax_roc, [0, 1], [0, 1])
for m in MODELS:
    fpr, tpr, _ = roc_curve(y, pred[m], sample_weight=w)
    auc = roc_auc_score(y, pred[m], sample_weight=w); printed[f"auroc_{m}"] = round(auc, 3)
    ax_roc.plot(fpr, tpr, color=MODEL_COL[m], lw=S.LW_DATA, label=f"{MODEL_LAB[m]}   {auc:.3f}")
ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
ax_roc.set_xticks(np.arange(0, 1.01, 0.2)); ax_roc.set_yticks(np.arange(0, 1.01, 0.2))
ax_roc.set_xticklabels([f"{t:.1f}" for t in np.arange(0, 1.01, 0.2)]); ax_roc.set_yticklabels([f"{t:.1f}" for t in np.arange(0, 1.01, 0.2)])
ax_roc.set_xlabel("1 − specificity"); ax_roc.set_ylabel("Sensitivity")
ax_roc.legend(loc="lower right", title="AUROC", alignment="left")

# (d) precision-recall ----------------------------------------------------------------------------
ax_pr.axhline(prev, color=MUTED, lw=S.LW_REF, ls=(0, (4, 2.5)), label=f"Base rate   {prev:.3f}", zorder=1)
printed["base_rate"] = round(prev, 3)
for m in MODELS:
    pr, rc, _ = precision_recall_curve(y, pred[m], sample_weight=w)
    ap = average_precision_score(y, pred[m], sample_weight=w); printed[f"auprc_{m}"] = round(ap, 3)
    ax_pr.plot(rc, pr, color=MODEL_COL[m], lw=S.LW_DATA, label=f"{MODEL_LAB[m]}   {ap:.3f}")
ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 0.45)
ax_pr.set_xticks(np.arange(0, 1.01, 0.2)); ax_pr.set_xticklabels([f"{t:.1f}" for t in np.arange(0, 1.01, 0.2)])
ax_pr.set_yticks(np.arange(0, 0.41, 0.1)); ax_pr.set_yticklabels([f"{t:.1f}" for t in np.arange(0, 0.41, 0.1)])
ax_pr.set_xlabel("Recall (sensitivity)"); ax_pr.set_ylabel("Precision (positive predictive value)")
h_, l_ = ax_pr.get_legend_handles_labels()
ax_pr.legend(h_[1:] + h_[:1], l_[1:] + l_[:1], loc="upper right", title="AUPRC", alignment="left")

print(printed)
S.save(fig, "fig2_performance", OUT)
