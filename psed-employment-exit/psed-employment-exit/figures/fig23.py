"""Fig 2 (calibration / DCA / ROC / PR) and Fig 3 (SHAP bar + beeswarm) — top-journal styling.
Run from a folder containing predictions_val.csv, shap_val.csv, X_val_transformed.csv."""
import os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from matplotlib.colors import LinearSegmentedColormap
import figstyle as F
from figstyle import plt, OK, MODEL_COL, MODEL_LAB, MODEL_MK, DOUBLE, MM, save, panel_label, flab, ref_line, INK2, MUTED

OUTD = os.environ.get("PSED_RESULTS", "results")
pred = pd.read_csv("predictions_val.csv")
y, w = pred.y.to_numpy(float), pred.wt01.to_numpy(float)
MODELS = ["lightgbm", "random_forest", "logistic"]
SHORT = {"lightgbm": "LightGBM", "random_forest": "Random forest", "logistic": "Logistic"}
rng = np.random.default_rng(0)


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


# =========================================================================== Fig 2
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 175 * MM))
(ax_cal, ax_dca), (ax_roc, ax_pr) = axes

# (a) calibration
ref_line(ax_cal, [0, 0.4], [0, 0.4])
for m in MODELS:
    mp, ob, lo, hi = calib_deciles(y, pred[m].to_numpy(), w, pred.pid.to_numpy())
    ax_cal.errorbar(mp, ob, yerr=[ob - lo, hi - ob], fmt=MODEL_MK[m] + "-", color=MODEL_COL[m], ms=6, lw=1.8,
                    capsize=2.5, elinewidth=1.0, mec="white", mew=0.9, label=MODEL_LAB[m], zorder=3)
ax_cal.set_xlim(0, 0.4); ax_cal.set_ylim(0, 0.4)
ax_cal.set_xlabel("Predicted probability of exit (decile mean)")
ax_cal.set_ylabel("Observed proportion (weighted)")
ax_cal.set_xticks(np.arange(0, 0.41, 0.1)); ax_cal.set_yticks(np.arange(0, 0.41, 0.1))
ax_cal.legend(loc="upper left")
ax_cal.text(0.98, 0.03, "Slope / intercept\nLightGBM 0.82 / −0.30\nRandom forest 1.03 / 0.10\nLogistic 0.79 / −0.44",
            transform=ax_cal.transAxes, ha="right", va="bottom", fontsize=9, color=INK2, linespacing=1.35,
            bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="none"))
panel_label(ax_cal, "a")

# (b) DCA
ths = np.arange(0.02, 0.501, 0.01)
prev = np.average(y, weights=w)
ax_dca.plot(ths, [prev - (1 - prev) * t / (1 - t) for t in ths], color=MUTED, lw=1.3, ls=(0, (4, 3)), label="Treat all")
ax_dca.axhline(0, color=MUTED, lw=1.3, ls=(0, (1.5, 2)), label="Treat none")
for m in MODELS:
    ax_dca.plot(ths, [net_benefit(y, pred[m].to_numpy(), t, w) for t in ths], color=MODEL_COL[m], label=MODEL_LAB[m])
ax_dca.set_xlim(0.02, 0.5); ax_dca.set_ylim(-0.02, 0.08)
ax_dca.set_xlabel("Threshold probability"); ax_dca.set_ylabel("Net benefit")
ax_dca.set_xticks([0.02, 0.1, 0.2, 0.3, 0.4, 0.5]); ax_dca.set_xticklabels(["2%", "10%", "20%", "30%", "40%", "50%"])
ax_dca.legend(loc="upper right")
panel_label(ax_dca, "b")

# (c) ROC
ref_line(ax_roc, [0, 1], [0, 1])
for m in MODELS:
    fpr, tpr, _ = roc_curve(y, pred[m], sample_weight=w)
    ax_roc.plot(fpr, tpr, color=MODEL_COL[m], label=f"{SHORT[m]} {roc_auc_score(y, pred[m], sample_weight=w):.3f}")
ax_roc.set_xlabel("1 − specificity"); ax_roc.set_ylabel("Sensitivity")
ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
ax_roc.legend(loc="lower right", title="AUROC", fontsize=9.5, title_fontsize=9.5, handlelength=1.6)
panel_label(ax_roc, "c")

# (d) PR
ax_pr.axhline(prev, color=MUTED, lw=1.3, ls=(0, (4, 3)), label=f"Base rate {prev:.3f}")
for m in MODELS:
    pr, rc, _ = precision_recall_curve(y, pred[m], sample_weight=w)
    ax_pr.plot(rc, pr, color=MODEL_COL[m], label=f"{SHORT[m]} {average_precision_score(y, pred[m], sample_weight=w):.3f}")
ax_pr.set_xlabel("Recall (sensitivity)"); ax_pr.set_ylabel("Precision (PPV)")
ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 0.45)
ax_pr.legend(loc="upper right", title="AUPRC", fontsize=9.5, title_fontsize=9.5, handlelength=1.6)
panel_label(ax_pr, "d")

for a_ in axes.ravel(): a_.set_box_aspect(0.9)
fig.tight_layout(w_pad=2.5, h_pad=2.5)
save(fig, "fig2_performance", OUTD, tiff=True)

# =========================================================================== Fig 3 SHAP
sv = pd.read_csv("shap_val.csv"); X = pd.read_csv("X_val_transformed.csv")
mabs = sv.abs().mean().sort_values(ascending=False)
top = mabs.index[:15]

fig, (ax_bar, ax_bee) = plt.subplots(1, 2, figsize=(DOUBLE, 135 * MM), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.12})
ax_bar.grid(False); ax_bee.grid(False)
# (a) bar
ax_bar.barh(range(len(top))[::-1], mabs[top].values, color=OK["blue"], height=0.68)
ax_bar.set_yticks(range(len(top))[::-1]); ax_bar.set_yticklabels([flab(f).replace("Sex", "Sex (female = high)") for f in top], fontsize=10)
ax_bar.set_xlabel("Mean |SHAP| (log-odds)")
for i, v in enumerate(mabs[top].values):
    ax_bar.text(v + 0.006, len(top) - 1 - i, f"{v:.2f}", va="center", fontsize=9, color=INK2)
ax_bar.set_xlim(0, mabs[top].max() * 1.2); ax_bar.set_ylim(-0.7, len(top) - 0.3)
ax_bar.tick_params(axis="y", length=0)
panel_label(ax_bar, "a", x=-0.62)

# (b) beeswarm — diverging: cool (low value) → neutral grey → warm (high value)
cmap = LinearSegmentedColormap.from_list("div", ["#2C7BB6", "#C9C9C9", "#D7301F"])
for row, f in enumerate(top):
    yy = len(top) - 1 - row
    s = sv[f].to_numpy(); x = X[f].to_numpy().astype(float)
    if set(np.unique(x[~np.isnan(x)])) <= {1.0, 2.0} and "gender" not in f:
        x = 3 - x  # yes/no items coded 1 = yes, 2 = no → high = yes (sex kept: high = female)
    xr = pd.Series(x).rank(pct=True).to_numpy()
    sel = rng.choice(len(s), min(1500, len(s)), replace=False); s, xr = s[sel], xr[sel]
    order = np.argsort(s); s = s[order]; xr = xr[order]
    bins = np.digitize(s, np.linspace(s.min(), s.max(), 60)); jit = np.zeros_like(s)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b); n = len(idx); amp = 0.42 * min(1, n / 60)
        jit[idx] = rng.permutation(np.linspace(-amp, amp, n))
    ax_bee.scatter(s, yy + jit, c=xr, cmap=cmap, s=7, alpha=0.8, linewidths=0, rasterized=True)
ax_bee.axvline(0, color=INK2, lw=1.0, zorder=0)
ax_bee.set_yticks(range(len(top))[::-1]); ax_bee.set_yticklabels([]); ax_bee.tick_params(axis="y", length=0)
ax_bee.set_xlabel("SHAP value (log-odds of exit)")
ax_bee.set_ylim(-0.7, len(top) - 0.3)
sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_array([0, 1])
cb = fig.colorbar(sm, ax=ax_bee, fraction=0.035, pad=0.03, ticks=[0, 1])
cb.ax.set_yticklabels(["Low", "High"]); cb.set_label("Feature value (percentile)", fontsize=10)
cb.outline.set_visible(False); cb.ax.tick_params(length=0)
panel_label(ax_bee, "b", x=-0.04)
fig.tight_layout(w_pad=3.0)
save(fig, "fig3_shap", OUTD, tiff=True)
