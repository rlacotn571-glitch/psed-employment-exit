"""Fig. 3 - basis of prediction in the LightGBM model: (a) mean |SHAP| of the 15 most influential predictors,
(b) SHAP beeswarm for the same predictors. Data: $PSED_WORK/shap_val.csv and X_val_transformed.csv, written
by scripts/repro_shap.py (refit predictions equal predictions_val.csv to 1e-16; subgroup means equal
shap_subgroups_recomputed.xlsx to 6e-17). All 4,530 validation person-waves are plotted."""
import os
RES = os.environ.get("PSED_RESULTS", "results")        # result tables and saved predictions
CFG = os.environ.get("PSED_CONFIG", "config")          # fixed hyperparameters, KoDLD mapping
SCR = os.environ.get("PSED_SCRIPTS", "scripts")        # pipeline scripts (02_model.py)
WORK = os.environ.get("PSED_WORK", "work")             # individual-level derived files (not distributed)
FIGOUT = os.environ.get("PSED_FIGURES", "figures_out") # figure output folder
import sys
import numpy as np, pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import pubstyle as S
from pubstyle import plt, MODEL_COL, INK, INK2, flab

OUT = sys.argv[1] if len(sys.argv) > 1 else FIGOUT
D = WORK  # reproduced individual-level inputs (kept out of the shared outputs)
sv = pd.read_csv(f"{D}/shap_val.csv"); X = pd.read_csv(f"{D}/X_val_transformed.csv")
mabs = sv.abs().mean().sort_values(ascending=False)
top = list(mabs.index[:15]); n = len(top)
rng = np.random.default_rng(20260918)

W, H = 170.0, 104.0
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
y0, ph = 13.0, 84.0                                    # plot bottom / height (mm)
ax_bar = fig.add_axes([50.0 / W, y0 / H, 38.0 / W, ph / H])
ax_bee = fig.add_axes([97.0 / W, y0 / H, 58.0 / W, ph / H])
cax = fig.add_axes([158.5 / W, (y0 + ph * 0.30) / H, 2.2 / W, ph * 0.40 / H])
ypos = np.arange(n)[::-1]

# (a) mean |SHAP| -----------------------------------------------------------------------------------
vals = mabs[top].to_numpy()
ax_bar.barh(ypos, vals, height=0.62, color=MODEL_COL["lightgbm"], lw=0)
for yy, v in zip(ypos, vals):
    ax_bar.text(v + 0.008, yy, f"{v:.2f}", va="center", ha="left", fontsize=S.FS, color=INK2)
labels = [flab(f) for f in top]
labels = [("Sex (female = high)" if l == "Sex" else l) for l in labels]
ax_bar.set_yticks(ypos); ax_bar.set_yticklabels(labels, linespacing=1.0)
ax_bar.tick_params(axis="y", length=0, pad=3)
ax_bar.set_xlim(0, 0.6); ax_bar.set_xticks([0, 0.2, 0.4, 0.6]); ax_bar.set_xticklabels(["0.0", "0.2", "0.4", "0.6"])
ax_bar.set_ylim(-0.6, n - 0.4)
ax_bar.spines["left"].set_visible(False)
ax_bar.set_xlabel("Mean |SHAP value| (log-odds)")

# (b) beeswarm (layout as in shap.summary_plot: 100 bins per row, alternating layers) -------------
cmap = LinearSegmentedColormap.from_list("lowhigh", S.DIV_BR)
for row, f in enumerate(top):
    yy = n - 1 - row
    s = sv[f].to_numpy(); x = X[f].to_numpy().astype(float)
    if set(np.unique(x[~np.isnan(x)])) <= {1.0, 2.0} and "gender" not in f:
        x = 3 - x                                   # yes/no items coded 1 = yes, 2 = no -> high = yes (sex: high = female)
    col = pd.Series(x).rank(pct=True).to_numpy()
    N = len(s)
    quant = np.round(100 * (s - s.min()) / (s.max() - s.min() + 1e-8))
    inds = np.argsort(quant + rng.standard_normal(N) * 1e-6)
    layer, last, off = 0, -1, np.zeros(N)
    for i in inds:
        if quant[i] != last: layer = 0
        off[i] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1); layer += 1; last = quant[i]
    off *= 0.40 / (np.max(np.abs(off)) + 1)
    order = rng.permutation(N)                     # draw in random order so neither colour hides the other
    # colour scale stretched to each row's own percentile range, as in the submitted figure (binary items: no = low, yes = high)
    ax_bee.scatter(s[order], yy + off[order], c=col[order], cmap=cmap, vmin=col.min(), vmax=col.max(), s=1.3, lw=0,
                   alpha=0.9, rasterized=True, zorder=2)
ax_bee.axvline(0, color=INK2, lw=0.6, zorder=1)
ax_bee.set_yticks(ypos); ax_bee.set_yticklabels([]); ax_bee.tick_params(axis="y", length=0)
ax_bee.spines["left"].set_visible(False)
ax_bee.set_ylim(-0.6, n - 0.4)
ax_bee.set_xlim(-1.1, 1.3); ax_bee.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
ax_bee.set_xticklabels(["−1.0", "−0.5", "0.0", "0.5", "1.0"])
ax_bee.set_xlabel("SHAP value (log-odds of exit)")
for yy in ypos:                                    # faint row guides help track a label across both panels
    ax_bee.axhline(yy, color=S.GRID, lw=0.5, zorder=0)
sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_array([0, 1])
cb = fig.colorbar(sm, cax=cax, ticks=[0, 1]); cb.ax.set_yticklabels(["Low", "High"])
cb.outline.set_visible(False); cb.ax.tick_params(length=0, pad=2)
cb.set_label("Predictor value (percentile)", fontsize=S.FS, labelpad=3)

S.panel_label(ax_bar, "a", xy_fig=(3.0 / W, (y0 + ph + 6.5) / H))
S.panel_label(ax_bee, "b", xy_fig=(92.0 / W, (y0 + ph + 6.5) / H))
print({flab(f).replace("\n", " "): round(float(v), 2) for f, v in zip(top, vals)})
S.save(fig, "fig3_shap", OUT)
