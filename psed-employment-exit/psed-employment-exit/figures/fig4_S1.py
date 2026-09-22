"""Fig 4 (predictor reliance within disability-type subgroups) and Fig S1 (by validation wave).
Recomputed directly from shap_val.csv + predictions_val.csv (row-aligned, temporal validation set),
so the figures no longer depend on results_main.xlsx. Weighted? No — mean |SHAP| is unweighted, as in the main text."""
import os
from pathlib import Path
import numpy as np, pandas as pd
import figstyle as F
from figstyle import plt, OK, DOUBLE, SINGLE, MM, save, flab, INK, INK2

OUTD = os.environ.get("PSED_RESULTS", "results")
pred = pd.read_csv("predictions_val.csv"); sv = pd.read_csv("shap_val.csv")
assert len(pred) == len(sv)
mabs = sv.abs().mean().sort_values(ascending=False)

# ---- subgroup tables (also written to xlsx so the numbers are auditable)
TYPE = {1: "Physical", 2: "Brain lesion", 3: "Visual", 4: "Hearing", 6: "Intellectual", 9: "Kidney"}
rows = []
for g, idx in pred.groupby("a0710").groups.items():
    m = sv.loc[idx].abs().mean().sort_values(ascending=False)
    for r, (f, v) in enumerate(m.items(), 1): rows.append((int(g), len(idx), f, v, r))
sub = pd.DataFrame(rows, columns=["subgroup", "n", "feature", "mean_abs_shap", "rank"])
rows = []
for wv, idx in pred.groupby("wave").groups.items():
    m = sv.loc[idx].abs().mean().sort_values(ascending=False)
    for r, (f, v) in enumerate(m.items(), 1): rows.append((int(wv), len(idx), f, v, r))
bw = pd.DataFrame(rows, columns=["wave", "n", "feature", "mean_abs_shap", "rank"])
with pd.ExcelWriter(f"{OUTD}/shap_subgroups_recomputed.xlsx") as xw:
    sub.to_excel(xw, sheet_name="SHAP_by_disability_type", index=False); bw.to_excel(xw, sheet_name="SHAP_by_wave", index=False)

# =========================================================================== Fig 4 heatmap
s6 = sub[(sub.n >= 100) & sub.subgroup.isin(TYPE)]
feats = list(mabs.index[:12])
mat = s6.pivot(index="feature", columns="subgroup", values="mean_abs_shap").reindex(feats)
rank = s6.pivot(index="feature", columns="subgroup", values="rank").reindex(feats)
ns = {c: int(s6[s6.subgroup == c].n.iloc[0]) for c in mat.columns}
mat.columns = [f"{TYPE[c]}\nn = {ns[c]:,}" for c in mat.columns]

fig, ax = plt.subplots(figsize=(DOUBLE, 125 * MM)); ax.grid(False)
im = ax.imshow(mat.values, cmap="Blues", aspect="auto", vmin=0, vmax=np.nanmax(mat.values))
ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, fontsize=9.5, linespacing=1.25)
ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels([flab(f) for f in mat.index], fontsize=10)
ax.xaxis.set_ticks_position("top")
vmax = np.nanmax(mat.values)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v, r = mat.values[i, j], rank.values[i, j]
        col = "white" if v > 0.55 * vmax else INK
        ax.text(j, i, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5, color=col, fontweight="medium")
        ax.text(j, i + 0.05, f"rank {int(r)}", ha="center", va="top", fontsize=7.5, color=col)
# thin white grid between cells
ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True); ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
ax.spines[:].set_visible(False); ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03); cb.set_label("Mean |SHAP value| (log-odds)", fontsize=10)
cb.outline.set_visible(False); cb.ax.tick_params(length=0)
fig.tight_layout()
save(fig, "fig4_shap_subgroups", OUTD, tiff=True)

# =========================================================================== Fig S1 by wave
feats8 = list(mabs.index[:8])
fig, ax = plt.subplots(figsize=(DOUBLE * 0.8, 100 * MM)); ax.grid(False)
waves = sorted(bw.wave.unique())
ypos = np.arange(len(feats8))[::-1]; height = 0.26
cols = ["#9ECAE1", "#4292C6", "#08306B"]
for k, wv in enumerate(waves):
    d = bw[bw.wave == wv].set_index("feature").reindex(feats8)["mean_abs_shap"]
    yr = {5: "2020→2021", 6: "2021→2022", 7: "2022→2023"}[int(wv)]
    ax.barh(ypos + (1 - k) * height, d.values, height, color=cols[k], label=f"Wave {int(wv)} ({yr})")
ax.set_yticks(ypos); ax.set_yticklabels([flab(f) for f in feats8], fontsize=10); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Mean |SHAP value| (log-odds)")
ax.xaxis.grid(True, color="#E4E4E4", lw=0.6); ax.set_axisbelow(True)
ax.legend(title="Validation index wave", loc="lower right")
fig.tight_layout()
save(fig, "figS1_shap_by_wave", OUTD)
print(sub[sub.subgroup.isin(TYPE)].groupby("subgroup").n.first().to_dict())
