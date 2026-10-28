"""Fig. 4 (predictor reliance within disability-type subgroups) and Fig. S1 (by validation wave).
Computed from $PSED_WORK/shap_val.csv (row-aligned with predictions_val.csv);
values and ranks equal shap_subgroups_recomputed.xlsx (max |diff| 6e-17, ranks identical)."""
import os
RES = os.environ.get("PSED_RESULTS", "results")        # result tables and saved predictions
CFG = os.environ.get("PSED_CONFIG", "config")          # fixed hyperparameters, KoDLD mapping
SCR = os.environ.get("PSED_SCRIPTS", "scripts")        # pipeline scripts (02_model.py)
WORK = os.environ.get("PSED_WORK", "work")             # individual-level derived files (not distributed)
FIGOUT = os.environ.get("PSED_FIGURES", "figures_out") # figure output folder
import sys
import numpy as np, pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
import pubstyle as S
from pubstyle import plt, INK, INK2, flab

OUT = sys.argv[1] if len(sys.argv) > 1 else FIGOUT
pred = pd.read_csv(os.path.join(RES, "predictions_val.csv"))
sv = pd.read_csv(os.path.join(WORK, "shap_val.csv"))
assert len(pred) == len(sv)
mabs = sv.abs().mean().sort_values(ascending=False)
TYPE = {1: "Physical", 2: "Brain lesion", 3: "Visual", 4: "Hearing", 6: "Intellectual", 9: "Kidney"}


def by(col):
    rows = []
    for g, idx in pred.groupby(col).groups.items():
        m = sv.loc[idx].abs().mean().sort_values(ascending=False)
        for r, (f, v) in enumerate(m.items(), 1):
            rows.append((int(g), len(idx), f, v, r))
    return pd.DataFrame(rows, columns=["grp", "n", "feature", "mean_abs_shap", "rank"])


sub, bw = by("a0710"), by("wave")
xl = pd.ExcelFile(os.path.join(RES, "shap_subgroups_recomputed.xlsx"))
ref = pd.read_excel(xl, "SHAP_by_disability_type").rename(columns={"subgroup": "grp"})
chk = sub.merge(ref, on=["grp", "feature"], suffixes=("", "_ref"))
assert len(chk) == len(sub) and (chk["mean_abs_shap"] - chk["mean_abs_shap_ref"]).abs().max() < 1e-12 and (chk["rank"] == chk["rank_ref"]).all()

# ============================================================================ Fig. 4 heatmap
s6 = sub[(sub.n >= 100) & sub.grp.isin(TYPE)]
feats = list(mabs.index[:12])
mat = s6.pivot(index="feature", columns="grp", values="mean_abs_shap").reindex(feats)
rank = s6.pivot(index="feature", columns="grp", values="rank").reindex(feats)
ns = {c: int(s6[s6.grp == c].n.iloc[0]) for c in mat.columns}
cmap = LinearSegmentedColormap.from_list("seqblue", S.SEQ_BLUE)
vmax = float(np.nanmax(mat.values))

W, H = 170.0, 96.0
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
CW, RH = 17.0, 6.6                       # cell width / row height (mm)
x0, ytop = 47.0, H - 12.5
nr, nc = mat.shape
ax = fig.add_axes([x0 / W, (ytop - nr * RH) / H, nc * CW / W, nr * RH / H])
ax.set_xlim(0, nc); ax.set_ylim(nr, 0); ax.axis("off")
GAP = 0.06                                # white surface gap between cells (cell units)
for i in range(nr):
    for j in range(nc):
        v, r = mat.values[i, j], int(rank.values[i, j])
        fc = cmap(v / vmax)
        ax.add_patch(Rectangle((j + GAP / 2, i + GAP / 2), 1 - GAP, 1 - GAP, fc=fc, ec="none"))
        lum = 0.2126 * fc[0] + 0.7152 * fc[1] + 0.0722 * fc[2]
        tc = "white" if lum < 0.45 else INK
        ax.text(j + 0.5, i + 0.47, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=tc)
        ax.text(j + 0.5, i + 0.52, f"rank {r}", ha="center", va="top", fontsize=S.FS, color=tc)
for i, f in enumerate(feats):
    ax.text(-0.12, i + 0.5, flab(f), ha="right", va="center", fontsize=S.FS_LAB, linespacing=1.0)
for j, c in enumerate(mat.columns):
    ax.text(j + 0.5, -0.62, TYPE[c], ha="center", va="bottom", fontsize=S.FS_LAB)
    ax.text(j + 0.5, -0.18, f"n = {ns[c]:,}", ha="center", va="bottom", fontsize=S.FS, color=INK2)
cax = fig.add_axes([(x0 + nc * CW + 5.0) / W, (ytop - nr * RH * 0.75) / H, 2.4 / W, nr * RH * 0.5 / H])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax)); sm.set_array([])
cb = fig.colorbar(sm, cax=cax, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
cb.ax.set_yticklabels(["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"])
cb.outline.set_visible(False); cb.ax.tick_params(length=2, width=0.5, pad=2)
cb.set_label("Mean |SHAP value| (log-odds)", fontsize=S.FS, labelpad=4)
print("Fig 4 n:", ns, "vmax", round(vmax, 3))
S.save(fig, "fig4_shap_subgroups", OUT)

# ============================================================================ Fig. S1 by wave
feats8 = list(mabs.index[:8])
W1, H1 = 85.0, 84.0
fig = plt.figure(figsize=(W1 * S.MM, H1 * S.MM))
ax = fig.add_axes([41.5 / W1, 12.0 / H1, 40.5 / W1, 69.0 / H1])
waves = sorted(bw.grp.unique())
ypos = np.arange(len(feats8))[::-1]; hb = 0.26
YR = {5: "2020→2021", 6: "2021→2022", 7: "2022→2023"}
for k, wv in enumerate(waves):
    d = bw[bw.grp == wv].set_index("feature").reindex(feats8)["mean_abs_shap"]
    n_w = int(bw[bw.grp == wv].n.iloc[0])
    ax.barh(ypos + (1 - k) * hb, d.values, hb * 0.92, color=S.ORD_BLUE3[k], lw=0, label=f"Wave {int(wv)} ({YR[int(wv)]})")
    print("S1 wave", wv, "n", n_w)
ax.set_yticks(ypos); ax.set_yticklabels([flab(f) for f in feats8], linespacing=1.0)
ax.tick_params(axis="y", length=0, pad=3); ax.spines["left"].set_visible(False)
ax.set_xlim(0, 0.5); ax.set_xticks(np.arange(0, 0.51, 0.1)); ax.set_xticklabels([f"{t:.1f}" for t in np.arange(0, 0.51, 0.1)])
ax.set_ylim(-0.6, len(feats8) - 0.4)
ax.xaxis.grid(True, color=S.GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_xlabel("Mean |SHAP value| (log-odds)")
ax.legend(title="Validation index wave", loc="lower right", alignment="left", handlelength=1.4, handleheight=0.9)
S.save(fig, "figS1_shap_by_wave", OUT)
