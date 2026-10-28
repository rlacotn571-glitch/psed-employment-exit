"""Fig. S2 - sensitivity and subgroup analyses (weighted AUROC, temporal validation).
Values transcribed from the sensitivity outputs in results/ (sensitivity.xlsx, sens2.json, sens_lbfgs.json, sens_sex.json)."""
import os
RES = os.environ.get("PSED_RESULTS", "results")        # result tables and saved predictions
CFG = os.environ.get("PSED_CONFIG", "config")          # fixed hyperparameters, KoDLD mapping
SCR = os.environ.get("PSED_SCRIPTS", "scripts")        # pipeline scripts (02_model.py)
WORK = os.environ.get("PSED_WORK", "work")             # individual-level derived files (not distributed)
FIGOUT = os.environ.get("PSED_FIGURES", "figures_out") # figure output folder
import sys
from matplotlib.lines import Line2D
import pubstyle as S
from pubstyle import plt, MODEL_COL, MODEL_MK, MODEL_MS, INK2, MUTED, SHADE

OUT = sys.argv[1] if len(sys.argv) > 1 else FIGOUT
rows = [
    ("Primary analysis (n=4,530; 326 exits)", 0.714, 0.665, 0.758, 0.654, 0.599, 0.700),
    ("Person-disjoint subset (n=335; 54)", 0.747, None, None, 0.673, None, None),
    ("Outcome C: incl. moves to non-wage work (355)", 0.700, None, None, 0.653, None, None),
    ("COVID-19 waves excluded (n=3,039; 230)", 0.684, None, None, 0.657, None, None),
    ("Index age <60 (n=3,920; 238)", 0.734, None, None, 0.715, None, None),
    ("Multiple imputation (5 imputations)", 0.706, None, None, 0.657, None, None),
    ("Weighted fitting + weighted evaluation", 0.669, None, None, 0.625, None, None),
    ("Pooled model, men (n=3,402; 191)", 0.747, 0.697, 0.800, 0.664, 0.595, 0.728),
    ("Pooled model, women (n=1,128; 135)", 0.584, 0.482, 0.694, 0.548, 0.442, 0.660),
    ("Re-fitted on men only", 0.740, None, None, 0.677, None, None),
    ("Re-fitted on women only", 0.590, None, None, 0.567, None, None),
    ("Pooled model, age <45 (n=2,126; 140)", 0.756, 0.688, 0.828, 0.757, 0.691, 0.809),
    ("Pooled model, age ≥45 (n=2,404; 186)", 0.700, 0.648, 0.754, 0.629, 0.575, 0.688),
    ("Pooled model, mild disability (n=3,610; 243)", 0.742, 0.696, 0.789, 0.673, None, None),
    ("Pooled model, severe disability (n=911; 81)", 0.628, 0.512, 0.763, 0.585, None, None),
    ("Re-fitted on mild only", 0.744, None, None, 0.688, None, None),
    ("Re-fitted on severe only", 0.547, None, None, 0.582, None, None),
]
n = len(rows)
W, H = 170.0, 118.0
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
ax = fig.add_axes([63.0 / W, 11.0 / H, 102.0 / W, 99.0 / H])
for i in range(n):
    if i % 2: ax.axhspan(i - 0.5, i + 0.5, color=SHADE, zorder=0, lw=0)
L, G = MODEL_COL["lightgbm"], MODEL_COL["logistic"]
for i, (lab, a, lo, hi, b, blo, bhi) in enumerate(rows):
    yy = n - 1 - i
    if lo is not None: ax.plot([lo, hi], [yy + 0.16, yy + 0.16], color=L, lw=0.9, solid_capstyle="butt", zorder=2)
    ax.plot(a, yy + 0.16, MODEL_MK["lightgbm"], color=L, ms=MODEL_MS["lightgbm"] + 0.4, mec="white", mew=0.5, zorder=3)
    if blo is not None: ax.plot([blo, bhi], [yy - 0.16, yy - 0.16], color=G, lw=0.9, solid_capstyle="butt", zorder=2)
    ax.plot(b, yy - 0.16, MODEL_MK["logistic"], color=G, ms=MODEL_MS["logistic"] + 0.4, mec="white", mew=0.5, zorder=3)
ax.axvline(0.5, color=MUTED, lw=S.LW_REF, ls=(0, (1, 1.6)), zorder=1)                 # dotted: AUROC 0.5
ax.axvline(0.714, color=L, lw=S.LW_REF, ls=(0, (4, 2.5)), alpha=0.75, zorder=1)        # dashed: primary LightGBM
for sep in (n - 7 - 0.5, n - 11 - 0.5, n - 13 - 0.5):
    ax.axhline(sep, color=INK2, lw=0.5, zorder=1)
ax.set_yticks(range(n)[::-1]); ax.set_yticklabels([r[0] for r in rows]); ax.tick_params(axis="y", length=0, pad=3)
ax.spines["left"].set_visible(False)
ax.set_xlim(0.42, 0.85); ax.set_ylim(-0.5, n - 0.5)
ticks = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
ax.set_xticks(ticks); ax.set_xticklabels([f"{t:.2f}" for t in ticks])
ax.set_xlabel("AUROC (weighted, temporal validation)")
ax.legend([Line2D([], [], marker=MODEL_MK[m], color=MODEL_COL[m], ls="-", lw=0.9, ms=MODEL_MS[m] + 0.4, mec="white", mew=0.5)
           for m in ("lightgbm", "logistic")], ["LightGBM", "Logistic regression (L-BFGS)"],
          loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2, handlelength=2.0, borderaxespad=0)
S.save(fig, "figS2_sensitivity_forest", OUT)
