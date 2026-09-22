import os
from pathlib import Path
import json
import pandas as pd
import figstyle as F
from figstyle import plt, OK, DOUBLE, MM, save, MUTED, INK2, SHADE
OUTD = os.environ.get("PSED_RESULTS", "results")

lj = json.load(open("logistic_lbfgs.json")); sens = json.load(open("sens_lbfgs.json"))
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
fig, ax = plt.subplots(figsize=(DOUBLE, 165 * MM)); ax.grid(False)
n = len(rows)
for i, (lab, a, lo, hi, b, blo, bhi) in enumerate(rows):
    yy = n - 1 - i
    if lo is not None: ax.plot([lo, hi], [yy + 0.15, yy + 0.15], color=OK["blue"], lw=1.6)
    ax.plot(a, yy + 0.15, "o", color=OK["blue"], ms=7, mec="white", mew=0.9, zorder=3)
    if b is not None:
        if blo is not None: ax.plot([blo, bhi], [yy - 0.15, yy - 0.15], color=OK["verm"], lw=1.6)
        ax.plot(b, yy - 0.15, "s", color=OK["verm"], ms=6.5, mec="white", mew=0.9, zorder=3)
ax.axvline(0.5, color=MUTED, lw=1.1, ls=(0, (1.5, 2)), zorder=1)
for sep in (n - 7 - 0.5, n - 11 - 0.5, n - 13 - 0.5):
    ax.axhline(sep, color=INK2, lw=0.8)
ax.axvline(0.714, color=OK["blue"], lw=1.0, ls=(0, (4, 3)), alpha=0.6, zorder=1)
ax.set_yticks(range(n)[::-1]); ax.set_yticklabels([r[0] for r in rows]); ax.tick_params(axis="y", length=0)
for i in range(n):
    if i % 2: ax.axhspan(i - 0.5, i + 0.5, color=SHADE, zorder=0, lw=0)
ax.set_xlim(0.42, 0.85); ax.set_xlabel("AUROC (weighted, temporal validation)")
ax.plot([], [], "o", color=OK["blue"], ms=7, mec="white", label="LightGBM"); ax.plot([], [], "s", color=OK["verm"], ms=6.5, mec="white", label="Logistic regression (L-BFGS)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.set_ylim(-0.6, n - 0.4)
fig.text(0.99, -0.02, "Bars: 95% person-clustered bootstrap CI where computed.", fontsize=9, color=INK2, ha="right")
save(fig, "figS2_sensitivity_forest", OUTD)
