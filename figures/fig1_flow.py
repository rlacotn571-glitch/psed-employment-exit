"""Fig. 1 - derivation of the analytic sample (CONSORT-style flow). 170 mm wide, drawn in mm coordinates.
Counts are those of the analytic sample derivation (see scripts/01_build_panel.py and scripts/attrition_wage_workers.py)."""
import os
RES = os.environ.get("PSED_RESULTS", "results")        # result tables and saved predictions
CFG = os.environ.get("PSED_CONFIG", "config")          # fixed hyperparameters, KoDLD mapping
SCR = os.environ.get("PSED_SCRIPTS", "scripts")        # pipeline scripts (02_model.py)
WORK = os.environ.get("PSED_WORK", "work")             # individual-level derived files (not distributed)
FIGOUT = os.environ.get("PSED_FIGURES", "figures_out") # figure output folder
import sys
from matplotlib.patches import FancyBboxPatch
import pubstyle as S
from pubstyle import plt, INK, INK2

OUT = sys.argv[1] if len(sys.argv) > 1 else FIGOUT
W, H = 170.0, 148.5
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

PT = 0.3528                     # mm per point
T_FS, B_FS, X_FS = 8.0, 7.5, 7.0
T_LH, B_LH, X_LH = T_FS * 1.32 * PT, B_FS * 1.32 * PT, X_FS * 1.30 * PT
PAD = 2.1


def main_box(xc, top, w, title, lines, lw=0.75):
    h = PAD * 2 + T_LH + B_LH * len(lines)
    ax.add_patch(FancyBboxPatch((xc - w / 2, top - h), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="white", ec=INK, lw=lw, zorder=2))
    y = top - PAD
    ax.text(xc, y, title, ha="center", va="top", fontsize=T_FS, fontweight="bold", color=INK, zorder=3)
    y -= T_LH
    for ln in lines:
        ax.text(xc, y, ln, ha="center", va="top", fontsize=B_FS, color=INK, zorder=3)
        y -= B_LH
    return dict(x0=xc - w / 2, x1=xc + w / 2, xc=xc, top=top, bot=top - h, yc=top - h / 2)


def excl_box(x0, yc, w, lines):
    h = 1.9 * 2 + X_LH * len(lines)
    ax.add_patch(FancyBboxPatch((x0, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc="#F2F2F2", ec="none", zorder=2))
    y = yc + h / 2 - 1.9
    for ln in lines:
        ax.text(x0 + 2.4, y, ln, ha="left", va="top", fontsize=X_FS, color=INK, zorder=3)
        y -= X_LH
    return dict(x0=x0, x1=x0 + w, top=yc + h / 2, bot=yc - h / 2)


def line(x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color=INK, lw=0.75, solid_capstyle="butt", zorder=1)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color=INK, lw=0.75,
                                shrinkA=0, shrinkB=0))


MX, MW = 48.0, 78.0             # main column centre / width
EX0, EW = 101.0, 60.0           # exclusion column left edge / width
GAP = 13.5                      # vertical gap between main boxes

b1 = main_box(MX, H - 3.0, MW, "PSED-II, waves 1–8 (2016–2023)", ["36,616 person-waves · 4,577 persons"])
b2 = main_box(MX, b1["bot"] - GAP - 2.5, MW, "Wage workers at index wave t", ["12,824 person-waves · 2,393 persons"])
b3 = main_box(MX, b2["bot"] - GAP, MW, "Wage workers at index waves 1–7", ["11,299 person-waves · 2,356 persons"])
b4 = main_box(MX, b3["bot"] - GAP, MW, "Analytic sample (risk set)",
              ["10,780 person-waves · 2,230 persons", "856 exits at t+1 (7.9%)"], lw=1.3)

for (a, b, txt) in [(b1, b2, ["Not a wage worker at index wave t", "(self-employed, unpaid family, unemployed",
                              "or inactive): 23,792 person-waves"]),
                    (b2, b3, ["Index wave 8 (final wave, no follow-up", "wave available): 1,525 person-waves"]),
                    (b3, b4, ["Not interviewed at wave t+1", "(attrition, 4.6%): 519 person-waves"])]:
    arrow(MX, a["bot"], MX, b["top"])
    yb = (a["bot"] + b["top"]) / 2
    e = excl_box(EX0, yb, EW, txt)
    arrow(MX, yb, e["x0"], yb)

# split into development / temporal validation (orthogonal connector)
DX, VX, SW = MX, 131.0, 60.0
ys = b4["bot"] - 4.5
top_sets = ys - 5.5
line(MX, b4["bot"], MX, ys); line(DX, ys, VX, ys)
arrow(DX, ys, DX, top_sets); arrow(VX, ys, VX, top_sets)
dv = main_box(DX, top_sets, SW, "Development set",
              ["index waves 1–4 (2016–2019)", "6,250 person-waves · 2,048 persons", "530 exits (8.5%)"])
vv = main_box(VX, top_sets, SW, "Temporal validation set",
              ["index waves 5–7 (2020–2022)", "4,530 person-waves · 1,783 persons", "326 exits (7.2%)"])
pd_top = vv["bot"] - 7.0
arrow(VX, vv["bot"], VX, pd_top)
pdis = main_box(VX, pd_top, SW, "Person-disjoint subset",
                ["335 person-waves · 54 exits", "(persons never in the development set)"])
note = ("Hyperparameters tuned by 5-fold\ncross-validation grouped by person\n"
        "within the development set; the\nvalidation set was inspected once.")
ax.text(dv["x0"] + 0.5, pd_top - 0.3, note, ha="left", va="top", fontsize=X_FS, color=INK2, linespacing=1.3)
print("lowest box bottom (mm):", round(pdis["bot"], 1))
S.save(fig, "fig1_sample_flow", OUT)
