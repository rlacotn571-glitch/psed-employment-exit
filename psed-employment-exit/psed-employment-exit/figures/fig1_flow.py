"""Fig 1 — sample flow diagram (PSED-II, waves 1-8). Okabe-Ito palette, 300 dpi."""
import os
from pathlib import Path
import figstyle as fs_
from figstyle import plt, OK, DOUBLE, MM, save
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, ORANGE, GREY, SKY = OK["blue"], "#C8451E", "#3A3A3A", "#2E86C1"
OUTD = os.environ.get("PSED_RESULTS", "results")

W, H = 12.8, 13.0
fig, ax = plt.subplots(figsize=(DOUBLE, 200 * MM))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.grid(False)


def box(x, y, w, h, title, lines, ec=BLUE, lw=1.6, fs=9.6, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw))
    n = len(lines)
    total = 0.36 + 0.3 * n
    top = y + h / 2 + total / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=fs + 1.0, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, top - 0.38 - i * 0.3, ln, ha="center", va="top", fontsize=fs, color=GREY)


def side(x, y, w, h, lines, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#F4F4F4", ec=GREY, lw=0.9))
    n = len(lines)
    top = y + h / 2 + (0.28 * n) / 2
    for i, ln in enumerate(lines):
        ax.text(x + 0.15, top - i * 0.28, ln, ha="left", va="top", fontsize=fs, color=GREY)


def arrow(x0, y0, x1, y1, color=BLUE):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=1.6, shrinkA=0, shrinkB=0))


cx, cw = 0.4, 6.0          # main column
sx, sw = 6.7, 5.9          # side column
mid = cx + cw / 2

# level 1
box(cx, 11.5, cw, 1.2, "PSED-II, waves 1–8 (2016–2023)",
    ["36,616 person-waves · 4,577 persons"])
arrow(mid, 11.5, mid, 10.55)
side(sx, 10.55, sw, 0.95, ["Not a wage worker at index wave t",
                           "(self-employed, unpaid family, unemployed",
                           "or inactive): 23,792 person-waves"])
arrow(cx + cw, 11.0, sx, 11.0, color=GREY)

# level 2
box(cx, 9.35, cw, 1.2, "Wage workers at index wave t",
    ["12,824 person-waves · 2,393 persons"])
arrow(mid, 9.35, mid, 8.4)
side(sx, 8.45, sw, 0.8, ["No observation at any later wave",
                         "(attrition; wave 8 index): 1,525 person-waves"])
arrow(cx + cw, 8.85, sx, 8.85, color=GREY)

# level 3
box(cx, 7.2, cw, 1.2, "Observed at a later wave",
    ["11,299 person-waves · 2,356 persons"])
arrow(mid, 7.2, mid, 6.25)
side(sx, 6.3, sw, 0.8, ["Next observation more than one wave later",
                        "or status unknown: 519 person-waves"])
arrow(cx + cw, 6.7, sx, 6.7, color=GREY)

# level 4
box(cx, 4.85, cw, 1.4, "Analytic sample (risk set)",
    ["10,780 person-waves · 2,230 persons", "856 exits at t+1 (7.9%)"], lw=2.0)

# split
bw = 5.6
arrow(cx + 1.3, 4.85, 0.3 + bw / 2, 3.95)
arrow(cx + cw - 1.3, 4.85, 6.9 + bw / 2, 3.95)
box(0.3, 2.35, bw, 1.6, "Development set",
    ["index waves 1–4 (2016–2019)", "6,250 person-waves · 2,048 persons", "530 exits (8.5%)"],
    ec=ORANGE, lw=1.8)
box(6.9, 2.35, bw, 1.6, "Temporal validation set",
    ["index waves 5–7 (2020–2022)", "4,530 person-waves · 1,783 persons", "326 exits (7.2%)"],
    ec=SKY, lw=1.8)

arrow(6.9 + bw / 2, 2.35, 6.9 + bw / 2, 1.6, color=SKY)
box(6.9, 0.4, bw, 1.2, "Person-disjoint subset",
    ["335 person-waves · 54 exits", "(persons never in the development set)"], ec=SKY, lw=1.4, fs=9.4)

ax.text(0.35, 1.55, "Hyperparameters tuned by 5-fold\ncross-validation grouped by person\nwithin the development set; the\nvalidation set was inspected once.",
        ha="left", va="top", fontsize=9, color=GREY)

save(fig, "fig1_sample_flow", OUTD, tiff=True)
