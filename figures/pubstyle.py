"""Shared publication style for the PSED manuscript figures (v2, BMC artwork sizes).

Final size: 85 mm (single column) or 170 mm (double column) wide, height <= 225 mm.
Type: Liberation Sans (metric-compatible with Arial), 7-8 pt at final size; panel labels 10 pt bold.
Lines >= 0.5 pt. Vector PDF with TrueType (Type 42) fonts embedded; 600-dpi TIFF (LZW) + 300-dpi PNG preview.

Model palette (dataviz validator, white surface, --pairs all: CVD dE 10.3 deutan, normal-vision dE 21.6,
all >= 3:1 contrast): LightGBM navy #1F4E99 (circles), logistic regression vermillion #C8451E (squares),
random forest teal #1B8A6B (diamonds) - unchanged from the submitted figures so model identity is stable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf"):
    try:
        font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{f}")
    except Exception:
        pass

MM = 1 / 25.4
SINGLE, DOUBLE = 85 * MM, 170 * MM

INK = "#1A1A1A"        # text, axes
INK2 = "#4D4D4D"       # secondary text
MUTED = "#8A8A8A"      # reference lines
GRID = "#E6E6E6"       # hairline grid
SHADE = "#F2F2F2"      # alternate-row band

MODEL_COL = {"lightgbm": "#1F4E99", "random_forest": "#1B8A6B", "logistic": "#C8451E"}
MODEL_LAB = {"lightgbm": "LightGBM", "random_forest": "Random forest", "logistic": "Logistic regression"}
MODEL_MK = {"lightgbm": "o", "random_forest": "D", "logistic": "s"}
MODEL_MS = {"lightgbm": 3.6, "random_forest": 3.2, "logistic": 3.3}   # visually equal marker areas

# one-hue sequential blue (dataviz reference ramp, 100 -> 700), near-surface start for "zero"
SEQ_BLUE = ["#F4F8FD", "#CDE2FB", "#9EC5F4", "#6DA7EC", "#3987E5", "#2A78D6", "#1C5CAB", "#104281", "#0D366B"]
# ordinal blue for three ordered waves (validator --ordinal: monotone L, light end 2.11:1)
ORD_BLUE3 = ["#86B6EF", "#2A78D6", "#104281"]
# diverging blue <-> red with a neutral grey midpoint (dots must stay visible on white)
DIV_BR = ["#2A78D6", "#C3C2B7", "#E34948"]

FS = 7.0       # base text
FS_LAB = 7.5   # axis labels
FS_PANEL = 10  # panel letters
LW_AX = 0.6
LW_DATA = 1.1
LW_REF = 0.75

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "FreeSans", "DejaVu Sans"],
    "font.size": FS, "axes.titlesize": FS_LAB, "axes.labelsize": FS_LAB, "axes.labelweight": "normal",
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS, "legend.title_fontsize": FS,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
    "axes.linewidth": LW_AX, "xtick.major.width": LW_AX, "ytick.major.width": LW_AX,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 3.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "legend.borderpad": 0.3, "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.35, "legend.borderaxespad": 0.4, "legend.columnspacing": 1.2,
    "lines.linewidth": LW_DATA, "lines.markersize": 3.5, "lines.markeredgewidth": 0.5, "lines.markeredgecolor": "white",
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "errorbar.capsize": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 600, "figure.dpi": 150, "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.unicode_minus": True, "mathtext.default": "regular",
})


def panel_label(ax, s, x=None, y=None, fig=None, xy_fig=None):
    """Bold lowercase panel letter at the top-left of the panel (outside the axes)."""
    if xy_fig is not None:
        (fig or ax.figure).text(*xy_fig, s, fontsize=FS_PANEL, fontweight="bold", va="top", ha="left", color=INK)
        return
    ax.text(-0.16 if x is None else x, 1.04 if y is None else y, s, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left", color=INK)


def ref_line(ax, *args, dashed=True, **kw):
    """Recessive reference line (perfect calibration, chance, base rate)."""
    kw = {"color": MUTED, "lw": LW_REF, "ls": (0, (4, 2.5)) if dashed else (0, (1, 1.6)), "zorder": 1, **kw}
    return ax.plot(*args, **kw)


def save(fig, stem, outdir, w_mm=None, h_mm=None):
    """Save at exact final size (no bbox cropping, so the page is exactly w x h mm)."""
    import io
    from PIL import Image
    if w_mm is not None and h_mm is not None:
        fig.set_size_inches(w_mm * MM, h_mm * MM)
    fig.savefig(f"{outdir}/{stem}.pdf")
    for dpi, ext in ((600, "tiff"), (300, "png")):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        im = Image.open(buf).convert("RGB")
        if ext == "tiff":
            im.save(f"{outdir}/{stem}.tiff", compression="tiff_lzw", dpi=(dpi, dpi))
        else:
            im.save(f"{outdir}/{stem}.png", dpi=(dpi, dpi), optimize=True)
    print("saved", stem, "%.1f x %.1f mm" % (fig.get_size_inches()[0] / MM, fig.get_size_inches()[1] / MM))


# feature name -> short English label (identical wording to the submitted figures)
FEAT = {
    "ca2603": "Monthly income", "aca0403": "Job tenure", "ca1203": "Workplace size",
    "ca1401": "Fixed-term contract", "a0710": "Disability type", "birthy": "Birth year", "age": "Age",
    "f0701": "Acceptance: cannot get along\nwith people because of disability", "ca2902": "Bonus entitlement",
    "ca2603_delta": "Change in income (1 yr)", "gender": "Sex", "f0602": "Self-esteem: good character",
    "ca1201": "Firm size", "ca2301": "Self-assessed non-regular", "g0101": "Self-rated health",
}


def flab(f):
    f = f.replace("num__", "").replace("cat__", "")
    return FEAT.get(f, f)
