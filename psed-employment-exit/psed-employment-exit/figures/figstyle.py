"""Shared figure style — top-journal look (Nature/Lancet/BMJ-type guidance).
Single column 88 mm, double column 180 mm; base font 11 pt at print size so text stays legible after
page-width reduction; 600 dpi raster + vector PDF with editable text.

Palette (validated colour-blind-safe, deutan ΔE 10.3, normal ΔE 25.8; dataviz validator 2026-09-22):
  navy #1F4E99 (LightGBM) · vermillion #C8451E (logistic) · teal-green #1B8A6B (random forest).
Reference/neutral marks use warm greys; text ink is near-black #1A1A1A rather than pure black."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
SINGLE, DOUBLE = 88 * MM, 180 * MM
INK, INK2, MUTED = "#1A1A1A", "#4A4A4A", "#8C8C8C"
GRID, SHADE = "#E4E4E4", "#F3F3F3"
OK = {"blue": "#1F4E99", "orange": "#E69F00", "sky": "#6BAED6", "green": "#1B8A6B",
      "yellow": "#F0E442", "verm": "#C8451E", "purple": "#8E5EA2", "black": INK, "grey": MUTED}
MODEL_COL = {"lightgbm": OK["blue"], "random_forest": OK["green"], "logistic": OK["verm"]}
MODEL_LAB = {"lightgbm": "LightGBM", "random_forest": "Random forest", "logistic": "Logistic regression"}
MODEL_MK = {"lightgbm": "o", "random_forest": "D", "logistic": "s"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["FreeSans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5, "axes.labelweight": "medium",
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10, "legend.title_fontsize": 10,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
    "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.major.size": 4, "ytick.major.size": 4, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.pad": 4, "ytick.major.pad": 4, "axes.labelpad": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    # opaque white legend plate: a data line or reference diagonal never shows through legend text
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.facecolor": "white", "legend.edgecolor": "none",
    "legend.borderpad": 0.45, "legend.handlelength": 2.0, "legend.handletextpad": 0.6, "legend.labelspacing": 0.45,
    "legend.borderaxespad": 0.5, "legend.fancybox": False,
    "lines.linewidth": 2.0, "lines.markersize": 6, "lines.markeredgewidth": 1.0, "lines.markeredgecolor": "white",
    "errorbar.capsize": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in PDF
    "savefig.dpi": 600, "figure.dpi": 100, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def panel_label(ax, s, x=-0.14, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom", ha="left", color=INK)


def ref_line(ax, *args, **kw):
    """Recessive reference line (chance diagonal, base rate, null)."""
    kw = {"color": MUTED, "lw": 1.1, "ls": (0, (4, 3)), "zorder": 1, **kw}
    return ax.plot(*args, **kw)


def save(fig, stem, outdir, tiff=False):
    fig.savefig(f"{outdir}/{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{outdir}/{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    if tiff:
        fig.savefig(f"{outdir}/{stem}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
    print("saved", stem)


# feature name → short English label for plots
FEAT = {
    "ca2603": "Monthly income", "aca0403": "Job tenure", "ca1203": "Workplace size",
    "ca1401": "Fixed-term contract", "a0710": "Disability type", "birthy": "Birth year", "age": "Age",
    "f0701": "Acceptance: cannot get along\nwith people because of disability", "ca2902": "Bonus entitlement",
    "ca2603_delta": "Change in income (1 yr)", "gender": "Sex", "f0602": "Self-esteem: good character",
    "ca1201": "Firm size", "ca1301": "Job designated for PWD", "ca2507": "Weekly hours worked",
    "ca2504": "Daily hours", "ca2506": "Days worked last month", "aca0403_delta": "Change in tenure (1 yr)",
    "ca2507_delta": "Change in hours (1 yr)", "ca2901": "Retirement pay", "ca2904": "Paid leave",
    "ca2801": "National pension", "ca2802": "Health insurance", "ca2803": "Employment insurance",
    "ca2804": "Accident insurance", "ca2601": "Pay basis", "ca1701": "Full-/part-time", "ca1101": "Job classification",
    "jobtype": "Job type", "emptype02": "Regular/temporary/daily", "emppme": "Non-regular worker",
    "grade02": "Severe disability", "g0101": "Self-rated health", "g0803": "Depression, past year",
    "g0804": "Happiness (0–10)", "g0801": "Daily stress", "g0802": "Disability stress", "ca2508": "Sick absence last month",
    "ae1508": "Job-search contacts", "ae0401": "Vocational certificate", "n_prior_waves": "Prior waves observed",
    "wage_hist2": "Wage work, prior 2 waves", "job_changed_last": "Changed job since last wave",
    "jobctotal": "Number of jobs held", "ca1205": "PWD employees at workplace", "ca2505": "Overtime frequency",
    "ca2510": "Shift work", "area": "Region", "aarea": "Region (3 groups)", "type04": "Disability type (4)",
    "type06": "Disability type (6)", "ca4103": "Discrimination: disability", "ca4001": "Disability known at work",
    "g0201": "Chronic disease", "g0901": "Needs help in daily life", "ca2202": "Dependent self-employed",
    "ca2301": "Self-assessed non-regular", "ca1501": "Daily-hire work", "ca1302": "Seasonal job",
    "ca2201": "Place of work", "ca1801": "Dispatch/agency", "ca2701": "Unpaid wages", "g0301": "Exercise days/wk",
    "g0401": "Sleep hours", "g0501": "Skipping meals", "g0502": "Regular meals", "g0601": "Smoking", "g0701": "Drinking",
    "g0102": "Health vs last year", "ca2501": "Fixed weekly days", "ca2502": "Days per week", "ca2503": "Fixed daily hours",
    "ca2903": "Overtime pay", "ca2905": "Sick leave", "ca2906": "Parental leave", "ca2907": "Training", "ca2908": "Accident cover",
    "wage_lag1": "Wage work, last wave", "wage_lag2": "Wage work, 2 waves ago",
}
for i in range(1, 11):
    FEAT.setdefault(f"f06{i:02d}", f"Self-esteem item {i}")
for i in range(1, 13):
    FEAT.setdefault(f"f07{i:02d}", f"Disability acceptance item {i}")
for i in range(1, 9):
    FEAT.setdefault(f"ca41{i:02d}", f"Discrimination item {i}")


def flab(f):
    f = f.replace("num__", "").replace("cat__", "")
    return FEAT.get(f, f)
