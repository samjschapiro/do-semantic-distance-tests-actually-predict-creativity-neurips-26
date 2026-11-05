"""Shared matplotlib style, test colours and the specificity-ceiling helper for the paper figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa: F401  (re-exported for the figure scripts)
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Per-test palette (Okabe-Ito, colour-blind safe).
C_DAT = "#0072B2"   # blue
C_CDAT = "#D55E00"  # vermillion (gated CDAT)
C_CNOV = "#009E73"  # bluish green (CDAT novelty)
C_CAPP = "#CC79A7"  # reddish purple (CDAT appropriateness)
C_PACE = "#E69F00"  # orange
C_GREY = "#4d4d4d"


def _panel_max_ceiling(R_list: list[float], v_grid: np.ndarray) -> np.ndarray:
    """Specificity ceiling of the panel benchmark with the largest |R|, where R is the benchmark's
    correlation with the control: ceiling(v) = v sqrt(1 - R^2) + |R| sqrt(1 - v^2)."""
    if not R_list:
        return np.zeros_like(v_grid)
    R = max(R_list, key=abs)
    root1mv2 = np.sqrt(np.clip(1 - v_grid ** 2, 0, None))
    return v_grid * np.sqrt(max(0.0, 1 - R ** 2)) + abs(R) * root1mv2
