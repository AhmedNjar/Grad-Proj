#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_theme.py — Shared plot theme for TechPulse Spindle RDO Suite
=================================================================

Provides a clean, light-background theme suitable for academic papers,
thesis figures, and conference presentations.

Usage in any plot function:
    from plot_theme import apply_paper_theme, C, savefig_paper
    apply_paper_theme()
    fig, ax = plt.subplots(...)
    ax.plot(..., color=C.BLUE)
    savefig_paper(fig, "output.png")
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (optimised for white background + print / IEEE / Elsevier)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Colours:
    # Main structural colours
    BG:      str = "#ffffff"       # figure / axes background
    AX_BG:   str = "#f7f7f7"       # axes face (very light gray)
    SPINE:   str = "#aaaaaa"       # axis borders
    GRID:    str = "#dddddd"       # grid lines
    TEXT:    str = "#1a1a2e"       # all text, labels, ticks
    SUBTEXT: str = "#555555"       # secondary text, annotations

    # Data colours  (tested for CMYK print, colorblind-safe)
    BLUE:    str = "#1f6eb5"       # primary   (replaces TEAL on dark)
    RED:     str = "#d62728"       # accent    (replaces CORAL on dark)
    ORANGE:  str = "#e08c1a"       # highlight (replaces GOLD on dark)
    GREEN:   str = "#2a7d4f"       # positive  (replaces MINT on dark)
    PURPLE:  str = "#7b2d8b"       # fifth series
    BROWN:   str = "#8c564b"       # sixth series
    GRAY:    str = "#7f7f7f"       # neutral

    # Semantic aliases kept for backward compat with existing code
    TEAL:    str = "#1f6eb5"       # alias → BLUE
    CORAL:   str = "#d62728"       # alias → RED
    GOLD:    str = "#e08c1a"       # alias → ORANGE
    MINT:    str = "#2a7d4f"       # alias → GREEN
    NAVY:    str = "#1a1a2e"       # alias → TEXT (dark accent)

    def cycle(self):
        """6-colour cycle for multi-series plots."""
        return [self.BLUE, self.RED, self.ORANGE,
                self.GREEN, self.PURPLE, self.BROWN]


C = _Colours()


# ─────────────────────────────────────────────────────────────────────────────
# rcParams
# ─────────────────────────────────────────────────────────────────────────────
PAPER_RC = {
    # --- backgrounds ---
    "figure.facecolor":   C.BG,
    "axes.facecolor":     C.BG,
    "savefig.facecolor":  C.BG,

    # --- spines / ticks ---
    "axes.edgecolor":     C.SPINE,
    "axes.labelcolor":    C.TEXT,
    "xtick.color":        C.TEXT,
    "ytick.color":        C.TEXT,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.titlesize":     11,
    "axes.labelsize":     10,

    # --- text ---
    "text.color":         C.TEXT,
    "font.size":          10,
    "font.family":        "sans-serif",

    # --- grid ---
    "axes.grid":          True,
    "grid.color":         C.GRID,
    "grid.alpha":         0.8,
    "grid.linewidth":     0.7,

    # --- lines ---
    "lines.linewidth":    1.8,
    "patch.linewidth":    0.8,

    # --- legend ---
    "legend.framealpha":  0.92,
    "legend.edgecolor":   C.SPINE,
    "legend.facecolor":   C.BG,
    "legend.labelcolor":  C.TEXT,
    "legend.fontsize":    8.5,

    # --- spines (keep only bottom + left for clean academic look) ---
    "axes.spines.top":    False,
    "axes.spines.right":  False,

    # --- figure ---
    "figure.dpi":         120,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
}


def apply_paper_theme() -> None:
    """
    Apply the light paper theme globally.

    Call once at the top of any plot function:
        apply_paper_theme()
    """
    plt.rcParams.update(PAPER_RC)


def savefig_paper(
    fig:      "mpl.figure.Figure",
    path:     str,
    dpi:      int   = 300,
    bbox:     str   = "tight",
) -> None:
    """
    Save figure with paper-quality settings.

    Parameters
    ----------
    fig   : matplotlib Figure
    path  : output file path (.png recommended for thesis; .pdf for LaTeX)
    dpi   : default 300 for print quality
    bbox  : "tight" trims surrounding whitespace
    """
    fig.savefig(path, dpi=dpi, bbox_inches=bbox, facecolor=C.BG)
    print(f"  Saved → {path}")


def patch_ax(ax) -> None:
    """
    Apply paper theme to a single existing axes.
    Useful when axes were created before apply_paper_theme().
    """
    ax.set_facecolor(C.BG)
    ax.tick_params(colors=C.TEXT)
    ax.title.set_color(C.TEXT)
    ax.xaxis.label.set_color(C.TEXT)
    ax.yaxis.label.set_color(C.TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(C.SPINE)


# ─────────────────────────────────────────────────────────────────────────────
# Dark-theme alias lookup (for in-function color variables)
# ─────────────────────────────────────────────────────────────────────────────
DARK_TO_LIGHT: dict = {
    # figure/axes backgrounds
    "#0d1b2a":  C.BG,
    "#112233":  C.BG,
    "#0a1628":  C.BG,
    "#1a2c3d":  C.BG,
    "#162436":  C.BG,
    # text colours that were white
    "white":    C.TEXT,
    "#ffffff":  C.TEXT,           # white text → dark text
    # data colours → light-bg equivalents
    "#00b4d8":  C.BLUE,           # TEAL → BLUE
    "#06d6a0":  C.GREEN,          # MINT → GREEN
    "#ffd166":  C.ORANGE,         # GOLD → ORANGE
    "#e63946":  C.RED,            # CORAL → RED (same hue, fine on white)
    "#7400b8":  C.PURPLE,         # PURPLE
    "#8d99ae":  C.GRAY,           # GRAY
    "#2d4060":  C.GRID,           # dark grid → light grid
    "#6b7c93":  C.GRAY,           # HOUSING
    "#4a6fa5":  C.BLUE,           # STEEL (shaft colour)
    "#1a3050":  "#e8eef4",        # BORE dark → very light blue
}


def remap(colour: str) -> str:
    """Return light-theme equivalent for a given dark-theme hex colour."""
    return DARK_TO_LIGHT.get(colour.lower(), colour)
