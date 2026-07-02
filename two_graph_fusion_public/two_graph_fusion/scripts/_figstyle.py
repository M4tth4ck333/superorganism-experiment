"""Shared styling for the thesis figures.

Single source of truth for the colourblind-safe (Okabe-Ito) dataset palette and
the clean-axes treatment, so every figure in the thesis matches. Imported by
make_thesis_figures.py and make_roc_pr.py.
"""

from __future__ import annotations

import matplotlib.patheffects as pe

# One muted Okabe-Ito hue per dataset; reused identically across all figures.
DATASET = {
    "Cresci-2015": "#E69F00",   # orange
    "Cresci-2017": "#0072B2",   # blue
    "Twitter-270k": "#009E73",  # green
    "TwiBot-20": "#D55E00",     # vermillion
}

# Neutral text + a couple of accents for non-dataset (semantic) series.
TXT, MUTE = "#2b2b2b", "#6b6b6b"
GRID, SPINE, GATE = "#dcdcdc", "#9a9a9a", "#7a7a7a"
ACCENT = "#CC79A7"        # reddish-purple, for a second semantic series
EDGE = "#1a1a1a"          # marker edge / neutral dark


def halo(lw: float = 2.4):
    """White stroke so labels stay legible over gridlines and markers."""
    return [pe.withStroke(linewidth=lw, foreground="white")]


def clean(ax) -> None:
    """Despine top/right, soften the remaining spines and the grid."""
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
