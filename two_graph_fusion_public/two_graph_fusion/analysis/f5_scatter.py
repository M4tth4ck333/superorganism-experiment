"""F5 plot: ``(p_T, p_B)`` scatter coloured by ground-truth label.

Section 8.4 of the pivot plan flags F5 as "the killer plot" - the
visual centrepiece that makes the fusion claim concrete. We render two
versions:

1. A density-shaded scatter with bots / humans on the same axes.
2. A diagonal-residual histogram (``p_T - p_B``) split by label - this
   is the silhouette of the "inconsistent corner" that variant B is
   supposed to exploit.

Both versions are written as PNGs alongside a tiny JSON summary of the
inconsistency statistics so they can be picked up by RESULTS.md.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Hashable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


def _to_bot_score(p_honest: Mapping[Hashable, float], nodes: list[Hashable]) -> np.ndarray:
    return np.asarray([1.0 - float(p_honest.get(n, 0.5)) for n in nodes])


def render_f5_scatter(
    p_t_honest: Mapping[Hashable, float],
    p_b_honest: Mapping[Hashable, float],
    labels: Mapping[Hashable, str],
    bot_label: str,
    honest_label: str,
    output_prefix: Path,
    title: str = "F5: trust- vs behavioral-view bot score",
) -> dict[str, float]:
    """Render the F5 scatter + residual plots and return summary stats.

    Args:
        p_t_honest: ``p_honest`` from SybilSCAR on the trust graph
            (indexed by user id).
        p_b_honest: ``p_honest`` from SybilSCAR on the behavioral
            graph, or the LR-prior probability of being honest.
        labels: Per-user label (``honest_label`` / ``bot_label``).
        bot_label: Label string for bots.
        honest_label: Label string for humans.
        output_prefix: Files written are ``<prefix>_scatter.png``,
            ``<prefix>_residual.png``, and ``<prefix>_summary.json``.
        title: Plot title.

    Returns:
        Dictionary of inconsistency statistics, also written to
        ``<prefix>_summary.json``.
    """
    common = [u for u in labels if u in p_t_honest and u in p_b_honest]
    if not common:
        raise ValueError("no common users between priors and labels")

    bot_score_t = _to_bot_score(p_t_honest, common)
    bot_score_b = _to_bot_score(p_b_honest, common)
    y = np.asarray([1 if labels[u] == bot_label else 0 for u in common])

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    for cls, marker, color, name in [
        (0, "o", "#3b82f6", honest_label),
        (1, "x", "#ef4444", bot_label),
    ]:
        mask = y == cls
        ax.scatter(
            bot_score_t[mask],
            bot_score_b[mask],
            s=8,
            alpha=0.4,
            marker=marker,
            color=color,
            label=f"{name} (n={mask.sum()})",
        )
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="y = x")
    ax.set_xlabel("Trust-view bot score (1 - p_T)")
    ax.set_ylabel("Behavioral-view bot score (1 - p_B)")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    scatter_path = output_prefix.with_name(output_prefix.name + "_scatter.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    residual = bot_score_t - bot_score_b
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-1.0, 1.0, 51)
    ax.hist(
        residual[y == 0],
        bins=bins,
        alpha=0.6,
        color="#3b82f6",
        label=f"{honest_label} (n={(y == 0).sum()})",
        density=True,
    )
    ax.hist(
        residual[y == 1],
        bins=bins,
        alpha=0.6,
        color="#ef4444",
        label=f"{bot_label} (n={(y == 1).sum()})",
        density=True,
    )
    ax.axvline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("View disagreement (trust bot-score - behavioral bot-score)")
    ax.set_ylabel("Density")
    ax.set_title("F5 residual: view disagreement by class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    residual_path = output_prefix.with_name(output_prefix.name + "_residual.png")
    fig.savefig(residual_path, dpi=150)
    plt.close(fig)

    summary = {
        "n_users": int(len(common)),
        "n_bot": int((y == 1).sum()),
        "n_human": int((y == 0).sum()),
        "mean_residual_human": float(residual[y == 0].mean()),
        "mean_residual_bot": float(residual[y == 1].mean()),
        "std_residual_human": float(residual[y == 0].std(ddof=0)),
        "std_residual_bot": float(residual[y == 1].std(ddof=0)),
        "scatter_path": str(scatter_path),
        "residual_path": str(residual_path),
    }
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    logger.info("wrote %s, %s, %s", scatter_path, residual_path, summary_path)
    return summary
