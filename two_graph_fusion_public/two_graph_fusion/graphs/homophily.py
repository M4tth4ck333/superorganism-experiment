"""Edge homophily statistics for labeled graphs.

SybilSCAR-style propagation assumes that neighbors tend to share the same
label. :func:`edge_homophily` measures how well a graph satisfies that
assumption and compares against the class-frequency chance baseline.

Milestone 3 on cross-class snowball sampling showed ``G_T`` homophily
lift near zero (RED-10). Milestone 4 uses this module as the go/no-go
check when trying alternative cohort constructions (separate-class
snowball, official train/test sampling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class HomophilyReport:
    """Edge homophily for a single labeled graph."""

    n_edges_labeled: int
    n_same_label: int
    n_cross_label: int
    n_edges_unlabeled: int
    homophily: float
    chance_baseline: float
    homophily_lift: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_edges_labeled": self.n_edges_labeled,
            "n_same_label": self.n_same_label,
            "n_cross_label": self.n_cross_label,
            "n_edges_unlabeled": self.n_edges_unlabeled,
            "homophily": self.homophily,
            "chance_baseline": self.chance_baseline,
            "homophily_lift": self.homophily_lift,
        }


def chance_baseline(label: dict[Hashable, str]) -> float:
    """Return ``sum_c p_c^2`` for the empirical class distribution."""
    if not label:
        return float("nan")
    proportions = pd.Series(list(label.values())).value_counts(normalize=True)
    return float((proportions**2).sum())


def edge_homophily(
    graph: nx.Graph,
    label: dict[Hashable, str],
) -> HomophilyReport:
    """Compute edge homophily and lift over chance for ``graph``.

    Only edges whose endpoints both appear in ``label`` are counted.
    """
    nodes = set(label.keys())
    same = 0
    cross = 0
    unlabeled = 0
    for u, v in graph.edges():
        if u not in nodes or v not in nodes:
            unlabeled += 1
            continue
        if label[u] == label[v]:
            same += 1
        else:
            cross += 1
    total = same + cross
    h = float(same / total) if total else float("nan")
    chance = chance_baseline(label)
    lift = h - chance if total else float("nan")
    return HomophilyReport(
        n_edges_labeled=total,
        n_same_label=same,
        n_cross_label=cross,
        n_edges_unlabeled=unlabeled,
        homophily=h,
        chance_baseline=chance,
        homophily_lift=lift,
    )


def homophily_from_dataframe(
    graph: nx.Graph,
    labels: pd.DataFrame,
    label_col: str = "label",
) -> HomophilyReport:
    """Convenience wrapper when labels live in a user-indexed DataFrame."""
    label_map = labels[label_col].to_dict()
    return edge_homophily(graph, label_map)


def format_homophily_row(name: str, report: HomophilyReport) -> str:
    """One-line summary suitable for logging or CLI tables."""
    return (
        f"{name:24s}  edges={report.n_edges_labeled:6d}  "
        f"h={report.homophily:.4f}  chance={report.chance_baseline:.4f}  "
        f"lift={report.homophily_lift:+.4f}"
    )
