"""Snowball / LCC sampling over labeled users in the TwiBot-22 follow graph.

Random sampling of users gives an induced trust subgraph that is
essentially empty (see ``RESULTS.md`` RED-7). To run a meaningful F6
overlap diagnostic - and, downstream, multi-relational SybilSCAR with a
non-trivial ``G_T`` - we need to sample users that are actually
connected in the follow graph.

Sampling strategies
-------------------

``lcc``
    Largest connected component of the labeled follow graph, then a
    balanced sub-sample.

``snowball`` (legacy cross-class)
    Single BFS seeded from the highest-degree user in **each** class.
    This mixes bot and human regions early and tends to produce
    near-chance ``G_T`` homophily (RED-10). Prefer ``snowball_separate``.

``snowball_separate`` (milestone 4 default)
    Two independent BFS runs - one seeded from bots, one from humans.
    Each run may traverse any node but only **collects** users of the
  target class. The final cohort is the union, balanced to
    ``n_per_class``. This avoids the cross-class bridge effect of a
    single mixed BFS while still exploring the full follow graph.

All samplers return a :class:`pandas.DataFrame` indexed by ``user_id``
with ``label`` and ``split`` columns, compatible with the rest of the
pipeline.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from two_graph_fusion.graphs.trust_graph import load_follow_edge_cache

logger = logging.getLogger(__name__)


def labeled_follow_graph(
    follow_cache_csv: Path,
    labels_splits: pd.DataFrame,
) -> nx.Graph:
    """Load the follow-edge cache restricted to labeled users."""
    labeled_ids = set(labels_splits.index)
    g = load_follow_edge_cache(follow_cache_csv, user_filter=labeled_ids)
    g.add_nodes_from(labeled_ids)
    logger.info(
        "labeled follow graph: nodes=%d, edges=%d",
        g.number_of_nodes(),
        g.number_of_edges(),
    )
    return g


def _balanced_sample(
    user_ids: Iterable[str],
    labels_splits: pd.DataFrame,
    n_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Sub-sample ``user_ids`` to ``n_per_class`` per label."""
    pool = labels_splits.loc[list(user_ids)]
    parts = []
    rng = np.random.default_rng(seed)
    for _, group in pool.groupby("label"):
        take = min(n_per_class, len(group))
        idx = rng.choice(len(group), size=take, replace=False)
        parts.append(group.iloc[idx])
    out = pd.concat(parts).sort_index()
    logger.info(
        "balanced sample: %d users (%s)",
        len(out),
        out["label"].value_counts().to_dict(),
    )
    return out


def _highest_degree_seed(
    g: nx.Graph,
    labels_splits: pd.DataFrame,
    label: str,
) -> str:
    """Return the highest-degree user with ``label`` in ``g``."""
    deg_series = pd.Series(dict(g.degree()))
    members = labels_splits.index[labels_splits["label"] == label]
    members_with_deg = deg_series.loc[members.intersection(deg_series.index)]
    if members_with_deg.empty:
        raise ValueError(f"no {label!r} user has a follow edge to seed BFS")
    return str(members_with_deg.idxmax())


def _bfs_collect_class(
    g: nx.Graph,
    labels_splits: pd.DataFrame,
    target_label: str,
    target_count: int,
    seed_user_ids: Iterable[str],
    max_hops: int,
) -> set[str]:
    """BFS through ``g`` and collect up to ``target_count`` users of one class.

    Traversal visits every neighbor (including other classes) so we can
    reach distant same-class nodes through the full follow graph. Only
    nodes whose ``label`` equals ``target_label`` are added to the
    collected set.
    """
    label_map = labels_splits["label"]
    collected: set[str] = set()
    visited: set[str] = set()
    frontier: deque[tuple[str, int]] = deque()

    for s in seed_user_ids:
        if s not in g.nodes() or s in visited:
            continue
        visited.add(s)
        frontier.append((s, 0))
        if label_map.get(s) == target_label:
            collected.add(s)

    while frontier and len(collected) < target_count:
        node, depth = frontier.popleft()
        if depth >= max_hops:
            continue
        for nb in g.neighbors(node):
            if nb in visited:
                continue
            visited.add(nb)
            if label_map.get(nb) == target_label:
                collected.add(nb)
                if len(collected) >= target_count:
                    break
            frontier.append((nb, depth + 1))

    logger.info(
        "BFS collect %s: collected=%d visited=%d (target=%d, max_hops=%d)",
        target_label,
        len(collected),
        len(visited),
        target_count,
        max_hops,
    )
    return collected


def _snowball_bfs_mixed(
    g: nx.Graph,
    seed_user_ids: Iterable[str],
    target_size: int,
    max_hops: int,
) -> set[str]:
    """Legacy cross-class BFS: collect any labeled user until ``target_size``."""
    visited: set[str] = set()
    frontier: deque[tuple[str, int]] = deque()
    for s in seed_user_ids:
        if s in g.nodes() and s not in visited:
            visited.add(s)
            frontier.append((s, 0))

    while frontier and len(visited) < target_size:
        node, depth = frontier.popleft()
        if depth >= max_hops:
            continue
        for nb in g.neighbors(node):
            if nb in visited:
                continue
            visited.add(nb)
            if len(visited) >= target_size:
                break
            frontier.append((nb, depth + 1))
    return visited


def sample_largest_component(
    follow_cache_csv: Path,
    labels_splits: pd.DataFrame,
    n_per_class: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample within the largest connected component of the labeled follow graph."""
    g = labeled_follow_graph(follow_cache_csv, labels_splits)
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    if not components:
        raise ValueError("labeled follow graph has no components")
    lcc = components[0]
    logger.info(
        "largest component: %d users (%.1f%% of labeled population)",
        len(lcc),
        100.0 * len(lcc) / g.number_of_nodes(),
    )
    return _balanced_sample(lcc, labels_splits, n_per_class, seed)


def snowball_sample(
    follow_cache_csv: Path,
    labels_splits: pd.DataFrame,
    target_size: int,
    n_per_class: int,
    seed_user_ids: Iterable[str] | None = None,
    max_hops: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    """Cross-class snowball (legacy). Prefer :func:`snowball_separate_classes`."""
    g = labeled_follow_graph(follow_cache_csv, labels_splits)

    if seed_user_ids is None:
        seed_user_ids = []
        for label in labels_splits["label"].unique():
            seed_user_ids.append(_highest_degree_seed(g, labels_splits, label))
    seed_user_ids = list(seed_user_ids)
    logger.info("snowball (cross-class) seeds: %s", seed_user_ids[:5])

    visited = _snowball_bfs_mixed(g, seed_user_ids, target_size, max_hops)
    logger.info(
        "snowball (cross-class): visited %d users (max_hops=%d)",
        len(visited),
        max_hops,
    )
    return _balanced_sample(visited, labels_splits, n_per_class, seed)


def snowball_separate_classes(
    follow_cache_csv: Path,
    labels_splits: pd.DataFrame,
    target_per_class: int,
    n_per_class: int,
    seed_user_ids: dict[str, list[str]] | None = None,
    max_hops: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    """Snowball each class independently, then union and balance.

    Args:
        follow_cache_csv: Path to the follow-edge cache.
        labels_splits: Labeled user table.
        target_per_class: Stop each class-specific BFS after collecting this
            many users of that class (before the final balanced sub-sample).
        n_per_class: Final balanced sample size per class.
        seed_user_ids: Optional ``{label: [user_id, ...]}`` seeds. When
            omitted, the highest-degree user per class is used.
        max_hops: BFS depth limit per class.
        seed: RNG seed for the balanced sub-sample.

    Returns:
        Balanced DataFrame of users drawn from the union of the two pools.
    """
    g = labeled_follow_graph(follow_cache_csv, labels_splits)
    pools: set[str] = set()

    for label in sorted(labels_splits["label"].unique()):
        if seed_user_ids is not None and label in seed_user_ids:
            seeds = seed_user_ids[label]
        else:
            seeds = [_highest_degree_seed(g, labels_splits, label)]
        pool = _bfs_collect_class(
            g,
            labels_splits,
            target_label=label,
            target_count=target_per_class,
            seed_user_ids=seeds,
            max_hops=max_hops,
        )
        pools |= pool

    logger.info(
        "snowball_separate: union pool=%d users (%s)",
        len(pools),
        labels_splits.loc[list(pools), "label"].value_counts().to_dict()
        if pools
        else {},
    )
    return _balanced_sample(pools, labels_splits, n_per_class, seed)
