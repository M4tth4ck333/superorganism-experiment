"""Approximate mutual k-NN via FAISS for large ``G_B`` builds.

At TwiBot-22 scale (hundreds of thousands of qualifying users) dense
``O(n^2)`` cosine k-NN is infeasible. This module uses FAISS
``IndexHNSWFlat`` on L2-normalised feature vectors (inner product ==
cosine similarity) and applies the same mutual-k-NN filter as
:func:`two_graph_fusion.graphs.behavioral_graph.build_mutual_knn_graph`.

Install on the server::

    uv pip install faiss-cpu

For GPU builds use ``faiss-gpu`` instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from two_graph_fusion.graphs.behavioral_graph import (
    BehavioralGraphBuildResult,
    _l2_normalise_rows,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaissKnnConfig:
    """Hyperparameters for the FAISS HNSW index.

    Attributes:
        k: Neighbours per node (mutual filter applied afterward).
        hnsw_m: HNSW ``M`` parameter (connectivity).
        ef_construction: HNSW build-time search depth.
        ef_search: HNSW query-time search depth; increase if recall is low.
        similarity_floor: Drop mutual edges whose cosine similarity is at
            or below this value (same semantics as the dense builder).
    """

    k: int = 10
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 128
    similarity_floor: float = 0.0


def _require_faiss():
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "faiss is required for large-scale G_B construction. "
            "Install with: uv pip install faiss-cpu"
        ) from exc
    return faiss


def build_mutual_knn_graph_faiss(
    features: pd.DataFrame,
    config: FaissKnnConfig | None = None,
) -> BehavioralGraphBuildResult:
    """Build mutual k-NN ``G_B`` with FAISS HNSW (cosine via normalised IP).

    Args:
        features: DataFrame indexed by user id, one row per feature column.
            Rows with any ``NaN`` are dropped.
        config: FAISS / k-NN settings.

    Returns:
        :class:`BehavioralGraphBuildResult`.
    """
    faiss = _require_faiss()
    cfg = config or FaissKnnConfig()
    k = cfg.k
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if features.empty:
        raise ValueError("features frame is empty")

    n_input = len(features)
    sub = features.dropna(how="any")
    n = len(sub)
    if n < k + 1:
        raise ValueError(
            f"Need at least k+1={k + 1} users with non-NaN features; got {n}"
        )

    user_ids = list(sub.index)
    X = sub.to_numpy(dtype=np.float32)
    Xn = _l2_normalise_rows(X.astype(np.float64)).astype(np.float32)

    dim = Xn.shape[1]
    index = faiss.IndexHNSWFlat(dim, cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = cfg.ef_construction
    index.hnsw.efSearch = cfg.ef_search
    index.add(Xn)

    # Query k+1 because the nearest neighbour is often the point itself.
    query_k = min(k + 1, n)
    sims, nbrs = index.search(Xn, query_k)

    # Per-row neighbour sets (excluding self).
    nbr_lists: list[list[int]] = []
    sim_lists: list[list[float]] = []
    for i in range(n):
        row_nbr: list[int] = []
        row_sim: list[float] = []
        for j in range(query_k):
            nb = int(nbrs[i, j])
            if nb == i:
                continue
            row_nbr.append(nb)
            row_sim.append(float(sims[i, j]))
            if len(row_nbr) >= k:
                break
        nbr_lists.append(row_nbr)
        sim_lists.append(row_sim)

    nbr_sets = [set(row) for row in nbr_lists]

    graph = nx.Graph()
    graph.add_nodes_from(user_ids)

    edges_added = 0
    for i in range(n):
        for j in nbr_lists[i]:
            if j <= i:
                continue
            if i not in nbr_sets[j]:
                continue
            # Recover similarity from i's neighbour list.
            try:
                pos = nbr_lists[i].index(j)
                weight = sim_lists[i][pos]
            except ValueError:
                weight = float(np.dot(Xn[i], Xn[j]))
            if weight <= cfg.similarity_floor:
                continue
            graph.add_edge(user_ids[i], user_ids[j], weight=weight)
            edges_added += 1

    logger.info(
        "G_B (FAISS): input_users=%d, used_users=%d, k=%d, mutual_edges=%d",
        n_input,
        n,
        k,
        edges_added,
    )
    return BehavioralGraphBuildResult(
        graph=graph,
        n_input_users=n_input,
        n_used_users=n,
        k=k,
        n_mutual_edges=edges_added,
    )
