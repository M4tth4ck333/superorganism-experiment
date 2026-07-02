"""Behavioral graph ``G_B`` construction via mutual k-NN.

The behavioral graph is the second view in the two-graph fusion algorithm
(section 5 of the pivot plan). Edges connect users whose 5-dimensional
behavioral feature vectors ``f(u)`` are *mutually* close in cosine space:
``(u, v)`` is kept iff ``v`` is in ``u``'s top-``k`` neighbours **and** ``u``
is in ``v``'s top-``k`` neighbours.

Why mutual k-NN
---------------

- One integer hyperparameter (``k``), robust to outliers.
- Standard manifold-learning construction; documented in the pivot plan
  as the recommended default (section 5.1).
- Sparsity is bounded above by ``n * k / 2`` edges, comparable in size to
  typical real-world trust graphs.

Implementation notes
--------------------

- For the milestone-2 subset (~2k users, 5-dim features) exact dense
  cosine k-NN is trivially fast (`O(n^2)` ~ 4.7e6 entries, ~40 MB). A
  FAISS / HNSW backend will become necessary at TwiBot-22 scale
  (~1M users); the public API is structured so swapping the backend
  later does not change call sites.
- Cosine similarities can be negative for z-scored vectors. The pivot
  plan recommends clipping to ``[0, 1]`` for edge weights; we follow
  that convention and treat clipped-to-zero similarities as "no edge"
  for the purpose of mutual-k-NN inclusion. This avoids spurious
  far-side mutual edges where both vectors are diametrically opposite.

References
----------
- Section 5 of ``thesis_writeup/pivot_two_graph_fusion.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BehavioralGraphBuildResult:
    """Output bundle from :func:`build_mutual_knn_graph`.

    Attributes:
        graph: Undirected NetworkX graph. Nodes are user ids (the index of
            the input DataFrame). Edges carry a single attribute
            ``weight`` equal to the cosine similarity in ``[0, 1]``.
        n_input_users: Number of users in the input feature frame.
        n_used_users: Number of users actually placed in the graph (after
            dropping rows with any NaN feature value).
        k: ``k`` used for the mutual k-NN construction.
        n_mutual_edges: Edge count of the returned graph.
    """

    graph: nx.Graph
    n_input_users: int
    n_used_users: int
    k: int
    n_mutual_edges: int


def _l2_normalise_rows(X: np.ndarray) -> np.ndarray:
    """Return ``X`` with each row rescaled to unit L2 norm.

    Zero-norm rows are returned unchanged (they would propagate ``NaN``
    through the cosine matrix otherwise); they will end up with zero
    cosine similarity to every other vector.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return X / norms


def _top_k_neighbours(sim: np.ndarray, k: int) -> np.ndarray:
    """Return the ``(n, k)`` indices of each row's top-``k`` columns.

    Uses :func:`numpy.argpartition` for ``O(n)`` per row, then a stable
    sort within the partition so the returned list is in descending
    similarity order.
    """
    n = sim.shape[0]
    if k >= n:
        order = np.argsort(-sim, axis=1)
        return order[:, :n]
    part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    # Stable in-row sort by similarity descending.
    row_idx = np.arange(n)[:, None]
    part_sims = sim[row_idx, part]
    order = np.argsort(-part_sims, axis=1)
    return np.take_along_axis(part, order, axis=1)


def build_mutual_knn_graph(
    features: pd.DataFrame,
    k: int = 10,
    similarity_floor: float = 0.0,
    metric: str = "cosine",
    distance_ceiling: float | None = None,
) -> BehavioralGraphBuildResult:
    """Build the behavioral graph ``G_B`` via mutual k-NN.

    Args:
        features: DataFrame indexed by user id. Each row is the
            z-scored behavioral feature vector ``f(u)``. Rows with any
            ``NaN`` value are silently dropped before graph construction.
        k: Number of nearest neighbours each node nominates. Recommended
            default ``10`` (pivot plan section 5.1).
        similarity_floor: For ``metric='cosine'``: cosine similarities at
            or below this value are dropped even if both endpoints
            mutually nominate each other. Default ``0.0`` matches the
            pivot plan's "clip to [0, 1]" prescription. For
            ``metric='euclidean'`` this argument is ignored - use
            ``distance_ceiling`` instead.
        metric: ``"cosine"`` (default; matches pivot plan section 5.2)
            or ``"euclidean"`` (RED-11 fallback for 5-D z-scored
            features whose cosines all live above 0.95 and so leave the
            ``similarity_floor`` knob inert).
        distance_ceiling: For ``metric='euclidean'`` only. Edges with
            Euclidean distance greater than this value are dropped.

    Returns:
        :class:`BehavioralGraphBuildResult`. Edge attribute ``weight``
        is always a non-negative similarity in ``[0, 1]``: cosine
        for ``metric='cosine'``, ``exp(-distance / median_distance)``
        for ``metric='euclidean'`` (so larger means "more similar").

    Raises:
        ValueError: If ``k < 1``, fewer than ``k + 1`` users remain
            after dropping NaN rows, the feature frame is empty, or
            ``metric`` is unsupported.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if features.empty:
        raise ValueError("features frame is empty")
    if metric not in {"cosine", "euclidean"}:
        raise ValueError(f"unknown metric {metric!r}")

    n_input = len(features)
    sub = features.dropna(how="any")
    n = len(sub)
    if n < k + 1:
        raise ValueError(
            f"Need at least k+1={k + 1} users with non-NaN features; got {n}"
        )

    user_ids = list(sub.index)
    X = sub.to_numpy(dtype=np.float64)

    if metric == "cosine":
        Xn = _l2_normalise_rows(X)
        score = Xn @ Xn.T  # similarity; higher == more similar
        np.fill_diagonal(score, -np.inf)
        topk_idx = _top_k_neighbours(score, k)
    else:  # euclidean
        # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v
        sq = (X * X).sum(axis=1)
        dist2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
        np.maximum(dist2, 0.0, out=dist2)
        dist = np.sqrt(dist2)
        np.fill_diagonal(dist, np.inf)
        # k smallest distances per row.
        score = -dist  # negate so that "higher" still means "more similar"
        topk_idx = _top_k_neighbours(score, k)

    nbrs: list[set[int]] = [set(topk_idx[i].tolist()) for i in range(n)]

    graph = nx.Graph()
    graph.add_nodes_from(user_ids)

    edges_added = 0
    if metric == "cosine":
        for i in range(n):
            for j in nbrs[i]:
                if j <= i:
                    continue
                if i in nbrs[j]:
                    weight = float(score[i, j])
                    if weight <= similarity_floor:
                        continue
                    graph.add_edge(user_ids[i], user_ids[j], weight=weight)
                    edges_added += 1
    else:  # euclidean
        # Use median pairwise mutual-k-NN distance as the kernel bandwidth so
        # edge weights live in (0, 1].
        mutual_dists: list[float] = []
        for i in range(n):
            for j in nbrs[i]:
                if j > i and i in nbrs[j]:
                    mutual_dists.append(float(dist[i, j]))
        bandwidth = float(np.median(mutual_dists)) if mutual_dists else 1.0
        if bandwidth <= 0.0:
            bandwidth = 1.0
        for i in range(n):
            for j in nbrs[i]:
                if j <= i:
                    continue
                if i in nbrs[j]:
                    d = float(dist[i, j])
                    if distance_ceiling is not None and d > distance_ceiling:
                        continue
                    weight = float(np.exp(-d / bandwidth))
                    graph.add_edge(user_ids[i], user_ids[j], weight=weight)
                    edges_added += 1

    logger.info(
        "G_B: input_users=%d, used_users=%d, k=%d, metric=%s, mutual_edges=%d",
        n_input,
        n,
        k,
        metric,
        edges_added,
    )
    return BehavioralGraphBuildResult(
        graph=graph,
        n_input_users=n_input,
        n_used_users=n,
        k=k,
        n_mutual_edges=edges_added,
    )


# Above this user count we switch to FAISS HNSW (section 5.2 of the pivot plan).
_FAISS_THRESHOLD: int = 10_000


def build_mutual_knn_graph_auto(
    features: pd.DataFrame,
    k: int = 10,
    similarity_floor: float = 0.0,
    metric: str = "cosine",
    distance_ceiling: float | None = None,
    faiss_threshold: int = _FAISS_THRESHOLD,
    force_dense: bool = False,
) -> BehavioralGraphBuildResult:
    """Build ``G_B``, using FAISS when ``n`` exceeds ``faiss_threshold``.

    For ``n <= faiss_threshold`` delegates to :func:`build_mutual_knn_graph`.
    For larger ``n`` uses :func:`two_graph_fusion.graphs.faiss_knn.build_mutual_knn_graph_faiss`
    (cosine / inner product only; ``metric`` must be ``"cosine"``).

    Args:
        features: Z-scored feature matrix indexed by user id.
        k: Mutual k-NN parameter.
        similarity_floor: Cosine similarity floor (dense and FAISS).
        metric: ``"cosine"`` or ``"euclidean"`` (dense only).
        distance_ceiling: Euclidean ceiling (dense only).
        faiss_threshold: Switch to FAISS above this many non-NaN users.
        force_dense: If True, always use the dense builder (debug only).

    Returns:
        :class:`BehavioralGraphBuildResult`.
    """
    sub = features.dropna(how="any")
    n = len(sub)
    if force_dense or n <= faiss_threshold:
        return build_mutual_knn_graph(
            features,
            k=k,
            similarity_floor=similarity_floor,
            metric=metric,
            distance_ceiling=distance_ceiling,
        )
    if metric != "cosine":
        logger.warning(
            "FAISS backend only supports cosine; got metric=%r, using cosine",
            metric,
        )
    from two_graph_fusion.graphs.faiss_knn import (
        FaissKnnConfig,
        build_mutual_knn_graph_faiss,
    )

    return build_mutual_knn_graph_faiss(
        features,
        config=FaissKnnConfig(k=k, similarity_floor=similarity_floor),
    )


def feature_matrix_for_qualifying_users(
    zscored: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return a clean feature frame ready to feed into G_B construction.

    Drops any non-numeric columns (e.g. ``label`` / ``split``) and any
    rows with missing values.

    Args:
        zscored: DataFrame produced by
            :func:`two_graph_fusion.features.compute_behavioral_features`
            (the ``zscored`` attribute) optionally annotated with label /
            split columns by the driver script.
        feature_columns: Optional explicit list of feature columns to
            keep. If ``None``, every numeric column is kept.

    Returns:
        A clean ``DataFrame`` indexed by user id.
    """
    if feature_columns is None:
        feature_columns = [
            c for c in zscored.columns if pd.api.types.is_numeric_dtype(zscored[c])
        ]
    sub = zscored[feature_columns].dropna(how="any")
    return sub
