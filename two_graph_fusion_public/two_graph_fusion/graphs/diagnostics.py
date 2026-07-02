"""F6 overlap diagnostic and standard graph summaries.

This module implements the F6 experiment defined in section 8.4 of the
pivot plan. F6 is the **go/no-go gate** for the whole fusion story
(section 12.2): if the behavioral graph ``G_B`` and the trust graph
``G_T`` overlap heavily, the two views are not independent and fusion
adds little signal over either view alone.

Three overlap statistics are reported:

1. **Edge Jaccard**: ``|E_T cap E_B| / |E_T cup E_B|``. Pass criterion
   (section 8.7): ``< 0.30``.
2. **Triangle Jaccard**: same Jaccard formula applied to the multisets
   of triangles in each graph. More demanding than edge Jaccard because
   it measures higher-order overlap.
3. **Spectral overlap**: cosine similarity between the leading
   eigenvectors of the (normalised) graph Laplacians. We report the
   mean and max pairwise cosine for the top ``k`` non-trivial
   eigenpairs.

The module also exposes two utility summaries that the pivot plan calls
out as soft gates:

- :func:`largest_connected_component_fraction`: section 10.1 asks for
  ``<= 80%`` of nodes in the largest CC of ``G_B`` so propagation does
  not wash out.
- :func:`degree_summary`: simple percentile breakdown of the degree
  distribution, useful for spotting hub-only artefacts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Hashable, Iterable

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class F6Report:
    """Bundle of F6 overlap statistics between ``G_B`` and ``G_T``.

    Attributes:
        n_common_nodes: Size of ``V(G_B) cap V(G_T)``.
        n_edges_b: Edge count of ``G_B`` restricted to the common nodes.
        n_edges_t: Edge count of ``G_T`` restricted to the common nodes.
        n_common_edges: ``|E_B cap E_T|`` after restriction.
        edge_jaccard: Edge Jaccard of the restricted graphs.
        n_triangles_b: Triangle count of restricted ``G_B``.
        n_triangles_t: Triangle count of restricted ``G_T``.
        n_common_triangles: Triangles common to both restricted graphs.
        triangle_jaccard: Triangle Jaccard.
        spectral_mean_cosine: Mean absolute cosine between the top-``k``
            eigenvectors of the two normalised Laplacians.
        spectral_max_cosine: Max absolute cosine across the same
            top-``k`` eigenpairs.
        spectral_k: Number of eigenpairs actually used (the request may
            be reduced when the smaller of the two graphs has too few
            non-trivial eigenvalues).
    """

    n_common_nodes: int
    n_edges_b: int
    n_edges_t: int
    n_common_edges: int
    edge_jaccard: float
    n_triangles_b: int
    n_triangles_t: int
    n_common_triangles: int
    triangle_jaccard: float
    spectral_mean_cosine: float
    spectral_max_cosine: float
    spectral_k: int


def _canonical_edges(graph: nx.Graph, node_filter: set[Hashable]) -> set[tuple]:
    """Return the set of edges in ``graph`` restricted to ``node_filter``.

    Edges are stored as sorted-tuple ``(min(u, v), max(u, v))`` so the
    set is order-insensitive.
    """
    out: set[tuple] = set()
    for u, v in graph.edges():
        if u in node_filter and v in node_filter:
            a, b = (u, v) if u <= v else (v, u)
            out.add((a, b))
    return out


def edge_jaccard(g_b: nx.Graph, g_t: nx.Graph) -> tuple[float, int, int, int]:
    """Compute the edge Jaccard between two graphs.

    Returns ``(jaccard, |E_B|, |E_T|, |E_B cap E_T|)`` evaluated on the
    intersection of their node sets.
    """
    common_nodes = set(g_b.nodes()) & set(g_t.nodes())
    edges_b = _canonical_edges(g_b, common_nodes)
    edges_t = _canonical_edges(g_t, common_nodes)
    cap = edges_b & edges_t
    cup = edges_b | edges_t
    j = float(len(cap) / len(cup)) if cup else 0.0
    return j, len(edges_b), len(edges_t), len(cap)


def _triangles(graph: nx.Graph, node_filter: set[Hashable]) -> set[tuple]:
    """Enumerate triangles as canonical sorted triples.

    Uses the standard "for each edge, intersect neighbour sets" idiom.
    Each triangle is emitted once.
    """
    tris: set[tuple] = set()
    nbr = {n: set(graph.neighbors(n)) & node_filter for n in node_filter}
    for u in node_filter:
        u_nbr = nbr[u]
        for v in u_nbr:
            if v <= u:
                continue
            common = u_nbr & nbr[v]
            for w in common:
                if w <= v:
                    continue
                tris.add((u, v, w))
    return tris


def triangle_jaccard(g_b: nx.Graph, g_t: nx.Graph) -> tuple[float, int, int, int]:
    """Compute the triangle Jaccard between two graphs.

    Returns ``(jaccard, |T_B|, |T_T|, |T_B cap T_T|)`` on the intersection
    of node sets. Note that triangle enumeration is ``O(sum_v d_v^2)``;
    for our G_B (mutual k-NN with k=10) and G_T (typically sparse follow
    graph on a small subset) this is fast.
    """
    common_nodes = set(g_b.nodes()) & set(g_t.nodes())
    tris_b = _triangles(g_b, common_nodes)
    tris_t = _triangles(g_t, common_nodes)
    cap = tris_b & tris_t
    cup = tris_b | tris_t
    j = float(len(cap) / len(cup)) if cup else 0.0
    return j, len(tris_b), len(tris_t), len(cap)


def _normalised_laplacian(graph: nx.Graph, node_order: list[Hashable]) -> sparse.csr_matrix:
    """Return the symmetric normalised Laplacian I - D^{-1/2} A D^{-1/2}.

    Isolated nodes contribute a 1.0 on the diagonal (zero off-diagonal)
    so the matrix stays well-formed.
    """
    n = len(node_order)
    idx = {nd: i for i, nd in enumerate(node_order)}
    rows: list[int] = []
    cols: list[int] = []
    for u, v in graph.edges():
        if u not in idx or v not in idx:
            continue
        i, j = idx[u], idx[v]
        if i == j:
            continue
        rows.append(i)
        cols.append(j)
        rows.append(j)
        cols.append(i)
    data = np.ones(len(rows), dtype=np.float64)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    L = sparse.eye(n, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt
    return L.tocsr()


def spectral_overlap(
    g_b: nx.Graph,
    g_t: nx.Graph,
    k: int = 10,
) -> tuple[float, float, int]:
    """Cosine similarity between the leading Laplacian eigenvectors.

    For each of the two graphs (restricted to the common node set with a
    common node ordering), we compute the bottom-``k+1`` eigenpairs of
    the symmetric normalised Laplacian. The 0-eigenvalue eigenvectors
    (one per connected component) carry no community structure; we
    discard them by skipping eigenvalues below ``1e-9``.

    Returns ``(mean_abs_cosine, max_abs_cosine, k_used)``. ``k_used``
    may be less than ``k`` if either graph has too few non-trivial
    eigenpairs (e.g. when ``G_T`` is nearly empty).
    """
    common_nodes = sorted(set(g_b.nodes()) & set(g_t.nodes()))
    if not common_nodes:
        return 0.0, 0.0, 0
    L_b = _normalised_laplacian(g_b, common_nodes)
    L_t = _normalised_laplacian(g_t, common_nodes)

    n = len(common_nodes)
    req = min(k + max(_count_components(g_b, common_nodes), _count_components(g_t, common_nodes)), n - 1)
    req = max(req, 2)

    def _eig(L: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        # eigsh with ``which='SM'`` is unstable; use sigma=0 shift-invert.
        try:
            vals, vecs = eigsh(L, k=req, sigma=0.0, which="LM")
        except Exception as exc:  # pragma: no cover
            logger.warning("eigsh shift-invert failed (%s); falling back to dense", exc)
            vals_full, vecs_full = np.linalg.eigh(L.toarray())
            vals = vals_full[:req]
            vecs = vecs_full[:, :req]
        order = np.argsort(vals)
        return vals[order], vecs[:, order]

    vals_b, vecs_b = _eig(L_b)
    vals_t, vecs_t = _eig(L_t)

    keep_b = vals_b > 1e-9
    keep_t = vals_t > 1e-9
    vecs_b = vecs_b[:, keep_b]
    vecs_t = vecs_t[:, keep_t]

    k_used = min(k, vecs_b.shape[1], vecs_t.shape[1])
    if k_used == 0:
        return 0.0, 0.0, 0

    vecs_b = vecs_b[:, :k_used]
    vecs_t = vecs_t[:, :k_used]

    # Pairwise cosine matrix; eigenvectors are L2-normalised by eigsh.
    cos_mat = np.abs(vecs_b.T @ vecs_t)
    mean_cos = float(cos_mat.mean())
    max_cos = float(cos_mat.max())
    return mean_cos, max_cos, k_used


def _count_components(graph: nx.Graph, node_filter: set[Hashable]) -> int:
    sub = graph.subgraph(node_filter)
    return nx.number_connected_components(sub)


def f6_report(
    g_b: nx.Graph,
    g_t: nx.Graph,
    spectral_k: int = 10,
    max_spectral_nodes: int = 50_000,
) -> F6Report:
    """Run the full F6 diagnostic and return a :class:`F6Report`.

    Spectral overlap uses sparse ``eigsh`` on normalised Laplacians and
  is skipped when the common node count exceeds ``max_spectral_nodes``
  (full TwiBot-22 cohorts are hundreds of thousands of nodes).
    """
    common_nodes = set(g_b.nodes()) & set(g_t.nodes())
    j_e, n_e_b, n_e_t, n_e_cap = edge_jaccard(g_b, g_t)
    j_t, n_t_b, n_t_t, n_t_cap = triangle_jaccard(g_b, g_t)
    if len(common_nodes) <= max_spectral_nodes:
        mean_cos, max_cos, k_used = spectral_overlap(g_b, g_t, k=spectral_k)
    else:
        logger.info(
            "F6: skip spectral overlap (%d nodes > max %d)",
            len(common_nodes),
            max_spectral_nodes,
        )
        mean_cos, max_cos, k_used = math.nan, math.nan, 0
    return F6Report(
        n_common_nodes=len(common_nodes),
        n_edges_b=n_e_b,
        n_edges_t=n_e_t,
        n_common_edges=n_e_cap,
        edge_jaccard=j_e,
        n_triangles_b=n_t_b,
        n_triangles_t=n_t_t,
        n_common_triangles=n_t_cap,
        triangle_jaccard=j_t,
        spectral_mean_cosine=mean_cos,
        spectral_max_cosine=max_cos,
        spectral_k=k_used,
    )


def largest_connected_component_fraction(graph: nx.Graph) -> float:
    """Return ``|LCC| / |V|``, or ``NaN`` for an empty graph.

    Section 10.1 of the pivot plan asks for this to be ``<= 0.8`` on
    ``G_B`` so propagation has structure to exploit.
    """
    if graph.number_of_nodes() == 0:
        return math.nan
    largest = max((len(c) for c in nx.connected_components(graph)), default=0)
    return float(largest / graph.number_of_nodes())


def degree_summary(graph: nx.Graph) -> dict[str, float]:
    """Return ``min/median/mean/max/p90/p99`` degree statistics."""
    degs = np.asarray([d for _, d in graph.degree()], dtype=np.float64)
    if degs.size == 0:
        nan = float("nan")
        return {k: nan for k in ("min", "median", "mean", "max", "p90", "p99")}
    return {
        "min": float(degs.min()),
        "median": float(np.median(degs)),
        "mean": float(degs.mean()),
        "max": float(degs.max()),
        "p90": float(np.percentile(degs, 90)),
        "p99": float(np.percentile(degs, 99)),
        "n_isolated": int((degs == 0).sum()),
    }


def graph_summary(graph: nx.Graph) -> dict[str, object]:
    """Compact text summary of a graph for logging / report writing."""
    n_components = nx.number_connected_components(graph)
    return {
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "n_components": int(n_components),
        "lcc_fraction": largest_connected_component_fraction(graph),
        "degree": degree_summary(graph),
    }


def acceptance_verdict(report: F6Report) -> dict[str, str]:
    """Map the F6 numbers to the section-8.7 acceptance criteria.

    Returns a dict whose values are ``"PASS"`` / ``"FAIL"`` strings so the
    final report doc can lift them directly.
    """
    return {
        "edge_jaccard_lt_0.30": "PASS"
        if (report.n_edges_b + report.n_edges_t == 0 or report.edge_jaccard < 0.30)
        else "FAIL"
    }


def restrict_to_common_nodes(g_b: nx.Graph, g_t: nx.Graph) -> tuple[nx.Graph, nx.Graph]:
    """Return both graphs subgraphed onto ``V(G_B) cap V(G_T)``.

    Useful for downstream propagation runs that need both graphs over the
    same vertex set with identical ordering. The returned graphs are
    fresh copies (not subgraph views) so callers can freely mutate them.
    """
    common = sorted(set(g_b.nodes()) & set(g_t.nodes()))
    return g_b.subgraph(common).copy(), g_t.subgraph(common).copy()


def _safe_jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    cap = sa & sb
    cup = sa | sb
    return float(len(cap) / len(cup)) if cup else 0.0
