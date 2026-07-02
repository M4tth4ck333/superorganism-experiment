"""Unit tests for the graph construction and diagnostics modules.

Run with::

    python -m unittest two_graph_fusion.tests.test_graphs
"""

from __future__ import annotations

import unittest

import networkx as nx
import numpy as np
import pandas as pd

from two_graph_fusion.graphs.behavioral_graph import (
    build_mutual_knn_graph,
    feature_matrix_for_qualifying_users,
)
from two_graph_fusion.graphs.diagnostics import (
    degree_summary,
    edge_jaccard,
    f6_report,
    largest_connected_component_fraction,
    spectral_overlap,
    triangle_jaccard,
)


def _square_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    return g


def _triangle_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    return g


class TestBehavioralGraph(unittest.TestCase):
    def test_two_clusters_yield_mutual_intra_edges(self) -> None:
        # Two well-separated clusters in 2D. Mutual k-NN should connect
        # within each cluster and (almost) never across.
        rng = np.random.default_rng(0)
        cluster_a = rng.normal(loc=(0.0, 0.0), scale=0.05, size=(8, 2))
        cluster_b = rng.normal(loc=(5.0, 5.0), scale=0.05, size=(8, 2))
        X = np.vstack([cluster_a, cluster_b])
        users = [f"a{i}" for i in range(8)] + [f"b{i}" for i in range(8)]
        df = pd.DataFrame(X, index=users, columns=["x", "y"])
        res = build_mutual_knn_graph(df, k=3)
        # Most edges are intra-cluster.
        intra = sum(
            1
            for u, v in res.graph.edges()
            if u[0] == v[0]
        )
        self.assertGreaterEqual(intra, res.graph.number_of_edges() - 1)

    def test_too_few_users_raises(self) -> None:
        df = pd.DataFrame([[1.0, 2.0]], index=["u1"], columns=["x", "y"])
        with self.assertRaises(ValueError):
            build_mutual_knn_graph(df, k=3)

    def test_drops_nan_rows(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(10, 3))
        users = [f"u{i}" for i in range(10)]
        df = pd.DataFrame(X, index=users, columns=["a", "b", "c"])
        df.loc["u0", "a"] = np.nan
        res = build_mutual_knn_graph(df, k=2)
        self.assertEqual(res.n_used_users, 9)
        self.assertNotIn("u0", res.graph.nodes())

    def test_similarity_floor_filters_edges(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(size=(20, 4))
        users = [f"u{i}" for i in range(20)]
        df = pd.DataFrame(X, index=users)
        loose = build_mutual_knn_graph(df, k=5, similarity_floor=-1.0)
        strict = build_mutual_knn_graph(df, k=5, similarity_floor=0.95)
        self.assertGreaterEqual(loose.n_mutual_edges, strict.n_mutual_edges)

    def test_feature_matrix_helper(self) -> None:
        df = pd.DataFrame(
            {
                "B_G_z": [0.1, 0.2, np.nan],
                "M_z": [0.0, 1.0, 2.0],
                "label": ["a", "b", "c"],
            },
            index=["u1", "u2", "u3"],
        )
        out = feature_matrix_for_qualifying_users(df, feature_columns=["B_G_z", "M_z"])
        self.assertListEqual(list(out.columns), ["B_G_z", "M_z"])
        self.assertEqual(len(out), 2)


class TestDiagnostics(unittest.TestCase):
    def test_edge_jaccard_identical(self) -> None:
        g = _square_graph()
        j, n_a, n_b, n_cap = edge_jaccard(g, g.copy())
        self.assertAlmostEqual(j, 1.0)
        self.assertEqual(n_a, n_b)
        self.assertEqual(n_cap, n_a)

    def test_edge_jaccard_disjoint(self) -> None:
        a = nx.Graph()
        a.add_edges_from([("x", "y"), ("y", "z")])
        b = nx.Graph()
        b.add_edges_from([("x", "z")])
        b.add_nodes_from(["x", "y", "z"])
        a.add_nodes_from(["x", "y", "z"])
        j, _, _, n_cap = edge_jaccard(a, b)
        self.assertEqual(n_cap, 0)
        self.assertAlmostEqual(j, 0.0)

    def test_triangle_jaccard(self) -> None:
        a = _triangle_graph()
        b = _triangle_graph()
        j, n_a, n_b, n_cap = triangle_jaccard(a, b)
        self.assertEqual(n_a, 1)
        self.assertEqual(n_b, 1)
        self.assertEqual(n_cap, 1)
        self.assertAlmostEqual(j, 1.0)

    def test_lcc_fraction(self) -> None:
        g = nx.Graph()
        g.add_edges_from([("a", "b"), ("c", "d"), ("d", "e")])
        # Components: {a, b}, {c, d, e}. LCC size = 3 / 5.
        self.assertAlmostEqual(largest_connected_component_fraction(g), 0.6)

    def test_degree_summary(self) -> None:
        g = nx.Graph()
        g.add_edges_from([("a", "b"), ("b", "c")])
        ds = degree_summary(g)
        self.assertEqual(ds["max"], 2)
        self.assertEqual(ds["min"], 1)

    def test_spectral_overlap_identical_graphs(self) -> None:
        rng = np.random.default_rng(3)
        g = nx.erdos_renyi_graph(40, 0.1, seed=int(rng.integers(0, 1000)))
        mean_cos, max_cos, k = spectral_overlap(g, g.copy(), k=5)
        self.assertGreater(k, 0)
        self.assertAlmostEqual(max_cos, 1.0, delta=1e-6)

    def test_f6_report_smoke(self) -> None:
        rng = np.random.default_rng(4)
        g_b = nx.erdos_renyi_graph(30, 0.2, seed=int(rng.integers(0, 1000)))
        g_t = nx.erdos_renyi_graph(30, 0.05, seed=int(rng.integers(0, 1000)))
        mapping = {i: f"u{i}" for i in range(30)}
        g_b = nx.relabel_nodes(g_b, mapping)
        g_t = nx.relabel_nodes(g_t, mapping)
        report = f6_report(g_b, g_t, spectral_k=3)
        self.assertEqual(report.n_common_nodes, 30)
        self.assertGreaterEqual(report.edge_jaccard, 0.0)
        self.assertLessEqual(report.edge_jaccard, 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
