"""Unit tests for edge homophily."""

from __future__ import annotations

import unittest

import networkx as nx

from two_graph_fusion.graphs.homophily import edge_homophily


class TestHomophily(unittest.TestCase):
    def test_all_same_label(self) -> None:
        g = nx.Graph()
        g.add_edges_from([("a", "b"), ("b", "c")])
        label = {"a": "bot", "b": "bot", "c": "bot"}
        rep = edge_homophily(g, label)
        self.assertEqual(rep.homophily, 1.0)
        self.assertEqual(rep.chance_baseline, 1.0)
        self.assertEqual(rep.homophily_lift, 0.0)

    def test_mixed_labels(self) -> None:
        g = nx.Graph()
        g.add_edge("a", "b")  # same
        g.add_edge("b", "c")  # cross
        label = {"a": "bot", "b": "bot", "c": "human"}
        rep = edge_homophily(g, label)
        self.assertEqual(rep.n_same_label, 1)
        self.assertEqual(rep.n_cross_label, 1)
        self.assertAlmostEqual(rep.homophily, 0.5)
        self.assertAlmostEqual(rep.chance_baseline, 5 / 9)  # 2/3^2 + 1/3^2


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
