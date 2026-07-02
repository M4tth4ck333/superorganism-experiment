"""Unit tests for the snowball / LCC samplers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from two_graph_fusion.datasets.snowball import (
    labeled_follow_graph,
    sample_largest_component,
    snowball_sample,
    snowball_separate_classes,
)
from two_graph_fusion.graphs.homophily import edge_homophily


def _write_cache(tmpdir: Path, edges: list[tuple[str, str]]) -> Path:
    out = tmpdir / "follow.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("source", "target"))
        for src, tgt in edges:
            writer.writerow((src, tgt))
    return out


class TestSnowball(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        # 10 users: u0..u4 form a clique-ish bot cluster,
        # u5..u9 form a chain of humans, with one bridge edge u4-u5.
        self.labels = pd.DataFrame(
            {
                "label": ["bot"] * 5 + ["human"] * 5,
                "split": ["train"] * 10,
            },
            index=[f"u{i}" for i in range(10)],
        )
        edges = [
            ("u0", "u1"), ("u1", "u2"), ("u2", "u3"), ("u3", "u4"),
            ("u0", "u2"), ("u1", "u3"),
            ("u4", "u5"),  # bridge
            ("u5", "u6"), ("u6", "u7"), ("u7", "u8"), ("u8", "u9"),
        ]
        self.cache = _write_cache(self.tmpdir, edges)

    def test_labeled_follow_graph_loads_all_edges(self) -> None:
        g = labeled_follow_graph(self.cache, self.labels)
        self.assertEqual(g.number_of_nodes(), 10)
        self.assertEqual(g.number_of_edges(), 11)

    def test_lcc_sample_returns_balanced_users(self) -> None:
        subset = sample_largest_component(
            self.cache, self.labels, n_per_class=3, seed=0
        )
        self.assertEqual(len(subset), 6)
        self.assertEqual(set(subset["label"]), {"bot", "human"})

    def test_snowball_visits_connected_block(self) -> None:
        subset = snowball_sample(
            self.cache,
            self.labels,
            target_size=10,
            n_per_class=3,
            max_hops=10,
            seed=0,
        )
        self.assertGreater(len(subset), 0)
        self.assertLessEqual(len(subset), 6)

    def test_separate_class_collects_per_label(self) -> None:
        subset = snowball_separate_classes(
            self.cache,
            self.labels,
            target_per_class=10,
            n_per_class=3,
            max_hops=10,
            seed=0,
        )
        self.assertEqual(len(subset), 6)
        self.assertEqual(set(subset["label"]), {"bot", "human"})

    def test_separate_class_higher_g_t_homophily_than_cross_class(self) -> None:
        """Separate-class union should have more same-label follow edges."""
        g = labeled_follow_graph(self.cache, self.labels)
        label = self.labels["label"].to_dict()

        cross = snowball_sample(
            self.cache, self.labels, target_size=10, n_per_class=5, max_hops=10, seed=0
        )
        separate = snowball_separate_classes(
            self.cache, self.labels, target_per_class=10, n_per_class=5, max_hops=10, seed=0
        )

        def induced_homophily(users: pd.DataFrame) -> float:
            ids = set(users.index)
            sub = g.subgraph(ids).copy()
            return edge_homophily(sub, label).homophily_lift

        lift_cross = induced_homophily(cross)
        lift_sep = induced_homophily(separate)
        self.assertGreaterEqual(lift_sep, lift_cross)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
