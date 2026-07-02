"""Unit tests for the milestone-3 propagation pipeline.

Run with::

    python -m unittest two_graph_fusion.tests.test_propagation
"""

from __future__ import annotations

import unittest

import networkx as nx
import numpy as np
import pandas as pd

from two_graph_fusion.propagation.prior import (
    DEFAULT_FEATURES,
    fit_predict_lr_prior,
)
from two_graph_fusion.propagation.sybilscar import (
    evaluation_priors,
    p_honest_to_dict,
    relation_from_graph,
    run_propagation,
    seed_priors_from_labels,
)


class TestSybilScarSingleRelation(unittest.TestCase):
    def test_isolated_nodes_keep_prior(self) -> None:
        # Three isolated nodes with distinct priors. With no edges,
        # propagation should return the priors unchanged (sigmoid(logit(p)) = p).
        nodes = ["a", "b", "c"]
        g = nx.Graph()
        g.add_nodes_from(nodes)
        rel = relation_from_graph("r", g, nodes)
        priors = {"a": 0.9, "b": 0.5, "c": 0.1}
        out = run_propagation(
            relations=[rel], node_order=nodes, priors=priors,
            max_iters=10, tol=1e-9,
        )
        p = p_honest_to_dict(out)
        self.assertAlmostEqual(p["a"], 0.9, places=4)
        self.assertAlmostEqual(p["b"], 0.5, places=4)
        self.assertAlmostEqual(p["c"], 0.1, places=4)

    def test_homophily_pulls_neighbours_toward_seed(self) -> None:
        # Star graph with an honest center; the leaves (uniform prior 0.5)
        # should drift toward "honest" (p > 0.5).
        nodes = [f"u{i}" for i in range(6)]
        g = nx.Graph()
        g.add_edges_from([("u0", f"u{i}") for i in range(1, 6)])
        rel = relation_from_graph("trust", g, nodes, weight=0.5)
        priors = {"u0": 0.98}  # honest center
        out = run_propagation(
            relations=[rel], node_order=nodes, priors=priors,
            default_prior=0.5, max_iters=50, tol=1e-7,
        )
        p = p_honest_to_dict(out)
        for leaf in nodes[1:]:
            self.assertGreater(p[leaf], 0.5)
        self.assertGreater(p["u0"], 0.98 - 0.01)

    def test_sybil_seed_drags_neighbours_down(self) -> None:
        nodes = [f"u{i}" for i in range(6)]
        g = nx.Graph()
        g.add_edges_from([("u0", f"u{i}") for i in range(1, 6)])
        rel = relation_from_graph("trust", g, nodes, weight=0.5)
        priors = {"u0": 0.02}  # sybil center
        out = run_propagation(
            relations=[rel], node_order=nodes, priors=priors,
            default_prior=0.5, max_iters=50, tol=1e-7,
        )
        p = p_honest_to_dict(out)
        for leaf in nodes[1:]:
            self.assertLess(p[leaf], 0.5)

    def test_multi_relation_combines_both_views(self) -> None:
        # Two leaf nodes share an honest seed on one relation and a sybil
        # seed on the other. The multi-relational posterior should sit
        # between the two single-relation extremes.
        nodes = ["a", "b", "h", "s"]
        g_pos = nx.Graph()
        g_pos.add_edges_from([("h", "a"), ("h", "b")])
        g_neg = nx.Graph()
        g_neg.add_edges_from([("s", "a"), ("s", "b")])
        node_order = nodes
        rel_pos = relation_from_graph("pos", g_pos, node_order, weight=0.5)
        rel_neg = relation_from_graph("neg", g_neg, node_order, weight=0.5)
        priors = {"h": 0.98, "s": 0.02}
        out_pos = run_propagation(
            relations=[rel_pos], node_order=node_order, priors=priors,
            default_prior=0.5, max_iters=50, tol=1e-7,
        )
        p_pos = p_honest_to_dict(out_pos)
        out_neg = run_propagation(
            relations=[rel_neg], node_order=node_order, priors=priors,
            default_prior=0.5, max_iters=50, tol=1e-7,
        )
        p_neg = p_honest_to_dict(out_neg)
        out_both = run_propagation(
            relations=[rel_pos, rel_neg], node_order=node_order,
            priors=priors, default_prior=0.5, max_iters=50, tol=1e-7,
        )
        p_both = p_honest_to_dict(out_both)
        # Single-view posteriors on the leaves are opposite-sign extremes.
        self.assertGreater(p_pos["a"], 0.6)
        self.assertLess(p_neg["a"], 0.4)
        # Joint should hover near 0.5 because the two views fully disagree.
        self.assertGreater(p_both["a"], 0.3)
        self.assertLess(p_both["a"], 0.7)

    def test_convergence_reaches_fixed_point(self) -> None:
        rng = np.random.default_rng(0)
        n = 20
        g = nx.erdos_renyi_graph(n, 0.3, seed=int(rng.integers(0, 1000)))
        node_order = [f"n{i}" for i in range(n)]
        g = nx.relabel_nodes(g, {i: f"n{i}" for i in range(n)})
        rel = relation_from_graph("r", g, node_order, weight=0.4)
        priors = {f"n{i}": 0.98 if i % 4 == 0 else 0.02 if i % 4 == 1 else 0.5
                  for i in range(n)}
        out = run_propagation(
            relations=[rel], node_order=node_order, priors=priors,
            max_iters=200, tol=1e-9,
        )
        self.assertTrue(out.converged)
        self.assertLess(out.max_delta, 1e-9)

    def test_evaluation_priors_uses_label_and_pi_b(self) -> None:
        labels = {"a": "human", "b": "bot"}
        pi_b = {"c": 0.7, "d": 0.3}
        out = evaluation_priors(
            labels=labels, honest_label="human", bot_label="bot",
            pi_b_honest=pi_b,
        )
        self.assertAlmostEqual(out["a"], 0.98)
        self.assertAlmostEqual(out["b"], 0.02)
        self.assertAlmostEqual(out["c"], 0.7)
        self.assertAlmostEqual(out["d"], 0.3)


class TestLRPrior(unittest.TestCase):
    def _make_separable(self, n: int = 80, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        humans = rng.normal(loc=0.0, scale=1.0, size=(n // 2, 5))
        bots = rng.normal(loc=2.5, scale=1.0, size=(n // 2, 5))
        rows = np.vstack([humans, bots])
        labels = ["human"] * (n // 2) + ["bot"] * (n // 2)
        idx = [f"u{i}" for i in range(n)]
        df = pd.DataFrame(rows, columns=list(DEFAULT_FEATURES), index=idx)
        df["label"] = labels
        # Shuffle so train/test slices contain both classes.
        df = df.sample(frac=1.0, random_state=seed)
        return df

    def test_lr_recovers_separable_signal(self) -> None:
        df = self._make_separable(n=120, seed=1)
        train = df.iloc[:90]
        test = df.iloc[90:]
        out = fit_predict_lr_prior(train, test, label_col="label")
        self.assertGreater(out.test_auc, 0.95)
        # All five coefficients should be negative (higher feature => more bot,
        # so logit(P(honest)) = -beta dot x; here labels y=1 means bot, so the LR
        # learned signed against the honest probability we return).
        # We're predicting bot probability; coefficients should be positive in
        # raw LR. The wrapper inverts to pi_b(honest); just sanity-check
        # magnitudes exceed zero.
        self.assertGreater(max(abs(c) for c in out.coefficients), 0.3)

    def test_drop_l_removes_column(self) -> None:
        df = self._make_separable(n=80, seed=2)
        # Manually rename one of the columns so we can test drop logic.
        df = df.rename(columns={"L_z": "L_z"})
        out_full = fit_predict_lr_prior(df.iloc[:60], df.iloc[60:], drop_l=False)
        out_no_l = fit_predict_lr_prior(df.iloc[:60], df.iloc[60:], drop_l=True)
        self.assertIn("L_z", out_full.feature_columns)
        self.assertNotIn("L_z", out_no_l.feature_columns)
        self.assertEqual(len(out_no_l.coefficients), 4)


class TestOfficialSplit(unittest.TestCase):
    def test_official_train_test_protocol(self) -> None:
        from two_graph_fusion.propagation.evaluation import run_official_split_evaluation

        rng = np.random.default_rng(7)
        humans_tr = rng.normal(loc=0.0, scale=1.0, size=(60, 5))
        bots_tr = rng.normal(loc=2.0, scale=1.0, size=(30, 5))
        humans_te = rng.normal(loc=0.0, scale=1.0, size=(20, 5))
        bots_te = rng.normal(loc=2.0, scale=1.0, size=(10, 5))
        rows = np.vstack([humans_tr, bots_tr, humans_te, bots_te])
        users = [f"u{i}" for i in range(120)]
        df = pd.DataFrame(rows, columns=list(DEFAULT_FEATURES), index=users)
        labels = ["human"] * 80 + ["bot"] * 40
        df["label"] = labels
        # Stratified split: 75% of each class in train.
        splits = []
        for lab in ("human", "bot"):
            idx = [i for i, u in enumerate(users) if labels[i] == lab]
            n_tr = int(0.75 * len(idx))
            tr_set = set(idx[:n_tr])
            for i in idx:
                splits.append("train" if i in tr_set else "test")
        df["split"] = splits
        n_train = sum(s == "train" for s in splits)
        n_test = sum(s == "test" for s in splits)

        n = len(users)
        g = nx.erdos_renyi_graph(n, 0.2, seed=42)
        g = nx.relabel_nodes(g, {i: users[i] for i in range(n)})
        result = run_official_split_evaluation(
            zscored=df, g_t=g, g_b=g.copy(),
            train_split="train", test_split="test",
            w_t=0.3, w_b=0.3, max_iters=30, tol=1e-5,
            alphas=(0.0, 1.0),
        )
        self.assertEqual(result.n_train, n_train)
        self.assertEqual(result.n_test, n_test)
        self.assertFalse(np.isnan(result.lr_with_l["test_auc"]))
        self.assertGreater(result.lr_with_l["test_auc"], 0.55)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
