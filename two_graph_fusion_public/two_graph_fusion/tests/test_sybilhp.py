"""Unit tests for the SybilHP directed adaptive-homophily LBP.

Run with::

    .venv/bin/python3 -m unittest two_graph_fusion.tests.test_sybilhp
"""

from __future__ import annotations

import unittest

import numpy as np

from two_graph_fusion.propagation.sybilhp import (
    _BIDIR,
    _HEAD,
    _TAIL,
    SybilHPParams,
    _build_message_graph,
    _edge_potentials,
    p_honest_to_dict,
    run_sybilhp,
    sybilhp_priors_from_honest,
)


class TestMessageGraphConstruction(unittest.TestCase):
    def test_single_directed_edge_types_and_reverse(self) -> None:
        # a -> b (a follows b). a is tail, b is head.
        nodes = ["a", "b"]
        mg = _build_message_graph(
            nodes, trust_edges=[("a", "b")], behavioral_edges=[],
            params=SybilHPParams(), w_behavioral=0.9,
        )
        self.assertEqual(mg.snd.shape[0], 2)  # one mutually-reverse pair
        # Locate the a->b message edge.
        ab = next(e for e in range(2) if mg.snd[e] == 0 and mg.rcv[e] == 1)
        ba = next(e for e in range(2) if mg.snd[e] == 1 and mg.rcv[e] == 0)
        self.assertEqual(mg.etype[ab], _TAIL)   # sender a is the tail
        self.assertEqual(mg.etype[ba], _HEAD)   # sender b is the head
        self.assertEqual(mg.rev[ab], ba)
        self.assertEqual(mg.rev[ba], ab)
        # Out/in adjacency: b in N_out(a), a in N_in(b).
        self.assertEqual(mg.a_out[0, 1], 1.0)
        self.assertEqual(mg.a_in[1, 0], 1.0)
        self.assertEqual(mg.a_out[1, 0], 0.0)

    def test_mutual_follow_is_bidirectional(self) -> None:
        nodes = ["a", "b"]
        mg = _build_message_graph(
            nodes, trust_edges=[("a", "b"), ("b", "a")], behavioral_edges=[],
            params=SybilHPParams(), w_behavioral=0.9,
        )
        self.assertEqual(mg.snd.shape[0], 2)
        self.assertTrue(np.all(mg.etype == _BIDIR))
        # Bidirectional neighbours count both ways.
        self.assertEqual(mg.a_out[0, 1], 1.0)
        self.assertEqual(mg.a_in[0, 1], 1.0)

    def test_behavioral_edge_is_bidirectional_with_w(self) -> None:
        nodes = ["a", "b"]
        mg = _build_message_graph(
            nodes, trust_edges=[], behavioral_edges=[("a", "b")],
            params=SybilHPParams(), w_behavioral=0.42,
        )
        self.assertEqual(mg.snd.shape[0], 2)
        self.assertTrue(np.all(mg.etype == _BIDIR))
        self.assertTrue(np.allclose(mg.w_bi, 0.42))

    def test_trust_and_behavioral_pair_yield_two_factors(self) -> None:
        # Same node pair carrying both a trust and a behavioral edge ->
        # two independent factor pairs (4 directed message edges).
        nodes = ["a", "b"]
        mg = _build_message_graph(
            nodes, trust_edges=[("a", "b")], behavioral_edges=[("a", "b")],
            params=SybilHPParams(), w_behavioral=0.9,
        )
        self.assertEqual(mg.snd.shape[0], 4)


class TestEdgePotentials(unittest.TestCase):
    def test_tail_potential_matches_eq10(self) -> None:
        nodes = ["a", "b"]
        params = SybilHPParams(wb_tail=0.9, ws_tail=0.7)
        mg = _build_message_graph(
            nodes, trust_edges=[("a", "b")], behavioral_edges=[],
            params=params, w_behavioral=0.9,
        )
        c_b = np.ones(2)
        c_s = np.ones(2)
        psi = _edge_potentials(mg, c_b, c_s)
        ab = next(e for e in range(2) if mg.snd[e] == 0 and mg.rcv[e] == 1)
        # Eq. 10 with c_s_b = 1: benign-tail rows use wb_tail; sybil-tail uses ws_tail.
        self.assertAlmostEqual(psi[ab, 0, 0], 0.9)        # benign sender, benign receiver
        self.assertAlmostEqual(psi[ab, 0, 1], 0.1)
        self.assertAlmostEqual(psi[ab, 1, 1], 0.7)        # sybil sender, sybil receiver
        self.assertAlmostEqual(psi[ab, 1, 0], 0.3)

    def test_head_potential_matches_eq11(self) -> None:
        nodes = ["a", "b"]
        params = SybilHPParams(wb_head=0.6, ws_head=0.95)
        mg = _build_message_graph(
            nodes, trust_edges=[("a", "b")], behavioral_edges=[],
            params=params, w_behavioral=0.9,
        )
        c_b = np.ones(2)
        c_s = np.ones(2)
        psi = _edge_potentials(mg, c_b, c_s)
        # b -> a message: sender b is the head.
        ba = next(e for e in range(2) if mg.snd[e] == 1 and mg.rcv[e] == 0)
        self.assertAlmostEqual(psi[ba, 0, 0], 0.6)        # benign head -> wb_head
        self.assertAlmostEqual(psi[ba, 1, 0], 0.4)
        self.assertAlmostEqual(psi[ba, 1, 1], 0.95)       # sybil head -> ws_head * c_s_s(=1)
        self.assertAlmostEqual(psi[ba, 0, 1], 0.05)

    def test_potentials_strictly_positive(self) -> None:
        # w=1, c=1 produces a zero entry pre-clip; ensure clip keeps log finite.
        nodes = ["a", "b"]
        mg = _build_message_graph(
            nodes, trust_edges=[], behavioral_edges=[("a", "b")],
            params=SybilHPParams(), w_behavioral=1.0,
        )
        psi = _edge_potentials(mg, np.ones(2), np.ones(2))
        self.assertTrue(np.all(psi > 0.0))
        self.assertTrue(np.all(np.isfinite(np.log(psi))))


class TestPropagationBehaviour(unittest.TestCase):
    def test_isolated_nodes_keep_prior(self) -> None:
        nodes = ["a", "b", "c"]
        q = {"a": 0.9, "b": 0.5, "c": 0.1}
        res = run_sybilhp(nodes, sybil_prior=q, max_iters=10, tol=1e-9)
        p = p_honest_to_dict(res)
        self.assertAlmostEqual(p["a"], 0.1, places=6)   # P(honest) = 1 - q
        self.assertAlmostEqual(p["b"], 0.5, places=6)
        self.assertAlmostEqual(p["c"], 0.9, places=6)

    def test_homophily_pulls_clusters_to_seed_labels(self) -> None:
        # Two behavioral triangles: {A,B,C} benign-seeded at A, {D,E,F}
        # sybil-seeded at D. Unlabeled members should follow their cluster.
        nodes = list("ABCDEF")
        beh = [
            ("A", "B"), ("B", "C"), ("A", "C"),
            ("D", "E"), ("E", "F"), ("D", "F"),
        ]
        q = {"A": 0.02, "D": 0.98}  # others default 0.5
        res = run_sybilhp(
            nodes, sybil_prior=q, behavioral_edges=beh,
            w_behavioral=0.9, max_iters=50, tol=1e-8,
        )
        p = p_honest_to_dict(res)  # P(honest)
        # Benign cluster: high P(honest).
        self.assertGreater(p["B"], 0.6)
        self.assertGreater(p["C"], 0.6)
        # Sybil cluster: low P(honest).
        self.assertLess(p["E"], 0.4)
        self.assertLess(p["F"], 0.4)

    def test_direction_sensitivity(self) -> None:
        # A strong benign seed u and an unlabeled v, linked by a single
        # directed trust edge. A benign *tail* (u follows v) transmits
        # homophily more strongly than a benign *head* (v follows u), so v
        # ends up more benign in the first configuration. We anchor u's
        # neighbourhood with two benign followees (w1, w2) in BOTH configs
        # so the adaptive estimator c_u_b is well-defined and stable, and
        # only the u-v edge direction differs between the two runs.
        params = SybilHPParams(wb_tail=0.97, ws_tail=0.77, wb_head=0.6, ws_head=0.95)
        nodes = ["u", "v", "w1", "w2"]
        q = {"u": 0.02, "w1": 0.02, "w2": 0.02}
        anchors = [("u", "w1"), ("u", "w2")]

        res_tail = run_sybilhp(
            nodes, sybil_prior=q, trust_edges=anchors + [("u", "v")],
            params=params, max_iters=80, tol=1e-9,
        )
        res_head = run_sybilhp(
            nodes, sybil_prior=q, trust_edges=anchors + [("v", "u")],
            params=params, max_iters=80, tol=1e-9,
        )
        psyb_v_tail = 1.0 - p_honest_to_dict(res_tail)["v"]
        psyb_v_head = 1.0 - p_honest_to_dict(res_head)["v"]
        # Both pull v toward benign, but the benign-tail config more so.
        self.assertLess(psyb_v_tail, 0.5)
        self.assertLess(psyb_v_head, 0.5)
        self.assertLess(psyb_v_tail, psyb_v_head - 0.05)

    def test_determinism(self) -> None:
        nodes = list("ABCDEF")
        beh = [("A", "B"), ("B", "C"), ("D", "E"), ("E", "F")]
        trust = [("A", "D"), ("C", "F")]
        q = {"A": 0.02, "D": 0.98}
        r1 = run_sybilhp(nodes, q, trust_edges=trust, behavioral_edges=beh, max_iters=30)
        r2 = run_sybilhp(nodes, q, trust_edges=trust, behavioral_edges=beh, max_iters=30)
        self.assertTrue(np.allclose(r1.p_honest, r2.p_honest))

    def test_converges_on_small_graph(self) -> None:
        nodes = list("ABCDEF")
        beh = [("A", "B"), ("B", "C"), ("A", "C"), ("D", "E"), ("E", "F"), ("D", "F")]
        q = {"A": 0.02, "D": 0.98}
        res = run_sybilhp(nodes, q, behavioral_edges=beh, max_iters=200, tol=1e-7)
        self.assertTrue(res.converged)

    def test_prior_conversion_roundtrip(self) -> None:
        honest = {"a": 0.98, "b": 0.3}
        q = sybilhp_priors_from_honest(honest)
        self.assertAlmostEqual(q["a"], 0.02)
        self.assertAlmostEqual(q["b"], 0.7)


if __name__ == "__main__":
    unittest.main()
