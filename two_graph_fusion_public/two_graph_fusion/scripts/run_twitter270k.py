"""Twitter-270k (SybilHP / SybilGAT benchmark): engine validation + diagnostic.

Structure-only directed follow graph (~269,640 nodes, 6.8M directed edges)
with benign/sybil seed lists — the dataset SybilHP (Lu et al. 2023) and
SybilGAT report on. There are NO tweets / timestamps / profile features, so
this exercises the **G_T-only engine** and serves three purposes:

1. **Engine external validation** — reproduce a SybilHP-class AUC with our
   re-implementation, confirming the engine is faithful.
2. **Directionality adjudication** — SybilHP (direction-sensitive potentials)
   vs SybilSCAR (direction-blind, symmetric adjacency) on a genuinely
   directed graph, the comparison Cresci (saturated/symmetric) could not make.
3. **Diagnostic spectrum point** — homophily lift on a large, independent,
   real graph where propagation is known to work.

Protocol (matching the benchmark): seeds = ``train.txt`` (20k benign + 10k
sybil); all other nodes get a neutral 0.5 prior; metrics computed on the
held-out ``test.txt`` nodes (158k benign + 81k sybil). AUC / PR-AUC are the
threshold-free headline; Acc / F1 use a 0.5 P(sybil) threshold.

File formats: ``graph.txt`` = "u v" directed edge per line (u follows v);
``train.txt`` / ``test.txt`` = two whitespace-separated lines, line 1 = benign
ids, line 2 = sybil ids.

Usage::

    .venv/bin/python3 -m two_graph_fusion.scripts.run_twitter270k \\
        --root ../GNN-sybil-detection/code/data/twitter-270k \\
        --engine both --output-dir two_graph_fusion/cache/twitter270k
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from two_graph_fusion.propagation.sybilhp import (
    SybilHPParams,
    run_sybilhp,
)
from two_graph_fusion.propagation.sybilscar import (
    Relation,
    run_propagation,
)

SEED_HI = 0.9   # P(honest) for benign seeds (SybilHP paper: benign prior 1-p, p=0.9)
SEED_LO = 0.1   # P(honest) for sybil seeds   (SybilHP paper: sybil prior p=0.9)


def _load_edges(graph_txt: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load directed edges as (src, dst) int arrays + node count."""
    arr = np.loadtxt(graph_txt, dtype=np.int64)
    src, dst = arr[:, 0], arr[:, 1]
    n = int(max(src.max(), dst.max())) + 1
    return np.ascontiguousarray(src), np.ascontiguousarray(dst), n


def _load_label_lists(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Two-line file -> (benign_ids, sybil_ids) int arrays."""
    with path.open() as fh:
        lines = [ln.split() for ln in fh if ln.strip()]
    benign = np.array(lines[0], dtype=np.int64)
    sybil = np.array(lines[1], dtype=np.int64)
    return benign, sybil


def _homophily_lift(src: np.ndarray, dst: np.ndarray, y: np.ndarray) -> dict:
    """Vectorised edge-homophily lift over the chance baseline ``sum_c p_c^2``.

    ``y`` is the full per-node label vector (0=benign, 1=sybil, -1=unlabeled).
    Only edges with both endpoints labelled are counted. Same definition as
    :func:`two_graph_fusion.graphs.edge_homophily`.
    """
    ls, ld = y[src], y[dst]
    both = (ls >= 0) & (ld >= 0)
    ls, ld = ls[both], ld[both]
    same = int((ls == ld).sum())
    total = int(both.sum())
    h = same / total if total else float("nan")
    n_lab = int((y >= 0).sum())
    p_bot = float((y == 1).sum()) / n_lab
    p_ben = float((y == 0).sum()) / n_lab
    chance = p_bot ** 2 + p_ben ** 2
    bb = int(((ls == 1) & (ld == 1)).sum())
    hh = int(((ls == 0) & (ld == 0)).sum())
    hb = total - bb - hh
    return {
        "n_edges_labeled": total, "homophily": h, "chance_baseline": chance,
        "homophily_lift": h - chance,
        "edge_breakdown": {"bot_bot": bb, "human_human": hh, "human_bot": hb},
    }


def _eval(p_sybil: np.ndarray, test_benign: np.ndarray, test_sybil: np.ndarray) -> dict:
    """AUC / PR-AUC / Acc / F1 on the held-out test nodes (sybil=positive)."""
    idx = np.concatenate([test_benign, test_sybil])
    y = np.concatenate([np.zeros(len(test_benign), int), np.ones(len(test_sybil), int)])
    s = p_sybil[idx]
    pred = (s >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "acc": float((pred == y).mean()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "n_test": int(len(y)),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Twitter-270k engine validation + diagnostic.")
    p.add_argument("--root", type=Path,
                   default=Path("../GNN-sybil-detection/code/data/twitter-270k"))
    p.add_argument("--output-dir", type=Path, default=Path("two_graph_fusion/cache/twitter270k"))
    p.add_argument("--engine", choices=["sybilscar", "sybilhp", "both"], default="both")
    p.add_argument("--w", type=float, default=0.01,
                   help="SybilSCAR homophily weight (small: hub-heavy graph diverges if large).")
    p.add_argument("--rule", choices=["logistic", "linear"], default="logistic",
                   help="SybilSCAR update rule: 'linear' = canonical published "
                        "SybilSCAR-D (additive prior + clamp); 'logistic' = "
                        "sigmoid-stabilised variant (default).")
    p.add_argument("--w-bi", type=float, default=0.99, help="SybilHP bidirectional strength (paper: 0.99).")
    p.add_argument("--damping", type=float, default=0.0,
                   help="SybilHP log-message damping in [0,1). Paper uses plain LBP (0.0) with iter=5.")
    p.add_argument("--max-iters", type=int, default=5,
                   help="SybilHP LBP iterations (paper: 5).")
    p.add_argument("--seed-hi", type=float, default=SEED_HI, help="P(honest) for benign seeds.")
    p.add_argument("--seed-lo", type=float, default=SEED_LO, help="P(honest) for sybil seeds.")
    p.add_argument("--flip-edges", action="store_true",
                   help="Treat 'u v' as v->u (test the follow-direction convention for SybilHP).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    log = logging.getLogger(__name__)
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    src, dst, n = _load_edges(args.root / "graph.txt")
    if args.flip_edges:
        src, dst = dst, src
        log.info("edge orientation flipped (treating 'u v' as v->u)")
    log.info("graph: %d nodes, %d directed edges", n, len(src))
    tr_ben, tr_syb = _load_label_lists(args.root / "train.txt")
    te_ben, te_syb = _load_label_lists(args.root / "test.txt")
    full_ben, full_syb = _load_label_lists(args.root / "test_full.txt")

    # Full label vector for the diagnostic (test_full ~ covers all nodes).
    y_full = np.full(n, -1, dtype=np.int8)
    y_full[full_ben] = 0
    y_full[full_syb] = 1

    # Reciprocity (how directed the graph is).
    eset = set(zip(src.tolist(), dst.tolist()))
    n_mutual = sum(1 for a, b in eset if a < b and (b, a) in eset)
    recip = 2 * n_mutual / len(src)
    log.info("reciprocity (fraction of directed edges that are reciprocated)=%.3f", recip)

    hom = _homophily_lift(src, dst, y_full)
    log.info("homophily lift=%.4f (h=%.4f chance=%.4f)",
             hom["homophily_lift"], hom["homophily"], hom["chance_baseline"])

    # Seed priors P(honest): train benign -> SEED_HI, train sybil -> SEED_LO.
    node_order = list(range(n))
    honest_prior = {}
    for u in tr_ben.tolist():
        honest_prior[u] = args.seed_hi
    for u in tr_syb.tolist():
        honest_prior[u] = args.seed_lo

    results: dict[str, dict] = {}

    if args.engine in ("sybilscar", "both"):
        log.info("building symmetric CSR for SybilSCAR ...")
        rows = np.concatenate([src, dst])
        cols = np.concatenate([dst, src])
        data = np.ones(len(rows), dtype=np.float64)
        adj = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        adj.data[:] = 1.0  # collapse any duplicate edges to binary
        rel = Relation(name="trust", adjacency=adj, weight=args.w)
        log.info("running SybilSCAR (%s rule, w=%.3f) ...", args.rule, args.w)
        t = time.time()
        res = run_propagation([rel], node_order, honest_prior,
                              default_prior=0.5, max_iters=args.max_iters,
                              linearized=(args.rule == "linear"))
        p_sybil = 1.0 - res.p_honest
        m = _eval(p_sybil, te_ben, te_syb)
        m.update(engine="sybilscar", rule=args.rule, w=args.w, n_iters=res.n_iters,
                 converged=res.converged, runtime_s=round(time.time() - t, 1))
        results["sybilscar"] = m
        log.info("SybilSCAR: AUC=%.4f PR-AUC=%.4f (%.1fs)", m["auc"], m["pr_auc"], m["runtime_s"])

    if args.engine in ("sybilhp", "both"):
        log.info("running SybilHP (w_bi=%.2f) ...", args.w_bi)
        t = time.time()
        sybil_prior = {u: 1.0 - ph for u, ph in honest_prior.items()}
        res = run_sybilhp(
            node_order=node_order, sybil_prior=sybil_prior,
            trust_edges=list(zip(src.tolist(), dst.tolist())),
            params=SybilHPParams(w_bi=args.w_bi),
            default_prior=0.5, max_iters=args.max_iters, damping=args.damping,
        )
        p_sybil = 1.0 - res.p_honest
        m = _eval(p_sybil, te_ben, te_syb)
        m.update(engine="sybilhp", w_bi=args.w_bi, n_iters=res.n_iters,
                 converged=res.converged, runtime_s=round(time.time() - t, 1))
        results["sybilhp"] = m
        log.info("SybilHP: AUC=%.4f PR-AUC=%.4f (%.1fs)", m["auc"], m["pr_auc"], m["runtime_s"])

    payload = {
        "dataset": "twitter-270k (SybilHP/SybilGAT benchmark)",
        "n_nodes": n, "n_directed_edges": int(len(src)),
        "reciprocity": round(recip, 4),
        "n_train_benign": len(tr_ben), "n_train_sybil": len(tr_syb),
        "n_test_benign": len(te_ben), "n_test_sybil": len(te_syb),
        "homophily": hom,
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = args.output_dir / f"results_{args.engine}.json"
    with out.open("w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    print(f"\n=== Twitter-270k ({n:,} nodes, {len(src):,} directed edges, "
          f"reciprocity={recip:.3f}) ===")
    bd = hom["edge_breakdown"]
    print(f"Homophily lift = {hom['homophily_lift']:+.4f}  "
          f"(h={hom['homophily']:.4f} chance={hom['chance_baseline']:.4f})  "
          f"bb={bd['bot_bot']:,} hh={bd['human_human']:,} hb={bd['human_bot']:,}")
    print(f"Seeds: train {len(tr_ben):,} benign + {len(tr_syb):,} sybil  |  "
          f"test {len(te_ben):,} benign + {len(te_syb):,} sybil")
    print(f"\n{'engine':<10}{'AUC':>8}{'PR-AUC':>9}{'Acc':>8}{'F1':>8}  iters")
    for name in ("sybilscar", "sybilhp"):
        if name in results:
            m = results[name]
            conv = "" if m["converged"] else " (max_iters)"
            print(f"{name:<10}{m['auc']:>8.4f}{m['pr_auc']:>9.4f}"
                  f"{m['acc']:>8.4f}{m['f1']:>8.4f}  {m['n_iters']}{conv}")
    print(f"\nwrote {out}")
    log.info("done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
