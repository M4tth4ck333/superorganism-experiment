"""Degree-core homophily on the full TwiBot-22 follow graph.

Prior threshold sweeps (Milestones 6–7) varied ``n_min_events`` — the
*tweet-activity* filter — and found follow-graph homophily lift never
cleared the +0.05 soft gate.  This script pulls the one remaining,
untested lever: **graph degree**.

LGB (Zhou et al. 2024, Fig. 2) report that on TwiBot-22 a GNN only beats
a text classifier for nodes with ≥2 follow-neighbours, and that the
densely-linked tail (>10 neighbours) is just 8.2% of the network.  The
question this script answers is whether that densely-linked core is
*homophilous* — i.e. whether SybilSCAR-style propagation could work if
restricted to it.

Unlike the pipeline's ``n_min`` cohorts, the cut here is purely
structural and computed over the **full** follow graph (all labelled
users), matching LGB's "number of neighbours" definition.  For each
degree threshold ``k`` we keep nodes with full-graph degree ≥ k, take the
induced subgraph, and measure edge homophily + class balance on it.

The computation is fully vectorised over the compact follow-edge cache;
no NetworkX graph is materialised, so the whole sweep runs in well under
a minute and a couple of GB of RAM.

Usage::

    python -m two_graph_fusion.scripts.degree_core_homophily \\
        --follow-cache two_graph_fusion/cache/twibot22_follow_edges.csv \\
        --label-csv twibot22/label.csv \\
        --output-dir two_graph_fusion/cache/degree_core
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50, 100)
SOFT_GATE = 0.05


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Degree-core homophily on the full TwiBot-22 follow graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--follow-cache",
        type=Path,
        default=Path("two_graph_fusion/cache/twibot22_follow_edges.csv"),
        help="Compact follow-edge cache (source,target).",
    )
    p.add_argument(
        "--label-csv",
        type=Path,
        default=Path("twibot22/label.csv"),
        help="TwiBot-22 label.csv (id,label).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("two_graph_fusion/cache/degree_core"),
    )
    p.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Minimum full-graph follow-degree for each core.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _load_unique_undirected_edges(cache_csv: Path) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Return deduplicated undirected edges as integer code columns.

    The follow cache keeps mirror rows (``u1->u2`` via ``following`` and
    ``u2->u1`` via ``followers``); we collapse them to undirected edges by
    canonicalising each pair to ``(min, max)`` and dropping duplicates and
    self-loops — exactly what :func:`load_follow_edge_cache` does via
    NetworkX, but vectorised.

    Returns ``(lo_codes, hi_codes, node_index)`` where ``node_index`` maps
    integer code -> user id.
    """
    edges = pd.read_csv(cache_csv, dtype={"source": str, "target": str})
    codes, uniques = pd.factorize(
        pd.concat([edges["source"], edges["target"]], ignore_index=True),
        sort=False,
    )
    n = len(edges)
    src = codes[:n]
    tgt = codes[n:]
    lo = np.minimum(src, tgt)
    hi = np.maximum(src, tgt)
    self_loop = lo == hi
    lo, hi = lo[~self_loop], hi[~self_loop]
    # Deduplicate (lo, hi) pairs.
    pair_view = np.stack([lo, hi], axis=1)
    pair_view = np.unique(pair_view, axis=0)
    return pair_view[:, 0], pair_view[:, 1], uniques


def _label_codes(node_index: pd.Index, label_csv: Path) -> np.ndarray:
    """Map each node code to {0: human, 1: bot, -1: unlabeled / other}."""
    labels = pd.read_csv(label_csv, dtype=str).set_index("id")["label"]
    mapping = {"human": 0, "bot": 1}
    node_labels = labels.reindex(node_index).map(mapping)
    return node_labels.to_numpy(dtype="float64")  # NaN where unlabeled


def _core_homophily(
    lo: np.ndarray,
    hi: np.ndarray,
    degree: np.ndarray,
    label: np.ndarray,
    k: int,
) -> dict:
    """Homophily + class balance on the induced subgraph of degree ≥ k nodes."""
    kept = degree >= k
    edge_kept = kept[lo] & kept[hi]
    e_lo, e_hi = lo[edge_kept], hi[edge_kept]

    # Node-level class balance over kept, labelled nodes.
    kept_codes = np.flatnonzero(kept)
    kept_labels = label[kept_codes]
    labelled = kept_labels[~np.isnan(kept_labels)]
    n_nodes = int(kept.sum())
    n_labelled = int(labelled.size)
    n_bot = int((labelled == 1).sum())
    n_human = int((labelled == 0).sum())
    bot_pct = (n_bot / n_labelled) if n_labelled else float("nan")
    p_h = n_human / n_labelled if n_labelled else float("nan")
    p_b = n_bot / n_labelled if n_labelled else float("nan")
    chance = (p_h**2 + p_b**2) if n_labelled else float("nan")

    # Edge-level homophily over edges with both endpoints labelled.
    la, lb = label[e_lo], label[e_hi]
    both_labelled = ~np.isnan(la) & ~np.isnan(lb)
    la, lb = la[both_labelled], lb[both_labelled]
    n_edges = int(la.size)
    same = la == lb
    bot_bot = int(((la == 1) & (lb == 1)).sum())
    hum_hum = int(((la == 0) & (lb == 0)).sum())
    hum_bot = n_edges - bot_bot - hum_hum
    h = float(same.mean()) if n_edges else float("nan")
    lift = (h - chance) if n_edges else float("nan")

    return {
        "k": k,
        "n_nodes": n_nodes,
        "n_labelled_nodes": n_labelled,
        "bot_pct": round(bot_pct, 4) if n_labelled else None,
        "n_edges_labelled": n_edges,
        "homophily": round(h, 4) if n_edges else None,
        "chance_baseline": round(chance, 4) if n_labelled else None,
        "homophily_lift": round(lift, 4) if n_edges else None,
        "gate_pass": bool(n_edges and lift >= SOFT_GATE),
        "edge_breakdown": {"bot_bot": bot_bot, "human_human": hum_hum, "human_bot": hum_bot},
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger(__name__)
    t0 = time.time()

    for path, name in [(args.follow_cache, "follow cache"), (args.label_csv, "label.csv")]:
        if not path.exists():
            log.error("missing %s: %s", name, path)
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading + deduplicating follow edges from %s", args.follow_cache)
    lo, hi, node_index = _load_unique_undirected_edges(args.follow_cache)
    n_nodes_total = len(node_index)
    n_edges_total = len(lo)
    log.info("full follow graph: nodes=%d  undirected edges=%d", n_nodes_total, n_edges_total)

    # Degree over the deduplicated undirected graph.
    degree = np.bincount(np.concatenate([lo, hi]), minlength=n_nodes_total)
    label = _label_codes(node_index, args.label_csv)
    n_labelled_total = int((~np.isnan(label)).sum())
    log.info(
        "labelled follow-graph nodes: %d / %d  (bot%%=%.1f)",
        n_labelled_total, n_nodes_total,
        100 * np.nansum(label == 1) / n_labelled_total,
    )

    # Degree distribution context (mirrors LGB Fig. 1 buckets).
    isolated_in_followgraph = n_nodes_total  # all cache nodes have ≥1 edge by construction
    deg_buckets = {
        "deg>=1": int((degree >= 1).sum()),
        "deg>=2": int((degree >= 2).sum()),
        "deg>=10": int((degree >= 10).sum()),
    }

    rows = [_core_homophily(lo, hi, degree, label, k) for k in sorted(args.thresholds)]

    payload = {
        "source": str(args.follow_cache),
        "full_graph": {
            "n_nodes_with_edges": n_nodes_total,
            "n_undirected_edges": n_edges_total,
            "n_labelled_nodes": n_labelled_total,
            "degree_buckets": deg_buckets,
        },
        "soft_gate": SOFT_GATE,
        "cores": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_json = args.output_dir / "homophily_degree_core.json"
    with out_json.open("w") as fh:
        json.dump(payload, fh, indent=2)
    log.info("wrote %s", out_json)

    # Human-readable table.
    print("\n=== Degree-core homophily: full TwiBot-22 follow graph ===")
    print(f"Full follow graph: {n_nodes_total:,} nodes (with ≥1 edge), "
          f"{n_edges_total:,} undirected edges, {n_labelled_total:,} labelled.")
    print(f"(For reference, the cache excludes follow-graph isolates; LGB report "
          f"~30.6% of all TwiBot-22 nodes are isolated.)\n")
    print(f"  {'deg≥k':>6} {'nodes':>10} {'bot%':>6} {'edges':>11} "
          f"{'h':>7} {'chance':>8} {'lift':>8} {'gate':>6} {'bot-bot':>9} {'hum-bot':>9}")
    for r in rows:
        gate = "PASS" if r["gate_pass"] else "fail"
        bd = r["edge_breakdown"]
        print(f"  {r['k']:>6} {r['n_nodes']:>10,} "
              f"{(100*r['bot_pct']):>5.1f} {r['n_edges_labelled']:>11,} "
              f"{r['homophily']:>7.4f} {r['chance_baseline']:>8.4f} "
              f"{r['homophily_lift']:>+8.4f} {gate:>6} "
              f"{bd['bot_bot']:>9,} {bd['human_bot']:>9,}")

    log.info("done in %.1f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
