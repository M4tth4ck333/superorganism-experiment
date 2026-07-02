"""M12 — TwiBot-20 follow-graph homophily gate.

Builds the directed follow graph among the 11,826 labelled TwiBot-20 users
and measures edge-homophily lift — the go/no-go check (soft gate +0.05)
for whether SybilSCAR / SybilHP propagation can work on this dataset.
This is the "gate first" deliverable of M12; it needs only the graph +
labels (no behavioral features, which TwiBot-20 cannot supply — see the
loader docstring).

Usage::

    .venv/bin/python3 -m two_graph_fusion.scripts.twibot20_gate \\
        --root Twibot-20-2 \\
        --output-dir two_graph_fusion/cache/twibot20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import networkx as nx

from two_graph_fusion.datasets.twibot20 import induced_directed_edges, load_twibot20
from two_graph_fusion.graphs import edge_homophily, graph_summary

SOFT_GATE = 0.05


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TwiBot-20 follow-graph homophily gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", type=Path, default=Path("Twibot-20-2"))
    p.add_argument("--output-dir", type=Path, default=Path("two_graph_fusion/cache/twibot20"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _homophily_report(g: nx.Graph, label_map: dict[str, str]) -> dict:
    rep = edge_homophily(g, label_map)
    bb = hh = hb = 0
    for u, v in g.edges():
        lu, lv = label_map.get(u), label_map.get(v)
        if lu == "bot" and lv == "bot":
            bb += 1
        elif lu == "human" and lv == "human":
            hh += 1
        elif lu is not None and lv is not None:
            hb += 1
    return {**rep.to_dict(), "edge_breakdown": {"bot_bot": bb, "human_human": hh, "human_bot": hb}}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger(__name__)
    t0 = time.time()
    if not args.root.exists():
        log.error("root not found: %s", args.root)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_twibot20(args.root, include_support=False)
    users = data.users
    labeled = users[users["label"].isin({"human", "bot"})]
    label_map = labeled["label"].to_dict()
    labeled_ids = set(labeled.index)
    n_bot = int((labeled["label"] == "bot").sum())
    n_human = int((labeled["label"] == "human").sum())
    log.info("labeled users: %d (human=%d bot=%d bot%%=%.1f)",
             len(labeled), n_human, n_bot, 100 * n_bot / len(labeled))

    # Directed follow edges among labeled users only.
    directed = induced_directed_edges(data.follow_edges, labeled_ids)
    # Undirected graph for the gate (consistent with M4–M9 homophily reports).
    g = nx.Graph()
    g.add_nodes_from(labeled_ids)
    g.add_edges_from(directed)
    gsum = graph_summary(g)

    # Mutual-edge fraction (how directed the graph is): both (a,b) and (b,a).
    dset = set(directed)
    mutual = sum(1 for (a, b) in dset if (b, a) in dset) // 2
    n_undirected = g.number_of_edges()

    hom = _homophily_report(g, label_map)
    gate = hom["homophily_lift"] >= SOFT_GATE

    # Split balance (for the eventual official-split eval).
    split_counts = (
        labeled.groupby("split")["label"].value_counts().unstack(fill_value=0).to_dict("index")
    )

    payload = {
        "n_labeled": len(labeled),
        "n_human": n_human,
        "n_bot": n_bot,
        "bot_pct": round(n_bot / len(labeled), 4),
        "split_label_counts": split_counts,
        "follow_graph_labeled_only": {
            "directed_edges_among_labeled": len(directed),
            "undirected_edges": n_undirected,
            "mutual_pairs": mutual,
            **gsum,
        },
        "homophily": hom,
        "soft_gate": SOFT_GATE,
        "gate_pass": bool(gate),
        "total_directed_refs": len(data.follow_edges),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = args.output_dir / "gate.json"
    with out.open("w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    log.info("wrote %s", out)

    bd = hom["edge_breakdown"]
    print("\n=== TwiBot-20 follow-graph homophily gate (labeled users) ===")
    print(f"Labeled users: {len(labeled):,}  (human={n_human:,} bot={n_bot:,} "
          f"bot%={100*n_bot/len(labeled):.1f}%)")
    print(f"Split balance: " + "  ".join(
        f"{s}:{d}" for s, d in split_counts.items()))
    print(f"\nFollow graph (labeled-induced):")
    print(f"  total directed follow refs (all targets): {len(data.follow_edges):,}")
    print(f"  directed edges among labeled users:       {len(directed):,}")
    print(f"  undirected edges:                         {n_undirected:,}  "
          f"(mutual pairs={mutual:,})")
    print(f"  lcc fraction={gsum['lcc_fraction']:.3f}  "
          f"isolated={gsum['degree'].get('n_isolated','?')}")
    print(f"\nHomophily (soft gate: lift ≥ {SOFT_GATE}):")
    print(f"  edges={hom['n_edges_labeled']:,}  h={hom['homophily']:.4f}  "
          f"chance={hom['chance_baseline']:.4f}  lift={hom['homophily_lift']:+.4f}  "
          f"{'PASS' if gate else 'fail'}")
    print(f"  bot-bot={bd['bot_bot']:,}  human-human={bd['human_human']:,}  "
          f"human-bot={bd['human_bot']:,}")

    log.info("done in %.1f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
