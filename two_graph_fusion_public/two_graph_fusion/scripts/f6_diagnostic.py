"""F6 overlap diagnostic between ``G_B`` and the TwiBot-22 trust graph.

Loads the behavioral-graph artefacts produced by ``build_gb``, streams
``edge.csv`` to assemble ``G_T`` on the same node set, then computes:

- Edge Jaccard (pivot plan section 8.7 pass criterion: ``< 0.30``).
- Triangle Jaccard.
- Spectral cosine overlap of the top-``k`` non-trivial Laplacian
  eigenvectors.
- Summary stats for each graph (nodes, edges, connected components,
  largest-component fraction, degree percentiles).

Outputs
-------

- ``<output_prefix>_summary.json`` - full report.
- Stdout - human-readable text summary.

Typical usage
-------------

    python -m two_graph_fusion.scripts.f6_diagnostic \
        --gb-prefix two_graph_fusion/cache/gb_9shard \
        --edge-csv twibot22/edge.csv \
        --output-prefix two_graph_fusion/cache/f6_9shard
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from two_graph_fusion.graphs import (
    DEFAULT_FOLLOW_RELATIONS,
    acceptance_verdict,
    build_trust_graph,
    f6_report,
    graph_summary,
    restrict_to_common_nodes,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the F6 overlap diagnostic between G_B and G_T.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gb-prefix",
        type=Path,
        required=True,
        help="Output prefix used by build_gb (reads _edges.csv and _nodes.csv).",
    )
    parser.add_argument(
        "--edge-csv",
        type=Path,
        default=Path("twibot22/edge.csv"),
        help="TwiBot-22 edge.csv.",
    )
    parser.add_argument(
        "--relations",
        nargs="+",
        default=sorted(DEFAULT_FOLLOW_RELATIONS),
        help="Edge relations to count as trust edges.",
    )
    parser.add_argument(
        "--spectral-k",
        type=int,
        default=10,
        help="Number of leading non-trivial eigenvectors to compare.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Prefix for the report JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _load_gb(prefix: Path) -> nx.Graph:
    edges_path = prefix.with_name(prefix.name + "_edges.csv")
    nodes_path = prefix.with_name(prefix.name + "_nodes.csv")
    nodes = pd.read_csv(nodes_path, index_col=0)
    edges = pd.read_csv(edges_path)
    g = nx.Graph()
    g.add_nodes_from(nodes.index)
    for _, row in edges.iterrows():
        g.add_edge(row["source"], row["target"], weight=float(row["weight"]))
    return g


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    g_b = _load_gb(args.gb_prefix)
    log.info(
        "loaded G_B: nodes=%d edges=%d",
        g_b.number_of_nodes(),
        g_b.number_of_edges(),
    )

    log.info("scanning %s ...", args.edge_csv)
    trust_result = build_trust_graph(
        args.edge_csv, user_ids=list(g_b.nodes()), relations=args.relations
    )
    g_t = trust_result.graph
    log.info(
        "G_T built: nodes=%d edges=%d (matched rows=%d, scanned=%d)",
        g_t.number_of_nodes(),
        g_t.number_of_edges(),
        trust_result.n_edges_matched,
        trust_result.n_edges_scanned,
    )

    g_b_c, g_t_c = restrict_to_common_nodes(g_b, g_t)
    report = f6_report(g_b_c, g_t_c, spectral_k=args.spectral_k)
    verdict = acceptance_verdict(report)

    summary = {
        "inputs": {
            "gb_prefix": str(args.gb_prefix),
            "edge_csv": str(args.edge_csv),
            "relations": list(args.relations),
            "spectral_k_requested": args.spectral_k,
        },
        "trust_graph_build": {
            "n_user_ids": trust_result.n_user_ids,
            "n_edges_scanned": trust_result.n_edges_scanned,
            "n_edges_matched": trust_result.n_edges_matched,
            "n_unique_edges": trust_result.n_unique_edges,
            "relations": sorted(trust_result.relations),
        },
        "summary_g_b": graph_summary(g_b_c),
        "summary_g_t": graph_summary(g_t_c),
        "f6": report.__dict__,
        "acceptance": verdict,
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_name(
        args.output_prefix.name + "_summary.json"
    )
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("wrote %s", summary_path)

    print("\n=== G_B summary ===")
    print(json.dumps(summary["summary_g_b"], indent=2))
    print("\n=== G_T summary ===")
    print(json.dumps(summary["summary_g_t"], indent=2))
    print("\n=== F6 ===")
    print(json.dumps(summary["f6"], indent=2))
    print("\n=== Acceptance ===")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
