"""Build the behavioral graph ``G_B`` from a z-scored feature CSV.

Reads ``<features_prefix>_zscored.csv`` (produced by
``compute_features_subset``), builds the mutual k-NN graph over the
five-dimensional feature vector, and writes:

- ``<output_prefix>_edges.csv``        edge list ``source,target,weight``.
- ``<output_prefix>_nodes.csv``        node list ``user_id,label,split``.
- ``<output_prefix>_summary.json``     build parameters + graph summary
                                       (nodes, edges, components, LCC, degree).

Typical usage
-------------

    python -m two_graph_fusion.scripts.build_gb \
        --features-prefix two_graph_fusion/cache/subset_features_9shard \
        --k 10 \
        --output-prefix two_graph_fusion/cache/gb_9shard
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from two_graph_fusion.features import ZSCORE_FEATURE_COLUMNS
from two_graph_fusion.graphs import (
    build_mutual_knn_graph,
    feature_matrix_for_qualifying_users,
    graph_summary,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build G_B as a mutual k-NN graph over the z-scored features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features-prefix",
        type=Path,
        required=True,
        help="Output prefix used by compute_features_subset.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbours each node nominates (mutual k-NN).",
    )
    parser.add_argument(
        "--similarity-floor",
        type=float,
        default=0.0,
        help=(
            "Cosine similarities at or below this value are dropped "
            "(ignored when --metric is 'euclidean')."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help=(
            "Distance metric for mutual k-NN. Cosine is the pivot-plan "
            "default; Euclidean is the RED-11 fallback for tightly "
            "clustered 5-D feature spaces."
        ),
    )
    parser.add_argument(
        "--distance-ceiling",
        type=float,
        default=None,
        help="(Euclidean only) Drop edges whose distance exceeds this.",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=list(ZSCORE_FEATURE_COLUMNS),
        help="Which z-scored columns to use as the embedding.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Prefix for output edges / nodes / summary files.",
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    z_path = args.features_prefix.with_name(
        args.features_prefix.name + "_zscored.csv"
    )
    zscored = pd.read_csv(z_path, index_col=0)
    log.info("loaded %s: %d rows", z_path, len(zscored))

    missing = [c for c in args.feature_columns if c not in zscored.columns]
    if missing:
        raise ValueError(
            f"feature columns missing from {z_path}: {missing}"
        )

    feat_df = feature_matrix_for_qualifying_users(
        zscored, feature_columns=args.feature_columns
    )
    log.info(
        "feature matrix: %d users x %d features",
        feat_df.shape[0],
        feat_df.shape[1],
    )

    result = build_mutual_knn_graph(
        feat_df,
        k=args.k,
        similarity_floor=args.similarity_floor,
        metric=args.metric,
        distance_ceiling=args.distance_ceiling,
    )
    log.info(
        "G_B built: nodes=%d edges=%d",
        result.graph.number_of_nodes(),
        result.graph.number_of_edges(),
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    edges_path = args.output_prefix.with_name(args.output_prefix.name + "_edges.csv")
    nodes_path = args.output_prefix.with_name(args.output_prefix.name + "_nodes.csv")
    summary_path = args.output_prefix.with_name(
        args.output_prefix.name + "_summary.json"
    )

    pd.DataFrame(
        [
            {"source": u, "target": v, "weight": float(d.get("weight", 1.0))}
            for u, v, d in result.graph.edges(data=True)
        ]
    ).to_csv(edges_path, index=False)

    keep_meta = [c for c in ("label", "split") if c in zscored.columns]
    nodes_df = zscored.loc[list(result.graph.nodes())][keep_meta].copy()
    nodes_df.index.name = "user_id"
    nodes_df.to_csv(nodes_path)

    summary = {
        "inputs": {
            "features_prefix": str(args.features_prefix),
            "feature_columns": list(args.feature_columns),
        },
        "build_params": {
            "k": args.k,
            "similarity_floor": args.similarity_floor,
            "metric": args.metric,
            "distance_ceiling": args.distance_ceiling,
        },
        "build_result": {
            "n_input_users": result.n_input_users,
            "n_used_users": result.n_used_users,
            "n_mutual_edges": result.n_mutual_edges,
        },
        "graph_summary": graph_summary(result.graph),
    }
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("wrote %s, %s, %s", edges_path, nodes_path, summary_path)

    print(json.dumps(summary["graph_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
