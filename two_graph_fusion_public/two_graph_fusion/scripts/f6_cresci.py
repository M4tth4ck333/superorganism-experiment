"""A9 — F6 overlap diagnostic (G_B vs G_T) for the Cresci datasets.

The TwiBot-22 F6 runs go through ``f6_diagnostic`` (cached ``build_gb``
artefacts + ``edge.csv``); the Cresci TVT runners never persisted their
graphs, so this script rebuilds them exactly as the runners do (canonical
config: n_min=10, mutual k-NN k=10 euclidean, circadian seed 0) and
computes edge Jaccard / triangle Jaccard / spectral cosine overlap.

Usage::

    .venv/bin/python3 -m two_graph_fusion.scripts.f6_cresci \\
        --dataset cresci2015 --output two_graph_fusion/cache/f6_cresci2015.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import networkx as nx
import pandas as pd

from two_graph_fusion.features import (
    ZSCORE_FEATURE_COLUMNS,
    CircadianConfig,
    PipelineConfig,
    compute_behavioral_features,
)
from two_graph_fusion.graphs import (
    build_mutual_knn_graph_auto,
    f6_report,
    feature_matrix_for_qualifying_users,
    graph_summary,
    restrict_to_common_nodes,
)
from two_graph_fusion.propagation import DEFAULT_FEATURES

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F6 overlap (G_B vs G_T) for Cresci-2015 / Cresci-2017.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", choices=["cresci2015", "cresci2017"], required=True)
    p.add_argument("--cresci2017-root", type=Path, default=Path("cresci_2017_datasets_full"))
    p.add_argument("--cresci2015-root", type=Path, default=Path("cresci-15-orig"))
    p.add_argument("--n-min-events", type=int, default=10)
    p.add_argument("--gb-k", type=int, default=10)
    p.add_argument("--spectral-k", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _qualifying(all_timestamps: dict, label_map: dict, n_min: int) -> pd.DataFrame:
    t_start = min(min(v) for v in all_timestamps.values())
    t_end = max(max(v) for v in all_timestamps.values())
    honest = frozenset(u for u, l in label_map.items() if l == "human")
    config = PipelineConfig(
        n_min_events=n_min,
        circadian=CircadianConfig(mapping="sigmoid", seed=0),
        reference_user_ids=honest,
        reference_iat_cap=200_000,
    )
    feats = compute_behavioral_features(
        user_timestamps=all_timestamps, window=(t_start, t_end), config=config,
    )
    zscored = feats.zscored.copy()
    zscored["label"] = zscored.index.map(label_map)
    return (
        zscored
        .dropna(how="any", subset=list(DEFAULT_FEATURES))
        .pipe(lambda df: df[df["label"].isin({"human", "bot"})])
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.time()

    if args.dataset == "cresci2015":
        from two_graph_fusion.datasets.cresci2015 import (
            load_follow_edges, load_labels, load_reply_edges,
            load_retweet_edges, load_timestamps,
        )
        root = args.cresci2015_root
        label_map = load_labels(root)
        ts_map = load_timestamps(root, user_filter=set(label_map.keys()))
    else:
        from two_graph_fusion.datasets.cresci2017 import (
            load_labels, load_reply_edges, load_retweet_edges, load_timestamps,
        )
        root = args.cresci2017_root
        label_map = load_labels(root, include_optional=False)
        ts_map = load_timestamps(root, user_filter=set(label_map.keys()),
                                 include_optional=False)

    all_ts = {u: t for u, t in ts_map.items() if u in label_map}
    qualifying = _qualifying(all_ts, label_map, args.n_min_events)
    users = set(qualifying.index)
    logger.info("qualifying users: %d", len(users))

    feat_df = feature_matrix_for_qualifying_users(
        qualifying, feature_columns=list(ZSCORE_FEATURE_COLUMNS)
    )
    g_b = build_mutual_knn_graph_auto(
        feat_df, k=args.gb_k, similarity_floor=0.0, metric="euclidean",
    ).graph
    g_b.add_nodes_from(users)

    interaction = (load_retweet_edges(root, user_filter=users)
                   + load_reply_edges(root, user_filter=users)) \
        if args.dataset == "cresci2015" else \
        (load_retweet_edges(root, user_filter=users, include_optional=False)
         + load_reply_edges(root, user_filter=users, include_optional=False))

    g_ts: dict[str, nx.Graph] = {}
    g_int = nx.Graph()
    g_int.add_nodes_from(users)
    g_int.add_edges_from(interaction)
    g_ts["g_t_interaction"] = g_int
    if args.dataset == "cresci2015":
        follow = load_follow_edges(root, user_filter=users)
        g_fol = nx.Graph()
        g_fol.add_nodes_from(users)
        g_fol.add_edges_from(follow)
        g_ts["g_t_follow"] = g_fol

    summary: dict = {
        "dataset": args.dataset,
        "n_qualifying": len(users),
        "gb_k": args.gb_k,
        "summary_g_b": graph_summary(g_b),
    }
    for name, g_t in g_ts.items():
        g_b_c, g_t_c = restrict_to_common_nodes(g_b, g_t)
        rep = f6_report(g_b_c, g_t_c, spectral_k=args.spectral_k)
        summary[name] = {
            "summary_g_t": graph_summary(g_t),
            "f6": rep.__dict__,
        }
        logger.info("%s: edge_jaccard=%.5f triangle_jaccard=%.5f spectral_mean_cos=%.4f",
                    name, rep.edge_jaccard, rep.triangle_jaccard,
                    rep.spectral_mean_cosine)

    summary["elapsed_s"] = round(time.time() - t0, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    logger.info("wrote %s", args.output)

    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("summary_g_b",)}, indent=2, default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
