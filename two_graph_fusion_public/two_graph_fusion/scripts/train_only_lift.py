"""A4 — train-only homophily lift vs. all-label lift, every dataset/graph.

The thesis computes the homophily-lift gate over ALL labeled edges
(including val/test). A deployed gate only has the seed (train) labels.
This script recomputes lift on the train-induced labeled subgraph — i.e.
``edge_homophily`` with the label map restricted to train users, which
also recomputes the chance baseline from train labels only — and compares
it with the all-label value.

Covered:

- Cresci-2017 (default subsets): G_T (retweet+reply), G_B — stratified
  splits, seeds 0-9 (mean ± std reported).
- Cresci-2015: G_T follow, G_T interaction, G_B — same protocol.
- TwiBot-22 official-split at-scale cohort (n_min=25): cached
  ``threshold25_tt`` graphs + split column.
- TwiBot-20: official split, G_T follow (no G_B; no timestamps).

Usage::

    .venv/bin/python3 -m two_graph_fusion.scripts.train_only_lift \\
        --output two_graph_fusion/cache/train_only_lift.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from two_graph_fusion.features import (
    ZSCORE_FEATURE_COLUMNS,
    CircadianConfig,
    PipelineConfig,
    compute_behavioral_features,
)
from two_graph_fusion.graphs import (
    build_mutual_knn_graph_auto,
    edge_homophily,
    feature_matrix_for_qualifying_users,
)
from two_graph_fusion.propagation import DEFAULT_FEATURES
from two_graph_fusion.propagation.tvt_evaluation import make_stratified_splits

logger = logging.getLogger(__name__)

SEEDS = tuple(range(10))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train-only vs all-label homophily lift (A4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cresci2017-root", type=Path, default=Path("cresci_2017_datasets_full"))
    p.add_argument("--cresci2015-root", type=Path, default=Path("cresci-15-orig"))
    p.add_argument("--twibot20-root", type=Path, default=Path("Twibot-20-2"))
    p.add_argument("--threshold25-dir", type=Path,
                   default=Path("two_graph_fusion/cache/threshold25_tt"))
    p.add_argument("--gb-k", type=int, default=10)
    p.add_argument("--n-min-events", type=int, default=10)
    p.add_argument("--output", type=Path,
                   default=Path("two_graph_fusion/cache/train_only_lift.json"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _lift_pair(g: nx.Graph, label_map: dict, train_users: set) -> dict:
    """All-label and train-only homophily reports for one graph."""
    full = edge_homophily(g, label_map)
    train_labels = {u: l for u, l in label_map.items() if u in train_users}
    train = edge_homophily(g, train_labels)
    return {"all_labels": full.to_dict(), "train_only": train.to_dict()}


def _multi_seed(g: nx.Graph, label_map: dict, qualifying: pd.DataFrame) -> dict:
    """Train-only lift across stratified splits (seeds 0-9) + all-label value."""
    full = edge_homophily(g, label_map)
    per_seed = {}
    for seed in SEEDS:
        splits = make_stratified_splits(
            qualifying, label_col="label", bot_label="bot", seed=seed,
        )
        train_labels = {u: label_map[u] for u in splits.train}
        per_seed[seed] = edge_homophily(g, train_labels).to_dict()
    lifts = np.array([per_seed[s]["homophily_lift"] for s in SEEDS])
    return {
        "all_labels": full.to_dict(),
        "train_only_per_seed": {str(s): per_seed[s] for s in SEEDS},
        "train_only_lift_mean": float(lifts.mean()),
        "train_only_lift_std": float(lifts.std(ddof=1)),
        "train_only_lift_seed0": float(lifts[0]),
    }


def _cresci_features(all_timestamps: dict, label_map: dict, n_min: int) -> pd.DataFrame:
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


def _build_gb(zscored_qualifying: pd.DataFrame, k: int) -> nx.Graph:
    feat_df = feature_matrix_for_qualifying_users(
        zscored_qualifying, feature_columns=list(ZSCORE_FEATURE_COLUMNS)
    )
    gb = build_mutual_knn_graph_auto(
        feat_df, k=k, similarity_floor=0.0, metric="euclidean",
    ).graph
    gb.add_nodes_from(zscored_qualifying.index)
    return gb


def _graph_from_edges(nodes, edges) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def cresci2017(root: Path, k: int, n_min: int) -> dict:
    from two_graph_fusion.datasets.cresci2017 import (
        load_labels, load_reply_edges, load_retweet_edges, load_timestamps,
    )
    label_map = load_labels(root, include_optional=False)
    ts_map = load_timestamps(root, user_filter=set(label_map.keys()),
                             include_optional=False)
    all_ts = {u: t for u, t in ts_map.items() if u in label_map}
    qualifying = _cresci_features(all_ts, label_map, n_min)
    users = set(qualifying.index)
    label_q = qualifying["label"].to_dict()

    g_b = _build_gb(qualifying, k)
    edges = (load_retweet_edges(root, user_filter=users, include_optional=False)
             + load_reply_edges(root, user_filter=users, include_optional=False))
    g_t = _graph_from_edges(users, edges)
    return {
        "g_t_interaction": _multi_seed(g_t, label_q, qualifying),
        "g_b": _multi_seed(g_b, label_q, qualifying),
    }


def cresci2015(root: Path, k: int, n_min: int) -> dict:
    from two_graph_fusion.datasets.cresci2015 import (
        load_follow_edges, load_labels, load_reply_edges,
        load_retweet_edges, load_timestamps,
    )
    label_map = load_labels(root)
    ts_map = load_timestamps(root, user_filter=set(label_map.keys()))
    all_ts = {u: t for u, t in ts_map.items() if u in label_map}
    qualifying = _cresci_features(all_ts, label_map, n_min)
    users = set(qualifying.index)
    label_q = qualifying["label"].to_dict()

    g_b = _build_gb(qualifying, k)
    inter = (load_retweet_edges(root, user_filter=users)
             + load_reply_edges(root, user_filter=users))
    follow = load_follow_edges(root, user_filter=users)
    return {
        "g_t_follow": _multi_seed(_graph_from_edges(users, follow), label_q, qualifying),
        "g_t_interaction": _multi_seed(_graph_from_edges(users, inter), label_q, qualifying),
        "g_b": _multi_seed(g_b, label_q, qualifying),
    }


def twibot22_threshold25(cache_dir: Path) -> dict:
    feats = pd.read_csv(cache_dir / "features_zscored.csv", index_col=0)
    # Restrict to the qualifying graph cohort: rows with all behavioral
    # features defined. features_zscored.csv retains NaN-feature users that
    # were dropped before graph construction; including them in the chance
    # baseline mismatches the edge population and inflates the lift (the
    # 697,710-node cohort is 7.2% bot, the 750,373 feature rows are 8.8%).
    feats = feats.dropna(subset=list(ZSCORE_FEATURE_COLUMNS))
    label_map = feats["label"].to_dict()
    train_users = set(feats.index[feats["split"] == "train"])
    out = {"split_counts": feats["split"].value_counts().to_dict()}
    for name, fname in [("g_b", "gb_edges.csv"),
                        ("g_t_follow", "gt_follow_edges.csv"),
                        ("g_t_interaction", "gt_interaction_edges.csv")]:
        edges_df = pd.read_csv(cache_dir / fname)
        g = _graph_from_edges(
            feats.index, zip(edges_df["source"], edges_df["target"])
        )
        out[name] = _lift_pair(g, label_map, train_users)
    return out


def twibot20(root: Path) -> dict:
    from two_graph_fusion.datasets.twibot20 import (
        induced_directed_edges, load_twibot20,
    )
    data = load_twibot20(root, include_support=False)
    labeled = data.users[data.users["label"].isin({"human", "bot"})]
    label_map = labeled["label"].to_dict()
    train_users = set(labeled.index[labeled["split"] == "train"])
    directed = induced_directed_edges(data.follow_edges, set(labeled.index))
    g_t = _graph_from_edges(labeled.index, directed)
    return {"g_t_follow": _lift_pair(g_t, label_map, train_users)}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    report: dict = {}

    for key, fn in [
        ("cresci2017", lambda: cresci2017(args.cresci2017_root, args.gb_k, args.n_min_events)),
        ("cresci2015", lambda: cresci2015(args.cresci2015_root, args.gb_k, args.n_min_events)),
        ("twibot22_threshold25", lambda: twibot22_threshold25(args.threshold25_dir)),
        ("twibot20", lambda: twibot20(args.twibot20_root)),
    ]:
        logger.info("=== %s ===", key)
        try:
            report[key] = fn()
        except Exception as exc:  # keep going; report what failed
            logger.exception("%s failed", key)
            report[key] = {"error": str(exc)}

    report["elapsed_s"] = round(time.time() - t0, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(report, fh, indent=2, default=float)
    logger.info("wrote %s", args.output)

    print(f"\n{'dataset/graph':<38} {'all-label lift':>15} {'train-only lift':>22}")
    for ds, graphs in report.items():
        if not isinstance(graphs, dict) or ds == "elapsed_s":
            continue
        for gname, rep in graphs.items():
            if not isinstance(rep, dict) or "all_labels" not in rep:
                continue
            full = rep["all_labels"]["homophily_lift"]
            if "train_only_lift_mean" in rep:
                to = (f"{rep['train_only_lift_mean']:+.4f} ± "
                      f"{rep['train_only_lift_std']:.4f}")
            else:
                to = f"{rep['train_only']['homophily_lift']:+.4f}"
            print(f"{ds + '/' + gname:<38} {full:>+15.4f} {to:>22}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
