"""Cresci-2015 full sweep: both engines, train/val/test protocol.

Raw "Fame for Sale" CSV release (``cresci-15-orig/``). Mirrors the
Cresci-2017 runner (M11): SybilSCAR + SybilHP over {G_T, G_B, fusion} ×
{neutral, LR prior} on a fixed stratified split, hyperparameters tuned on
val, metrics reported on test. Behavioral G_B is available here because
the tweets carry timestamps — unlike the BotRGCN tensor release.

G_T for the fusion matrix defaults to the reply+retweet *interaction*
graph (for direct comparability with Cresci-2017); ``--gt follow`` uses
the friends/followers *follow* graph instead. The follow-graph homophily
gate is always reported as well.

Usage::

    .venv/bin/python3 -m two_graph_fusion.scripts.run_cresci2015_tvt \\
        --cresci-root cresci-15-orig \\
        --output-dir two_graph_fusion/cache/cresci2015_tvt \\
        --n-min-events 10 --gb-k 10 --seed 0
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

from two_graph_fusion.datasets.cresci2015 import (
    load_follow_edges,
    load_labels,
    load_reply_edges,
    load_retweet_edges,
    load_timestamps,
)
from two_graph_fusion.features import (
    ZSCORE_FEATURE_COLUMNS,
    BehavioralFeatures,
    CircadianConfig,
    PipelineConfig,
    compute_behavioral_features,
)
from two_graph_fusion.graphs import (
    build_mutual_knn_graph_auto,
    edge_homophily,
    feature_matrix_for_qualifying_users,
    graph_summary,
)
from two_graph_fusion.propagation import DEFAULT_FEATURES
from two_graph_fusion.propagation.tvt_evaluation import (
    make_stratified_splits,
    run_tvt_evaluation,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cresci-2015 train/val/test sweep (SybilSCAR + SybilHP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cresci-root", type=Path, default=Path("cresci-15-orig"))
    p.add_argument("--output-dir", type=Path, default=Path("two_graph_fusion/cache/cresci2015_tvt"))
    p.add_argument("--n-min-events", type=int, default=10)
    p.add_argument("--gb-k", type=int, default=10)
    p.add_argument("--gb-metric", choices=["cosine", "euclidean"], default="euclidean")
    p.add_argument("--gt", choices=["interaction", "follow"], default="interaction",
                   help="Which graph feeds the fusion matrix (homophily reported for both).")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rule", choices=["logistic", "linear"], default="logistic",
                   help="SybilSCAR update rule: 'linear' = canonical published rule, "
                        "'logistic' = sigmoid-stabilised variant (default).")
    p.add_argument("--hygiene", action="store_true",
                   help="Leakage-hygiene mode: D_shape honest reference restricted to "
                        "train-split honest users; feature z-scoring fit on train only.")
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
    if not args.cresci_root.exists():
        log.error("cresci root not found: %s", args.cresci_root)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== Cresci-2015 TVT sweep (gt=%s, seed=%d) ===", args.gt, args.seed)

    # ------------------------------------------------------------------
    # 1. Labels, timestamps, behavioral features
    # ------------------------------------------------------------------
    label_map = load_labels(args.cresci_root)
    ts_map = load_timestamps(args.cresci_root, user_filter=set(label_map.keys()))
    all_timestamps = {uid: ts for uid, ts in ts_map.items() if uid in label_map}
    t_start = min(min(v) for v in all_timestamps.values())
    t_end = max(max(v) for v in all_timestamps.values())
    honest_users = frozenset(uid for uid, lbl in label_map.items() if lbl == "human")

    config = PipelineConfig(
        n_min_events=args.n_min_events,
        circadian=CircadianConfig(mapping="sigmoid", seed=0),
        reference_user_ids=honest_users,
        reference_iat_cap=200_000,
    )
    features: BehavioralFeatures = compute_behavioral_features(
        user_timestamps=all_timestamps, window=(t_start, t_end), config=config,
    )
    zscored = features.zscored.copy()
    zscored["label"] = zscored.index.map(label_map)

    qualifying = (
        zscored
        .dropna(how="any", subset=list(DEFAULT_FEATURES))
        .pipe(lambda df: df[df["label"].isin({"human", "bot"})])
    )
    qualifying_users = set(qualifying.index)
    label_map_q = qualifying["label"].to_dict()
    n_q_human = int((qualifying["label"] == "human").sum())
    n_q_bot = int((qualifying["label"] == "bot").sum())
    log.info("qualifying: %d (human=%d bot=%d bot%%=%.1f)",
             len(qualifying), n_q_human, n_q_bot, 100 * n_q_bot / len(qualifying))

    # ------------------------------------------------------------------
    # 1b. Stratified train/val/test split (depends only on the qualifying
    #     index + labels, so it is identical with or without --hygiene)
    # ------------------------------------------------------------------
    splits = make_stratified_splits(
        qualifying, label_col="label", bot_label="bot",
        seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac,
    )
    log.info("splits: train=%d val=%d test=%d", len(splits.train), len(splits.val), len(splits.test))

    if args.hygiene:
        # Second pass: refit the D_shape honest reference on train-split
        # honest users only, and fit the z-scaling on train rows only.
        train_set = set(splits.train)
        train_honest = frozenset(
            u for u in train_set if label_map_q.get(u) == "human"
        )
        log.info("hygiene: %d train-honest reference users (of %d labeled honest)",
                 len(train_honest), len(honest_users))
        config_h = PipelineConfig(
            n_min_events=args.n_min_events,
            circadian=CircadianConfig(mapping="sigmoid", seed=0),
            reference_user_ids=train_honest,
            reference_iat_cap=200_000,
        )
        features_h: BehavioralFeatures = compute_behavioral_features(
            user_timestamps=all_timestamps, window=(t_start, t_end), config=config_h,
        )
        raw_h = features_h.raw
        train_rows = raw_h.index.isin(train_set)
        zscored_h = pd.DataFrame(index=raw_h.index)
        for z_col in ZSCORE_FEATURE_COLUMNS:
            raw_col = z_col[:-2]
            col = raw_h[raw_col]
            mu = col[train_rows].mean(skipna=True)
            sigma = col[train_rows].std(skipna=True, ddof=0)
            zscored_h[z_col] = (col - mu) / sigma
        zscored_h["label"] = zscored_h.index.map(label_map)
        qualifying_h = (
            zscored_h
            .dropna(how="any", subset=list(DEFAULT_FEATURES))
            .pipe(lambda df: df[df["label"].isin({"human", "bot"})])
        )
        if set(qualifying_h.index) != qualifying_users:
            raise RuntimeError("hygiene pass changed the qualifying set")
        qualifying = qualifying_h.loc[qualifying.index]
        zscored = zscored_h

    # ------------------------------------------------------------------
    # 2. Graphs: G_B (behavioral k-NN), interaction + follow G_T
    # ------------------------------------------------------------------
    feat_df = feature_matrix_for_qualifying_users(
        zscored, feature_columns=list(ZSCORE_FEATURE_COLUMNS)
    ).loc[lambda df: df.index.isin(qualifying_users)]
    gb_result = build_mutual_knn_graph_auto(
        feat_df, k=args.gb_k, similarity_floor=0.0, metric=args.gb_metric,
    )
    g_b = gb_result.graph
    g_b.add_nodes_from(qualifying_users)

    retweet_edges = load_retweet_edges(args.cresci_root, user_filter=qualifying_users)
    reply_edges = load_reply_edges(args.cresci_root, user_filter=qualifying_users)
    interaction_edges = retweet_edges + reply_edges
    follow_edges = load_follow_edges(args.cresci_root, user_filter=qualifying_users)

    g_t_interaction = nx.Graph()
    g_t_interaction.add_nodes_from(qualifying_users)
    g_t_interaction.add_edges_from(interaction_edges)
    g_t_follow = nx.Graph()
    g_t_follow.add_nodes_from(qualifying_users)
    g_t_follow.add_edges_from(follow_edges)

    if args.gt == "follow":
        g_t, trust_edges_directed = g_t_follow, follow_edges
    else:
        g_t, trust_edges_directed = g_t_interaction, interaction_edges

    gb_sum = graph_summary(g_b)
    gt_int_sum = graph_summary(g_t_interaction)
    gt_fol_sum = graph_summary(g_t_follow)
    log.info("G_B: nodes=%d edges=%d lcc=%.3f", gb_sum["n_nodes"], gb_sum["n_edges"], gb_sum["lcc_fraction"])
    log.info("G_T interaction: edges=%d lcc=%.3f (directed=%d)",
             gt_int_sum["n_edges"], gt_int_sum["lcc_fraction"], len(interaction_edges))
    log.info("G_T follow: edges=%d lcc=%.3f (directed=%d)",
             gt_fol_sum["n_edges"], gt_fol_sum["lcc_fraction"], len(follow_edges))

    hom = {
        "g_b": _homophily_report(g_b, label_map_q),
        "g_t_interaction": _homophily_report(g_t_interaction, label_map_q),
        "g_t_follow": _homophily_report(g_t_follow, label_map_q),
    }

    # ------------------------------------------------------------------
    # 3. Full matrix (on the selected G_T)
    # ------------------------------------------------------------------
    cfg_results, extras = run_tvt_evaluation(
        qualifying=qualifying, g_t=g_t, g_b=g_b,
        trust_edges_directed=trust_edges_directed, splits=splits,
        linearized=(args.rule == "linear"),
        return_extras=True,
    )

    # ------------------------------------------------------------------
    # 4. Persist + print
    # ------------------------------------------------------------------
    payload = {
        "dataset": "cresci-2015 (raw Fame-for-Sale CSV)",
        "gt_for_matrix": args.gt,
        "rule": args.rule,
        "hygiene": bool(args.hygiene),
        "seed": args.seed,
        "n_qualifying": len(qualifying),
        "n_human": n_q_human,
        "n_bot": n_q_bot,
        "bot_pct": round(n_q_bot / len(qualifying), 4),
        "splits": {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
        "gb": {"k": args.gb_k, "metric": args.gb_metric, **gb_sum},
        "gt_interaction": {"directed_edges": len(interaction_edges), **gt_int_sum},
        "gt_follow": {"directed_edges": len(follow_edges), **gt_fol_sum},
        "homophily": hom,
        "results": [r.to_dict() for r in cfg_results],
        "lr_prior": extras["lr_prior"],
        "lr_prior_drop_l": extras["lr_prior_drop_l"],
        "test_scores": extras["test_scores"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    rule_tag = "" if args.rule == "logistic" else f"_{args.rule}"
    k_tag = "" if args.gb_k == 10 else f"_k{args.gb_k}"
    hyg_tag = "_hygiene" if args.hygiene else ""
    out_json = args.output_dir / f"results_gt{args.gt}{rule_tag}{k_tag}{hyg_tag}_seed{args.seed}.json"
    with out_json.open("w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    log.info("wrote %s", out_json)

    print(f"\n=== Cresci-2015 TVT sweep (gt={args.gt}, seed={args.seed}) ===")
    print(f"Qualifying: {len(qualifying):,}  (human={n_q_human:,} bot={n_q_bot:,} "
          f"bot%={100*n_q_bot/len(qualifying):.1f}%)  "
          f"split {len(splits.train)}/{len(splits.val)}/{len(splits.test)}")
    print("\nHomophily (soft gate: lift ≥ 0.05):")
    for gname, lbl in [("g_t_interaction", "G_T interaction"),
                       ("g_t_follow", "G_T follow"), ("g_b", "G_B")]:
        h = hom[gname]
        gate = "PASS" if h["homophily_lift"] >= 0.05 else "fail"
        bd = h["edge_breakdown"]
        print(f"  {lbl:<16} edges={h['n_edges_labeled']:>7,}  h={h['homophily']:.4f}  "
              f"chance={h['chance_baseline']:.4f}  lift={h['homophily_lift']:+.4f}  {gate}  "
              f"(bb={bd['bot_bot']} hh={bd['human_human']} hb={bd['human_bot']})")

    print("\nTest metrics (val-tuned hyperparameters, val-selected threshold):")
    print(f"  {'engine':<10} {'graph':<7} {'prior':<8} {'AUC':>7} {'PR-AUC':>7} "
          f"{'Acc':>7} {'F1':>7}  hparams")
    for r in cfg_results:
        print(f"  {r.engine:<10} {r.graph:<7} {r.prior:<8} "
              f"{r.test_auc:>7.4f} {r.test_pr_auc:>7.4f} "
              f"{r.test_acc:>7.4f} {r.test_f1:>7.4f}  {r.best_hparams}")

    log.info("done in %.1f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
