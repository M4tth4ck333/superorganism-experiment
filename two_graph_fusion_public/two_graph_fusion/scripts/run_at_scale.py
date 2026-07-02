"""Server-oriented end-to-end pipeline for TwiBot-22 at scale.

Runs the full two-graph fusion validation stack on a large fraction of
the labeled TwiBot-22 population. Stages are resumable via marker files
under ``<work-dir>/stages/``.

Stages
------

1. ``extract``     - checkpointed tweet scan -> ``timestamps.parquet``
2. ``features``    - five features + z-scores -> ``features_*_zscored.csv``
3. ``follow_cache`` - (optional) one-time ``edge.csv`` scan if cache missing
4. ``gb``          - FAISS mutual k-NN ``G_B`` -> ``gb_*_edges.csv``
5. ``gt``          - induced ``G_T`` from follow-edge cache -> ``gt_*_edges.csv``
6. ``eval``        - official train/test evaluation (default) or K-fold
7. ``f6``          - overlap diagnostic

Recommended server profile
--------------------------

- **Do not** use cross-class snowball sampling at this scale; use
  ``--user-scope train_test`` so official splits apply and G_T homophily
  is measured on the benchmark cohort.
- Install: ``uv pip install faiss-cpu pyarrow scikit-learn``
- Disk: ~50 GB for checkpoints + features + graphs (varies with qualifying
  user count).
- RAM: 32-64 GB recommended for propagation on 100k+ node graphs.

Example (full pipeline, official benchmark protocol)::

    python -m two_graph_fusion.scripts.run_at_scale \
        --twibot22-root /data/twibot22 \
        --work-dir /data/tgf_at_scale \
        --user-scope train_test \
        --stages all

Example (resume after extraction finished)::

    python -m two_graph_fusion.scripts.run_at_scale \
        --work-dir /data/tgf_at_scale \
        --stages features gb gt eval f6
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

from two_graph_fusion.analysis import render_f5_scatter
from two_graph_fusion.datasets.twibot22 import discover_twibot22, load_labels_and_splits
from two_graph_fusion.features import ZSCORE_FEATURE_COLUMNS
from two_graph_fusion.graphs import (
    build_mutual_knn_graph_auto,
    f6_report,
    feature_matrix_for_qualifying_users,
    graph_summary,
    restrict_to_common_nodes,
)
from two_graph_fusion.graphs.trust_graph import (
    cache_all_user_follow_edges,
    load_follow_edge_cache,
)
from two_graph_fusion.propagation import (
    DEFAULT_FEATURES,
    dump_results_json,
    official_result_to_dict,
    run_kfold_evaluation,
    run_official_split_evaluation,
    run_propagation,
    summarise_with_stderr,
    relation_from_graph,
    fit_predict_lr_prior,
)
from two_graph_fusion.scripts import compute_features_at_scale
from two_graph_fusion.scripts import extract_twibot22_at_scale


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="At-scale TwiBot-22 two-graph fusion pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--twibot22-root", type=Path, default=Path("twibot22"))
    parser.add_argument("--work-dir", type=Path, default=Path("two_graph_fusion/cache/at_scale"))
    parser.add_argument(
        "--follow-cache",
        type=Path,
        default=None,
        help="User-user follow cache CSV (default: <work-dir>/twibot22_follow_edges.csv).",
    )
    parser.add_argument(
        "--user-scope",
        choices=["all_labeled", "train_only", "train_test"],
        default="train_test",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        help=(
            "Stages to run: extract, features, follow_cache, gb, gt, eval, f6, "
            "or 'all'. Skips stages whose marker file exists unless --force."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-tweets-per-user", type=int, default=500)
    parser.add_argument(
        "--incremental-merge",
        action="store_true",
        help="Pass through to extract: rolling merge to save disk on local runs.",
    )
    parser.add_argument("--gb-k", type=int, default=10)
    parser.add_argument("--gb-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--gb-similarity-floor", type=float, default=0.0)
    parser.add_argument(
        "--eval-protocol",
        choices=["official", "kfold"],
        default="official",
        help=(
            "'official' = TwiBot-22 train seeds / test AUC (recommended at "
            "scale). 'kfold' = stratified 5-fold CV (subset-scale only)."
        ),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--w-t", type=float, default=0.5)
    parser.add_argument("--w-b", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _stage_marker(work_dir: Path, name: str) -> Path:
    return work_dir / "stages" / f"{name}.done"


def _mark_done(work_dir: Path, name: str, info: dict) -> None:
    path = _stage_marker(work_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(info, fh, indent=2, default=float)


def _should_run(work_dir: Path, name: str, stages: set[str], force: bool) -> bool:
    if name not in stages and "all" not in stages:
        return False
    return force or not _stage_marker(work_dir, name).exists()


def _save_graph_edges(graph: nx.Graph, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    edges_path = prefix.with_name(prefix.name + "_edges.csv")
    pd.DataFrame(
        [
            {"source": u, "target": v, "weight": float(d.get("weight", 1.0))}
            for u, v, d in graph.edges(data=True)
        ]
    ).to_csv(edges_path, index=False)
    nodes_path = prefix.with_name(prefix.name + "_nodes.csv")
    pd.DataFrame(index=list(graph.nodes())).to_csv(nodes_path)


def _load_graph(prefix: Path) -> nx.Graph:
    edges = pd.read_csv(prefix.with_name(prefix.name + "_edges.csv"))
    nodes = pd.read_csv(prefix.with_name(prefix.name + "_nodes.csv"), index_col=0)
    g = nx.Graph()
    g.add_nodes_from(nodes.index.astype(str))
    for _, row in edges.iterrows():
        g.add_edge(str(row["source"]), str(row["target"]), weight=float(row["weight"]))
    return g


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger(__name__)
    t0 = time.time()

    stages = set(args.stages)
    follow_cache = args.follow_cache or (args.work_dir / "twibot22_follow_edges.csv")
    features_prefix = args.work_dir / "features"
    gb_prefix = args.work_dir / "gb"
    gt_prefix = args.work_dir / "gt"
    eval_prefix = args.work_dir / "eval"

    # --- extract ---
    if _should_run(args.work_dir, "extract", stages, args.force):
        log.info("=== stage: extract ===")
        extract_argv = [
            "--twibot22-root", str(args.twibot22_root),
            "--work-dir", str(args.work_dir),
            "--user-scope", args.user_scope,
            "--max-tweets-per-user", str(args.max_tweets_per_user),
        ]
        if args.incremental_merge:
            extract_argv.append("--incremental-merge")
        extract_twibot22_at_scale.main(extract_argv)
        _mark_done(args.work_dir, "extract", {"user_scope": args.user_scope})

    # --- features ---
    if _should_run(args.work_dir, "features", stages, args.force):
        log.info("=== stage: features ===")
        paths = discover_twibot22(args.twibot22_root)
        compute_features_at_scale.main(
            [
                "--work-dir", str(args.work_dir),
                "--twibot22-root", str(args.twibot22_root),
                "--output-prefix", str(features_prefix),
                "--humanlike-mapping", "sigmoid",
                "--d-shape-reference", "honest",
            ]
        )
        _mark_done(args.work_dir, "features", {"prefix": str(features_prefix)})

    z_path = features_prefix.with_name(features_prefix.name + "_zscored.csv")
    if not z_path.exists():
        raise FileNotFoundError(
            f"missing {z_path}; run the features stage first"
        )
    zscored = pd.read_csv(z_path, index_col=0)
    zscored.index = zscored.index.astype(str)

    # Merge official labels/splits from TwiBot-22 tables.
    paths = discover_twibot22(args.twibot22_root)
    labels_splits = load_labels_and_splits(paths)
    zscored["label"] = labels_splits["label"].reindex(zscored.index)
    zscored["split"] = labels_splits["split"].reindex(zscored.index)

    qualifying_users = set(
        zscored.dropna(how="any", subset=list(DEFAULT_FEATURES)).index
    )
    log.info("qualifying users for graphs/eval: %d", len(qualifying_users))

    # --- follow_cache ---
    if _should_run(args.work_dir, "follow_cache", stages, args.force):
        log.info("=== stage: follow_cache ===")
        if not follow_cache.exists():
            edge_csv = paths.root / "edge.csv"
            cache_all_user_follow_edges(edge_csv, follow_cache)
        _mark_done(args.work_dir, "follow_cache", {"path": str(follow_cache)})

    # --- gb ---
    if _should_run(args.work_dir, "gb", stages, args.force):
        log.info("=== stage: gb (FAISS if n > 10k) ===")
        feat_df = feature_matrix_for_qualifying_users(
            zscored, feature_columns=list(ZSCORE_FEATURE_COLUMNS)
        )
        build = build_mutual_knn_graph_auto(
            feat_df,
            k=args.gb_k,
            similarity_floor=args.gb_similarity_floor,
            metric=args.gb_metric,
        )
        _save_graph_edges(build.graph, gb_prefix)
        summary = graph_summary(build.graph)
        with (args.work_dir / "gb_summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2, default=float)
        _mark_done(args.work_dir, "gb", summary)

    # --- gt ---
    if _should_run(args.work_dir, "gt", stages, args.force):
        log.info("=== stage: gt (induced follow graph) ===")
        if not follow_cache.exists():
            raise FileNotFoundError(
                f"follow cache missing: {follow_cache}; run follow_cache stage"
            )
        g_t = load_follow_edge_cache(follow_cache, user_filter=qualifying_users)
        g_t.add_nodes_from(qualifying_users)
        _save_graph_edges(g_t, gt_prefix)
        summary = graph_summary(g_t)
        with (args.work_dir / "gt_summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2, default=float)
        _mark_done(args.work_dir, "gt", summary)

    gb_edges = gb_prefix.with_name(gb_prefix.name + "_edges.csv")
    gt_edges = gt_prefix.with_name(gt_prefix.name + "_edges.csv")
    need_graphs = any(
        s in stages or "all" in stages for s in ("eval", "f6", "f5")
    )
    if need_graphs and (not gb_edges.exists() or not gt_edges.exists()):
        raise FileNotFoundError(
            f"missing graph edges (gb={gb_edges.exists()}, gt={gt_edges.exists()}); "
            "run gb and gt stages first"
        )
    g_b = _load_graph(gb_prefix) if gb_edges.exists() else nx.Graph()
    g_t = _load_graph(gt_prefix) if gt_edges.exists() else nx.Graph()

    # --- eval ---
    if _should_run(args.work_dir, "eval", stages, args.force):
        log.info("=== stage: eval (protocol=%s) ===", args.eval_protocol)
        if args.eval_protocol == "official":
            result = run_official_split_evaluation(
                zscored=zscored,
                g_t=g_t,
                g_b=g_b,
                w_t=args.w_t,
                w_b=args.w_b,
            )
            payload = official_result_to_dict(result)
            out_path = eval_prefix.with_name(eval_prefix.name + "_official.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as fh:
                json.dump(payload, fh, indent=2, default=float)
            log.info("wrote %s", out_path)
            _mark_done(args.work_dir, "eval", payload)
        else:
            summary_df, fold_records = run_kfold_evaluation(
                zscored=zscored,
                g_t=g_t,
                g_b=g_b,
                n_splits=args.n_splits,
                seed=args.seed,
                w_t=args.w_t,
                w_b=args.w_b,
            )
            summary_df.to_csv(
                eval_prefix.with_name(eval_prefix.name + "_kfold_summary.csv"),
                index=False,
            )
            dump_results_json(
                summary_df,
                fold_records,
                eval_prefix.with_name(eval_prefix.name + "_kfold.json"),
            )
            agg = summarise_with_stderr(summary_df)
            log.info("\n%s", agg.to_string())
            _mark_done(args.work_dir, "eval", {"protocol": "kfold"})

    # --- f6 ---
    if _should_run(args.work_dir, "f6", stages, args.force):
        log.info("=== stage: f6 ===")
        g_b_c, g_t_c = restrict_to_common_nodes(g_b, g_t)
        report = f6_report(
            g_b_c,
            g_t_c,
            spectral_k=10,
            max_spectral_nodes=50_000,
        )
        f6_path = eval_prefix.with_name(eval_prefix.name + "_f6.json")
        with f6_path.open("w") as fh:
            json.dump(report.__dict__, fh, indent=2, default=float)
        log.info("F6 edge_jaccard=%.4f", report.edge_jaccard)
        _mark_done(args.work_dir, "f6", report.__dict__)

    # --- optional F5 on official split (lightweight) ---
    if "f5" in stages or "all" in stages:
        log.info("=== stage: f5 (LR prior vs p_T scatter) ===")
        qualifying = zscored.dropna(how="any", subset=list(DEFAULT_FEATURES))
        qualifying = qualifying[qualifying["split"].isin(["train", "test"])]
        train_df = qualifying[qualifying["split"] == "train"]
        test_df = qualifying[qualifying["split"] == "test"]
        lr = fit_predict_lr_prior(train_df, test_df)
        node_order = list(qualifying.index)
        relation_t = relation_from_graph("trust", g_t, node_order, weight=args.w_t)
        priors = {u: float(lr.pi_b_test.get(u, lr.pi_b_train.get(u, 0.5)))
                  for u in qualifying.index}
        for u in train_df.index:
            priors[u] = 0.98 if qualifying.loc[u, "label"] == "human" else 0.02
        prop = run_propagation(
            relations=[relation_t],
            node_order=node_order,
            priors=priors,
            max_iters=50,
            tol=1e-5,
        )
        p_t = {nd: float(p) for nd, p in zip(prop.node_order, prop.p_honest)}
        labels = qualifying["label"].to_dict()
        render_f5_scatter(
            p_t_honest=p_t,
            p_b_honest={**lr.pi_b_train, **lr.pi_b_test},
            labels=labels,
            bot_label="bot",
            honest_label="human",
            output_prefix=eval_prefix.with_name(eval_prefix.name + "_f5"),
        )

    log.info("pipeline finished in %.0f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())