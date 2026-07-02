"""Milestone 5: interaction-edge G_T experiment.

Replaces the follow-graph G_T with a user-user interaction graph derived
from retweet, quote, and mention edges, then re-runs the full evaluation
pipeline (homophily, F6, official train/test split).

Motivation
----------

The at-scale follow-edge experiment (Milestone 4 / RESULTS.md) showed
that the follow graph has near-zero homophily lift (+0.002) within the
qualifying user population: bots systematically follow real users to
avoid detection, making the follow graph structurally heterophilic.

Interaction edges (retweet, quote, mention) reflect content affinity
rather than follower-building strategy.  Bots tend to retweet and quote
other bots' content, so these edges should form denser same-class
clusters. The experiment tests whether this produces a usable trust
graph for SybilSCAR propagation.

Reuse
-----

All heavy precomputed artefacts from the at-scale follow-edge run are
reused without change:

- ``features_zscored.csv``: five z-scored behavioural features +
  label/split for 750 k tweet-having users.
- ``gb_edges.csv`` / ``gb_nodes.csv``: the FAISS mutual k-NN
  behavioural graph G_B (k=10, cosine, n=432 k).

Only G_T is rebuilt from scratch using
:func:`two_graph_fusion.graphs.interaction_graph.build_interaction_gt`.
The interaction edge cache is saved for reproducibility.

Usage
-----

    python -m two_graph_fusion.scripts.run_interaction_experiment \\
        --twibot22-root twibot22 \\
        --atscale-cache two_graph_fusion/cache/at_scale_full_tt \\
        --output-dir two_graph_fusion/cache/interaction_experiment

Stages are idempotent: if the output directory already contains a stage
marker the stage is skipped unless ``--force`` is passed.
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

from two_graph_fusion.features import ZSCORE_FEATURE_COLUMNS
from two_graph_fusion.graphs import (
    f6_report,
    graph_summary,
    restrict_to_common_nodes,
    edge_homophily,
)
from two_graph_fusion.graphs.interaction_graph import (
    DEFAULT_INTERACTION_RELATIONS,
    build_interaction_gt,
)
from two_graph_fusion.propagation import (
    DEFAULT_FEATURES,
    official_result_to_dict,
    run_official_split_evaluation,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Milestone 5: interaction-edge G_T experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--twibot22-root",
        type=Path,
        default=Path("twibot22"),
        help="Directory containing TwiBot-22 edge.csv and label/split CSVs.",
    )
    p.add_argument(
        "--atscale-cache",
        type=Path,
        default=Path("two_graph_fusion/cache/at_scale_full_tt"),
        help="Directory with at-scale features_zscored.csv and gb_*.csv.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("two_graph_fusion/cache/interaction_experiment"),
    )
    p.add_argument(
        "--w-t",
        type=float,
        default=0.5,
        help="SybilSCAR trust-graph weight w_T.",
    )
    p.add_argument(
        "--w-b",
        type=float,
        default=0.5,
        help="SybilSCAR behavioural-graph weight w_B.",
    )
    p.add_argument("--force", action="store_true", help="Re-run completed stages.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _marker(out: Path, stage: str) -> Path:
    return out / "stages" / f"{stage}.done"


def _should_run(out: Path, stage: str, force: bool) -> bool:
    return force or not _marker(out, stage).exists()


def _mark_done(out: Path, stage: str, info: dict) -> None:
    m = _marker(out, stage)
    m.parent.mkdir(parents=True, exist_ok=True)
    with m.open("w") as fh:
        json.dump(info, fh, indent=2, default=float)


def _load_gb(cache_dir: Path) -> nx.Graph:
    """Load the cached G_B from the at-scale run."""
    edges_path = cache_dir / "gb_edges.csv"
    nodes_path = cache_dir / "gb_nodes.csv"
    edges = pd.read_csv(edges_path, dtype={"source": str, "target": str})
    nodes = pd.read_csv(nodes_path, index_col=0)
    nodes.index = nodes.index.astype(str)
    g = nx.Graph()
    g.add_nodes_from(nodes.index)
    # Use from_pandas_edgelist — much faster than iterrows for millions of edges.
    g = nx.from_pandas_edgelist(
        edges.astype(str),
        source="source",
        target="target",
        create_using=nx.Graph(),
    )
    # Ensure isolated nodes from the node list are included.
    g.add_nodes_from(nodes.index)
    return g


def _save_graph_edges(graph: nx.Graph, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    edges_path = prefix.with_name(prefix.name + "_edges.csv")
    nodes_path = prefix.with_name(prefix.name + "_nodes.csv")
    edge_rows = [
        {"source": u, "target": v}
        for u, v in graph.edges()
    ]
    pd.DataFrame(edge_rows).to_csv(edges_path, index=False)
    pd.DataFrame(index=list(graph.nodes())).to_csv(nodes_path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger(__name__)
    t0 = time.time()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    edge_csv = args.twibot22_root / "edge.csv"
    features_csv = args.atscale_cache / "features_zscored.csv"
    gb_edges_csv = args.atscale_cache / "gb_edges.csv"

    for path in (edge_csv, features_csv, gb_edges_csv):
        if not path.exists():
            log.error("missing required file: %s", path)
            return 1

    # ------------------------------------------------------------------
    # Load features and determine qualifying users.
    # ------------------------------------------------------------------
    log.info("loading features from %s ...", features_csv)
    zscored = pd.read_csv(features_csv, index_col=0)
    zscored.index = zscored.index.astype(str)

    qualifying = zscored.dropna(how="any", subset=list(DEFAULT_FEATURES))
    qualifying = qualifying[qualifying["label"].isin({"human", "bot"})]
    qualifying = qualifying[qualifying["split"].isin({"train", "test"})]
    qualifying_users: set[str] = set(qualifying.index)

    log.info(
        "qualifying users (train+test, non-NaN features): %d  "
        "(human=%d  bot=%d)",
        len(qualifying_users),
        (qualifying["label"] == "human").sum(),
        (qualifying["label"] == "bot").sum(),
    )

    # ------------------------------------------------------------------
    # Stage: build interaction G_T.
    # ------------------------------------------------------------------
    gt_prefix = args.output_dir / "gt_interaction"
    gt_edges_path = args.output_dir / "gt_interaction_edges.csv"

    if _should_run(args.output_dir, "gt", args.force):
        log.info("=== stage: gt (interaction edges) ===")
        result = build_interaction_gt(
            edge_csv=edge_csv,
            qualifying_users=qualifying_users,
            relations=DEFAULT_INTERACTION_RELATIONS,
        )
        g_t = result.graph
        gt_summary = graph_summary(g_t)
        gt_info = {
            "n_qualifying_users": result.n_qualifying_users,
            "n_post_edges_scanned": result.n_post_edges_scanned,
            "n_tweet_author_pairs": result.n_tweet_author_pairs,
            "n_raw_pairs": result.n_raw_pairs,
            "n_resolved_edges": result.n_resolved_edges,
            "relations": sorted(result.relations),
            "relation_edge_counts": result.relation_edge_counts,
            "graph_summary": gt_summary,
        }
        with (args.output_dir / "gt_interaction_summary.json").open("w") as fh:
            json.dump(gt_info, fh, indent=2, default=float)
        _save_graph_edges(g_t, gt_prefix)
        _mark_done(args.output_dir, "gt", gt_info)
        log.info(
            "G_T built: nodes=%d  edges=%d  relations=%s",
            g_t.number_of_nodes(),
            g_t.number_of_edges(),
            sorted(result.relations),
        )
    else:
        log.info("stage gt already done; loading from %s", gt_edges_path)
        edges = pd.read_csv(gt_edges_path, dtype={"source": str, "target": str})
        g_t = nx.from_pandas_edgelist(
            edges, source="source", target="target", create_using=nx.Graph()
        )
        g_t.add_nodes_from(qualifying_users)
        log.info("G_T loaded: nodes=%d  edges=%d", g_t.number_of_nodes(), g_t.number_of_edges())

    # ------------------------------------------------------------------
    # Stage: load G_B (from at-scale run).
    # ------------------------------------------------------------------
    log.info("loading G_B from %s ...", args.atscale_cache)
    g_b = _load_gb(args.atscale_cache)
    log.info("G_B loaded: nodes=%d  edges=%d", g_b.number_of_nodes(), g_b.number_of_edges())

    # ------------------------------------------------------------------
    # Stage: homophily.
    # ------------------------------------------------------------------
    if _should_run(args.output_dir, "homophily", args.force):
        log.info("=== stage: homophily ===")
        label_map = qualifying["label"].to_dict()

        h_t = edge_homophily(g_t, label_map)
        h_b = edge_homophily(g_b, label_map)

        # Bot-bot and human-human breakdown for G_T
        bot_bot = 0
        human_human = 0
        human_bot = 0
        for u, v in g_t.edges():
            lu = label_map.get(u)
            lv = label_map.get(v)
            if lu == "bot" and lv == "bot":
                bot_bot += 1
            elif lu == "human" and lv == "human":
                human_human += 1
            elif lu is not None and lv is not None:
                human_bot += 1

        hom_info = {
            "g_t_interaction": h_t.to_dict(),
            "g_b": h_b.to_dict(),
            "g_t_interaction_edge_breakdown": {
                "bot_bot": bot_bot,
                "human_human": human_human,
                "human_bot": human_bot,
            },
        }
        with (args.output_dir / "homophily.json").open("w") as fh:
            json.dump(hom_info, fh, indent=2, default=float)
        _mark_done(args.output_dir, "homophily", hom_info)
        log.info(
            "G_T_interaction homophily: h=%.4f  chance=%.4f  lift=%+.4f",
            h_t.homophily,
            h_t.chance_baseline,
            h_t.homophily_lift,
        )
        log.info(
            "G_B homophily:             h=%.4f  chance=%.4f  lift=%+.4f",
            h_b.homophily,
            h_b.chance_baseline,
            h_b.homophily_lift,
        )
        log.info(
            "G_T edge breakdown: bot-bot=%d  human-human=%d  human-bot=%d",
            bot_bot, human_human, human_bot,
        )
    else:
        log.info("stage homophily already done; skipping")

    # ------------------------------------------------------------------
    # Stage: F6 overlap diagnostic.
    # ------------------------------------------------------------------
    if _should_run(args.output_dir, "f6", args.force):
        log.info("=== stage: f6 (G_T_interaction vs G_B overlap) ===")
        g_b_c, g_t_c = restrict_to_common_nodes(g_b, g_t)
        report = f6_report(g_b_c, g_t_c, spectral_k=10, max_spectral_nodes=50_000)
        f6_info = report.__dict__
        with (args.output_dir / "f6_summary.json").open("w") as fh:
            json.dump(f6_info, fh, indent=2, default=float)
        _mark_done(args.output_dir, "f6", f6_info)
        log.info(
            "F6: edge_jaccard=%.5f  triangle_jaccard=%.5f",
            report.edge_jaccard,
            report.triangle_jaccard,
        )
    else:
        log.info("stage f6 already done; skipping")

    # ------------------------------------------------------------------
    # Stage: official split evaluation.
    # ------------------------------------------------------------------
    if _should_run(args.output_dir, "eval", args.force):
        log.info("=== stage: eval (official train/test split) ===")
        result = run_official_split_evaluation(
            zscored=zscored,
            g_t=g_t,
            g_b=g_b,
            w_t=args.w_t,
            w_b=args.w_b,
        )
        payload = official_result_to_dict(result)
        out_path = args.output_dir / "eval_official.json"
        with out_path.open("w") as fh:
            json.dump(payload, fh, indent=2, default=float)
        _mark_done(args.output_dir, "eval", payload)
        log.info(
            "eval results: LR=%.3f  p_T=%.3f  p_B=%.3f  variantB=%.3f",
            result.lr_with_l["test_auc"],
            result.p_t_auc,
            result.p_b_auc,
            result.variant_b_auc,
        )
    else:
        log.info("stage eval already done; loading from eval_official.json")

    # ------------------------------------------------------------------
    # Consolidated results JSON.
    # ------------------------------------------------------------------
    results_path = args.output_dir / "results.json"
    hom_path = args.output_dir / "homophily.json"
    f6_path = args.output_dir / "f6_summary.json"
    eval_path = args.output_dir / "eval_official.json"
    gt_sum_path = args.output_dir / "gt_interaction_summary.json"

    results: dict = {"elapsed_s": round(time.time() - t0, 1)}
    for key, path in [
        ("gt_build", gt_sum_path),
        ("homophily", hom_path),
        ("f6", f6_path),
        ("eval", eval_path),
    ]:
        if path.exists():
            with path.open() as fh:
                results[key] = json.load(fh)

    with results_path.open("w") as fh:
        json.dump(results, fh, indent=2, default=float)

    log.info("wrote %s", results_path)
    log.info("pipeline finished in %.0f s", time.time() - t0)

    # ------------------------------------------------------------------
    # Human-readable summary.
    # ------------------------------------------------------------------
    print("\n=== Milestone 5 summary ===")
    if hom_path.exists():
        with hom_path.open() as fh:
            h = json.load(fh)
        ht = h["g_t_interaction"]
        hb = h["g_b"]
        bd = h["g_t_interaction_edge_breakdown"]
        print(
            f"G_T_interaction  edges={ht['n_edges_labeled']:,}  "
            f"h={ht['homophily']:.4f}  chance={ht['chance_baseline']:.4f}  "
            f"lift={ht['homophily_lift']:+.4f}"
        )
        print(
            f"  breakdown: bot-bot={bd['bot_bot']:,}  "
            f"human-human={bd['human_human']:,}  "
            f"human-bot={bd['human_bot']:,}"
        )
        print(
            f"G_B              edges={hb['n_edges_labeled']:,}  "
            f"h={hb['homophily']:.4f}  chance={hb['chance_baseline']:.4f}  "
            f"lift={hb['homophily_lift']:+.4f}"
        )
    if eval_path.exists():
        with eval_path.open() as fh:
            ev = json.load(fh)
        print()
        print(f"LR prior (behavioral features): test AUC = {ev['lr_with_l']['test_auc']:.3f}")
        print(f"SybilSCAR on G_T_interaction:   test AUC = {ev['p_t_auc']:.3f}")
        print(f"SybilSCAR on G_B:               test AUC = {ev['p_b_auc']:.3f}")
        print(f"Variant B (fusion):             test AUC = {ev['variant_b_auc']:.3f}")
        best_a = max(ev["variant_a"].items(), key=lambda x: x[1]["auc"])
        print(
            f"Variant A best (α={best_a[0]}):         test AUC = {best_a[1]['auc']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
