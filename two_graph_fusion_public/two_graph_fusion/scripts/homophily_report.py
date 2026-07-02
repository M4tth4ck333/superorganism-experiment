"""Compare edge homophily across TwiBot-22 cohorts and graph variants.

Milestone 4 uses this script as the fast go/no-go check before running
the full feature extraction + milestone-3 evaluation pipeline. It
answers: does ``G_T`` on this cohort have enough label homophily for
SybilSCAR propagation to matter?

Typical usage
-------------

After building cohort CSVs and (optionally) a feature prefix with
qualifying users::

    python -m two_graph_fusion.scripts.homophily_report \
        --follow-cache two_graph_fusion/cache/twibot22_follow_edges.csv \
        --features-prefix two_graph_fusion/cache/subset_features_snowball_sep \
        --cohort-name snowball_separate \
        --output two_graph_fusion/cache/m4_homophily_snowball_sep.json

Compare two cohort definitions on the same qualifying users (subset
CSV only, no features yet)::

    python -m two_graph_fusion.scripts.homophily_report \
        --follow-cache two_graph_fusion/cache/twibot22_follow_edges.csv \
        --subset-csv two_graph_fusion/cache/subset_snowball_sep.csv \
        --cohort-name snowball_separate \
        --subset-csv two_graph_fusion/cache/subset_snowball.csv \
        --cohort-name snowball_cross_class
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
    homophily_from_dataframe,
)
from two_graph_fusion.graphs.homophily import HomophilyReport, format_homophily_row
from two_graph_fusion.graphs.trust_graph import load_follow_edge_cache
from two_graph_fusion.propagation import DEFAULT_FEATURES


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report G_T and G_B edge homophily for TwiBot-22 cohorts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--follow-cache",
        type=Path,
        default=Path("two_graph_fusion/cache/twibot22_follow_edges.csv"),
    )
    parser.add_argument(
        "--features-prefix",
        type=Path,
        default=None,
        help="If set, restrict to qualifying users from <prefix>_zscored.csv.",
    )
    parser.add_argument(
        "--subset-csv",
        type=Path,
        action="append",
        default=None,
        help="Cohort user list (index=user_id). Repeatable.",
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        action="append",
        default=None,
        help="Label for each --subset-csv (same order).",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--gb-metric",
        choices=["cosine", "euclidean"],
        default="euclidean",
        help="Metric for a freshly built G_B (milestone 4 default).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _load_qualifying(features_prefix: Path | None, subset_csv: Path | None) -> pd.DataFrame:
    """Return a user table with ``label`` for homophily (qualifying if possible)."""
    if features_prefix is not None:
        z_path = features_prefix.with_name(features_prefix.name + "_zscored.csv")
        zscored = pd.read_csv(z_path, index_col=0)
        zscored.index = zscored.index.astype(str)
        qualifying = zscored.dropna(how="any", subset=list(DEFAULT_FEATURES))
        qualifying = qualifying[qualifying["label"].isin({"human", "bot"})]
        return qualifying

    if subset_csv is None:
        raise ValueError("provide --features-prefix or at least one --subset-csv")

    subset = pd.read_csv(subset_csv, index_col=0)
    subset.index = subset.index.astype(str)
    return subset[subset["label"].isin({"human", "bot"})]


def _has_zscored_features(users: pd.DataFrame) -> bool:
    return all(c in users.columns for c in ZSCORE_FEATURE_COLUMNS)


def report_for_cohort(
    name: str,
    users: pd.DataFrame,
    follow_cache: Path,
    k: int,
    gb_metric: str,
) -> dict:
    """Build induced G_T (and G_B when z-scores exist); return a JSON report."""
    user_ids = set(users.index)

    g_t = load_follow_edge_cache(follow_cache, user_filter=user_ids)
    g_t.add_nodes_from(user_ids)
    h_t = homophily_from_dataframe(g_t, users)

    out: dict = {
        "cohort": name,
        "n_users": len(users),
        "class_counts": users["label"].value_counts().to_dict(),
        "g_t": {
            **h_t.to_dict(),
            "summary": graph_summary(g_t),
        },
        "gates": {
            "g_t_homophily_lift_ge_0.05": h_t.homophily_lift >= 0.05,
        },
    }

    if _has_zscored_features(users):
        feat_df = feature_matrix_for_qualifying_users(
            users, feature_columns=list(ZSCORE_FEATURE_COLUMNS)
        )
        gb_build = build_mutual_knn_graph(feat_df, k=k, metric=gb_metric)
        g_b = gb_build.graph
        h_b = homophily_from_dataframe(g_b, users)
        out["g_b"] = {
            **h_b.to_dict(),
            "summary": graph_summary(g_b),
            "metric": gb_metric,
            "k": k,
        }
        out["gates"]["g_b_homophily_lift_ge_0.05"] = h_b.homophily_lift >= 0.05
    else:
        out["g_b"] = {"skipped": True, "reason": "z-scored features not in input"}
        out["gates"]["g_b_homophily_lift_ge_0.05"] = None

    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    cohorts: list[tuple[str, Path | None, pd.DataFrame]] = []

    if args.features_prefix is not None:
        users = _load_qualifying(args.features_prefix, None)
        name = args.cohort_name[0] if args.cohort_name else args.features_prefix.name
        cohorts.append((name, None, users))

    if args.subset_csv:
        names = args.cohort_name or []
        if names and len(names) != len(args.subset_csv):
            print("error: --cohort-name count must match --subset-csv count", file=sys.stderr)
            return 2
        for i, path in enumerate(args.subset_csv):
            nm = names[i] if i < len(names) else path.stem
            users = _load_qualifying(None, path)
            cohorts.append((nm, path, users))

    if not cohorts:
        print("error: provide --features-prefix and/or --subset-csv", file=sys.stderr)
        return 2

    reports = [
        report_for_cohort(nm, users, args.follow_cache, args.k, args.gb_metric)
        for nm, _, users in cohorts
    ]

    payload = {"cohorts": reports}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.output}")

    print("=== Homophily comparison (milestone 4) ===")
    for rep in reports:
        print(f"\n--- {rep['cohort']} (n={rep['n_users']}) ---")
        h_t = HomophilyReport(
            **{k: rep["g_t"][k] for k in (
                "n_edges_labeled", "n_same_label", "n_cross_label",
                "n_edges_unlabeled", "homophily", "chance_baseline", "homophily_lift",
            )}
        )
        print(format_homophily_row("G_T (follow)", h_t))
        if rep["g_b"].get("skipped"):
            print(f"  G_B: skipped ({rep['g_b'].get('reason', '')})")
        else:
            h_b = HomophilyReport(
                **{k: rep["g_b"][k] for k in (
                    "n_edges_labeled", "n_same_label", "n_cross_label",
                    "n_edges_unlabeled", "homophily", "chance_baseline",
                    "homophily_lift",
                )}
            )
            print(format_homophily_row(f"G_B ({args.gb_metric}, k={args.k})", h_b))
        gates = rep["gates"]
        gb_gate = gates.get("g_b_homophily_lift_ge_0.05")
        gb_str = "n/a" if gb_gate is None else str(gb_gate)
        print(
            f"  gates: G_T lift>=0.05 -> {gates['g_t_homophily_lift_ge_0.05']}  "
            f"G_B lift>=0.05 -> {gb_str}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
