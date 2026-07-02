"""Build a user cohort for TwiBot-22 fusion experiments.

Milestone 2 introduced snowball / LCC sampling to fix RED-7 (empty
``G_T`` on random subsets). Milestone 4 adds **separate-class snowball**
and **train/test balanced** sampling to address RED-10 (``G_T``
homophily at chance under cross-class BFS).

Strategies
----------

``snowball_separate`` (recommended)
    Independent BFS per class; see :func:`snowball_separate_classes`.

``train_test_balanced``
    Random balanced sample from official train+test splits (no BFS).

``snowball``
    Legacy cross-class BFS (discouraged; kept for regression comparison).

``lcc``
    Largest connected component + balanced sample.

Examples
--------

Separate-class snowball (milestone 4 default)::

    python -m two_graph_fusion.scripts.build_connected_subset \
        --strategy snowball_separate \
        --target-per-class 15000 \
        --n-per-class 2500 \
        --output two_graph_fusion/cache/subset_snowball_sep.csv

Train/test balanced baseline (no follow-graph sampling)::

    python -m two_graph_fusion.scripts.build_connected_subset \
        --strategy train_test_balanced \
        --n-per-class 2500 \
        --output two_graph_fusion/cache/subset_train_test.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from two_graph_fusion.datasets.snowball import (
    sample_largest_component,
    snowball_sample,
    snowball_separate_classes,
)
from two_graph_fusion.datasets.twibot22 import (
    discover_twibot22,
    load_labels_and_splits,
    sample_balanced_train_test,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TwiBot-22 user cohort for fusion experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--twibot22-root",
        type=Path,
        default=Path("twibot22"),
    )
    parser.add_argument(
        "--follow-cache",
        type=Path,
        default=Path("two_graph_fusion/cache/twibot22_follow_edges.csv"),
        help="Required for snowball and lcc strategies.",
    )
    parser.add_argument(
        "--strategy",
        choices=[
            "lcc",
            "snowball",
            "snowball_separate",
            "train_test_balanced",
        ],
        default="snowball_separate",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=2500,
        help="Final balanced sample size per class.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=30_000,
        help="(snowball only) Total BFS visit cap before balancing.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=15_000,
        help="(snowball_separate only) Users to collect per class before balancing.",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=8,
        help="BFS depth limit for snowball strategies.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def build_subset(args: argparse.Namespace, labels_splits: pd.DataFrame) -> pd.DataFrame:
    """Dispatch on ``args.strategy`` and return the cohort DataFrame."""
    if args.strategy == "train_test_balanced":
        return sample_balanced_train_test(
            labels_splits,
            n_per_class=args.n_per_class,
            seed=args.seed,
        )
    if args.strategy == "lcc":
        return sample_largest_component(
            follow_cache_csv=args.follow_cache,
            labels_splits=labels_splits,
            n_per_class=args.n_per_class,
            seed=args.seed,
        )
    if args.strategy == "snowball":
        return snowball_sample(
            follow_cache_csv=args.follow_cache,
            labels_splits=labels_splits,
            target_size=args.target_size,
            n_per_class=args.n_per_class,
            max_hops=args.max_hops,
            seed=args.seed,
        )
    if args.strategy == "snowball_separate":
        return snowball_separate_classes(
            follow_cache_csv=args.follow_cache,
            labels_splits=labels_splits,
            target_per_class=args.target_per_class,
            n_per_class=args.n_per_class,
            max_hops=args.max_hops,
            seed=args.seed,
        )
    raise ValueError(f"unknown strategy {args.strategy!r}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger(__name__)

    paths = discover_twibot22(args.twibot22_root)
    labels_splits = load_labels_and_splits(paths)

    if args.strategy in {"lcc", "snowball", "snowball_separate"}:
        if not args.follow_cache.exists():
            log.error(
                "follow cache missing: %s\n"
                "Run: python -m two_graph_fusion.scripts.cache_follow_edges "
                "--edge-csv twibot22/edge.csv --output %s",
                args.follow_cache,
                args.follow_cache,
            )
            return 1

    subset = build_subset(args, labels_splits)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output)
    log.info(
        "wrote %s: %d users (%s)",
        args.output,
        len(subset),
        subset["label"].value_counts().to_dict(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
