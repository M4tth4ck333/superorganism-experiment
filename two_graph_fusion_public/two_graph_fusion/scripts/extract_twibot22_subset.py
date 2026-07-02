"""Extract per-user timestamps for a small TwiBot-22 subset.

This script picks a class-balanced subset of users, scans one or more of the
``tweet_*.json`` shards, and writes a JSON file mapping user id to the list of
Unix-second timestamps for tweets the user authored.

Typical usage
-------------

    python -m two_graph_fusion.scripts.extract_twibot22_subset \
        --twibot22-root twibot22 \
        --n-per-class 1000 \
        --shards 0 \
        --output two_graph_fusion/cache/subset_timestamps.json

Run with ``--help`` for the full option list.

Output schema
-------------

A JSON dict with two top-level keys::

    {
      "metadata": {
        "twibot22_root": "...",
        "shards_scanned": [0],
        "n_per_class_requested": 1000,
        "splits": ["train"],
        "seed": 0,
        "max_tweets_per_user": null,
        "window_start_unix": <float>,
        "window_end_unix":   <float>,
        "users_total": <int>,
        "users_with_tweets": <int>
      },
      "labels": { "uXXX": "human", ... },
      "splits": { "uXXX": "train", ... },
      "timestamps": { "uXXX": [unix_seconds, ...], ... }
    }

The window is derived from the min/max of all collected timestamps, so the
activity-duration feature ``L`` is computed against the *observed* window of
the scanned shards. If you want a fixed window across runs, override the
two ``window_*`` fields downstream.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from two_graph_fusion.datasets.twibot22 import (
    discover_twibot22,
    load_labels_and_splits,
    sample_balanced_subset,
    stream_timestamps_for_users,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a small TwiBot-22 subset of user timestamps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--twibot22-root",
        type=Path,
        default=Path("twibot22"),
        help="Path to the TwiBot-22 release directory.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=1000,
        help="Number of users to sample per class (human, bot).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "valid", "test"],
        help="Which dataset splits to draw the subset from.",
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        type=int,
        default=[0],
        help="Indices of tweet_*.json shards to scan (0 through 8).",
    )
    parser.add_argument(
        "--max-tweets-per-user",
        type=int,
        default=None,
        help=(
            "Early-stop after this many tweets are collected per user. "
            "Default: no cap."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subset sampling.",
    )
    parser.add_argument(
        "--subset-csv",
        type=Path,
        default=None,
        help=(
            "Optional pre-computed subset CSV with index 'user_id' and "
            "columns 'label' and 'split' (the format produced by "
            "two_graph_fusion.datasets.snowball.sample_largest_component "
            "and friends). If set, the script skips the balanced random "
            "sampling step and uses these users directly."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("two_graph_fusion/cache/subset_timestamps.json"),
        help="Destination JSON file.",
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

    paths = discover_twibot22(args.twibot22_root)
    log.info("twibot22 root: %s", paths.root)

    labels_splits = load_labels_and_splits(paths)
    if args.subset_csv is not None:
        import pandas as pd

        subset = pd.read_csv(args.subset_csv, index_col=0)
        if "label" not in subset.columns or "split" not in subset.columns:
            raise ValueError(
                f"--subset-csv {args.subset_csv} must have 'label' and "
                "'split' columns; got " + str(list(subset.columns))
            )
        log.info(
            "loaded pre-computed subset: %d users (%s)",
            len(subset),
            subset["label"].value_counts().to_dict(),
        )
    else:
        subset = sample_balanced_subset(
            labels_splits,
            n_per_class=args.n_per_class,
            splits=args.splits,
            seed=args.seed,
        )

    shard_paths = []
    for shard_idx in args.shards:
        match = [p for p in paths.tweet_files if p.stem == f"tweet_{shard_idx}"]
        if not match:
            raise FileNotFoundError(f"shard tweet_{shard_idx}.json not found")
        shard_paths.append(match[0])
    log.info("scanning shards: %s", [p.name for p in shard_paths])

    timestamps = stream_timestamps_for_users(
        shard_paths,
        user_ids=set(subset.index),
        max_tweets_per_user=args.max_tweets_per_user,
    )

    if timestamps:
        all_ts = [t for ts in timestamps.values() for t in ts]
        window_start = min(all_ts)
        window_end = max(all_ts)
    else:
        window_start = 0.0
        window_end = 0.0
        log.warning("no tweets matched the subset; output will be empty")

    out = {
        "metadata": {
            "twibot22_root": str(paths.root),
            "shards_scanned": list(args.shards),
            "n_per_class_requested": args.n_per_class,
            "splits": list(args.splits),
            "seed": args.seed,
            "max_tweets_per_user": args.max_tweets_per_user,
            "window_start_unix": window_start,
            "window_end_unix": window_end,
            "users_total": int(len(subset)),
            "users_with_tweets": int(len(timestamps)),
        },
        "labels": subset["label"].to_dict(),
        "splits": subset["split"].to_dict(),
        "timestamps": timestamps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(out, fh)
    log.info(
        "wrote %s (users_total=%d users_with_tweets=%d)",
        args.output,
        out["metadata"]["users_total"],
        out["metadata"]["users_with_tweets"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
