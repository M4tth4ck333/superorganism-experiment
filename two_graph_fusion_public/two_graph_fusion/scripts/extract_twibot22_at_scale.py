"""Checkpointed timestamp extraction for all (or most) TwiBot-22 labeled users.

Unlike ``extract_twibot22_subset``, this script is meant for server runs:

- Processes one tweet shard at a time and writes a pickle checkpoint.
- Resumes automatically if a shard checkpoint already exists.
- Merges checkpoints into ``timestamps.parquet`` (compact) and an optional
  JSON bundle compatible with ``compute_features_subset``.

Typical usage (all labeled users, all 9 shards, cap 500 tweets/user):

    python -m two_graph_fusion.scripts.extract_twibot22_at_scale \
        --twibot22-root twibot22 \
        --work-dir two_graph_fusion/cache/at_scale \
        --user-scope all_labeled \
        --max-tweets-per-user 500

Estimated wall time: ~12-15 min per shard on a fast SSD (~2 hours total).
Peak RAM: depends on ``--max-tweets-per-user``; 500 tweets x 1M users is
~4 GB for the timestamp lists alone.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from two_graph_fusion.datasets.checkpoint_extract import (
    is_shard_done,
    load_meta,
    load_partial_merged,
    merge_shard_checkpoints,
    merge_timestamp_dicts,
    partial_merged_path,
    save_meta,
    save_partial_merged,
    save_shard_checkpoint,
    write_timestamps_json_bundle,
    write_timestamps_parquet,
)
from two_graph_fusion.datasets.twibot22 import (
    discover_twibot22,
    load_labels_and_splits,
    stream_timestamps_for_users,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TwiBot-22 timestamps at scale with per-shard checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--twibot22-root", type=Path, default=Path("twibot22"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("two_graph_fusion/cache/at_scale"),
        help="Directory for checkpoints and merged outputs.",
    )
    parser.add_argument(
        "--user-scope",
        choices=["all_labeled", "train_only", "train_test"],
        default="all_labeled",
        help=(
            "Which users to include. 'all_labeled' = every user in "
            "label.csv (~1M). 'train_only' / 'train_test' restrict to "
            "official splits (recommended for benchmark runs)."
        ),
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        type=int,
        default=list(range(9)),
        help="Tweet shard indices to scan (0-8).",
    )
    parser.add_argument(
        "--max-tweets-per-user",
        type=int,
        default=500,
        help="Cap tweets collected per user (bounds RAM).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-scan shards even if checkpoints exist.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Only run shard scans; do not write parquet/json.",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Also write timestamps.json (large; parquet is preferred).",
    )
    parser.add_argument(
        "--incremental-merge",
        action="store_true",
        help=(
            "After each shard scan, merge into checkpoints/merged_partial.pkl "
            "and delete the per-shard pickle. Lowers peak disk use (~2x one "
            "shard instead of 9x) for local runs with limited free space."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _select_users(labels_splits, scope: str):
    if scope == "all_labeled":
        return labels_splits
    if scope == "train_only":
        return labels_splits[labels_splits["split"] == "train"]
    if scope == "train_test":
        return labels_splits[labels_splits["split"].isin(["train", "test"])]
    raise ValueError(scope)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger(__name__)
    t0 = time.time()

    paths = discover_twibot22(args.twibot22_root)
    labels_splits = load_labels_and_splits(paths)
    subset = _select_users(labels_splits, args.user_scope)
    user_ids = set(subset.index.astype(str))
    log.info(
        "user scope=%s: %d users (%s)",
        args.user_scope,
        len(user_ids),
        subset["label"].value_counts().to_dict(),
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.work_dir)
    shard_files = list(paths.tweet_files)

    completed_shards: set[int] = set(int(s) for s in meta.get("completed_shards", []))

    for shard_idx in args.shards:
        if shard_idx < 0 or shard_idx >= len(shard_files):
            raise ValueError(f"shard index {shard_idx} out of range")
        if args.incremental_merge:
            already = shard_idx in completed_shards and not args.force
        else:
            already = is_shard_done(args.work_dir, shard_idx) and not args.force
        if already:
            log.info("shard %d: already done, skipping", shard_idx)
            continue
        shard_path = shard_files[shard_idx]
        log.info("shard %d: scanning %s ...", shard_idx, shard_path.name)
        partial = stream_timestamps_for_users(
            [shard_path],
            user_ids=user_ids,
            max_tweets_per_user=args.max_tweets_per_user,
        )
        if args.incremental_merge:
            merged = load_partial_merged(args.work_dir)
            timestamps = merge_timestamp_dicts(
                merged,
                partial,
                args.max_tweets_per_user,
            )
            save_partial_merged(args.work_dir, timestamps)
            log.info(
                "shard %d: incremental merge -> %d users in partial",
                shard_idx,
                len(timestamps),
            )
        else:
            save_shard_checkpoint(args.work_dir, shard_idx, partial)
        meta.setdefault("completed_shards", [])
        if shard_idx not in meta["completed_shards"]:
            meta["completed_shards"].append(shard_idx)
        meta.setdefault("shard_stats", {})[str(shard_idx)] = {
            "users_with_data": len(partial),
            "shard": shard_path.name,
            "incremental_merge": args.incremental_merge,
        }
        save_meta(args.work_dir, meta)
        completed_shards.add(shard_idx)

    if args.skip_merge:
        log.info("skip-merge set; done after shard scans")
        return 0

    completed = sorted(
        int(s) for s in meta.get("completed_shards", []) if int(s) in args.shards
    )
    if len(completed) != len(args.shards):
        log.warning(
            "only %d/%d shard checkpoints present; merge may be incomplete",
            len(completed),
            len(args.shards),
        )

    if args.incremental_merge:
        log.info("finalising incremental merge (%d shards) ...", len(completed))
        timestamps = load_partial_merged(args.work_dir)
    else:
        log.info("merging %d shard checkpoints ...", len(completed))
        timestamps = merge_shard_checkpoints(args.work_dir, completed)
        # Re-apply per-user cap after merge (users may appear in multiple shards).
        if args.max_tweets_per_user is not None:
            cap = args.max_tweets_per_user
            for uid in list(timestamps.keys()):
                if len(timestamps[uid]) > cap:
                    timestamps[uid] = sorted(timestamps[uid])[-cap:]

    parquet_path = args.work_dir / "timestamps.parquet"
    write_timestamps_parquet(timestamps, parquet_path)

    labels = {str(u): str(subset.loc[u, "label"]) for u in timestamps if u in subset.index}
    splits = {str(u): str(subset.loc[u, "split"]) for u in timestamps if u in subset.index}
    all_ts = [t for ts in timestamps.values() for t in ts]
    metadata = {
        "twibot22_root": str(args.twibot22_root.resolve()),
        "user_scope": args.user_scope,
        "shards_scanned": completed,
        "max_tweets_per_user": args.max_tweets_per_user,
        "users_total": len(user_ids),
        "users_with_tweets": len(timestamps),
        "window_start_unix": float(min(all_ts)) if all_ts else None,
        "window_end_unix": float(max(all_ts)) if all_ts else None,
        "elapsed_seconds": time.time() - t0,
    }
    meta["merge"] = metadata
    save_meta(args.work_dir, meta)

    if args.incremental_merge and partial_merged_path(args.work_dir).exists():
        partial_merged_path(args.work_dir).unlink()
        log.info("removed incremental partial checkpoint after parquet write")

    if args.write_json:
        json_path = args.work_dir / "timestamps.json"
        write_timestamps_json_bundle(
            timestamps, labels, splits, metadata, json_path
        )

    log.info(
        "done: users_with_tweets=%d parquet=%s elapsed=%.0fs",
        len(timestamps),
        parquet_path,
        time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
