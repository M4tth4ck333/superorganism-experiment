"""Checkpointed tweet extraction for at-scale TwiBot-22 runs.

A single JSON dump of timestamps for ~1M users is fragile (hours of work
lost on crash). This module writes one pickle checkpoint per tweet shard
and merges them into a final artefact when all shards finish.

Checkpoint layout::

    <work_dir>/checkpoints/shard_00.pkl   # dict[user_id, list[float]]
    <work_dir>/checkpoints/meta.json      # progress metadata
    <work_dir>/timestamps.parquet         # final merged table (optional)
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def checkpoint_path(work_dir: Path, shard_index: int) -> Path:
    return work_dir / "checkpoints" / f"shard_{shard_index:02d}.pkl"


def meta_path(work_dir: Path) -> Path:
    return work_dir / "checkpoints" / "meta.json"


def load_meta(work_dir: Path) -> dict:
    path = meta_path(work_dir)
    if not path.exists():
        return {"completed_shards": [], "shard_stats": {}}
    with path.open() as fh:
        return json.load(fh)


def save_meta(work_dir: Path, meta: dict) -> None:
    path = meta_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(meta, fh, indent=2)


def is_shard_done(work_dir: Path, shard_index: int) -> bool:
    return checkpoint_path(work_dir, shard_index).exists()


def save_shard_checkpoint(
    work_dir: Path,
    shard_index: int,
    timestamps: dict[str, list[float]],
) -> None:
    path = checkpoint_path(work_dir, shard_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(timestamps, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("wrote checkpoint %s (%d users)", path, len(timestamps))


def load_shard_checkpoint(work_dir: Path, shard_index: int) -> dict[str, list[float]]:
    path = checkpoint_path(work_dir, shard_index)
    with path.open("rb") as fh:
        return pickle.load(fh)


def merge_shard_checkpoints(
    work_dir: Path,
    shard_indices: Iterable[int],
) -> dict[str, list[float]]:
    """Merge per-shard dicts; later shards append timestamps for shared users."""
    merged: dict[str, list[float]] = defaultdict(list)
    for idx in shard_indices:
        partial = load_shard_checkpoint(work_dir, idx)
        for uid, ts_list in partial.items():
            merged[uid].extend(ts_list)
    return dict(merged)


def partial_merged_path(work_dir: Path) -> Path:
    """Rolling merge checkpoint used by ``--incremental-merge`` extraction."""
    return work_dir / "checkpoints" / "merged_partial.pkl"


def load_partial_merged(work_dir: Path) -> dict[str, list[float]]:
    path = partial_merged_path(work_dir)
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return pickle.load(fh)


def save_partial_merged(work_dir: Path, timestamps: dict[str, list[float]]) -> None:
    path = partial_merged_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(timestamps, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("wrote partial merge %s (%d users)", path, len(timestamps))


def merge_timestamp_dicts(
    base: dict[str, list[float]],
    incoming: dict[str, list[float]],
    max_tweets_per_user: int | None,
) -> dict[str, list[float]]:
    """Append ``incoming`` into ``base`` and optionally cap list length per user."""
    for uid, ts_list in incoming.items():
        if uid in base:
            base[uid].extend(ts_list)
        else:
            base[uid] = list(ts_list)
        if max_tweets_per_user is not None and len(base[uid]) > max_tweets_per_user:
            base[uid] = sorted(base[uid])[-max_tweets_per_user:]
    return base


def write_timestamps_parquet(
    timestamps: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Write a parquet file with columns ``user_id`` and ``timestamps`` (list)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "user_id": list(timestamps.keys()),
            "timestamps": [timestamps[u] for u in timestamps],
        }
    )
    df.to_parquet(output_path, index=False)
    logger.info("wrote %s (%d users)", output_path, len(df))


def read_timestamps_parquet(path: Path) -> dict[str, list[float]]:
    df = pd.read_parquet(path)
    return {
        str(row.user_id): list(row.timestamps)
        for row in df.itertuples(index=False)
    }


def write_timestamps_json_bundle(
    timestamps: dict[str, list[float]],
    labels: dict[str, str],
    splits: dict[str, str],
    metadata: dict,
    output_path: Path,
) -> None:
    """Write the same JSON schema as ``extract_twibot22_subset`` for compatibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "labels": labels,
        "splits": splits,
        "timestamps": timestamps,
    }
    with output_path.open("w") as fh:
        json.dump(payload, fh)
    logger.info("wrote %s (%d users)", output_path, len(timestamps))
