"""TwiBot-22 dataset loaders.

This module knows the on-disk layout of the TwiBot-22 release (see
``twibot22/readme.md``) and exposes helpers to:

1. load the user split and label tables (``split.csv``, ``label.csv``);
2. stream the giant ``tweet_*.json`` files and extract per-user timestamps
   for an arbitrary subset of users.

Two design notes
----------------

- TwiBot-22 stores user ids as ``"u<digits>"`` in ``label.csv`` / ``split.csv``
  but as bare ``int`` in tweet ``author_id`` fields. The loaders normalise
  everything to the prefixed string form so downstream code only sees one
  type.
- Tweet timestamps are formatted as ``"YYYY-MM-DD HH:MM:SS+00:00"``. Parsing
  this with :func:`datetime.fromisoformat` is roughly 4x faster than
  :func:`pandas.to_datetime` on a per-row basis; for ten million tweets the
  difference is the wall-clock budget for a small-subset run.

The streaming code uses ``ijson`` with the C backend (``yajl2_c``) when
available, falling back to the pure-Python parser otherwise.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import ijson.backends.yajl2_c as ijson_backend  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - rare
    import ijson as ijson_backend  # type: ignore[no-redef]
    logger.warning("ijson C backend unavailable, falling back to pure Python")


@dataclass(frozen=True)
class TwiBot22Paths:
    """Absolute paths to the TwiBot-22 release files we care about.

    Attributes:
        root: Directory containing ``label.csv``, ``split.csv``, and the
            nine ``tweet_*.json`` shards.
        label_csv: ``root / "label.csv"``.
        split_csv: ``root / "split.csv"``.
        tweet_files: Sorted list of the nine ``tweet_*.json`` shards.
    """

    root: Path
    label_csv: Path
    split_csv: Path
    tweet_files: tuple[Path, ...]


def discover_twibot22(root: str | Path) -> TwiBot22Paths:
    """Locate the TwiBot-22 release files under ``root``.

    Args:
        root: Path to the directory that contains the dataset.

    Returns:
        A :class:`TwiBot22Paths` bundle.

    Raises:
        FileNotFoundError: If any expected file is missing.
    """
    root_path = Path(root).expanduser().resolve()
    label_csv = root_path / "label.csv"
    split_csv = root_path / "split.csv"
    tweet_files = tuple(sorted(root_path.glob("tweet_*.json")))

    for required in (label_csv, split_csv):
        if not required.exists():
            raise FileNotFoundError(f"missing required TwiBot-22 file: {required}")
    if not tweet_files:
        raise FileNotFoundError(f"no tweet_*.json files under {root_path}")

    return TwiBot22Paths(
        root=root_path,
        label_csv=label_csv,
        split_csv=split_csv,
        tweet_files=tweet_files,
    )


def load_labels_and_splits(paths: TwiBot22Paths) -> pd.DataFrame:
    """Load ``label.csv`` and ``split.csv`` and join on ``user_id``.

    Args:
        paths: Resolved paths returned by :func:`discover_twibot22`.

    Returns:
        DataFrame indexed by ``user_id`` (``"u<digits>"``) with columns
        ``["label", "split"]``. ``label`` is ``"human"`` or ``"bot"``;
        ``split`` is ``"train"``, ``"valid"``, or ``"test"``.
    """
    labels = pd.read_csv(paths.label_csv, dtype={"id": "string", "label": "string"})
    splits = pd.read_csv(paths.split_csv, dtype={"id": "string", "split": "string"})
    df = labels.merge(splits, on="id", how="inner").rename(columns={"id": "user_id"})
    df = df.set_index("user_id")
    logger.info(
        "loaded %d users (%s)",
        len(df),
        df["label"].value_counts().to_dict(),
    )
    return df


UserScope = str  # "all_labeled" | "train_only" | "train_test"


def filter_user_scope(
    labels_splits: pd.DataFrame,
    scope: UserScope,
) -> pd.DataFrame:
    """Restrict ``labels_splits`` to a TwiBot-22 evaluation cohort.

    Args:
        labels_splits: Full labeled population.
        scope:
            - ``"all_labeled"``: no filtering.
            - ``"train_only"``: official train split.
            - ``"train_test"``: train and test splits (recommended benchmark
              cohort; avoids the cross-class snowball bridge, RED-10).
    """
    if scope == "all_labeled":
        return labels_splits
    if scope == "train_only":
        return labels_splits[labels_splits["split"] == "train"]
    if scope == "train_test":
        return labels_splits[labels_splits["split"].isin(["train", "test"])]
    raise ValueError(f"unknown user scope {scope!r}")


def sample_balanced_subset(
    labels_splits: pd.DataFrame,
    n_per_class: int,
    splits: Iterable[str] = ("train",),
    seed: int = 0,
) -> pd.DataFrame:
    """Sample a class-balanced subset of users.

    Args:
        labels_splits: DataFrame from :func:`load_labels_and_splits`.
        n_per_class: Number of users to sample per class. Samples are
            without replacement; if a class has fewer than ``n_per_class``
            users the full class is returned.
        splits: Iterable of split names to draw from. Default ``("train",)``.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with the same columns as ``labels_splits``, restricted to
        the sampled users.
    """
    pool = labels_splits[labels_splits["split"].isin(list(splits))]
    sampled_parts = []
    for label, group in pool.groupby("label"):
        take = min(n_per_class, len(group))
        sampled_parts.append(group.sample(n=take, random_state=seed))
    sampled = pd.concat(sampled_parts).sort_index()
    logger.info(
        "sampled subset: %d users (%s)",
        len(sampled),
        sampled["label"].value_counts().to_dict(),
    )
    return sampled


def sample_balanced_train_test(
    labels_splits: pd.DataFrame,
    n_per_class: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Balanced random sample from official train and test splits.

    No follow-graph structure is imposed. Use this cohort when you want
    benchmark-compatible users without snowball BFS (milestone 4).
    """
    pool = filter_user_scope(labels_splits, "train_test")
    return sample_balanced_subset(pool, n_per_class=n_per_class, splits=("train", "test"), seed=seed)


def _parse_created_at(value: str) -> float:
    """Parse a TwiBot-22 ``created_at`` string into a Unix-second float.

    The dataset stores timestamps as ``"YYYY-MM-DD HH:MM:SS+00:00"``.
    :func:`datetime.fromisoformat` accepts this format directly in Python
    3.11+ and returns a tz-aware datetime, whose ``.timestamp()`` gives the
    Unix seconds.
    """
    return datetime.fromisoformat(value).timestamp()


def stream_timestamps_for_users(
    tweet_files: Iterable[Path],
    user_ids: set[str],
    progress_every: int = 5_000_000,
    max_tweets_per_user: int | None = None,
) -> dict[str, list[float]]:
    """Stream tweet shards and collect Unix-second timestamps per user.

    Args:
        tweet_files: Iterable of paths to ``tweet_*.json`` shards. Each
            shard is a top-level JSON array of tweet objects with at least
            the keys ``author_id`` and ``created_at``.
        user_ids: Set of user ids to keep. Each id should be in the
            ``"u<digits>"`` form used elsewhere in this package; the
            integer ``author_id`` from each tweet is prefixed with ``"u"``
            before look-up.
        progress_every: How often to log progress (in tweets scanned).
        max_tweets_per_user: Optional early-stopping cap. Once a user has
            accumulated this many timestamps further tweets for them are
            ignored. Useful when the goal is feature extraction and a few
            hundred IATs per user are enough.

    Returns:
        Dict ``user_id -> list of Unix-second timestamps``. Users with no
        matching tweets are absent from the dict.
    """
    out: dict[str, list[float]] = defaultdict(list)
    total = 0
    matched = 0
    for shard in tweet_files:
        logger.info("scanning %s ...", shard.name)
        with shard.open("rb") as fh:
            total, matched = _scan_shard(
                fh,
                user_ids=user_ids,
                out=out,
                total=total,
                matched=matched,
                progress_every=progress_every,
                max_tweets_per_user=max_tweets_per_user,
            )
        logger.info(
            "  done: scanned=%d matched=%d users_with_data=%d",
            total,
            matched,
            len(out),
        )
    return dict(out)


def _scan_shard(
    fh: IO[bytes],
    user_ids: set[str],
    out: dict[str, list[float]],
    total: int,
    matched: int,
    progress_every: int,
    max_tweets_per_user: int | None,
) -> tuple[int, int]:
    """Inner streaming loop for a single shard.

    Returns the updated ``(total_scanned, total_matched)`` counters.
    """
    items: Iterator[dict] = ijson_backend.items(fh, "item")
    for tweet in items:
        total += 1
        if total % progress_every == 0:
            logger.info(
                "    scanned=%d matched=%d users_with_data=%d",
                total,
                matched,
                len(out),
            )
        author_id = tweet.get("author_id")
        if author_id is None:
            continue
        uid = f"u{author_id}"
        if uid not in user_ids:
            continue
        if max_tweets_per_user is not None and len(out[uid]) >= max_tweets_per_user:
            continue
        created_at = tweet.get("created_at")
        if not created_at:
            continue
        try:
            ts = _parse_created_at(created_at)
        except ValueError:
            continue
        out[uid].append(ts)
        matched += 1
    return total, matched
