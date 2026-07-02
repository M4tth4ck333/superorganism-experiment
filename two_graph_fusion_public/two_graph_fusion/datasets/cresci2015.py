"""Cresci-2015 dataset loader ("Fame for Sale", raw CSV release).

On-disk layout (``cresci-15-orig/``)::

    <root>/
        TFP.csv/   users.csv tweets.csv friends.csv followers.csv   (genuine)
        E13.csv/   ...                                              (genuine)
        FSF.csv/   ...                                              (fake followers)
        INT.csv/   ...                                              (fake followers)
        TWT.csv/   ...                                              (fake followers)

Labels are assigned by group: TFP + E13 are genuine human accounts; FSF
(fastfollowerz) + INT (intertwitter) + TWT (twittertechnology) are fake
followers (bots). This matches the 1,950 human / 3,351 bot split of the
standardised tensor release.

Unlike Cresci-2017, every tweets.csv here carries a **header** with named
columns and uses standard CSV quoting, and the dataset additionally ships
``friends.csv`` / ``followers.csv`` — a real **follow graph** (in addition
to the reply/retweet interaction graph reconstructable from tweets).

Tweet columns (named): ``created_at, id, text, source, user_id, truncated,
in_reply_to_status_id, in_reply_to_user_id, in_reply_to_screen_name,
retweeted_status_id, ..., timestamp``. ``timestamp`` is ISO
``YYYY-MM-DD HH:MM:SS``; this loader exposes the same interface as
:mod:`two_graph_fusion.datasets.cresci2017` so the TVT runner is shared.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Group folder -> label.
_GROUPS: dict[str, str] = {
    "TFP.csv": "human",
    "E13.csv": "human",
    "FSF.csv": "bot",
    "INT.csv": "bot",
    "TWT.csv": "bot",
}

_NA = ["\\N", "NULL", "null", ""]


def _parse_iso_timestamp(s: str) -> float | None:
    """Parse 'YYYY-MM-DD HH:MM:SS' -> Unix timestamp (UTC)."""
    try:
        dt = datetime.strptime(str(s).strip(), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, AttributeError, TypeError):
        return None


def _read_csv(path: Path, usecols: list[str]) -> pd.DataFrame | None:
    """Read a headered Cresci-2015 CSV, keeping only ``usecols`` (C engine)."""
    try:
        return pd.read_csv(
            path, dtype=str, usecols=usecols, on_bad_lines="skip",
            na_values=_NA, encoding="utf-8", encoding_errors="replace",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read %s: %s", path, exc)
        return None


def load_labels(root: Path) -> dict[str, str]:
    """Return user_id (str) -> 'human' / 'bot', assigned by group folder."""
    label_map: dict[str, str] = {}
    for folder, label in _GROUPS.items():
        users_csv = root / folder / "users.csv"
        if not users_csv.exists():
            logger.warning("missing %s, skipping", users_csv)
            continue
        df = _read_csv(users_csv, usecols=["id"])
        if df is None:
            continue
        for uid in df["id"].dropna():
            uid_str = str(uid).strip()
            if uid_str and uid_str != "nan":
                label_map[uid_str] = label
        logger.info("loaded %d %s users from %s", len(df), label, folder)
    return label_map


def load_timestamps(
    root: Path, user_filter: set[str] | None = None,
) -> dict[str, list[float]]:
    """Return user_id (str) -> sorted list of Unix timestamps (ISO column)."""
    ts_map: dict[str, list[float]] = {}
    for folder in _GROUPS:
        tweets_csv = root / folder / "tweets.csv"
        if not tweets_csv.exists():
            continue
        df = _read_csv(tweets_csv, usecols=["user_id", "timestamp"])
        if df is None:
            continue
        pairs = df.dropna()
        n_kept = 0
        for uid_raw, ts_raw in zip(pairs["user_id"], pairs["timestamp"]):
            uid = str(uid_raw).strip()
            if not uid or uid == "nan":
                continue
            if user_filter is not None and uid not in user_filter:
                continue
            ts = _parse_iso_timestamp(ts_raw)
            if ts is None:
                continue
            ts_map.setdefault(uid, []).append(ts)
            n_kept += 1
        logger.info("  %s: scanned=%d kept=%d users_so_far=%d",
                    folder, len(df), n_kept, len(ts_map))
    for uid in ts_map:
        ts_map[uid].sort()
    return ts_map


def load_reply_edges(
    root: Path, user_filter: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Directed reply edges (user_id -> in_reply_to_user_id)."""
    edges: list[tuple[str, str]] = []
    for folder in _GROUPS:
        tweets_csv = root / folder / "tweets.csv"
        if not tweets_csv.exists():
            continue
        df = _read_csv(tweets_csv, usecols=["user_id", "in_reply_to_user_id"])
        if df is None:
            continue
        sub = df.dropna()
        sub = sub[sub["in_reply_to_user_id"] != "0"]
        for src_raw, dst_raw in zip(sub["user_id"], sub["in_reply_to_user_id"]):
            src, dst = str(src_raw).strip(), str(dst_raw).strip()
            if not src or not dst or src in {"nan"} or dst in {"nan"} or src == dst:
                continue
            if user_filter is not None and (src not in user_filter or dst not in user_filter):
                continue
            edges.append((src, dst))
        logger.info("reply edges after %s: %d", folder, len(edges))
    return edges


def load_retweet_edges(
    root: Path, user_filter: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Directed retweet edges (retweeter -> original author).

    Two passes: build tweet_id -> author across all groups, then resolve
    each ``retweeted_status_id`` to its author.
    """
    tweet_author: dict[str, str] = {}
    for folder in _GROUPS:
        tweets_csv = root / folder / "tweets.csv"
        if not tweets_csv.exists():
            continue
        df = _read_csv(tweets_csv, usecols=["id", "user_id"])
        if df is None:
            continue
        pairs = df.dropna()
        for tid, uid in zip(pairs["id"], pairs["user_id"]):
            tweet_author[str(tid).strip()] = str(uid).strip()
        logger.info("retweet pass1 %s: author map=%d", folder, len(tweet_author))

    edges: list[tuple[str, str]] = []
    for folder in _GROUPS:
        tweets_csv = root / folder / "tweets.csv"
        if not tweets_csv.exists():
            continue
        df = _read_csv(tweets_csv, usecols=["user_id", "retweeted_status_id"])
        if df is None:
            continue
        sub = df.dropna()
        sub = sub[sub["retweeted_status_id"] != "0"]
        for src_raw, rt_raw in zip(sub["user_id"], sub["retweeted_status_id"]):
            src = str(src_raw).strip()
            dst = tweet_author.get(str(rt_raw).strip())
            if dst is None or src == dst:
                continue
            if user_filter is not None and (src not in user_filter or dst not in user_filter):
                continue
            edges.append((src, dst))
        logger.info("retweet edges after %s: %d", folder, len(edges))
    return edges


def load_follow_edges(
    root: Path, user_filter: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Directed follow edges (source_id follows target_id).

    Reads both ``friends.csv`` and ``followers.csv`` from every group; both
    encode ``(source_id, target_id)`` = source follows target. Deduplicated.
    """
    edges: set[tuple[str, str]] = set()
    for folder in _GROUPS:
        for fname in ("friends.csv", "followers.csv"):
            path = root / folder / fname
            if not path.exists():
                continue
            df = _read_csv(path, usecols=["source_id", "target_id"])
            if df is None:
                continue
            sub = df.dropna()
            for s_raw, t_raw in zip(sub["source_id"], sub["target_id"]):
                s, t = str(s_raw).strip(), str(t_raw).strip()
                if not s or not t or s == "nan" or t == "nan" or s == t:
                    continue
                if user_filter is not None and (s not in user_filter or t not in user_filter):
                    continue
                edges.add((s, t))
            logger.info("follow edges after %s/%s: %d", folder, fname, len(edges))
    return list(edges)
