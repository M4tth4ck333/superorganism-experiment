"""Cresci-2017 dataset loader.

Reads the Cresci-2017 "The Fake Project" dataset from its on-disk layout::

    <root>/
        genuine_accounts.csv/
            users.csv       # no header; same schema as bot users.csv
            tweets.csv      # no header; same schema as bot tweets.csv
        social_spambots_1.csv/
            users.csv
            tweets.csv
        social_spambots_2.csv/
        social_spambots_3.csv/
        traditional_spambots_1.csv/   # optional
        fake_followers.csv/           # optional

Subsets used by default:
    human  — genuine_accounts
    bot    — social_spambots_1, social_spambots_2, social_spambots_3

Tweet CSV schema (both with and without header, same column order):
    col 0: tweet id
    col 1: text
    col 2: source
    col 3: user_id
    col 4: truncated
    col 5: in_reply_to_status_id
    col 6: in_reply_to_user_id
    col 7: in_reply_to_screen_name
    col 8: retweeted_status_id
    ...
    col 22: created_at  (Twitter format "Fri May 01 00:18:11 +0000 2015")
    col 23: timestamp   (ISO "2015-05-01 02:18:11")

``\\N`` values (MySQL NULL) are treated as NaN/missing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Tweet column indices (no header)
_COL_TWEET_ID = 0
_COL_USER_ID = 3
_COL_REPLY_TO_USER_ID = 6
_COL_CREATED_AT = 22
_COL_TIMESTAMP_ISO = 23

_TWEET_COLUMNS = [
    "tweet_id", "text", "source", "user_id", "truncated",
    "in_reply_to_status_id", "in_reply_to_user_id", "in_reply_to_screen_name",
    "retweeted_status_id", "geo", "place", "contributors",
    "retweet_count", "reply_count", "favorite_count", "favorited",
    "retweeted", "possibly_sensitive", "num_hashtags", "num_urls",
    "num_mentions", "created_at", "timestamp", "crawled_at", "updated",
]

_SUBSETS = {
    "genuine_accounts.csv": "human",
    "social_spambots_1.csv": "bot",
    "social_spambots_2.csv": "bot",
    "social_spambots_3.csv": "bot",
}

_OPTIONAL_SUBSETS = {
    "traditional_spambots_1.csv": "bot",
    "fake_followers.csv": "bot",
}


def _parse_iso_timestamp(s: str) -> float | None:
    """Parse 'YYYY-MM-DD HH:MM:SS' -> Unix timestamp (UTC). Returns None on failure."""
    try:
        dt = datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, AttributeError):
        return None


def load_labels(root: Path, include_optional: bool = False) -> dict[str, str]:
    """Return mapping user_id (str) -> label ('human' or 'bot').

    Args:
        root: Path to the cresci_2017_datasets_full directory.
        include_optional: If True, also load traditional_spambots_1 and fake_followers.
    """
    subsets = dict(_SUBSETS)
    if include_optional:
        subsets.update(_OPTIONAL_SUBSETS)

    label_map: dict[str, str] = {}
    for folder, label in subsets.items():
        users_csv = root / folder / "users.csv"
        if not users_csv.exists():
            logger.warning("missing %s, skipping", users_csv)
            continue
        try:
            df = pd.read_csv(users_csv, dtype=str, on_bad_lines="skip")
        except Exception as exc:
            logger.warning("could not read %s: %s", users_csv, exc)
            continue
        id_col = "id" if "id" in df.columns else df.columns[0]
        for uid in df[id_col].dropna():
            uid_str = str(uid).strip()
            if uid_str:
                label_map[uid_str] = label
        logger.info("loaded %d %s users from %s", len(df), label, folder)

    return label_map


def load_timestamps(
    root: Path,
    user_filter: set[str] | None = None,
    include_optional: bool = False,
) -> dict[str, list[float]]:
    """Return mapping user_id (str) -> sorted list of Unix timestamps.

    Reads tweets.csv from each subset with tweets. Skips subsets that
    have no tweets.csv. Timestamps come from the ISO 'timestamp' column
    (col 23); rows where this is missing/invalid are dropped.

    Args:
        root: Path to the cresci_2017_datasets_full directory.
        user_filter: If provided, only collect timestamps for these user ids.
        include_optional: If True, also load traditional_spambots_1 and
            fake_followers.
    """
    subsets = dict(_SUBSETS)
    if include_optional:
        subsets.update(_OPTIONAL_SUBSETS)

    ts_map: dict[str, list[float]] = {}

    for folder, _label in subsets.items():
        tweets_csv = root / folder / "tweets.csv"
        if not tweets_csv.exists():
            logger.info("no tweets.csv in %s, skipping", folder)
            continue

        logger.info("reading tweets from %s ...", folder)
        n_rows = n_kept = 0

        # Detect header and CSV dialect from first line.
        # genuine_accounts uses MySQL-dump backslash escaping (\"...\");
        # bot files use standard CSV double-quote escaping.
        with tweets_csv.open(encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline()
        has_header = first_line.strip().startswith('"id"') or first_line.strip().startswith("id,")
        use_escapechar = r'\"' in first_line

        try:
            with tweets_csv.open(encoding="utf-8", errors="replace") as fh:
                df = pd.read_csv(
                    fh,
                    header=0 if has_header else None,
                    dtype=str,
                    on_bad_lines="skip",
                    na_values=["\\N", "NULL", ""],
                    engine="python",
                    escapechar="\\" if use_escapechar else None,
                )
        except Exception as exc:
            logger.warning("could not read %s: %s", tweets_csv, exc)
            continue

        # Column resolution: header files use named columns; no-header files use
        # positional indices (genuine_accounts has 26 cols, bots have 25).
        if has_header and "user_id" in df.columns and "timestamp" in df.columns:
            uid_series = df["user_id"]
            ts_series = df["timestamp"]
        elif not has_header and df.shape[1] >= 24:
            # positional: user_id=3, ISO timestamp=23 for 26-col genuine_accounts
            uid_series = df.iloc[:, _COL_USER_ID]
            ts_series = df.iloc[:, _COL_TIMESTAMP_ISO]
        else:
            logger.warning("unrecognised column layout in %s (ncols=%d)", folder, df.shape[1])
            continue

        n_rows = len(df)
        pairs = pd.DataFrame({"uid": uid_series, "ts": ts_series}).dropna()

        for uid_raw, ts_raw in zip(pairs["uid"], pairs["ts"]):
            uid = str(uid_raw).strip()
            if not uid or uid == "nan":
                continue
            if user_filter is not None and uid not in user_filter:
                continue
            ts = _parse_iso_timestamp(str(ts_raw))
            if ts is None:
                continue
            if uid not in ts_map:
                ts_map[uid] = []
            ts_map[uid].append(ts)
            n_kept += 1

        logger.info("  %s: scanned=%d  kept=%d  users_so_far=%d", folder, n_rows, n_kept, len(ts_map))

    # Sort each user's timestamps
    for uid in ts_map:
        ts_map[uid].sort()

    return ts_map


def load_reply_edges(
    root: Path,
    user_filter: set[str] | None = None,
    include_optional: bool = False,
) -> list[tuple[str, str]]:
    """Return list of (src_user_id, dst_user_id) directed reply edges.

    Uses ``in_reply_to_user_id`` (col 6) which is a direct user-to-user
    reference — no tweet-author resolution pass needed.

    Both endpoints must be in ``user_filter`` (if provided).

    Args:
        root: Path to the cresci_2017_datasets_full directory.
        user_filter: If provided, only keep edges where both endpoints are
            in this set.
        include_optional: If True, also load traditional_spambots_1 and
            fake_followers.
    """
    subsets = dict(_SUBSETS)
    if include_optional:
        subsets.update(_OPTIONAL_SUBSETS)

    edges: list[tuple[str, str]] = []

    for folder, _label in subsets.items():
        tweets_csv = root / folder / "tweets.csv"
        if not tweets_csv.exists():
            continue

        with tweets_csv.open(encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline()
        has_header = first_line.strip().startswith('"id"') or first_line.strip().startswith("id,")
        use_escapechar = r'\"' in first_line

        try:
            with tweets_csv.open(encoding="utf-8", errors="replace") as fh:
                df = pd.read_csv(
                    fh,
                    header=0 if has_header else None,
                    names=None if has_header else _TWEET_COLUMNS,
                    dtype=str,
                    on_bad_lines="skip",
                    na_values=["\\N", "NULL", ""],
                    engine="python",
                    escapechar="\\" if use_escapechar else None,
                )
        except Exception as exc:
            logger.warning("could not read %s: %s", tweets_csv, exc)
            continue

        if has_header and "user_id" in df.columns and "in_reply_to_user_id" in df.columns:
            src_series = df["user_id"]
            dst_series = df["in_reply_to_user_id"]
        elif not has_header and df.shape[1] >= 7:
            src_series = df.iloc[:, _COL_USER_ID]
            dst_series = df.iloc[:, _COL_REPLY_TO_USER_ID]
        else:
            continue

        sub = pd.DataFrame({"src": src_series, "dst": dst_series}).dropna()
        # Drop rows where in_reply_to_user_id is '0' (not a real reply)
        sub = sub[sub["dst"] != "0"]
        n_before = len(sub)
        for src_raw, dst_raw in zip(sub["src"], sub["dst"]):
            src = str(src_raw).strip()
            dst = str(dst_raw).strip()
            if not src or not dst or src == "nan" or dst == "nan" or src == dst:
                continue
            if user_filter is not None and (src not in user_filter or dst not in user_filter):
                continue
            edges.append((src, dst))

        logger.info("reply edges from %s: %d raw reply rows -> %d kept edges", folder, n_before, len(edges))

    return edges


def load_retweet_edges(
    root: Path,
    user_filter: set[str] | None = None,
    include_optional: bool = False,
) -> list[tuple[str, str]]:
    """Return list of (retweeter_user_id, original_author_user_id) retweet edges.

    Two-pass over all tweet files:
      Pass 1 — build tweet_id -> user_id map from every tweet.
      Pass 2 — for each tweet with a retweeted_status_id, emit an edge
                (src_user_id, original_author_user_id).

    Both endpoints must be in ``user_filter`` if provided.
    """
    subsets = dict(_SUBSETS)
    if include_optional:
        subsets.update(_OPTIONAL_SUBSETS)

    def _read_tweets(tweets_csv: Path) -> pd.DataFrame | None:
        with tweets_csv.open(encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline()
        has_header = first_line.strip().startswith('"id"') or first_line.strip().startswith("id,")
        use_escapechar = r'\"' in first_line
        try:
            with tweets_csv.open(encoding="utf-8", errors="replace") as fh:
                return pd.read_csv(
                    fh,
                    header=0 if has_header else None,
                    dtype=str,
                    on_bad_lines="skip",
                    na_values=["\\N", "NULL", ""],
                    engine="python",
                    escapechar="\\" if use_escapechar else None,
                )
        except Exception as exc:
            logger.warning("could not read %s: %s", tweets_csv, exc)
            return None

    # Determine column positions for each file (header vs positional)
    def _get_cols(df: pd.DataFrame, has_header: bool) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (tweet_id_series, user_id_series, retweeted_status_id_series)."""
        if has_header and "id" in df.columns:
            return df["id"], df["user_id"], df.get("retweeted_status_id", pd.Series(dtype=str))
        else:
            # positional: 0=tweet_id, 3=user_id, 8=retweeted_status_id
            tid = df.iloc[:, 0]
            uid = df.iloc[:, 3]
            rt_id = df.iloc[:, 8] if df.shape[1] > 8 else pd.Series(dtype=str)
            return tid, uid, rt_id

    # Pass 1: build tweet_id -> user_id across all subsets
    logger.info("retweet edges pass 1: building tweet_id -> user_id map ...")
    tweet_author: dict[str, str] = {}

    for folder in subsets:
        tweets_csv = root / folder / "tweets.csv"
        if not tweets_csv.exists():
            continue
        with tweets_csv.open(encoding="utf-8", errors="replace") as fh:
            fl = fh.readline()
        has_header = fl.strip().startswith('"id"') or fl.strip().startswith("id,")
        df = _read_tweets(tweets_csv)
        if df is None:
            continue
        tid_s, uid_s, _ = _get_cols(df, has_header)
        pairs = pd.DataFrame({"tid": tid_s, "uid": uid_s}).dropna()
        for tid, uid in zip(pairs["tid"], pairs["uid"]):
            tweet_author[str(tid).strip()] = str(uid).strip()
        logger.info("  %s: %d tweets in author map", folder, len(pairs))

    logger.info("pass 1 done: %d tweet-author pairs", len(tweet_author))

    # Pass 2: emit retweet edges
    logger.info("retweet edges pass 2: resolving retweet -> author edges ...")
    edges: list[tuple[str, str]] = []

    for folder in subsets:
        tweets_csv = root / folder / "tweets.csv"
        if not tweets_csv.exists():
            continue
        with tweets_csv.open(encoding="utf-8", errors="replace") as fh:
            fl = fh.readline()
        has_header = fl.strip().startswith('"id"') or fl.strip().startswith("id,")
        df = _read_tweets(tweets_csv)
        if df is None:
            continue
        _, uid_s, rt_s = _get_cols(df, has_header)
        pairs = pd.DataFrame({"uid": uid_s, "rt_id": rt_s}).dropna()
        pairs = pairs[pairs["rt_id"] != "0"]
        n_before = len(pairs)
        for src_raw, rt_id_raw in zip(pairs["uid"], pairs["rt_id"]):
            src = str(src_raw).strip()
            rt_id = str(rt_id_raw).strip()
            dst = tweet_author.get(rt_id)
            if dst is None or src == dst:
                continue
            if user_filter is not None and (src not in user_filter or dst not in user_filter):
                continue
            edges.append((src, dst))
        logger.info("  %s: %d retweet rows -> %d new edges (total=%d)",
                    folder, n_before, len(edges), len(edges))

    return edges
