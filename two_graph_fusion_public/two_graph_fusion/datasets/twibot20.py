"""TwiBot-20 dataset loader (labels, official split, follow graph, profile).

On-disk layout (the standard GabrielHam/TwiBot-20 release)::

    <root>/
        train.json      # list of labelled user objects
        dev.json        # labelled (our validation split)
        test.json       # labelled
        support.json    # ~5 GB, UNLABELLED neighbourhood nodes

Each user object has keys ``ID``, ``profile`` (Twitter user object),
``tweet`` (list of up to 200 tweet **text** strings — *no timestamps*),
``neighbor`` (``{"following": [...ids], "follower": [...ids]}``; ``None``
for some users), ``domain``, ``label`` (``"1"`` bot / ``"0"`` human).

Important limitation
--------------------

TwiBot-20 tweets are bare text with **no per-tweet timestamps**, so the
five temporal behavioral features used elsewhere in this codebase cannot
be computed here. This loader therefore exposes the **profile/statistical**
fields (the signal BotRGCN / SEGCN use) and the directed **follow graph**;
the choice of how to build a second graph ``G_B`` on TwiBot-20 is a
modelling decision documented in ``RESULTS.md`` (M12).

The files are large; users are streamed with :mod:`ijson` so peak memory
stays modest. ``support.json`` is only scanned when explicitly requested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import ijson
import pandas as pd

logger = logging.getLogger(__name__)

# Split file -> our split label. dev.json is used as the validation split.
_SPLIT_FILES: dict[str, str] = {
    "train.json": "train",
    "dev.json": "val",
    "test.json": "test",
}

_LABEL_MAP = {"0": "human", "1": "bot", 0: "human", 1: "bot"}

# Profile fields used downstream (statistical + categorical). These mirror
# the feature set common to BotRGCN / SEGCN on TwiBot-20.
_NUMERIC_PROFILE_FIELDS = (
    "followers_count", "friends_count", "listed_count",
    "favourites_count", "statuses_count",
)
_BOOL_PROFILE_FIELDS = (
    "verified", "protected", "geo_enabled",
    "default_profile", "default_profile_image",
)


@dataclass
class TwiBot20Data:
    """Loaded TwiBot-20 user table + directed follow edges."""

    users: pd.DataFrame                    # indexed by user id (str)
    follow_edges: list[tuple[str, str]]    # directed (tail, head): tail follows head
    n_labeled: int
    n_support_nodes: int = 0
    label_col: str = "label"
    split_col: str = "split"
    meta: dict = field(default_factory=dict)


def _parse_created_at(s: str | None) -> float | None:
    """Twitter ``created_at`` ('Wed Oct 10 ...') -> Unix timestamp (UTC)."""
    if not s:
        return None
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%a %b %d %H:%M:%S +0000 %Y"):
        try:
            dt = datetime.strptime(s.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError):
            continue
    return None


def _to_float(v) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def _to_bool_int(v) -> int:
    """Coerce Twitter's bool-ish profile flags to {0, 1}."""
    if v in (True, "True", "true", 1, "1"):
        return 1
    return 0


def _profile_row(profile: dict, observation_ts: float | None) -> dict:
    """Extract numeric + categorical profile features from a user object."""
    row: dict[str, float | int | None] = {}
    for fld in _NUMERIC_PROFILE_FIELDS:
        row[fld] = _to_float(profile.get(fld))
    for fld in _BOOL_PROFILE_FIELDS:
        row[fld] = _to_bool_int(profile.get(fld))
    created = _parse_created_at(profile.get("created_at"))
    if created is not None and observation_ts is not None:
        age_days = max((observation_ts - created) / 86400.0, 1.0)
        row["account_age_days"] = age_days
        sc = row.get("statuses_count")
        row["tweets_per_day"] = (sc / age_days) if sc is not None else None
    else:
        row["account_age_days"] = None
        row["tweets_per_day"] = None
    return row


def _iter_users(path: Path):
    with path.open("rb") as fh:
        yield from ijson.items(fh, "item")


def load_twibot20(
    root: Path,
    include_support: bool = False,
    observation_date: str = "2020-09-01",
) -> TwiBot20Data:
    """Load TwiBot-20 labels, official split, profile features, follow edges.

    Args:
        root: Directory containing ``train.json`` / ``dev.json`` /
            ``test.json`` (and optionally ``support.json``).
        include_support: If ``True``, also stream ``support.json`` (~5 GB) to
            add unlabeled neighbourhood nodes + their follow edges. Off by
            default — labeled-only is enough for the homophily gate.
        observation_date: Reference date for ``account_age_days`` /
            ``tweets_per_day`` (TwiBot-20 was crawled in 2020).

    Returns:
        :class:`TwiBot20Data`.
    """
    obs_ts = _parse_created_at(
        datetime.strptime(observation_date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc).strftime("%a %b %d %H:%M:%S +0000 %Y")
    )

    rows: dict[str, dict] = {}
    edges: list[tuple[str, str]] = []

    def _ingest(obj: dict, split: str | None) -> None:
        uid = str(obj.get("ID")).strip()
        if not uid or uid == "None":
            return
        profile = obj.get("profile") or {}
        row = _profile_row(profile, obs_ts)
        if split is not None:
            row["label"] = _LABEL_MAP.get(obj.get("label"), None)
            row["split"] = split
        else:
            row.setdefault("label", None)
            row.setdefault("split", "support")
        # Keep the first occurrence's label/split; merge edges regardless.
        if uid not in rows:
            rows[uid] = row
        nb = obj.get("neighbor")
        if isinstance(nb, dict):
            for f in (nb.get("following") or []):
                fid = str(f).strip()
                if fid and fid != uid:
                    edges.append((uid, fid))          # uid follows fid
            for g in (nb.get("follower") or []):
                gid = str(g).strip()
                if gid and gid != uid:
                    edges.append((gid, uid))          # gid follows uid

    n_labeled = 0
    for fname, split in _SPLIT_FILES.items():
        path = root / fname
        if not path.exists():
            logger.warning("missing %s, skipping", path)
            continue
        cnt = 0
        for obj in _iter_users(path):
            _ingest(obj, split)
            cnt += 1
        n_labeled += cnt
        logger.info("loaded %d users from %s (split=%s)", cnt, fname, split)

    n_support = 0
    if include_support:
        path = root / "support.json"
        if path.exists():
            for obj in _iter_users(path):
                _ingest(obj, None)
                n_support += 1
                if n_support % 50_000 == 0:
                    logger.info("  support streamed=%d", n_support)
            logger.info("loaded %d support users", n_support)

    users = pd.DataFrame.from_dict(rows, orient="index")
    users.index.name = "id"

    # Deduplicate directed edges.
    edges = list({e for e in edges})
    logger.info(
        "TwiBot-20: %d user rows, %d unique directed follow edges",
        len(users), len(edges),
    )
    return TwiBot20Data(
        users=users,
        follow_edges=edges,
        n_labeled=n_labeled,
        n_support_nodes=n_support,
        meta={"observation_date": observation_date, "include_support": include_support},
    )


def induced_directed_edges(
    follow_edges: list[tuple[str, str]],
    node_set: set[str],
) -> list[tuple[str, str]]:
    """Keep only directed edges with both endpoints in ``node_set``."""
    return [(u, v) for (u, v) in follow_edges if u in node_set and v in node_set]
