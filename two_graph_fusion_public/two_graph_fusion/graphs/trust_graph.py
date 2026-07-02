"""TwiBot-22 trust graph ``G_T`` construction.

The TwiBot-22 release ships a single 6.6 GB ``edge.csv`` listing 170M
heterogeneous edges (14 relation types between users, tweets, lists, and
hashtags). For sybil detection we want the *trust graph*: a homogeneous
graph between user nodes whose edges encode a follow / followed-by /
follower relationship.

This module provides:

- :func:`iter_user_user_edges` / :func:`build_trust_graph`: stream
  ``edge.csv`` and filter to user-user follow edges in a caller-supplied
  user set. Used by the F6 diagnostic.
- :func:`cache_all_user_follow_edges`: stream ``edge.csv`` *once* and
  write a compact CSV of every user-user follow edge in the dataset.
  This is a one-time investment that lets downstream sampling routines
  (snowball, LCC selection) skip the 70-second full scan.
- :func:`load_follow_edge_cache`: read the cache back into a NetworkX
  graph.

Stream-filter rationale
-----------------------

Loading the full 170M-row file with :func:`pandas.read_csv` would burn
~25 GB of RAM. The streaming approach with a per-row Python tuple costs
roughly 5 GB peak memory and runs in ~3-4 minutes on a fast SSD; the
matched-edge count is tiny (hundreds to a few thousand) because the
qualifying subset is a thin slice of the dataset.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)

DEFAULT_FOLLOW_RELATIONS: frozenset[str] = frozenset(
    {"following", "followers", "followed"}
)

# The CSV is gigantic; a generous per-row buffer noticeably speeds up
# Python's CSV parser without raising memory dramatically.
_CSV_BUFFER_BYTES = 1 << 22  # 4 MiB


@dataclass(frozen=True)
class TrustGraphBuildResult:
    """Output bundle from :func:`build_trust_graph`.

    Attributes:
        graph: Undirected NetworkX graph. Nodes are user ids (a subset of
            ``user_ids``); isolated users with no follow edges in the
            scanned file are still added as nodes (so ``G_B`` and ``G_T``
            share the same node set).
        n_user_ids: Number of input user ids.
        n_edges_scanned: How many rows of ``edge.csv`` were considered.
        n_edges_matched: Rows whose relation was in
            ``relations`` and whose both endpoints were in ``user_ids``.
        n_unique_edges: Edge count of the returned graph (after
            deduplication of multi-relation pairs).
        relations: Frozenset of relation strings the filter accepted.
    """

    graph: nx.Graph
    n_user_ids: int
    n_edges_scanned: int
    n_edges_matched: int
    n_unique_edges: int
    relations: frozenset[str]


def iter_user_user_edges(
    edge_csv: Path,
    user_ids: set[str],
    relations: Iterable[str] = DEFAULT_FOLLOW_RELATIONS,
    progress_every: int = 20_000_000,
) -> Iterator[tuple[str, str, str]]:
    """Stream-filter ``edge.csv`` to user-user edges in the requested set.

    Args:
        edge_csv: Path to the TwiBot-22 ``edge.csv``.
        user_ids: Set of user ids (``"u<digits>"``) to keep.
        relations: Iterable of relation strings to accept.
        progress_every: How often to log scan progress.

    Yields:
        ``(source_id, target_id, relation)`` tuples. Self-loops are
        suppressed. The (source, target) ordering reflects the file; the
        caller is responsible for treating the graph as undirected if
        desired.
    """
    rel_set = frozenset(relations)
    scanned = 0
    matched = 0
    with edge_csv.open("r", newline="", buffering=_CSV_BUFFER_BYTES) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        try:
            src_idx = header.index("source_id")
            rel_idx = header.index("relation")
            tgt_idx = header.index("target_id")
        except ValueError as exc:
            raise ValueError(
                f"edge.csv missing expected columns: {header}"
            ) from exc

        for row in reader:
            scanned += 1
            if scanned % progress_every == 0:
                logger.info(
                    "    edge.csv scanned=%d matched=%d",
                    scanned,
                    matched,
                )
            if len(row) < 3:
                continue
            relation = row[rel_idx]
            if relation not in rel_set:
                continue
            source = row[src_idx]
            target = row[tgt_idx]
            if source == target:
                continue
            if source not in user_ids or target not in user_ids:
                continue
            matched += 1
            yield source, target, relation

    logger.info(
        "edge.csv done: scanned=%d matched=%d",
        scanned,
        matched,
    )


def build_trust_graph(
    edge_csv: Path,
    user_ids: Iterable[str],
    relations: Iterable[str] = DEFAULT_FOLLOW_RELATIONS,
    progress_every: int = 20_000_000,
) -> TrustGraphBuildResult:
    """Build an undirected user-user trust graph from ``edge.csv``.

    Args:
        edge_csv: Path to ``edge.csv``.
        user_ids: Iterable of user ids to include as nodes (even if no
            edges are recovered for them).
        relations: Relation strings accepted as trust edges.
        progress_every: How often to log scan progress.

    Returns:
        :class:`TrustGraphBuildResult`.
    """
    uid_set = set(user_ids)
    relations_fs = frozenset(relations)

    graph = nx.Graph()
    graph.add_nodes_from(uid_set)

    scanned = 0
    matched = 0
    with edge_csv.open("r", newline="", buffering=_CSV_BUFFER_BYTES) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        try:
            src_idx = header.index("source_id")
            rel_idx = header.index("relation")
            tgt_idx = header.index("target_id")
        except ValueError as exc:
            raise ValueError(
                f"edge.csv missing expected columns: {header}"
            ) from exc

        for row in reader:
            scanned += 1
            if scanned % progress_every == 0:
                logger.info(
                    "    edge.csv scanned=%d matched=%d unique_edges=%d",
                    scanned,
                    matched,
                    graph.number_of_edges(),
                )
            if len(row) < 3:
                continue
            relation = row[rel_idx]
            if relation not in relations_fs:
                continue
            source = row[src_idx]
            target = row[tgt_idx]
            if source == target:
                continue
            if source not in uid_set or target not in uid_set:
                continue
            matched += 1
            # NetworkX dedupes (u, v) and (v, u) for an undirected Graph.
            graph.add_edge(source, target)

    logger.info(
        "G_T: scanned=%d matched_rows=%d unique_edges=%d nodes=%d",
        scanned,
        matched,
        graph.number_of_edges(),
        graph.number_of_nodes(),
    )
    return TrustGraphBuildResult(
        graph=graph,
        n_user_ids=len(uid_set),
        n_edges_scanned=scanned,
        n_edges_matched=matched,
        n_unique_edges=graph.number_of_edges(),
        relations=relations_fs,
    )


def cache_all_user_follow_edges(
    edge_csv: Path,
    output_csv: Path,
    relations: Iterable[str] = DEFAULT_FOLLOW_RELATIONS,
    progress_every: int = 20_000_000,
) -> tuple[int, int]:
    """Write every user-user follow edge from ``edge.csv`` into a compact CSV.

    The output CSV has the header ``source,target`` and one row per
    accepted edge, with no relation column. Mirror rows
    (``u1 -> u2`` via ``following`` and ``u2 -> u1`` via ``followers``)
    are kept verbatim; the loader collapses them to undirected edges.

    Args:
        edge_csv: Path to the TwiBot-22 ``edge.csv``.
        output_csv: Destination CSV for the compact cache.
        relations: Relation strings to keep.
        progress_every: How often to log scan progress.

    Returns:
        ``(rows_scanned, rows_written)``.
    """
    rel_set = frozenset(relations)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scanned = 0
    written = 0
    with edge_csv.open("r", newline="", buffering=_CSV_BUFFER_BYTES) as fin, \
         output_csv.open("w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        try:
            src_idx = header.index("source_id")
            rel_idx = header.index("relation")
            tgt_idx = header.index("target_id")
        except ValueError as exc:
            raise ValueError(
                f"edge.csv missing expected columns: {header}"
            ) from exc
        writer.writerow(("source", "target"))

        for row in reader:
            scanned += 1
            if scanned % progress_every == 0:
                logger.info(
                    "    cache scan: rows=%d written=%d",
                    scanned,
                    written,
                )
            if len(row) < 3:
                continue
            relation = row[rel_idx]
            if relation not in rel_set:
                continue
            source = row[src_idx]
            target = row[tgt_idx]
            # Only user-user edges (`u` prefix on both ends).
            if not (source.startswith("u") and target.startswith("u")):
                continue
            if source == target:
                continue
            writer.writerow((source, target))
            written += 1

    logger.info(
        "follow-edge cache done: rows=%d written=%d",
        scanned,
        written,
    )
    return scanned, written


def load_follow_edge_cache(
    cache_csv: Path,
    user_filter: set[str] | None = None,
) -> nx.Graph:
    """Load the compact follow-edge cache into an undirected NetworkX graph.

    Args:
        cache_csv: Path to the cache produced by
            :func:`cache_all_user_follow_edges`.
        user_filter: Optional set of user ids; if provided, only edges
            with both endpoints in this set are loaded.

    Returns:
        Undirected :class:`networkx.Graph` with mirror-row deduplication.
    """
    graph = nx.Graph()
    with cache_csv.open("r", newline="", buffering=_CSV_BUFFER_BYTES) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header[:2] != ["source", "target"]:
            raise ValueError(
                f"unexpected follow-edge cache header: {header}"
            )
        for row in reader:
            if len(row) < 2:
                continue
            src, tgt = row[0], row[1]
            if user_filter is not None:
                if src not in user_filter or tgt not in user_filter:
                    continue
            graph.add_edge(src, tgt)
    return graph
