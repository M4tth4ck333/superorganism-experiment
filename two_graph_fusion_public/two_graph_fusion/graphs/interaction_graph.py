"""User-user interaction graph G_T built from tweet-level interaction edges.

TwiBot-22's ``edge.csv`` stores interactions at the tweet level rather
than the user level:

- ``post``      (user  → tweet): user authored this tweet.
- ``retweeted`` (tweet → tweet): tweet A is a retweet of tweet B.
- ``quoted``    (tweet → tweet): tweet A is a quoted retweet of tweet B.
- ``mentioned`` (tweet → user):  tweet A mentions user B.

To obtain a *user-user* interaction graph we resolve these through the
``post`` edge type:

  retweeted / quoted: author(tweet_A) ——interacted——> author(tweet_B)
  mentioned:          author(tweet_A) ——interacted——> mentioned_user

The resulting undirected graph has an edge between user A and user B
whenever A retweeted, quoted, or mentioned B (or vice versa, since we
treat it as undirected for SybilSCAR).

Why interaction edges instead of follow edges
---------------------------------------------

At-scale experiments (Milestone 5 in RESULTS.md) showed that the follow
graph (``following`` / ``followers``) has near-zero homophily lift
(+0.002) within the qualifying user population because sophisticated
bots systematically follow real users to appear legitimate.  Interaction
edges (retweets, quotes, mentions) reflect content affinity: bots
typically retweet and quote *other bots'* content, creating detectable
clusters.

Two-pass algorithm
------------------

A naive approach would build a tweet-to-author map for all 88 M ``post``
edges, requiring ~3.4 GB of RAM.  Instead we use two focused passes:

Pass 1 — interaction sweep (``retweeted``, ``quoted``, ``mentioned``):
  collect all tweet IDs that appear as endpoints in these edges, and
  store raw interaction pairs. Only mentions whose target is a qualifying
  user are kept, so we never materialise the full tweet-to-author map.

Pass 2 — post sweep:
  for tweet IDs collected in pass 1 that were authored by qualifying
  users, record tweet_id → author_id.

Resolution: look up both endpoints of every stored pair in the
tweet-to-author map; emit a user-user edge only when both endpoints are
qualifying users.

Memory budget (full TwiBot-22 scale)
-------------------------------------

- Pass-1 pairs in RAM: ≤ 6.35 M pairs × ~60 B = ~380 MB.
- needed_tweets set: ≤ 8.5 M IDs × ~35 B = ~300 MB.
- tweet_to_author dict (pass 2): ≤ 8.5 M entries × ~60 B = ~510 MB.
- Total peak: ~1.2 GB, well within a 16 GB laptop.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)

# Edge types that represent user-to-tweet authorship.
_POST_RELATION = "post"

# Default set of tweet-level interaction types to resolve.
DEFAULT_INTERACTION_RELATIONS: frozenset[str] = frozenset(
    {"retweeted", "quoted", "mentioned"}
)

_CSV_BUFFER_BYTES = 1 << 22  # 4 MiB


@dataclass(frozen=True)
class InteractionGraphBuildResult:
    """Output bundle from :func:`build_interaction_gt`.

    Attributes:
        graph: Undirected NetworkX graph over the qualifying user set.
            Every qualifying user is present as a node; isolated users
            (no interaction edges to other qualifying users) have degree 0.
        n_qualifying_users: Size of the input user set.
        n_post_edges_scanned: ``post`` rows read in pass 2.
        n_tweet_author_pairs: Entries in the tweet-to-author map after
            pass 2 (only tweets in ``needed_tweets`` and authored by
            qualifying users).
        n_raw_pairs: Total raw interaction pairs collected in pass 1,
            before resolution.
        n_resolved_edges: Unique undirected user-user edges after
            resolution.
        relations: Frozenset of interaction relation types used.
        relation_edge_counts: Per-relation count of resolved user-user
            edges (edges contributed by retweet, quote, mention
            separately — may sum to more than ``n_resolved_edges`` since
            multiple relations can produce the same undirected pair).
    """

    graph: nx.Graph
    n_qualifying_users: int
    n_post_edges_scanned: int
    n_tweet_author_pairs: int
    n_raw_pairs: int
    n_resolved_edges: int
    relations: frozenset[str]
    relation_edge_counts: dict[str, int]


def build_interaction_gt(
    edge_csv: Path,
    qualifying_users: set[str],
    relations: frozenset[str] = DEFAULT_INTERACTION_RELATIONS,
    progress_every: int = 20_000_000,
) -> InteractionGraphBuildResult:
    """Build a user-user interaction graph from TwiBot-22's ``edge.csv``.

    Args:
        edge_csv: Path to the TwiBot-22 ``edge.csv`` (170 M rows).
        qualifying_users: Set of user ids (``"u<digits>"``) that should
            appear as nodes.  Only edges with both endpoints in this set
            are kept.
        relations: Interaction relation types to resolve.  Default:
            ``{"retweeted", "quoted", "mentioned"}``.  Do not include
            ``"replied_to"``; replies are typically bot→human (spam /
            harassment) and are heterophilic.
        progress_every: Row count between progress log lines.

    Returns:
        :class:`InteractionGraphBuildResult`.
    """
    tweet_tweet_relations = relations - {"mentioned"}
    tweet_user_relations = relations & {"mentioned"}

    # ------------------------------------------------------------------
    # Pass 1: collect raw interaction pairs and needed tweet IDs.
    # ------------------------------------------------------------------
    logger.info(
        "interaction G_T pass 1: scanning %s for relations %s ...",
        edge_csv.name,
        sorted(relations),
    )
    # tweet-tweet pairs: (src_tweet, tgt_tweet, relation)
    tt_pairs: list[tuple[str, str, str]] = []
    # tweet-user pairs: (src_tweet, tgt_user, relation) — pre-filtered to qualifying users
    tu_pairs: list[tuple[str, str, str]] = []
    # all tweet IDs we need an author for
    needed_tweets: set[str] = set()

    scanned_p1 = 0
    with edge_csv.open("r", newline="", buffering=_CSV_BUFFER_BYTES) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        src_idx = header.index("source_id")
        rel_idx = header.index("relation")
        tgt_idx = header.index("target_id")

        for row in reader:
            scanned_p1 += 1
            if scanned_p1 % progress_every == 0:
                logger.info(
                    "  pass 1: scanned=%d  tt_pairs=%d  tu_pairs=%d  needed_tweets=%d",
                    scanned_p1,
                    len(tt_pairs),
                    len(tu_pairs),
                    len(needed_tweets),
                )
            if len(row) < 3:
                continue
            rel = row[rel_idx]
            src = row[src_idx]
            tgt = row[tgt_idx]

            if rel in tweet_tweet_relations:
                # Both endpoints are tweet IDs (t... prefix).
                if src.startswith("t") and tgt.startswith("t"):
                    tt_pairs.append((src, tgt, rel))
                    needed_tweets.add(src)
                    needed_tweets.add(tgt)

            elif rel in tweet_user_relations:
                # Source is a tweet, target is a user.
                if src.startswith("t") and tgt in qualifying_users:
                    tu_pairs.append((src, tgt, rel))
                    needed_tweets.add(src)

    logger.info(
        "pass 1 done: scanned=%d  tt_pairs=%d  tu_pairs=%d  needed_tweets=%d",
        scanned_p1,
        len(tt_pairs),
        len(tu_pairs),
        len(needed_tweets),
    )

    # ------------------------------------------------------------------
    # Pass 2: build tweet-to-author map for needed tweets by qualifying users.
    # ------------------------------------------------------------------
    logger.info(
        "interaction G_T pass 2: scanning %s for post edges ...",
        edge_csv.name,
    )
    tweet_to_author: dict[str, str] = {}
    scanned_p2 = 0

    with edge_csv.open("r", newline="", buffering=_CSV_BUFFER_BYTES) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        src_idx = header.index("source_id")
        rel_idx = header.index("relation")
        tgt_idx = header.index("target_id")

        for row in reader:
            scanned_p2 += 1
            if scanned_p2 % progress_every == 0:
                logger.info(
                    "  pass 2: scanned=%d  tweet_author_pairs=%d",
                    scanned_p2,
                    len(tweet_to_author),
                )
            if len(row) < 3:
                continue
            if row[rel_idx] != _POST_RELATION:
                continue
            author = row[src_idx]
            tweet_id = row[tgt_idx]
            if author in qualifying_users and tweet_id in needed_tweets:
                tweet_to_author[tweet_id] = author

    logger.info(
        "pass 2 done: scanned=%d  tweet_author_pairs=%d",
        scanned_p2,
        len(tweet_to_author),
    )

    # ------------------------------------------------------------------
    # Resolution: translate raw pairs to user-user edges.
    # ------------------------------------------------------------------
    graph = nx.Graph()
    graph.add_nodes_from(qualifying_users)

    relation_edge_counts: dict[str, int] = {rel: 0 for rel in relations}

    for src_tweet, tgt_tweet, rel in tt_pairs:
        author_a = tweet_to_author.get(src_tweet)
        author_b = tweet_to_author.get(tgt_tweet)
        if (
            author_a is not None
            and author_b is not None
            and author_a != author_b
        ):
            if not graph.has_edge(author_a, author_b):
                graph.add_edge(author_a, author_b)
            relation_edge_counts[rel] = relation_edge_counts.get(rel, 0) + 1

    for src_tweet, tgt_user, rel in tu_pairs:
        author_a = tweet_to_author.get(src_tweet)
        if author_a is not None and author_a != tgt_user:
            if not graph.has_edge(author_a, tgt_user):
                graph.add_edge(author_a, tgt_user)
            relation_edge_counts[rel] = relation_edge_counts.get(rel, 0) + 1

    n_edges = graph.number_of_edges()
    logger.info(
        "interaction G_T resolved: user-user edges=%d  nodes=%d",
        n_edges,
        graph.number_of_nodes(),
    )

    return InteractionGraphBuildResult(
        graph=graph,
        n_qualifying_users=len(qualifying_users),
        n_post_edges_scanned=scanned_p2,
        n_tweet_author_pairs=len(tweet_to_author),
        n_raw_pairs=len(tt_pairs) + len(tu_pairs),
        n_resolved_edges=n_edges,
        relations=relations,
        relation_edge_counts=relation_edge_counts,
    )
