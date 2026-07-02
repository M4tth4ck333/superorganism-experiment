"""One-time builder for the compact user-user follow-edge cache.

Streams ``twibot22/edge.csv`` (6.6 GB) once and writes a much smaller
CSV containing only the user-user follow edges. Downstream scripts
(e.g. snowball / LCC samplers) can then assemble the trust graph in a
fraction of a second from this cache instead of re-scanning the full
file.

Typical usage
-------------

    python -m two_graph_fusion.scripts.cache_follow_edges \
        --edge-csv twibot22/edge.csv \
        --output two_graph_fusion/cache/twibot22_follow_edges.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from two_graph_fusion.graphs import DEFAULT_FOLLOW_RELATIONS
from two_graph_fusion.graphs.trust_graph import cache_all_user_follow_edges


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache the user-user follow edges from edge.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--edge-csv",
        type=Path,
        default=Path("twibot22/edge.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("two_graph_fusion/cache/twibot22_follow_edges.csv"),
    )
    parser.add_argument(
        "--relations",
        nargs="+",
        default=sorted(DEFAULT_FOLLOW_RELATIONS),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    scanned, written = cache_all_user_follow_edges(
        edge_csv=args.edge_csv,
        output_csv=args.output,
        relations=args.relations,
    )
    print(f"scanned={scanned} written={written} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
