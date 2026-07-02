"""Compute behavioral features from at-scale ``timestamps.parquet``.

Reads the parquet produced by ``extract_twibot22_at_scale`` and writes
the same ``_raw.csv``, ``_zscored.csv``, and ``_summary.json`` artefacts
as ``compute_features_subset``.

For very large user counts, feature computation is chunked in memory
but the pipeline still pools reference IATs globally (honest-only by
default). Expect tens of minutes for ~200k qualifying users.

Typical usage::

    python -m two_graph_fusion.scripts.compute_features_at_scale \
        --work-dir two_graph_fusion/cache/at_scale \
        --output-prefix two_graph_fusion/cache/at_scale/features \
        --humanlike-mapping sigmoid \
        --d-shape-reference honest
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from two_graph_fusion.datasets.checkpoint_extract import read_timestamps_parquet
from two_graph_fusion.datasets.twibot22 import discover_twibot22, load_labels_and_splits
from two_graph_fusion.features import (
    CircadianConfig,
    PipelineConfig,
    RAW_FEATURE_COLUMNS,
    ZSCORE_FEATURE_COLUMNS,
    compute_behavioral_features,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute features from at-scale timestamps.parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("two_graph_fusion/cache/at_scale"),
    )
    parser.add_argument(
        "--timestamps-parquet",
        type=Path,
        default=None,
        help="Override path to timestamps.parquet (default: <work-dir>/timestamps.parquet).",
    )
    parser.add_argument(
        "--twibot22-root",
        type=Path,
        default=Path("twibot22"),
        help="TwiBot-22 root for label/split join and honest D_shape reference.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--n-min-events", type=int, default=50)
    parser.add_argument("--n-perm", type=int, default=100)
    parser.add_argument(
        "--humanlike-mapping",
        choices=["sigmoid", "gaussian"],
        default="sigmoid",
    )
    parser.add_argument(
        "--d-shape-reference",
        choices=["pooled", "honest"],
        default="honest",
    )
    parser.add_argument("--honest-label", default="human")
    parser.add_argument(
        "--chunk-users",
        type=int,
        default=50_000,
        help="Log progress every N users during dict conversion (metadata only).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger(__name__)
    t0 = time.time()

    parquet_path = args.timestamps_parquet or (args.work_dir / "timestamps.parquet")
    log.info("loading %s ...", parquet_path)
    timestamps = read_timestamps_parquet(parquet_path)
    log.info("loaded %d users with tweets", len(timestamps))

    paths = discover_twibot22(args.twibot22_root)
    labels_splits = load_labels_and_splits(paths)
    labels = labels_splits["label"].astype(str).to_dict()
    splits = labels_splits["split"].astype(str).to_dict()

    reference_user_ids = None
    if args.d_shape_reference == "honest" and labels:
        reference_user_ids = frozenset(
            uid for uid, lab in labels.items() if lab == args.honest_label
        )
        log.info(
            "D_shape reference: honest-only (%d candidate users)",
            len(reference_user_ids),
        )

    config = PipelineConfig(
        n_min_events=args.n_min_events,
        circadian=CircadianConfig(
            mapping=args.humanlike_mapping,
            n_perm=args.n_perm,
        ),
        reference_user_ids=reference_user_ids,
    )

    all_ts = [t for ts_list in timestamps.values() for t in ts_list]
    if not all_ts:
        raise ValueError(f"no timestamps in {parquet_path}")
    window = (float(min(all_ts)), float(max(all_ts)))
    if window[1] <= window[0]:
        raise ValueError(f"degenerate observation window {window}")

    log.info(
        "computing features for %d users (window %.0f .. %.0f) ...",
        len(timestamps),
        window[0],
        window[1],
    )
    result = compute_behavioral_features(
        timestamps, window=window, config=config
    )

    raw = result.raw.copy()
    zscored = result.zscored.copy()
    if labels:
        raw["label"] = pd.Series(labels)
        zscored["label"] = pd.Series(labels)
    if splits:
        raw["split"] = pd.Series(splits)
        zscored["split"] = pd.Series(splits)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_prefix.with_name(args.output_prefix.name + "_raw.csv")
    z_path = args.output_prefix.with_name(args.output_prefix.name + "_zscored.csv")
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")

    raw.to_csv(raw_path)
    zscored.to_csv(z_path)

    qualifying = zscored.dropna(how="any", subset=list(ZSCORE_FEATURE_COLUMNS))
    summary = {
        "n_users_input": len(timestamps),
        "n_users_raw": len(raw),
        "n_users_qualifying": len(qualifying),
        "label_counts_qualifying": (
            qualifying["label"].value_counts().to_dict()
            if "label" in qualifying.columns else {}
        ),
        "split_counts_qualifying": (
            qualifying["split"].value_counts().to_dict()
            if "split" in qualifying.columns else {}
        ),
        "config": {
            "n_min_events": args.n_min_events,
            "humanlike_mapping": args.humanlike_mapping,
            "d_shape_reference": args.d_shape_reference,
        },
        "elapsed_seconds": time.time() - t0,
    }
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    log.info(
        "wrote %s, %s, %s (qualifying=%d, elapsed=%.0fs)",
        raw_path,
        z_path,
        summary_path,
        len(qualifying),
        time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
