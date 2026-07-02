"""Compute the five behavioral features for a TwiBot-22 subset.

This script consumes the JSON produced by
``extract_twibot22_subset`` and writes:

- ``<output_prefix>_raw.csv``        Raw feature values + n_events + label.
- ``<output_prefix>_zscored.csv``    Z-scored feature vector f(u) in R^5.
- ``<output_prefix>_summary.json``   Per-label feature statistics.

Typical usage
-------------

    python -m two_graph_fusion.scripts.compute_features_subset \
        --input two_graph_fusion/cache/subset_timestamps.json \
        --output-prefix two_graph_fusion/cache/subset_features
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from two_graph_fusion.features import (
    CircadianConfig,
    PipelineConfig,
    RAW_FEATURE_COLUMNS,
    compute_behavioral_features,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute behavioral features for an extracted TwiBot-22 subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("two_graph_fusion/cache/subset_timestamps.json"),
        help="JSON produced by extract_twibot22_subset.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("two_graph_fusion/cache/subset_features"),
        help="Prefix for the output CSV / JSON files.",
    )
    parser.add_argument(
        "--n-min-events",
        type=int,
        default=50,
        help="Minimum events required to compute features for a user.",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=100,
        help="Number of permutations for the circadian shuffle null.",
    )
    parser.add_argument(
        "--target-z",
        type=float,
        default=2.0,
        help=(
            "Centre of the C_24 -> humanlike mapping. For sigmoid this is "
            "the 0.5 crossover; for gaussian this is the peak."
        ),
    )
    parser.add_argument(
        "--sigma-z",
        type=float,
        default=1.0,
        help=(
            "Slope (sigmoid) or width (gaussian) of the humanlike mapping."
        ),
    )
    parser.add_argument(
        "--humanlike-mapping",
        choices=["sigmoid", "gaussian"],
        default="sigmoid",
        help=(
            "Functional form for C_24 -> humanlike. The default sigmoid is "
            "monotone; the gaussian form is the original section-4.3 spec."
        ),
    )
    parser.add_argument(
        "--circadian-seed",
        type=int,
        default=0,
        help="Seed for the circadian shuffle null.",
    )
    parser.add_argument(
        "--reference-iat-cap",
        type=int,
        default=200_000,
        help="Optional cap on pooled IATs for fitting the reference bins.",
    )
    parser.add_argument(
        "--d-shape-reference",
        choices=["pooled", "honest"],
        default="honest",
        help=(
            "Source of IATs for the D_shape reference. 'honest' pools only "
            "users with the honest label from the timestamps JSON, matching "
            "section 11.7 of the pivot plan. 'pooled' falls back to all "
            "qualifying users (the milestone-1 RED-2 default)."
        ),
    )
    parser.add_argument(
        "--honest-label",
        default="human",
        help="Label value treated as honest when --d-shape-reference=honest.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _per_label_summary(raw: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """Return a nested dict ``label -> feature -> {mean, std, median, n}``."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for label, group in raw.groupby("label"):
        out[str(label)] = {}
        for feat in RAW_FEATURE_COLUMNS:
            values = group[feat].dropna()
            out[str(label)][feat] = {
                "n": int(values.size),
                "mean": float(values.mean()) if values.size else float("nan"),
                "std": float(values.std(ddof=0)) if values.size else float("nan"),
                "median": float(values.median()) if values.size else float("nan"),
                "p10": float(values.quantile(0.10)) if values.size else float("nan"),
                "p90": float(values.quantile(0.90)) if values.size else float("nan"),
            }
    return out


def _feature_correlations(raw: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Spearman correlation between raw features, NaN-tolerant."""
    sub = raw[list(RAW_FEATURE_COLUMNS)].dropna()
    if len(sub) < 3:
        return {}
    corr = sub.corr(method="spearman")
    return {col: corr[col].to_dict() for col in corr.columns}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    with args.input.open() as fh:
        payload = json.load(fh)
    metadata = payload["metadata"]
    labels = payload["labels"]
    splits = payload["splits"]
    timestamps = {uid: np.asarray(ts, dtype=np.float64) for uid, ts in payload["timestamps"].items()}
    log.info(
        "loaded subset: users_total=%d users_with_tweets=%d",
        metadata["users_total"],
        metadata["users_with_tweets"],
    )

    window = (
        float(metadata["window_start_unix"]),
        float(metadata["window_end_unix"]),
    )
    if window[1] <= window[0]:
        raise ValueError(
            f"degenerate window {window}; was the extraction step empty?"
        )

    reference_user_ids: frozenset[str] | None
    if args.d_shape_reference == "honest":
        honest_ids = {
            uid for uid, lab in labels.items() if lab == args.honest_label
        }
        if not honest_ids:
            raise ValueError(
                f"--d-shape-reference=honest but no users have label "
                f"{args.honest_label!r} in the timestamps JSON"
            )
        reference_user_ids = frozenset(honest_ids)
        log.info(
            "D_shape reference: honest-only (%d candidate users)",
            len(reference_user_ids),
        )
    else:
        reference_user_ids = None
        log.info("D_shape reference: pooled across all qualifying users")

    cfg = PipelineConfig(
        n_min_events=args.n_min_events,
        circadian=CircadianConfig(
            n_perm=args.n_perm,
            target_z=args.target_z,
            sigma_z=args.sigma_z,
            mapping=args.humanlike_mapping,
            seed=args.circadian_seed,
        ),
        reference_iat_cap=args.reference_iat_cap,
        reference_user_ids=reference_user_ids,
    )
    features = compute_behavioral_features(timestamps, window=window, config=cfg)

    raw = features.raw.copy()
    raw["label"] = raw.index.map(labels.get)
    raw["split"] = raw.index.map(splits.get)
    zscored = features.zscored.copy()
    zscored["label"] = zscored.index.map(labels.get)
    zscored["split"] = zscored.index.map(splits.get)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_prefix.with_name(args.output_prefix.name + "_raw.csv")
    z_path = args.output_prefix.with_name(args.output_prefix.name + "_zscored.csv")
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")

    raw.to_csv(raw_path)
    zscored.to_csv(z_path)
    summary = {
        "metadata": {
            **metadata,
            "n_min_events": args.n_min_events,
            "n_perm": args.n_perm,
            "target_z": args.target_z,
            "sigma_z": args.sigma_z,
            "humanlike_mapping": args.humanlike_mapping,
            "d_shape_reference": args.d_shape_reference,
            "honest_label": args.honest_label,
            "reference_iat_cap": args.reference_iat_cap,
            "window_used": list(features.window),
            "shape_bins_count": int(features.bins.edges.size - 1),
        },
        "n_users_qualifying": int(
            raw[list(RAW_FEATURE_COLUMNS)].dropna(how="all").shape[0]
        ),
        "per_label": _per_label_summary(raw),
        "spearman_corr": _feature_correlations(raw),
    }
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    log.info("wrote %s, %s, %s", raw_path, z_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
