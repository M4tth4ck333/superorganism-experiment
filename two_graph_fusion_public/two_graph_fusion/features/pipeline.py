"""End-to-end behavioral feature pipeline.

This module ties the five feature implementations together so that downstream
code can treat the feature extraction as a single function call::

    features = compute_behavioral_features(user_timestamps, window=(t0, t1))

The output is a tidy :class:`pandas.DataFrame` indexed by ``user_id`` with one
column per raw feature, plus their z-scored versions which are used as the
node feature vector ``f(u)`` in section 4.6 of the pivot plan.

Conventions
-----------

- All timestamps are Unix seconds (UTC).
- The user-id type is unconstrained as long as it hashes; strings are
  recommended (TwiBot-22 uses ``"u<digits>"``).
- Per-user IATs are computed after sorting timestamps and dropping duplicate
  timestamps. Duplicates would yield ``IAT = 0`` and bias every downstream
  feature (especially ``B_G`` and ``M``).
- Users with fewer than ``n_min_events`` events return ``NaN`` for all
  features. The default of ``50`` events matches section 7.1.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from two_graph_fusion.features.activity_duration import activity_duration
from two_graph_fusion.features.burstiness import burstiness
from two_graph_fusion.features.circadian import (
    CircadianConfig,
    circadian_zscore,
    humanlike,
)
from two_graph_fusion.features.memory import memory_coefficient
from two_graph_fusion.features.shape_divergence import (
    ShapeBins,
    fit_reference_bins,
    shape_divergence,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BehavioralFeatures:
    """Result bundle returned by :func:`compute_behavioral_features`.

    Attributes:
        raw: DataFrame indexed by user_id with columns
            ``["B_G", "M", "C_24", "C_24_humanlike", "D_shape", "L",
            "n_events"]``.
        zscored: DataFrame indexed by user_id with columns
            ``["B_G_z", "M_z", "C_24_humanlike_z", "D_shape_z", "L_z"]``,
            i.e. the node feature vector ``f(u) in R^5``. NaNs are
            preserved (z-scoring ignores NaN users).
        bins: The pooled reference bins used to compute ``D_shape``.
        window: The ``(start, end)`` Unix-second window used for ``L``.
    """

    raw: pd.DataFrame
    zscored: pd.DataFrame
    bins: ShapeBins
    window: tuple[float, float]


# The order of feature columns is part of the public API.
RAW_FEATURE_COLUMNS: tuple[str, ...] = (
    "B_G",
    "M",
    "C_24",
    "C_24_humanlike",
    "D_shape",
    "L",
)
ZSCORE_FEATURE_COLUMNS: tuple[str, ...] = (
    "B_G_z",
    "M_z",
    "C_24_humanlike_z",
    "D_shape_z",
    "L_z",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for :func:`compute_behavioral_features`.

    Attributes:
        n_min_events: Minimum number of events required to compute features
            for a user; users below the threshold return all-NaN rows.
        circadian: Configuration for the circadian z-score and human-likeness
            mapping. See :class:`CircadianConfig`.
        epsilon: Floor used when constructing reference and per-user
            histograms for ``D_shape``. Must be small relative to ``1/n_bins``.
        reference_iat_cap: Optional cap on the number of reference IATs
            pooled to fit the shape bins. Useful when the dataset is large
            but FD bin estimation needs only a few hundred thousand IATs.
            ``None`` means use all available IATs.
        reference_seed: Optional seed for sub-sampling the reference pool.
        reference_user_ids: Optional set of user ids whose IATs form the
            reference for ``D_shape``. ``None`` means pool IATs from every
            qualifying user (the "dataset_pooled" mode of the existing
            GitHub pipeline). For labeled datasets, passing the labeled
            honest cohort here is the section-11.7 recommendation; the
            milestone-1 RED-2 finding shows the pooled default is
            uninformative on TwiBot-22 heavy-tweet users.
    """

    n_min_events: int = 50
    circadian: CircadianConfig = field(default_factory=CircadianConfig)
    epsilon: float = 1e-9
    reference_iat_cap: int | None = 200_000
    reference_seed: int | None = 0
    reference_user_ids: frozenset[object] | None = None


def _iats_from_timestamps(timestamps: Iterable[float]) -> np.ndarray:
    """Sort, deduplicate, and difference a single user's timestamps."""
    arr = np.asarray(list(timestamps), dtype=np.float64)
    arr.sort()
    arr = np.unique(arr)
    if arr.size < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(arr)


def _pool_reference_iats(
    user_iats: Mapping[object, np.ndarray],
    cap: int | None,
    rng: np.random.Generator,
    reference_user_ids: frozenset[object] | None = None,
) -> np.ndarray:
    """Concatenate per-user IATs, optionally restricted and sub-sampled.

    Args:
        user_iats: Per-user IAT arrays.
        cap: Optional sub-sample cap.
        rng: RNG for sub-sampling.
        reference_user_ids: If provided, only pool IATs from users whose id
            is in this set. If ``None``, all users contribute.
    """
    if not user_iats:
        raise ValueError("Cannot fit reference bins on an empty corpus")
    if reference_user_ids is not None:
        filtered = {
            uid: arr for uid, arr in user_iats.items() if uid in reference_user_ids
        }
        if not filtered:
            raise ValueError(
                "reference_user_ids did not match any user with qualifying IATs"
            )
        source = filtered
    else:
        source = user_iats

    total = sum(arr.size for arr in source.values())
    pooled = np.empty(total, dtype=np.float64)
    pos = 0
    for arr in source.values():
        pooled[pos : pos + arr.size] = arr
        pos += arr.size
    if cap is not None and pooled.size > cap:
        idx = rng.choice(pooled.size, size=cap, replace=False)
        pooled = pooled[idx]
    return pooled


def _zscore(series: pd.Series) -> pd.Series:
    """Z-score a series, NaN-safe, with zero-variance fallback to NaN."""
    mu = series.mean(skipna=True)
    sigma = series.std(skipna=True, ddof=0)
    if not math.isfinite(sigma) or sigma == 0.0:
        return pd.Series(np.full(len(series), np.nan), index=series.index)
    return (series - mu) / sigma


def compute_behavioral_features(
    user_timestamps: Mapping[object, npt.ArrayLike],
    window: tuple[float, float],
    config: PipelineConfig = PipelineConfig(),
) -> BehavioralFeatures:
    """Compute the five behavioral features for every user.

    Args:
        user_timestamps: Mapping ``user_id -> iterable of Unix-second
            timestamps``. Timestamps are sorted and de-duplicated internally;
            the caller does not need to pre-sort.
        window: ``(T_start, T_end)`` Unix-second observation window used by
            the activity-duration feature ``L``.
        config: Pipeline configuration. Defaults match the values in the
            pivot plan (section 7.1 and section 4.3).

    Returns:
        A :class:`BehavioralFeatures` bundle with raw and z-scored feature
        DataFrames. Users with fewer than ``config.n_min_events`` events have
        all-NaN rows but are still present in the index so callers can keep
        a stable user list.

    Raises:
        ValueError: If ``window`` is degenerate, or if no user meets the
            ``n_min_events`` threshold (FD reference cannot be fit).
    """
    if window[1] <= window[0]:
        raise ValueError(f"Window must be increasing, got {window}")

    qualifying_iats: dict[object, np.ndarray] = {}
    qualifying_timestamps: dict[object, np.ndarray] = {}
    n_events_map: dict[object, int] = {}
    user_ids: list[object] = []

    for uid, ts in user_timestamps.items():
        user_ids.append(uid)
        ts_arr = np.asarray(list(ts), dtype=np.float64)
        ts_arr.sort()
        ts_arr = np.unique(ts_arr)
        n_events_map[uid] = int(ts_arr.size)
        if ts_arr.size >= config.n_min_events:
            qualifying_timestamps[uid] = ts_arr
            iats = np.diff(ts_arr)
            qualifying_iats[uid] = iats

    if not qualifying_iats:
        raise ValueError(
            f"No user has >= {config.n_min_events} events; "
            f"reduce n_min_events or extract more data."
        )

    logger.info(
        "qualifying users: %d / %d (n_min_events=%d)",
        len(qualifying_iats),
        len(user_ids),
        config.n_min_events,
    )

    rng = np.random.default_rng(config.reference_seed)
    pooled = _pool_reference_iats(
        qualifying_iats,
        config.reference_iat_cap,
        rng,
        reference_user_ids=config.reference_user_ids,
    )
    bins = fit_reference_bins(pooled, epsilon=config.epsilon)
    logger.info(
        "fit reference shape bins: n_bins=%d, pooled_iats=%d, reference_mode=%s",
        bins.edges.size - 1,
        pooled.size,
        "subset" if config.reference_user_ids is not None else "pooled",
    )

    rows: list[dict[str, float]] = []
    for uid in user_ids:
        row: dict[str, float] = {
            "user_id": uid,
            "n_events": n_events_map[uid],
        }
        if uid in qualifying_iats:
            iats = qualifying_iats[uid]
            ts_arr = qualifying_timestamps[uid]
            row["B_G"] = burstiness(iats)
            row["M"] = memory_coefficient(iats)
            z = circadian_zscore(ts_arr, config.circadian)
            row["C_24"] = z
            row["C_24_humanlike"] = humanlike(z, config.circadian)
            row["D_shape"] = shape_divergence(iats, bins, epsilon=config.epsilon)
            row["L"] = activity_duration(ts_arr, window[0], window[1])
        else:
            for col in RAW_FEATURE_COLUMNS:
                row[col] = math.nan
        rows.append(row)

    raw = pd.DataFrame(rows).set_index("user_id")
    raw = raw[list(RAW_FEATURE_COLUMNS) + ["n_events"]]

    zscored = pd.DataFrame(
        {
            "B_G_z": _zscore(raw["B_G"]),
            "M_z": _zscore(raw["M"]),
            "C_24_humanlike_z": _zscore(raw["C_24_humanlike"]),
            "D_shape_z": _zscore(raw["D_shape"]),
            "L_z": _zscore(raw["L"]),
        },
        index=raw.index,
    )
    zscored = zscored[list(ZSCORE_FEATURE_COLUMNS)]

    return BehavioralFeatures(
        raw=raw, zscored=zscored, bins=bins, window=(float(window[0]), float(window[1]))
    )
