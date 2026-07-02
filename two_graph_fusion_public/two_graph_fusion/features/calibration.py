"""Label-aware calibration helpers for the behavioral features.

These helpers address two milestone-1 red flags called out in
``RESULTS.md``:

1. ``D_shape`` is uninformative when its reference IAT pmf is pooled across
   both classes; section 11.7 of the pivot plan calls for an honest-only
   reference. :func:`refit_d_shape_with_reference` rebuilds ``D_shape``
   given a chosen reference subset.
2. The circadian humanlike mapping uses fixed defaults (``target_z=2.0``,
   ``sigma_z=1.0``) that bear no relation to the actual honest-user
   distribution; section 4.3 of the plan asks for a calibration on
   labeled honest users. :func:`calibrate_circadian_humanlike` derives
   ``target_z`` and ``sigma_z`` from a labeled honest cohort and applies
   the new mapping to every user.

Both helpers operate on the :class:`BehavioralFeatures` raw frame
produced by :func:`compute_behavioral_features` plus the per-user
timestamps (for the ``D_shape`` refit) - they do not need the original
``ShapeBins`` object because they fit a fresh one from the chosen
reference.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from two_graph_fusion.features.circadian import CircadianConfig, humanlike
from two_graph_fusion.features.shape_divergence import (
    ShapeBins,
    fit_reference_bins,
    shape_divergence,
)


def refit_d_shape_with_reference(
    user_timestamps: Mapping[str, np.ndarray],
    reference_user_ids: Iterable[str],
    epsilon: float = 1e-9,
    cap: int | None = 200_000,
    seed: int = 0,
) -> tuple[pd.Series, ShapeBins]:
    """Recompute ``D_shape`` using a chosen subset of users as reference.

    Args:
        user_timestamps: Mapping ``user_id -> sorted unique Unix-second
            timestamps`` (the same input given to
            :func:`compute_behavioral_features`).
        reference_user_ids: Iterable of user ids whose IATs to pool into
            the reference. Users outside this set will still receive a
            ``D_shape`` value, but their IATs will not influence the
            reference.
        epsilon: Floor on histogram bins for KL.
        cap: Optional sub-sample cap on pooled reference IATs.
        seed: RNG seed for the sub-sample.

    Returns:
        ``(d_shape, bins)`` where ``d_shape`` is a :class:`pandas.Series`
        indexed by ``user_id`` containing the recomputed ``D_shape`` values
        (``NaN`` for users with fewer than 5 IATs) and ``bins`` is the
        refitted :class:`ShapeBins`.

    Raises:
        ValueError: If no reference user has >= 10 IATs.
    """
    ref_ids = set(reference_user_ids)
    pooled: list[float] = []
    for uid, ts in user_timestamps.items():
        if uid not in ref_ids:
            continue
        ts_arr = np.asarray(ts, dtype=np.float64)
        if ts_arr.size < 2:
            continue
        pooled.extend(np.diff(ts_arr).tolist())
    if len(pooled) < 10:
        raise ValueError(
            f"Need >= 10 reference IATs to refit D_shape; got {len(pooled)}"
        )

    pooled_arr = np.asarray(pooled, dtype=np.float64)
    if cap is not None and pooled_arr.size > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pooled_arr.size, size=cap, replace=False)
        pooled_arr = pooled_arr[idx]

    bins = fit_reference_bins(pooled_arr, epsilon=epsilon)

    out: dict[str, float] = {}
    for uid, ts in user_timestamps.items():
        ts_arr = np.asarray(ts, dtype=np.float64)
        if ts_arr.size < 2:
            out[uid] = math.nan
            continue
        iats = np.diff(ts_arr)
        out[uid] = shape_divergence(iats, bins, epsilon=epsilon)

    return pd.Series(out, name="D_shape_honest_ref"), bins


def calibrate_circadian_humanlike(
    raw: pd.DataFrame,
    label_col: str,
    honest_label: str,
    z_col: str = "C_24",
    min_honest: int = 30,
    mapping: str = "sigmoid",
) -> tuple[pd.Series, CircadianConfig]:
    """Recompute the human-likeness score using a label-derived mapping.

    The new ``target_z`` is derived from the labeled honest cohort:

    - For ``mapping="sigmoid"`` (default), ``target_z`` is the 25th
      percentile of honest ``C_24``. This places the sigmoid's 0.5
      crossover *below* the bulk of honest users so that bots, which the
      milestone-1 data shows concentrate on the lower z side, end up in
      the bottom half of the mapping. ``sigma_z`` is the IQR of honest
      ``C_24`` divided by 4 (so that the bulk of honest users score
      noticeably above 0.5).
    - For ``mapping="gaussian"`` (the section-4.3 form), ``target_z`` is
      the honest median and ``sigma_z`` is half the IQR. This mirrors the
      milestone-1 calibration that empirically failed (RED-3) and is kept
      only for sensitivity analyses.

    Args:
        raw: ``BehavioralFeatures.raw`` DataFrame, indexed by user_id and
            containing at least the columns ``[z_col, label_col]``.
        label_col: Column in ``raw`` carrying class labels.
        honest_label: Value of ``label_col`` that identifies honest users.
        z_col: Column with the raw ``C_24`` z-score.
        min_honest: Minimum number of honest users with non-NaN ``z_col``
            required to perform the calibration.
        mapping: ``"sigmoid"`` or ``"gaussian"``.

    Returns:
        ``(humanlike, config)`` where ``humanlike`` is a Series indexed by
        user_id with the new ``C_24_humanlike`` values, and ``config`` is
        the :class:`CircadianConfig` that produced them.

    Raises:
        ValueError: If fewer than ``min_honest`` honest users have a
            non-NaN ``z_col``, or if their IQR collapses to zero.
    """
    honest_z = raw.loc[raw[label_col] == honest_label, z_col].dropna()
    if len(honest_z) < min_honest:
        raise ValueError(
            f"Need >= {min_honest} honest users with non-NaN {z_col!r}; "
            f"got {len(honest_z)}"
        )
    q25, q50, q75 = honest_z.quantile([0.25, 0.5, 0.75])
    iqr = float(q75 - q25)
    if iqr <= 0.0 or not math.isfinite(iqr):
        raise ValueError(
            f"Cannot calibrate C_24 humanlike: honest IQR is {iqr}"
        )

    if mapping == "sigmoid":
        target_z = float(q25)
        sigma_z = max(iqr / 4.0, 1e-6)
    elif mapping == "gaussian":
        target_z = float(q50)
        sigma_z = max(iqr / 2.0, 1e-6)
    else:
        raise ValueError(f"unknown mapping {mapping!r}")

    config = CircadianConfig(target_z=target_z, sigma_z=sigma_z, mapping=mapping)

    out = raw[z_col].apply(lambda z: humanlike(z, config) if pd.notna(z) else math.nan)
    out.name = "C_24_humanlike_calibrated"
    return out, config
