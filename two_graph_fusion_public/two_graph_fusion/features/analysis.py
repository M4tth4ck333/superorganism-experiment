"""Univariate analysis of behavioral features stratified by label.

The functions here are intentionally side-effect free so they can be reused
across datasets and from notebooks. ``feature_separation_table`` is the main
entry point: it returns one row per feature with the Mann-Whitney U
statistic, the corresponding univariate AUC, and the two-sample
Kolmogorov-Smirnov statistic.

Conventions
-----------

We treat ``positive_label`` as the *target* class for AUC (sybils / bots in
our case). The univariate AUC is computed both for the *raw* feature and
for ``-feature``; we then report the larger of the two, together with the
sign that achieved it. This is the standard "best-of-direction" univariate
AUC, useful when we do not know a priori which direction of a feature
indicates the positive class.

The implementation uses the well-known identity

    AUC(feature, label) = U / (n_pos * n_neg)

where ``U`` is the Mann-Whitney U statistic on ``feature`` with the positive
class as the first sample. This is exact (no bootstrap) and tolerates ties
via the average-rank convention in :func:`scipy.stats.mannwhitneyu`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class FeatureSeparation:
    """One row of the per-feature separation report.

    Attributes:
        feature: Feature column name.
        n_pos: Number of positive (target) samples with a non-NaN value.
        n_neg: Number of negative samples with a non-NaN value.
        median_pos: Median value of the feature on the positive class.
        median_neg: Median value of the feature on the negative class.
        mw_u: Mann-Whitney U statistic with the positive class as sample 1.
        mw_pvalue: Two-sided Mann-Whitney p-value.
        ks_stat: Two-sample KS statistic.
        ks_pvalue: Two-sided KS p-value.
        auc: Best-of-direction univariate AUC, in ``[0.5, 1.0]``.
        auc_direction: ``"+"`` if higher feature -> positive class wins,
            ``"-"`` if lower feature -> positive class wins.
    """

    feature: str
    n_pos: int
    n_neg: int
    median_pos: float
    median_neg: float
    mw_u: float
    mw_pvalue: float
    ks_stat: float
    ks_pvalue: float
    auc: float
    auc_direction: str


def _univariate_auc_from_mw(u: float, n_pos: int, n_neg: int) -> float:
    """Convert a Mann-Whitney U statistic to a univariate AUC."""
    denom = n_pos * n_neg
    if denom == 0:
        return math.nan
    return u / denom


def feature_separation(
    raw: pd.DataFrame,
    feature: str,
    label_col: str,
    positive_label: str,
) -> FeatureSeparation:
    """Compute MW-U, KS, and univariate AUC for one feature column.

    Args:
        raw: DataFrame with at least ``feature`` and ``label_col`` columns.
        feature: Column name of the numeric feature to evaluate.
        label_col: Column name of the class label.
        positive_label: Value in ``label_col`` to treat as the positive
            class (the target of AUC).

    Returns:
        :class:`FeatureSeparation` with sample counts, medians, test
        statistics and AUC. Rows with NaN feature values are dropped before
        any computation.
    """
    sub = raw[[feature, label_col]].dropna()
    pos = sub.loc[sub[label_col] == positive_label, feature].to_numpy(dtype=np.float64)
    neg = sub.loc[sub[label_col] != positive_label, feature].to_numpy(dtype=np.float64)
    n_pos = pos.size
    n_neg = neg.size

    if n_pos == 0 or n_neg == 0:
        return FeatureSeparation(
            feature=feature,
            n_pos=n_pos,
            n_neg=n_neg,
            median_pos=math.nan,
            median_neg=math.nan,
            mw_u=math.nan,
            mw_pvalue=math.nan,
            ks_stat=math.nan,
            ks_pvalue=math.nan,
            auc=math.nan,
            auc_direction="",
        )

    mw = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    ks = stats.ks_2samp(pos, neg, alternative="two-sided", mode="auto")
    auc_pos = _univariate_auc_from_mw(float(mw.statistic), n_pos, n_neg)
    auc = auc_pos if auc_pos >= 0.5 else (1.0 - auc_pos)
    direction = "+" if auc_pos >= 0.5 else "-"

    return FeatureSeparation(
        feature=feature,
        n_pos=n_pos,
        n_neg=n_neg,
        median_pos=float(np.median(pos)),
        median_neg=float(np.median(neg)),
        mw_u=float(mw.statistic),
        mw_pvalue=float(mw.pvalue),
        ks_stat=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        auc=auc,
        auc_direction=direction,
    )


def feature_separation_table(
    raw: pd.DataFrame,
    features: list[str],
    label_col: str,
    positive_label: str,
) -> pd.DataFrame:
    """Apply :func:`feature_separation` over a list of feature columns."""
    rows = [
        feature_separation(raw, feat, label_col, positive_label).__dict__
        for feat in features
    ]
    return pd.DataFrame(rows)
