"""Supervised behavioral prior ``pi_b(u)`` from the five features.

This module implements the LR-on-``f(u)`` prior described in
section 7.1 of the pivot plan:

1. Class-weighted L2-regularised logistic regression on the
   z-scored feature vector ``f(u) in R^5``.
2. Sigmoid (logistic) calibration via :class:`sklearn.calibration.CalibratedClassifierCV`
   so that the output ``pi_b(u)`` is a calibrated probability of being
   honest.
3. An "ablate-L" variant that drops the ``L_z`` column to address
   RED-1 (the activity-span confound).

The output ``pi_b(u) in [0, 1]`` is the probability of being **honest**,
matching the SybilSCAR convention in :mod:`two_graph_fusion.propagation.sybilscar`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Hashable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)

DEFAULT_FEATURES: tuple[str, ...] = (
    "B_G_z",
    "M_z",
    "C_24_humanlike_z",
    "D_shape_z",
    "L_z",
)


@dataclass(frozen=True)
class LRPriorResult:
    """Output bundle from :func:`fit_predict_lr_prior`.

    Attributes:
        feature_columns: Columns of ``f(u)`` actually used (after the
            optional L drop).
        coefficients: Mean LR coefficients across the calibrated
            ensemble, one per feature.
        intercept: Mean intercept across the ensemble.
        pi_b_train: Predicted honest probability on the training rows.
        pi_b_test: Predicted honest probability on the held-out rows.
        train_auc: ROC-AUC against ``label = bot`` on the training rows.
        train_pr_auc: PR-AUC against ``label = bot`` on the training rows.
        test_auc: ROC-AUC against ``label = bot`` on the held-out rows.
        test_pr_auc: PR-AUC against ``label = bot`` on the held-out rows.
        n_train: Training-set size.
        n_test: Held-out-set size.
    """

    feature_columns: list[str]
    coefficients: list[float]
    intercept: float
    pi_b_train: dict[Hashable, float]
    pi_b_test: dict[Hashable, float]
    train_auc: float
    train_pr_auc: float
    test_auc: float
    test_pr_auc: float
    n_train: int
    n_test: int


def _select_features(
    zscored: pd.DataFrame,
    drop_l: bool,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURES,
) -> list[str]:
    """Return the column list after optionally dropping the L feature."""
    cols = [c for c in feature_columns if c in zscored.columns]
    if drop_l:
        cols = [c for c in cols if c not in {"L_z", "L"}]
    if not cols:
        raise ValueError("no usable feature columns remain")
    return cols


def fit_predict_lr_prior(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "label",
    honest_label: str = "human",
    bot_label: str = "bot",
    drop_l: bool = False,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURES,
    C: float = 1.0,
    n_calibration_folds: int = 3,
    random_state: int = 0,
) -> LRPriorResult:
    """Fit a calibrated LR on the training rows and predict on the test rows.

    Args:
        train_df: DataFrame indexed by user id; must contain
            ``feature_columns`` and ``label_col``.
        test_df: DataFrame indexed by user id; must contain
            ``feature_columns`` and ``label_col``.
        label_col: Name of the label column in both frames.
        honest_label: String value for the honest class.
        bot_label: String value for the bot class.
        drop_l: When ``True``, drop ``L_z`` / ``L`` from the feature set
            for the RED-1 ablation.
        feature_columns: Candidate feature columns (intersected with the
            actual frames).
        C: Inverse L2 regularisation strength.
        n_calibration_folds: Inner-CV folds used by
            ``CalibratedClassifierCV``.
        random_state: Seed for the base LR's stochastic solver path.

    Returns:
        :class:`LRPriorResult`.
    """
    cols = _select_features(train_df, drop_l=drop_l, feature_columns=feature_columns)

    train_X = train_df[cols].dropna(how="any")
    train_y_raw = train_df.loc[train_X.index, label_col]
    train_mask = train_y_raw.isin({honest_label, bot_label})
    train_X = train_X[train_mask]
    train_y_raw = train_y_raw[train_mask]
    y_train = (train_y_raw == bot_label).astype(int).to_numpy()

    test_X = test_df[cols].dropna(how="any")
    test_y_raw = test_df.loc[test_X.index, label_col]
    test_mask = test_y_raw.isin({honest_label, bot_label})
    test_X = test_X[test_mask]
    test_y_raw = test_y_raw[test_mask]
    y_test = (test_y_raw == bot_label).astype(int).to_numpy()

    if len(train_X) < 10 or len(test_X) < 1:
        raise ValueError(
            f"insufficient rows for LR prior: train={len(train_X)} test={len(test_X)}"
        )

    base = LogisticRegression(
        C=C,
        penalty="l2",
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
        max_iter=1000,
    )
    n_folds = min(n_calibration_folds, int(min(np.bincount(y_train))))
    if n_folds >= 2:
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=n_folds)
    else:
        # Too few minority samples for calibration; fall back to raw LR.
        clf = base
    clf.fit(train_X.to_numpy(), y_train)

    p_bot_train = clf.predict_proba(train_X.to_numpy())[:, 1]
    p_bot_test = clf.predict_proba(test_X.to_numpy())[:, 1]

    train_auc = float(roc_auc_score(y_train, p_bot_train)) if len(np.unique(y_train)) > 1 else float("nan")
    train_pr_auc = float(average_precision_score(y_train, p_bot_train)) if len(np.unique(y_train)) > 1 else float("nan")
    test_auc = float(roc_auc_score(y_test, p_bot_test)) if len(np.unique(y_test)) > 1 else float("nan")
    test_pr_auc = float(average_precision_score(y_test, p_bot_test)) if len(np.unique(y_test)) > 1 else float("nan")

    if isinstance(clf, CalibratedClassifierCV):
        coefs_stack = np.stack(
            [c.estimator.coef_.ravel() for c in clf.calibrated_classifiers_], axis=0
        )
        intercepts = np.array(
            [c.estimator.intercept_.ravel()[0] for c in clf.calibrated_classifiers_]
        )
        coef_mean = coefs_stack.mean(axis=0)
        intercept_mean = float(intercepts.mean())
    else:
        coef_mean = clf.coef_.ravel()
        intercept_mean = float(clf.intercept_.ravel()[0])

    pi_b_train = {
        idx: float(1.0 - p) for idx, p in zip(train_X.index, p_bot_train)
    }
    pi_b_test = {
        idx: float(1.0 - p) for idx, p in zip(test_X.index, p_bot_test)
    }

    return LRPriorResult(
        feature_columns=list(cols),
        coefficients=[float(x) for x in coef_mean],
        intercept=intercept_mean,
        pi_b_train=pi_b_train,
        pi_b_test=pi_b_test,
        train_auc=train_auc,
        train_pr_auc=train_pr_auc,
        test_auc=test_auc,
        test_pr_auc=test_pr_auc,
        n_train=int(len(train_X)),
        n_test=int(len(test_X)),
    )
