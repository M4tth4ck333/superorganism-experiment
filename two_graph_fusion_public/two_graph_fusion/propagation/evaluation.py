"""Stratified K-fold evaluation harness for the milestone-3 pipeline.

The TwiBot-22 official train/val/test split on our snowball subset is
heavily skewed by class (RED-9): train is 89% human, val/test are
60-80% bot. To get a defensible AUC estimate at this subset size we use
**stratified K-fold cross-validation** over the qualifying users. Each
fold serves once as the held-out evaluation set; the other ``K-1``
folds act as the "labeled" population (LR training rows + propagation
seeds).

Per fold the harness produces:

- ``pi_b``: the LR-prior probability on the held-out fold (after
  training the LR on the seed folds).
- ``p_T``: SybilSCAR posterior on ``G_T`` alone with the seed nodes
  pinned to their labels.
- ``p_B``: SybilSCAR posterior on ``G_B`` alone.
- ``p_combined_A``: a simple alpha-blended ``alpha * (1 - p_T) + (1 - alpha) * (1 - p_B)``
  score (variant A baseline). Reported across an ``alpha`` grid.
- ``p_combined_B``: multi-relational SybilSCAR over ``G_T + G_B``
  with ``q(u) = logit(pi_b(u))`` for the held-out fold and pinned
  priors for the seed folds (variant B, the headline).

For each posterior we record AUC and PR-AUC against the bot label.

The aggregated table is returned both as a pandas DataFrame and a JSON
record.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from two_graph_fusion.propagation.prior import (
    DEFAULT_FEATURES,
    LRPriorResult,
    fit_predict_lr_prior,
)
from two_graph_fusion.propagation.sybilscar import (
    DEFAULT_W,
    Relation,
    evaluation_priors,
    p_honest_to_dict,
    relation_from_graph,
    run_propagation,
)

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """One fold's worth of evaluation metrics."""

    fold: int
    n_train: int
    n_test: int
    lr_with_l: dict[str, float] = field(default_factory=dict)
    lr_without_l: dict[str, float] = field(default_factory=dict)
    lr_coefficients: dict[str, dict[str, float]] = field(default_factory=dict)
    p_t_auc: float = float("nan")
    p_t_pr_auc: float = float("nan")
    p_b_auc: float = float("nan")
    p_b_pr_auc: float = float("nan")
    variant_a: dict[float, dict[str, float]] = field(default_factory=dict)
    variant_b_auc: float = float("nan")
    variant_b_pr_auc: float = float("nan")
    n_iters_t: int = 0
    n_iters_b: int = 0
    n_iters_mr: int = 0


def _score_p_honest(
    p_honest: dict[Hashable, float],
    labels: dict[Hashable, str],
    test_users: list[Hashable],
    bot_label: str,
) -> tuple[float, float]:
    """Return ``(AUC, PR-AUC)`` against the bot label on ``test_users``.

    AUC is computed on ``1 - p_honest`` (so larger means more bot-like).
    """
    y = np.asarray([1 if labels[u] == bot_label else 0 for u in test_users])
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    s = np.asarray([1.0 - p_honest.get(u, 0.5) for u in test_users])
    auc = float(roc_auc_score(y, s))
    pr = float(average_precision_score(y, s))
    return auc, pr


def _run_single_relation(
    relation: Relation,
    node_order: list[Hashable],
    seed_priors: dict[Hashable, float],
    pi_b_honest: dict[Hashable, float],
    default_prior: float,
    max_iters: int,
    tol: float,
) -> tuple[dict[Hashable, float], int]:
    """Run SybilSCAR on a single relation and return ``(p_honest, n_iters)``."""
    priors = dict(seed_priors)
    for u, pi in pi_b_honest.items():
        if u not in priors:
            priors[u] = pi
    result = run_propagation(
        relations=[relation],
        node_order=node_order,
        priors=priors,
        default_prior=default_prior,
        max_iters=max_iters,
        tol=tol,
    )
    return p_honest_to_dict(result), result.n_iters


def _run_multi_relation(
    relations: Sequence[Relation],
    node_order: list[Hashable],
    seed_priors: dict[Hashable, float],
    pi_b_honest: dict[Hashable, float],
    default_prior: float,
    max_iters: int,
    tol: float,
) -> tuple[dict[Hashable, float], int]:
    """Run multi-relational SybilSCAR and return ``(p_honest, n_iters)``."""
    priors = dict(seed_priors)
    for u, pi in pi_b_honest.items():
        if u not in priors:
            priors[u] = pi
    result = run_propagation(
        relations=list(relations),
        node_order=node_order,
        priors=priors,
        default_prior=default_prior,
        max_iters=max_iters,
        tol=tol,
    )
    return p_honest_to_dict(result), result.n_iters


def _alpha_grid_scores(
    p_t: dict[Hashable, float],
    pi_b_test: dict[Hashable, float],
    labels: dict[Hashable, str],
    test_users: list[Hashable],
    bot_label: str,
    alphas: Sequence[float],
) -> dict[float, dict[str, float]]:
    """Score variant A across an ``alpha`` grid.

    ``score_A(u) = alpha * (1 - p_T(u)) + (1 - alpha) * (1 - pi_b(u))``
    where ``pi_b`` is the LR-honest probability on the eval set.

    For consistency with variant B we keep the "honest probability"
    convention; AUC is taken against ``label == bot``.
    """
    out: dict[float, dict[str, float]] = {}
    y = np.asarray([1 if labels[u] == bot_label else 0 for u in test_users])
    if len(np.unique(y)) < 2:
        return out
    pt_score = np.asarray([1.0 - p_t.get(u, 0.5) for u in test_users])
    pb_score = np.asarray([1.0 - pi_b_test.get(u, 0.5) for u in test_users])
    for alpha in alphas:
        s = alpha * pt_score + (1.0 - alpha) * pb_score
        out[float(alpha)] = {
            "auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
        }
    return out


def run_kfold_evaluation(
    zscored: pd.DataFrame,
    g_t: nx.Graph,
    g_b: nx.Graph,
    label_col: str = "label",
    honest_label: str = "human",
    bot_label: str = "bot",
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
    n_splits: int = 5,
    seed: int = 0,
    w_t: float = DEFAULT_W,
    w_b: float = DEFAULT_W,
    max_iters: int = 50,
    tol: float = 1e-5,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Run the full E1/E2/E3 + variant A pipeline under stratified K-fold CV.

    Args:
        zscored: DataFrame of qualifying users (indexed by user id) with
            ``feature_columns`` + ``label_col``.
        g_t: Trust graph; nodes must be a superset of ``zscored.index``.
        g_b: Behavioral graph; same node constraint.
        label_col / honest_label / bot_label: Label conventions.
        feature_columns: LR feature columns.
        n_splits: Number of stratified folds.
        seed: RNG seed for ``StratifiedKFold``.
        w_t / w_b: Per-relation homophily strengths for SybilSCAR.
        max_iters / tol: Propagation convergence params.
        alphas: Variant-A blend weights to evaluate.

    Returns:
        ``(summary_df, fold_results)``:

        - ``summary_df``: per-fold metrics for every variant
          (one row per fold).
        - ``fold_results``: rich :class:`FoldResult` objects with the
          full coefficient ablations.
    """
    qualifying = zscored.dropna(how="any", subset=list(feature_columns)).copy()
    qualifying = qualifying[qualifying[label_col].isin({honest_label, bot_label})]
    logger.info(
        "K-fold harness: %d qualifying users (%s)",
        len(qualifying),
        qualifying[label_col].value_counts().to_dict(),
    )

    node_order = list(qualifying.index)
    labels = qualifying[label_col].to_dict()

    relation_t = relation_from_graph("trust", g_t, node_order, weight=w_t)
    relation_b = relation_from_graph("behavioral", g_b, node_order, weight=w_b)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_strat = (qualifying[label_col] == bot_label).astype(int).to_numpy()

    fold_records: list[FoldResult] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(qualifying, y_strat)):
        train_df = qualifying.iloc[train_idx]
        test_df = qualifying.iloc[test_idx]

        rec = FoldResult(
            fold=fold_idx, n_train=len(train_df), n_test=len(test_df)
        )

        lr_full: LRPriorResult = fit_predict_lr_prior(
            train_df, test_df, label_col=label_col,
            honest_label=honest_label, bot_label=bot_label,
            drop_l=False, feature_columns=tuple(feature_columns),
        )
        lr_no_l: LRPriorResult = fit_predict_lr_prior(
            train_df, test_df, label_col=label_col,
            honest_label=honest_label, bot_label=bot_label,
            drop_l=True, feature_columns=tuple(feature_columns),
        )
        rec.lr_with_l = {
            "train_auc": lr_full.train_auc,
            "test_auc": lr_full.test_auc,
            "train_pr_auc": lr_full.train_pr_auc,
            "test_pr_auc": lr_full.test_pr_auc,
            "n_train": float(lr_full.n_train),
            "n_test": float(lr_full.n_test),
        }
        rec.lr_without_l = {
            "train_auc": lr_no_l.train_auc,
            "test_auc": lr_no_l.test_auc,
            "train_pr_auc": lr_no_l.train_pr_auc,
            "test_pr_auc": lr_no_l.test_pr_auc,
            "n_train": float(lr_no_l.n_train),
            "n_test": float(lr_no_l.n_test),
        }
        rec.lr_coefficients = {
            "with_l": {
                "intercept": lr_full.intercept,
                **dict(zip(lr_full.feature_columns, lr_full.coefficients)),
            },
            "without_l": {
                "intercept": lr_no_l.intercept,
                **dict(zip(lr_no_l.feature_columns, lr_no_l.coefficients)),
            },
        }

        # Seed labels visible to propagation = the training fold's labels.
        seed_priors = {
            u: (0.98 if labels[u] == honest_label else 0.02)
            for u in train_df.index
        }
        # Eval users: their LR-prior (calibrated honest probability).
        pi_b_test_honest = lr_full.pi_b_test

        # E1: SybilSCAR on G_T alone.
        p_t, n_iter_t = _run_single_relation(
            relation_t, node_order, seed_priors, pi_b_test_honest,
            default_prior=0.5, max_iters=max_iters, tol=tol,
        )
        rec.p_t_auc, rec.p_t_pr_auc = _score_p_honest(
            p_t, labels, list(test_df.index), bot_label=bot_label
        )
        rec.n_iters_t = n_iter_t

        # SybilSCAR on G_B alone (companion baseline).
        p_b_prop, n_iter_b = _run_single_relation(
            relation_b, node_order, seed_priors, pi_b_test_honest,
            default_prior=0.5, max_iters=max_iters, tol=tol,
        )
        rec.p_b_auc, rec.p_b_pr_auc = _score_p_honest(
            p_b_prop, labels, list(test_df.index), bot_label=bot_label
        )
        rec.n_iters_b = n_iter_b

        # Variant A: score combination across an alpha grid.
        rec.variant_a = _alpha_grid_scores(
            p_t, pi_b_test_honest, labels,
            list(test_df.index), bot_label=bot_label, alphas=alphas,
        )

        # Variant B: multi-relational SybilSCAR.
        p_mr, n_iter_mr = _run_multi_relation(
            [relation_t, relation_b], node_order, seed_priors, pi_b_test_honest,
            default_prior=0.5, max_iters=max_iters, tol=tol,
        )
        rec.variant_b_auc, rec.variant_b_pr_auc = _score_p_honest(
            p_mr, labels, list(test_df.index), bot_label=bot_label
        )
        rec.n_iters_mr = n_iter_mr

        logger.info(
            "fold %d/%d: LR_with_L AUC=%.3f LR_no_L AUC=%.3f p_T AUC=%.3f "
            "p_B AUC=%.3f variantB AUC=%.3f",
            fold_idx + 1, n_splits,
            rec.lr_with_l["test_auc"], rec.lr_without_l["test_auc"],
            rec.p_t_auc, rec.p_b_auc, rec.variant_b_auc,
        )
        fold_records.append(rec)

    summary_df = _summary_dataframe(fold_records, alphas)
    return summary_df, fold_records


def _summary_dataframe(
    folds: list[FoldResult],
    alphas: Sequence[float],
) -> pd.DataFrame:
    """Flatten the per-fold records into a tidy table."""
    rows = []
    for f in folds:
        row = {
            "fold": f.fold,
            "n_train": f.n_train,
            "n_test": f.n_test,
            "lr_with_l_auc": f.lr_with_l["test_auc"],
            "lr_with_l_pr_auc": f.lr_with_l["test_pr_auc"],
            "lr_no_l_auc": f.lr_without_l["test_auc"],
            "lr_no_l_pr_auc": f.lr_without_l["test_pr_auc"],
            "p_t_auc": f.p_t_auc,
            "p_t_pr_auc": f.p_t_pr_auc,
            "p_b_auc": f.p_b_auc,
            "p_b_pr_auc": f.p_b_pr_auc,
            "variant_b_auc": f.variant_b_auc,
            "variant_b_pr_auc": f.variant_b_pr_auc,
            "n_iters_t": f.n_iters_t,
            "n_iters_b": f.n_iters_b,
            "n_iters_mr": f.n_iters_mr,
        }
        for a in alphas:
            ent = f.variant_a.get(float(a), {})
            row[f"variant_a_alpha={a}_auc"] = ent.get("auc", float("nan"))
            row[f"variant_a_alpha={a}_pr_auc"] = ent.get("pr_auc", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_with_stderr(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a per-fold table into a ``(mean, stderr)`` table.

    Returns a DataFrame whose index is the metric name and whose
    columns are ``mean``, ``stderr``, ``min``, ``max``.
    """
    metrics = [
        c for c in summary_df.columns
        if c.endswith("_auc") or c.endswith("_pr_auc")
    ]
    out = []
    for m in metrics:
        vals = summary_df[m].dropna().to_numpy()
        if vals.size == 0:
            continue
        out.append(
            {
                "metric": m,
                "mean": float(vals.mean()),
                "stderr": float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan"),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "n_folds": int(vals.size),
            }
        )
    return pd.DataFrame(out).set_index("metric")


def dump_results_json(
    summary_df: pd.DataFrame,
    fold_records: list[FoldResult],
    output_path: Path,
    extra: dict | None = None,
) -> None:
    """Serialise the milestone-3 results to a JSON report."""
    aggregate = summarise_with_stderr(summary_df).reset_index().to_dict(orient="records")
    payload = {
        "aggregate": aggregate,
        "per_fold": summary_df.to_dict(orient="records"),
        "lr_coefficients_per_fold": [
            {
                "fold": f.fold,
                **f.lr_coefficients,
            }
            for f in fold_records
        ],
    }
    if extra is not None:
        payload["meta"] = extra
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=float)


@dataclass(frozen=True)
class OfficialSplitResult:
    """Metrics from a single train/test evaluation (TwiBot-22 official splits).

    This is the recommended protocol at full dataset scale where K-fold CV
    over hundreds of thousands of nodes is too expensive.
    """

    n_train: int
    n_test: int
    n_graph_nodes: int
    lr_with_l: dict[str, float]
    lr_without_l: dict[str, float]
    lr_coefficients: dict[str, dict[str, float]]
    p_t_auc: float
    p_t_pr_auc: float
    p_b_auc: float
    p_b_pr_auc: float
    variant_b_auc: float
    variant_b_pr_auc: float
    variant_a: dict[float, dict[str, float]]
    n_iters_t: int
    n_iters_b: int
    n_iters_mr: int
    train_split: str
    test_split: str


def run_official_split_evaluation(
    zscored: pd.DataFrame,
    g_t: nx.Graph,
    g_b: nx.Graph,
    label_col: str = "label",
    split_col: str = "split",
    honest_label: str = "human",
    bot_label: str = "bot",
    train_split: str = "train",
    test_split: str = "test",
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
    w_t: float = DEFAULT_W,
    w_b: float = DEFAULT_W,
    max_iters: int = 50,
    tol: float = 1e-5,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> OfficialSplitResult:
    """Evaluate on the official TwiBot-22 train/test partition.

    Protocol:

    1. Restrict to users present in both graphs with non-NaN features.
    2. Train the LR prior on ``train_split`` rows; predict on ``test_split``.
    3. Run SybilSCAR on the union graph with **train** users pinned to
       seed priors (0.02 / 0.98) and **test** users initialised from the
       held-out LR prior ``pi_b(u)``.
    4. Score posteriors on ``test_split`` users only.

    Args:
        zscored: Feature table indexed by user id with ``label_col`` and
            ``split_col``.
        g_t / g_b: Trust and behavioral graphs (may contain extra nodes;
            intersected with qualifying users).
        train_split / test_split: Values of ``split_col`` (TwiBot-22 uses
            ``"train"`` and ``"test"``; validation is ``"valid"``).

    Returns:
        :class:`OfficialSplitResult`.
    """
    qualifying = zscored.dropna(how="any", subset=list(feature_columns)).copy()
    qualifying = qualifying[qualifying[label_col].isin({honest_label, bot_label})]
    qualifying = qualifying[qualifying[split_col].isin({train_split, test_split})]

    graph_nodes = (
        set(g_t.nodes()) & set(g_b.nodes()) & set(qualifying.index)
    )
    qualifying = qualifying.loc[sorted(graph_nodes)]
    if qualifying.empty:
        raise ValueError("no qualifying users in graph intersection")

    train_df = qualifying[qualifying[split_col] == train_split]
    test_df = qualifying[qualifying[split_col] == test_split]
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"empty train ({len(train_df)}) or test ({len(test_df)}) after filtering"
        )

    logger.info(
        "official split: train=%d test=%d graph_nodes=%d",
        len(train_df),
        len(test_df),
        len(qualifying),
    )

    lr_full = fit_predict_lr_prior(
        train_df, test_df,
        label_col=label_col,
        honest_label=honest_label,
        bot_label=bot_label,
        drop_l=False,
        feature_columns=tuple(feature_columns),
    )
    lr_no_l = fit_predict_lr_prior(
        train_df, test_df,
        label_col=label_col,
        honest_label=honest_label,
        bot_label=bot_label,
        drop_l=True,
        feature_columns=tuple(feature_columns),
    )

    labels = qualifying[label_col].to_dict()
    node_order = list(qualifying.index)
    seed_priors = {
        u: (0.98 if labels[u] == honest_label else 0.02)
        for u in train_df.index
    }
    pi_b_test = lr_full.pi_b_test

    relation_t = relation_from_graph("trust", g_t, node_order, weight=w_t)
    relation_b = relation_from_graph("behavioral", g_b, node_order, weight=w_b)

    p_t, n_iter_t = _run_single_relation(
        relation_t, node_order, seed_priors, pi_b_test,
        default_prior=0.5, max_iters=max_iters, tol=tol,
    )
    p_t_auc, p_t_pr_auc = _score_p_honest(
        p_t, labels, list(test_df.index), bot_label=bot_label
    )

    p_b_prop, n_iter_b = _run_single_relation(
        relation_b, node_order, seed_priors, pi_b_test,
        default_prior=0.5, max_iters=max_iters, tol=tol,
    )
    p_b_auc, p_b_pr_auc = _score_p_honest(
        p_b_prop, labels, list(test_df.index), bot_label=bot_label
    )

    variant_a = _alpha_grid_scores(
        p_t, pi_b_test, labels,
        list(test_df.index), bot_label=bot_label, alphas=alphas,
    )

    p_mr, n_iter_mr = _run_multi_relation(
        [relation_t, relation_b], node_order, seed_priors, pi_b_test,
        default_prior=0.5, max_iters=max_iters, tol=tol,
    )
    variant_b_auc, variant_b_pr_auc = _score_p_honest(
        p_mr, labels, list(test_df.index), bot_label=bot_label
    )

    logger.info(
        "official split results: LR=%.3f p_T=%.3f p_B=%.3f variantB=%.3f",
        lr_full.test_auc, p_t_auc, p_b_auc, variant_b_auc,
    )

    return OfficialSplitResult(
        n_train=len(train_df),
        n_test=len(test_df),
        n_graph_nodes=len(qualifying),
        lr_with_l={
            "train_auc": lr_full.train_auc,
            "test_auc": lr_full.test_auc,
            "train_pr_auc": lr_full.train_pr_auc,
            "test_pr_auc": lr_full.test_pr_auc,
        },
        lr_without_l={
            "train_auc": lr_no_l.train_auc,
            "test_auc": lr_no_l.test_auc,
            "train_pr_auc": lr_no_l.train_pr_auc,
            "test_pr_auc": lr_no_l.test_pr_auc,
        },
        lr_coefficients={
            "with_l": {
                "intercept": lr_full.intercept,
                **dict(zip(lr_full.feature_columns, lr_full.coefficients)),
            },
            "without_l": {
                "intercept": lr_no_l.intercept,
                **dict(zip(lr_no_l.feature_columns, lr_no_l.coefficients)),
            },
        },
        p_t_auc=p_t_auc,
        p_t_pr_auc=p_t_pr_auc,
        p_b_auc=p_b_auc,
        p_b_pr_auc=p_b_pr_auc,
        variant_b_auc=variant_b_auc,
        variant_b_pr_auc=variant_b_pr_auc,
        variant_a=variant_a,
        n_iters_t=n_iter_t,
        n_iters_b=n_iter_b,
        n_iters_mr=n_iter_mr,
        train_split=train_split,
        test_split=test_split,
    )


def official_result_to_dict(result: OfficialSplitResult) -> dict:
    """JSON-serialisable view of :class:`OfficialSplitResult`."""
    return {
        "n_train": result.n_train,
        "n_test": result.n_test,
        "n_graph_nodes": result.n_graph_nodes,
        "train_split": result.train_split,
        "test_split": result.test_split,
        "lr_with_l": result.lr_with_l,
        "lr_without_l": result.lr_without_l,
        "lr_coefficients": result.lr_coefficients,
        "p_t_auc": result.p_t_auc,
        "p_t_pr_auc": result.p_t_pr_auc,
        "p_b_auc": result.p_b_auc,
        "p_b_pr_auc": result.p_b_pr_auc,
        "variant_b_auc": result.variant_b_auc,
        "variant_b_pr_auc": result.variant_b_pr_auc,
        "variant_a": {str(k): v for k, v in result.variant_a.items()},
        "n_iters_t": result.n_iters_t,
        "n_iters_b": result.n_iters_b,
        "n_iters_mr": result.n_iters_mr,
    }
