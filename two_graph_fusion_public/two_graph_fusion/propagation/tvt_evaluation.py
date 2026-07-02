"""Train / validation / test evaluation harness (both propagation engines).

Milestones 4–8 used either the TwiBot-22 official split (single train/test)
or stratified K-fold CV (Cresci). For the M10+ comparison against SOTA we
need a single, leakage-free protocol that also tunes hyperparameters
honestly:

1. Split qualifying users into **train / val / test** (stratified; the
   caller supplies either an explicit split column or a seed for a fresh
   stratified split).
2. Fit the calibrated LR behavioral prior on **train** only.
3. Propagation seeds = **train** labels pinned to ``0.02 / 0.98``. Val and
   test nodes are unlabeled to the propagation (initialised from the LR
   prior or a neutral 0.5).
4. For every (engine, graph configuration, prior mode) cell, grid-search
   the homophily-strength hyperparameters on **val** (ROC-AUC), then report
   the chosen configuration's metrics on **test**.
5. Accuracy / F1 use a decision threshold selected on **val** (max-F1).

Engines:

- ``sybilscar`` — linearised SybilSCAR (:mod:`.sybilscar`).
- ``sybilhp``   — directed adaptive-homophily LBP (:mod:`.sybilhp`).

Graph configurations: ``g_t`` (trust only), ``g_b`` (behavioral only),
``fusion`` (both). Prior modes: ``neutral`` (pure graph baseline) and
``lr`` (the behavioral LR prior — our method).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Hashable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from two_graph_fusion.propagation.prior import (
    DEFAULT_FEATURES,
    fit_predict_lr_prior,
)
from two_graph_fusion.propagation.sybilhp import (
    SybilHPParams,
    p_honest_to_dict as sybilhp_p_honest_to_dict,
    run_sybilhp,
)
from two_graph_fusion.propagation.sybilscar import (
    Relation,
    p_honest_to_dict,
    relation_from_graph,
    run_propagation,
)

logger = logging.getLogger(__name__)

# Hyperparameter grids (coarse; propagation is cheap at Cresci scale).
SYBILSCAR_W_GRID: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)
# The canonical linear rule diverges at the logistic weights above on graphs with
# hub nodes (it has no sigmoid soft-clamp), so it needs its own much smaller grid;
# validation AUC rejects any w that saturates.
# Capped at 0.5 (standard SybilSCAR homophily-strength upper bound). On hub-bearing
# graphs the strong-LR-prior cells sit on a flat val-AUC plateau and drift to this
# cap; the scientifically load-bearing neutral-prior cells select interior weights.
SYBILSCAR_W_GRID_LINEAR: tuple[float, ...] = (
    0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5,
)
SYBILHP_WBI_GRID: tuple[float, ...] = (0.7, 0.9, 0.99)
SYBILHP_WBEH_GRID: tuple[float, ...] = (0.5, 0.7, 0.9)


@dataclass(frozen=True)
class ConfigResult:
    """Metrics for one (engine, graph, prior) configuration."""

    engine: str           # "lr" | "sybilscar" | "sybilhp"
    graph: str            # "-" | "g_t" | "g_b" | "fusion"
    prior: str            # "-" | "neutral" | "lr"
    best_hparams: dict[str, float]
    val_auc: float
    test_auc: float
    test_pr_auc: float
    test_acc: float
    test_f1: float
    threshold: float

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "graph": self.graph,
            "prior": self.prior,
            "best_hparams": self.best_hparams,
            "val_auc": self.val_auc,
            "test_auc": self.test_auc,
            "test_pr_auc": self.test_pr_auc,
            "test_acc": self.test_acc,
            "test_f1": self.test_f1,
            "threshold": self.threshold,
        }


@dataclass
class Splits:
    """Train/val/test user-id partitions over the qualifying set."""

    train: list[Hashable]
    val: list[Hashable]
    test: list[Hashable]
    node_order: list[Hashable]
    labels: dict[Hashable, str]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def make_stratified_splits(
    qualifying: pd.DataFrame,
    label_col: str,
    bot_label: str,
    seed: int,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> Splits:
    """Stratified train/val/test split over the qualifying users."""
    y = (qualifying[label_col] == bot_label).astype(int).to_numpy()
    idx = np.asarray(qualifying.index)
    train_idx, hold_idx, y_train, y_hold = train_test_split(
        idx, y, test_size=val_frac + test_frac, stratify=y, random_state=seed,
    )
    rel_test = test_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        hold_idx, test_size=rel_test, stratify=y_hold, random_state=seed,
    )
    return Splits(
        train=list(train_idx),
        val=list(val_idx),
        test=list(test_idx),
        node_order=list(qualifying.index),
        labels=qualifying[label_col].to_dict(),
    )


def splits_from_column(
    qualifying: pd.DataFrame,
    label_col: str,
    split_col: str,
    train_val_test: tuple[str, str, str] = ("train", "val", "test"),
) -> Splits:
    """Build :class:`Splits` from an explicit split column (e.g. TwiBot-20)."""
    tr, va, te = train_val_test
    return Splits(
        train=list(qualifying.index[qualifying[split_col] == tr]),
        val=list(qualifying.index[qualifying[split_col] == va]),
        test=list(qualifying.index[qualifying[split_col] == te]),
        node_order=list(qualifying.index),
        labels=qualifying[label_col].to_dict(),
    )


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

def _honest_priors(
    splits: Splits,
    honest_label: str,
    pi_b_honest: dict[Hashable, float] | None,
    seed_lo: float = 0.02,
    seed_hi: float = 0.98,
) -> dict[Hashable, float]:
    """Per-node P(honest): train seeds pinned; others from LR prior or omitted.

    When ``pi_b_honest`` is ``None`` (neutral mode), only train seeds are
    returned and non-seed nodes fall back to the propagation default (0.5).
    """
    priors: dict[Hashable, float] = {
        u: (seed_hi if splits.labels[u] == honest_label else seed_lo)
        for u in splits.train
    }
    if pi_b_honest is not None:
        for u in splits.val + splits.test:
            if u in pi_b_honest:
                priors[u] = float(pi_b_honest[u])
    return priors


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _scores(p_honest: dict[Hashable, float], users: Sequence[Hashable]) -> np.ndarray:
    """Bot-likelihood scores ``1 - P(honest)`` for ``users``."""
    return np.asarray([1.0 - p_honest.get(u, 0.5) for u in users])


def _labels01(splits: Splits, users: Sequence[Hashable], bot_label: str) -> np.ndarray:
    return np.asarray([1 if splits.labels[u] == bot_label else 0 for u in users])


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def _best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold on the score that maximises F1 over the candidate set."""
    candidates = np.unique(s)
    if candidates.size > 200:
        candidates = np.quantile(s, np.linspace(0.0, 1.0, 200))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        pred = (s >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _test_metrics(
    p_honest: dict[Hashable, float],
    splits: Splits,
    bot_label: str,
    threshold: float,
) -> tuple[float, float, float, float]:
    """Return ``(test_auc, test_pr_auc, test_acc, test_f1)``."""
    y = _labels01(splits, splits.test, bot_label)
    s = _scores(p_honest, splits.test)
    auc = _auc(y, s)
    pr = float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    pred = (s >= threshold).astype(int)
    acc = float((pred == y).mean())
    f1 = float(f1_score(y, pred, zero_division=0))
    return auc, pr, acc, f1


# ---------------------------------------------------------------------------
# Engine runners (return p_honest dict over all nodes)
# ---------------------------------------------------------------------------

def _run_sybilscar(
    relations: Sequence[Relation],
    splits: Splits,
    priors: dict[Hashable, float],
    max_iters: int,
    tol: float,
    linearized: bool = False,
) -> dict[Hashable, float]:
    result = run_propagation(
        relations=list(relations),
        node_order=splits.node_order,
        priors=priors,
        default_prior=0.5,
        max_iters=max_iters,
        tol=tol,
        linearized=linearized,
    )
    return p_honest_to_dict(result)


def _run_sybilhp(
    splits: Splits,
    honest_priors: dict[Hashable, float],
    trust_edges: Sequence[tuple[Hashable, Hashable]],
    behavioral_edges: Sequence[tuple[Hashable, Hashable]],
    params: SybilHPParams,
    w_behavioral: float,
    max_iters: int,
    tol: float,
) -> dict[Hashable, float]:
    sybil_prior = {u: 1.0 - p for u, p in honest_priors.items()}
    result = run_sybilhp(
        node_order=splits.node_order,
        sybil_prior=sybil_prior,
        trust_edges=trust_edges,
        behavioral_edges=behavioral_edges,
        params=params,
        w_behavioral=w_behavioral,
        default_prior=0.5,
        max_iters=max_iters,
        tol=tol,
    )
    return sybilhp_p_honest_to_dict(result)


# ---------------------------------------------------------------------------
# Tuning + evaluation per cell
# ---------------------------------------------------------------------------

def _select_and_eval(
    candidates: list[tuple[dict[str, float], dict[Hashable, float]]],
    splits: Splits,
    bot_label: str,
) -> tuple[dict[str, float], float, dict[Hashable, float]]:
    """Pick the candidate with best val AUC; return (hparams, val_auc, p_honest)."""
    y_val = _labels01(splits, splits.val, bot_label)
    best = None
    for hparams, p_honest in candidates:
        val_auc = _auc(y_val, _scores(p_honest, splits.val))
        if best is None or val_auc > best[1]:
            best = (hparams, val_auc, p_honest)
    assert best is not None
    return best


def _finalise(
    engine: str,
    graph: str,
    prior: str,
    selection: tuple[dict[str, float], float, dict[Hashable, float]],
    splits: Splits,
    bot_label: str,
) -> ConfigResult:
    hparams, val_auc, p_honest = selection
    y_val = _labels01(splits, splits.val, bot_label)
    thr = _best_f1_threshold(y_val, _scores(p_honest, splits.val))
    test_auc, test_pr, test_acc, test_f1 = _test_metrics(
        p_honest, splits, bot_label, thr
    )
    return ConfigResult(
        engine=engine, graph=graph, prior=prior,
        best_hparams=hparams, val_auc=val_auc,
        test_auc=test_auc, test_pr_auc=test_pr,
        test_acc=test_acc, test_f1=test_f1, threshold=thr,
    )


def run_tvt_evaluation(
    qualifying: pd.DataFrame,
    g_t: nx.Graph,
    g_b: nx.Graph,
    trust_edges_directed: Sequence[tuple[Hashable, Hashable]],
    splits: Splits,
    label_col: str = "label",
    honest_label: str = "human",
    bot_label: str = "bot",
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
    sybilhp_params: SybilHPParams = SybilHPParams(),
    max_iters: int = 50,
    tol: float = 1e-5,
    linearized: bool = False,
    return_extras: bool = False,
) -> list[ConfigResult] | tuple[list[ConfigResult], dict]:
    """Run the full engine × graph × prior matrix under train/val/test.

    Args:
        qualifying: Feature table (indexed by user id) for qualifying users
            with ``label_col`` and ``feature_columns``.
        g_t / g_b: Undirected trust / behavioral graphs (SybilSCAR engine).
        trust_edges_directed: Directed ``(tail, head)`` trust edges
            (SybilHP engine). The behavioral layer for SybilHP is taken
            from ``g_b`` edges.
        splits: Train/val/test partition (see :func:`make_stratified_splits`).
        sybilhp_params: Trust-layer homophily parameters for SybilHP.
        return_extras: When ``True``, also return a dict with the fitted LR
            coefficients (full and drop-L) and per-node bot scores on the
            test split for every cell (for bootstrap / DeLong CIs).

    Returns:
        One :class:`ConfigResult` per configuration, plus the LR baseline
        (and its drop-L ablation). With ``return_extras=True``, a
        ``(results, extras)`` tuple.
    """
    results: list[ConfigResult] = []
    extras: dict = {"test_scores": {}}

    def _record_scores(cell: str, p_honest: dict[Hashable, float]) -> None:
        extras["test_scores"][cell] = {
            str(u): float(1.0 - p_honest.get(u, 0.5)) for u in splits.test
        }

    train_df = qualifying.loc[splits.train]
    val_df = qualifying.loc[splits.val]
    test_df = qualifying.loc[splits.test]

    # --- LR behavioral prior (trained on train, predicts val + test) ---
    lr_valtest = fit_predict_lr_prior(
        train_df, pd.concat([val_df, test_df]),
        label_col=label_col, honest_label=honest_label, bot_label=bot_label,
        drop_l=False, feature_columns=tuple(feature_columns),
    )
    pi_b = lr_valtest.pi_b_test  # P(honest) on val + test
    # LR-only baseline metrics.
    sel_lr = ({}, _auc(_labels01(splits, splits.val, bot_label),
                       _scores(pi_b, splits.val)), pi_b)
    results.append(_finalise("lr", "-", "-", sel_lr, splits, bot_label))
    _record_scores("lr|-|-", pi_b)
    extras["lr_prior"] = {
        "feature_columns": lr_valtest.feature_columns,
        "coefficients": lr_valtest.coefficients,
        "intercept": lr_valtest.intercept,
        "train_auc": lr_valtest.train_auc,
    }

    # --- L-ablation of the LR prior (drop the activity-span feature) ---
    lr_valtest_nol = fit_predict_lr_prior(
        train_df, pd.concat([val_df, test_df]),
        label_col=label_col, honest_label=honest_label, bot_label=bot_label,
        drop_l=True, feature_columns=tuple(feature_columns),
    )
    pi_b_nol = lr_valtest_nol.pi_b_test
    sel_lr_nol = ({}, _auc(_labels01(splits, splits.val, bot_label),
                           _scores(pi_b_nol, splits.val)), pi_b_nol)
    results.append(_finalise("lr", "-", "drop_l", sel_lr_nol, splits, bot_label))
    _record_scores("lr|-|drop_l", pi_b_nol)
    extras["lr_prior_drop_l"] = {
        "feature_columns": lr_valtest_nol.feature_columns,
        "coefficients": lr_valtest_nol.coefficients,
        "intercept": lr_valtest_nol.intercept,
        "train_auc": lr_valtest_nol.train_auc,
    }

    behavioral_edges = list(g_b.edges())

    # Precompute neutral / lr honest-prior dicts.
    priors_neutral = _honest_priors(splits, honest_label, pi_b_honest=None)
    priors_lr = _honest_priors(splits, honest_label, pi_b_honest=pi_b)

    # The canonical linear rule needs a much smaller weight grid (no sigmoid
    # soft-clamp) than the logistic variant.
    w_grid = SYBILSCAR_W_GRID_LINEAR if linearized else SYBILSCAR_W_GRID

    for prior_mode, base_priors in [("neutral", priors_neutral), ("lr", priors_lr)]:

        # ===================== SybilSCAR =====================
        # g_t only
        cands = []
        for w in w_grid:
            rel = relation_from_graph("trust", g_t, splits.node_order, weight=w)
            p = _run_sybilscar([rel], splits, base_priors, max_iters, tol, linearized)
            cands.append(({"w_t": w}, p))
        sel = _select_and_eval(cands, splits, bot_label)
        results.append(_finalise("sybilscar", "g_t", prior_mode, sel, splits, bot_label))
        _record_scores(f"sybilscar|g_t|{prior_mode}", sel[2])
        # g_b only
        cands = []
        for w in w_grid:
            rel = relation_from_graph("behavioral", g_b, splits.node_order, weight=w)
            p = _run_sybilscar([rel], splits, base_priors, max_iters, tol, linearized)
            cands.append(({"w_b": w}, p))
        sel = _select_and_eval(cands, splits, bot_label)
        results.append(_finalise("sybilscar", "g_b", prior_mode, sel, splits, bot_label))
        _record_scores(f"sybilscar|g_b|{prior_mode}", sel[2])
        # fusion
        cands = []
        for w_t in w_grid:
            rel_t = relation_from_graph("trust", g_t, splits.node_order, weight=w_t)
            for w_b in w_grid:
                rel_b = relation_from_graph("behavioral", g_b, splits.node_order, weight=w_b)
                p = _run_sybilscar([rel_t, rel_b], splits, base_priors, max_iters, tol, linearized)
                cands.append(({"w_t": w_t, "w_b": w_b}, p))
        sel = _select_and_eval(cands, splits, bot_label)
        results.append(_finalise("sybilscar", "fusion", prior_mode, sel, splits, bot_label))
        _record_scores(f"sybilscar|fusion|{prior_mode}", sel[2])

        # ===================== SybilHP =====================
        # g_t only (directed trust layer)
        cands = []
        for w_bi in SYBILHP_WBI_GRID:
            params = SybilHPParams(
                w_bi=w_bi, wb_tail=sybilhp_params.wb_tail, ws_tail=sybilhp_params.ws_tail,
                wb_head=sybilhp_params.wb_head, ws_head=sybilhp_params.ws_head,
            )
            p = _run_sybilhp(splits, base_priors, trust_edges_directed, [],
                             params, 0.9, max_iters, tol)
            cands.append(({"w_bi": w_bi}, p))
        sel = _select_and_eval(cands, splits, bot_label)
        results.append(_finalise("sybilhp", "g_t", prior_mode, sel, splits, bot_label))
        _record_scores(f"sybilhp|g_t|{prior_mode}", sel[2])
        # g_b only (behavioral bidirectional layer)
        cands = []
        for w_beh in SYBILHP_WBEH_GRID:
            p = _run_sybilhp(splits, base_priors, [], behavioral_edges,
                             sybilhp_params, w_beh, max_iters, tol)
            cands.append(({"w_beh": w_beh}, p))
        sel = _select_and_eval(cands, splits, bot_label)
        results.append(_finalise("sybilhp", "g_b", prior_mode, sel, splits, bot_label))
        _record_scores(f"sybilhp|g_b|{prior_mode}", sel[2])
        # fusion (directed trust + behavioral)
        cands = []
        for w_bi in (0.9, 0.99):
            params = SybilHPParams(
                w_bi=w_bi, wb_tail=sybilhp_params.wb_tail, ws_tail=sybilhp_params.ws_tail,
                wb_head=sybilhp_params.wb_head, ws_head=sybilhp_params.ws_head,
            )
            for w_beh in (0.5, 0.9):
                p = _run_sybilhp(splits, base_priors, trust_edges_directed,
                                 behavioral_edges, params, w_beh, max_iters, tol)
                cands.append(({"w_bi": w_bi, "w_beh": w_beh}, p))
        sel = _select_and_eval(cands, splits, bot_label)
        results.append(_finalise("sybilhp", "fusion", prior_mode, sel, splits, bot_label))
        _record_scores(f"sybilhp|fusion|{prior_mode}", sel[2])

    if return_extras:
        return results, extras
    return results
