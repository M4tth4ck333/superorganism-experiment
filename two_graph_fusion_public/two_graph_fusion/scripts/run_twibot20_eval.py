"""M12b — TwiBot-20 G_T-only propagation eval (official split).

TwiBot-20 has no per-tweet timestamps, so the behavioral ``G_B`` cannot be
built (M12). This run therefore evaluates the **directed follow graph G_T
only**, with a **profile-feature LR prior** (the BotRGCN/SEGCN statistical
signal) in place of the behavioral prior, under the official train/val/test
split. It closes the 5.4 gap: confirm what the +0.025 sub-gate lift predicts —
propagation cannot beat the feature prior — and put an AUC number on it.

Cells: LR-only baseline; {SybilSCAR, SybilHP} × G_T × {neutral, lr}.
Hparams tuned on val (ROC-AUC); threshold val-selected (max-F1); test-reported.
SybilHP is included because the labelled TwiBot-20 graph is small and sparse
(11.8k nodes, 16k edges, 98.6% unidirectional) — the genuinely-directed regime,
without the dense-graph divergence seen on Twitter-270k (M14).

Usage::

    .venv/bin/python3 -m two_graph_fusion.scripts.run_twibot20_eval \\
        --root Twibot-20-2 --output-dir two_graph_fusion/cache/twibot20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from two_graph_fusion.datasets.twibot20 import induced_directed_edges, load_twibot20
from two_graph_fusion.propagation.prior import fit_predict_lr_prior
from two_graph_fusion.propagation.sybilhp import SybilHPParams, p_honest_to_dict as hp_dict, run_sybilhp
from two_graph_fusion.propagation.sybilscar import (
    p_honest_to_dict, relation_from_graph, run_propagation,
)
from two_graph_fusion.propagation.tvt_evaluation import (
    Splits, _best_f1_threshold, _honest_priors, _labels01, _scores, splits_from_column,
)

# Profile features: log1p-transform the heavy-tailed counts, keep bool flags as-is.
_COUNT_FIELDS = (
    "followers_count", "friends_count", "listed_count",
    "favourites_count", "statuses_count", "account_age_days", "tweets_per_day",
)
_BOOL_FIELDS = (
    "verified", "protected", "geo_enabled", "default_profile", "default_profile_image",
)
SYBILSCAR_W_GRID = (0.3, 0.5, 0.7, 0.9)
# Canonical linear rule needs a much smaller grid (no sigmoid soft-clamp).
SYBILSCAR_W_GRID_LINEAR = (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
SYBILHP_WBI_GRID = (0.7, 0.9, 0.99)


def _zscore_profile(users: pd.DataFrame, train_ids: list) -> tuple[pd.DataFrame, list[str]]:
    """Train-fit z-scored profile feature frame (log1p counts + bool flags)."""
    feat = pd.DataFrame(index=users.index)
    cols: list[str] = []
    train = users.loc[train_ids]
    for c in _COUNT_FIELDS:
        if c not in users.columns:
            continue
        x = pd.to_numeric(users[c], errors="coerce")
        med = pd.to_numeric(train[c], errors="coerce").median()
        x = np.log1p(x.fillna(med).clip(lower=0))
        mu, sd = x.loc[train_ids].mean(), x.loc[train_ids].std() or 1.0
        feat[c + "_z"] = (x - mu) / sd
        cols.append(c + "_z")
    for c in _BOOL_FIELDS:
        if c not in users.columns:
            continue
        x = pd.to_numeric(users[c], errors="coerce").fillna(0.0)
        mu, sd = x.loc[train_ids].mean(), x.loc[train_ids].std() or 1.0
        feat[c + "_z"] = (x - mu) / sd
        cols.append(c + "_z")
    return feat, cols


def _metrics(p_honest: dict, splits: Splits, thr: float) -> dict:
    y = _labels01(splits, splits.test, "bot")
    s = _scores(p_honest, splits.test)
    pred = (s >= thr).astype(int)
    return {
        "auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
        "acc": float((pred == y).mean()), "f1": float(f1_score(y, pred, zero_division=0)),
    }


def _finalise(name: str, prior: str, p_honest: dict, splits: Splits, hp: dict) -> dict:
    y_val = _labels01(splits, splits.val, "bot")
    s_val = _scores(p_honest, splits.val)
    val_auc = float(roc_auc_score(y_val, s_val))
    thr = _best_f1_threshold(y_val, s_val)
    m = _metrics(p_honest, splits, thr)
    return {"engine": name, "graph": "g_t", "prior": prior, "hparams": hp,
            "val_auc": val_auc, **m}


def _select(cands: list[tuple[dict, dict]], splits: Splits) -> tuple[dict, dict]:
    """Pick candidate with best val AUC -> (hparams, p_honest)."""
    y_val = _labels01(splits, splits.val, "bot")
    best = max(cands, key=lambda c: roc_auc_score(y_val, _scores(c[1], splits.val)))
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("Twibot-20-2"))
    ap.add_argument("--output-dir", type=Path, default=Path("two_graph_fusion/cache/twibot20"))
    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--rule", choices=["logistic", "linear"], default="logistic",
                    help="SybilSCAR update rule: 'linear' = canonical published rule, "
                         "'logistic' = sigmoid-stabilised variant (default).")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger(__name__)
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_twibot20(args.root, include_support=False)
    users = data.users
    labeled = users[users["label"].isin({"human", "bot"})].copy()
    labeled_ids = set(labeled.index)
    log.info("labeled=%d (bot=%d human=%d)", len(labeled),
             int((labeled.label == "bot").sum()), int((labeled.label == "human").sum()))

    splits = splits_from_column(labeled, "label", "split")
    log.info("split: train=%d val=%d test=%d", len(splits.train), len(splits.val), len(splits.test))

    # Directed follow edges among labelled users (tail follows head).
    directed = induced_directed_edges(data.follow_edges, labeled_ids)
    g_t = nx.Graph()
    g_t.add_nodes_from(splits.node_order)
    g_t.add_edges_from(directed)

    # Profile-feature LR prior (z-scored on train; predicts val+test).
    feat, cols = _zscore_profile(labeled, splits.train)
    feat["label"] = labeled["label"]
    lr = fit_predict_lr_prior(
        feat.loc[splits.train], feat.loc[splits.val + splits.test],
        label_col="label", honest_label="human", bot_label="bot",
        feature_columns=tuple(cols),
    )
    pi_b = lr.pi_b_test  # P(honest) on val+test
    log.info("profile-LR prior: test_auc=%.4f (n_feat=%d)", lr.test_auc, len(cols))

    results: list[dict] = []
    # LR-only baseline.
    results.append(_finalise("lr", "-", pi_b, splits, {}))

    priors_neutral = _honest_priors(splits, "human", pi_b_honest=None)
    priors_lr = _honest_priors(splits, "human", pi_b_honest=pi_b)

    linearized = args.rule == "linear"
    w_grid = SYBILSCAR_W_GRID_LINEAR if linearized else SYBILSCAR_W_GRID
    for prior_mode, base in [("neutral", priors_neutral), ("lr", priors_lr)]:
        # SybilSCAR on undirected G_T.
        cands = []
        for w in w_grid:
            rel = relation_from_graph("trust", g_t, splits.node_order, weight=w)
            r = run_propagation([rel], splits.node_order, base, default_prior=0.5,
                                max_iters=args.max_iters, linearized=linearized)
            cands.append(({"w_t": w}, p_honest_to_dict(r)))
        hp_, p = _select(cands, splits)
        results.append(_finalise("sybilscar", prior_mode, p, splits, hp_))

        # SybilHP on directed G_T.
        cands = []
        for w_bi in SYBILHP_WBI_GRID:
            sybil_prior = {u: 1.0 - v for u, v in base.items()}
            r = run_sybilhp(node_order=splits.node_order, sybil_prior=sybil_prior,
                            trust_edges=directed, params=SybilHPParams(w_bi=w_bi),
                            default_prior=0.5, max_iters=args.max_iters)
            cands.append(({"w_bi": w_bi}, hp_dict(r)))
        hp_, p = _select(cands, splits)
        results.append(_finalise("sybilhp", prior_mode, p, splits, hp_))

    payload = {
        "dataset": "twibot-20 (G_T-only, official split, profile-LR prior)",
        "rule": args.rule,
        "n_labeled": len(labeled), "split_sizes": {
            "train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
        "g_t": {"nodes": g_t.number_of_nodes(), "edges": g_t.number_of_edges(),
                "directed_edges": len(directed)},
        "lr_prior_test_auc": lr.test_auc, "homophily_lift": 0.0251,
        "results": results, "elapsed_s": round(time.time() - t0, 1),
    }
    out = args.output_dir / ("eval_gt.json" if args.rule == "logistic"
                             else f"eval_gt_{args.rule}.json")
    out.write_text(json.dumps(payload, indent=2, default=float))

    print(f"\n=== TwiBot-20 G_T-only eval (official split, lift +0.025 sub-gate) ===")
    print(f"labeled={len(labeled):,}  test={len(splits.test):,}  "
          f"G_T: {g_t.number_of_nodes():,} nodes / {g_t.number_of_edges():,} edges")
    print(f"\n{'engine':<10}{'prior':<9}{'AUC':>8}{'PR-AUC':>9}{'Acc':>8}{'F1':>8}   hparams")
    for r in results:
        print(f"{r['engine']:<10}{r['prior']:<9}{r['auc']:>8.4f}{r['pr_auc']:>9.4f}"
              f"{r['acc']:>8.4f}{r['f1']:>8.4f}   {r['hparams']}")
    print(f"\nwrote {out}  ({payload['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
