"""Figure 7 — ROC + PR curves for the headline SybilSCAR configs.

Per-test-node (label, score) pairs for three headline cells, plotted as ROC
and Precision-Recall curves:

- Cresci-2017  : fusion (G_T + G_B), neutral prior      (≈ AUC 0.994)
- Cresci-2015  : fusion (G_T follow + G_B), neutral prior (≈ AUC 0.976)
- TwiBot-20    : G_T only, profile-LR prior             (≈ AUC 0.786)

All canonical (linear) SybilSCAR, weights val-tuned over ``W_GRID``.

The two Cresci cells are read directly from the per-user ``test_scores``
persisted in the deterministic seed-0 TVT artifacts
(``cache/cresci201{5,7}_tvt/``), so the curves are exactly the runs the
thesis tables report. TwiBot-20 (no behavioral layer, fully deterministic)
is regenerated in place.

    .venv/bin/python3 -m two_graph_fusion.scripts.make_roc_pr
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import (
    auc as sk_auc, average_precision_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)

from two_graph_fusion.propagation.prior import fit_predict_lr_prior
from two_graph_fusion.propagation.sybilscar import (
    p_honest_to_dict, relation_from_graph, run_propagation,
)
from two_graph_fusion.propagation.tvt_evaluation import (
    Splits, _honest_priors, _labels01, _scores, splits_from_column,
)

OUT = Path("thesis_writeup/figures")
# Canonical (linear) SybilSCAR weight grid, capped at 0.5 — matches
# SYBILSCAR_W_GRID_LINEAR in tvt_evaluation.py.
W_GRID = (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
log = logging.getLogger("make_roc_pr")


# --------------------------------------------------------------------------
# Per-config run: grid-tune on val, return test (y, score) for the best config
# --------------------------------------------------------------------------

def _best_test_scores(relations_for, splits: Splits, base_priors: dict,
                      bot_label: str) -> tuple[np.ndarray, np.ndarray]:
    """Grid-search relation weights on val AUC; return (y_test, score_test)."""
    y_val = _labels01(splits, splits.val, bot_label)
    best = None
    for hp, rels in relations_for():
        r = run_propagation(rels, splits.node_order, base_priors,
                            default_prior=0.5, max_iters=50, linearized=True)
        ph = p_honest_to_dict(r)
        va = roc_auc_score(y_val, _scores(ph, splits.val))
        if best is None or va > best[0]:
            best = (va, hp, ph)
    _, hp, ph = best
    y_test = _labels01(splits, splits.test, bot_label)
    s_test = _scores(ph, splits.test)
    log.info("  selected %s  val_auc=%.4f  test_auc=%.4f", hp, best[0],
             roc_auc_score(y_test, s_test))
    return y_test, s_test


# --------------------------------------------------------------------------
# Dataset builders -> (y_test, score_test, label)
# --------------------------------------------------------------------------

def _scores_from_artifact(artifact: Path, cell: str,
                          label_map: dict) -> tuple[np.ndarray, np.ndarray]:
    """(y_test, score_test) for one cell from a TVT runner JSON."""
    import json
    with artifact.open() as fh:
        run = json.load(fh)
    ts = run["test_scores"][cell]
    users = sorted(ts)
    y = np.array([1 if label_map[u] == "bot" else 0 for u in users])
    s = np.array([ts[u] for u in users])
    log.info("  %s %s: %d test users, test_auc=%.4f",
             artifact.name, cell, len(users), roc_auc_score(y, s))
    return y, s


def cresci2017_fusion(root: Path) -> tuple[np.ndarray, np.ndarray]:
    from two_graph_fusion.datasets.cresci2017 import load_labels
    return _scores_from_artifact(
        Path("two_graph_fusion/cache/cresci2017_tvt/results_default_linear_seed0.json"),
        "sybilscar|fusion|neutral",
        load_labels(root, include_optional=False),
    )


def cresci2015_fusion(root: Path) -> tuple[np.ndarray, np.ndarray]:
    from two_graph_fusion.datasets.cresci2015 import load_labels
    return _scores_from_artifact(
        Path("two_graph_fusion/cache/cresci2015_tvt/results_gtfollow_linear_seed0.json"),
        "sybilscar|fusion|neutral",
        load_labels(root),
    )


def twibot20_gt_lr(root: Path) -> tuple[np.ndarray, np.ndarray]:
    from two_graph_fusion.datasets.twibot20 import induced_directed_edges, load_twibot20
    from two_graph_fusion.scripts.run_twibot20_eval import _zscore_profile
    data = load_twibot20(root, include_support=False)
    labeled = data.users[data.users["label"].isin({"human", "bot"})].copy()
    ids = set(labeled.index)
    splits = splits_from_column(labeled, "label", "split")
    directed = induced_directed_edges(data.follow_edges, ids)
    g_t = nx.Graph(); g_t.add_nodes_from(splits.node_order); g_t.add_edges_from(directed)
    feat, cols = _zscore_profile(labeled, splits.train)
    feat["label"] = labeled["label"]
    lr = fit_predict_lr_prior(feat.loc[splits.train], feat.loc[splits.val + splits.test],
                              label_col="label", honest_label="human", bot_label="bot",
                              feature_columns=tuple(cols))
    base = _honest_priors(splits, "human", pi_b_honest=lr.pi_b_test)  # LR prior
    log.info("TwiBot-20 G_T edges=%d  profile-LR test_auc=%.4f",
             g_t.number_of_edges(), lr.test_auc)

    def rels():
        for w in W_GRID:
            yield {"w_t": w}, [relation_from_graph("trust", g_t, splits.node_order, weight=w)]

    return _best_test_scores(rels, splits, base, "bot")


# --------------------------------------------------------------------------

def plot(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    from two_graph_fusion.scripts import _figstyle
    colors = _figstyle.DATASET  # shared colourblind-safe palette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _figstyle.clean(ax1)
    _figstyle.clean(ax2)
    for name, (y, s) in curves.items():
        c = colors[name]
        fpr, tpr, _ = roc_curve(y, s)
        ax1.plot(fpr, tpr, color=c, lw=2.2, label=f"{name} (AUC {sk_auc(fpr, tpr):.4f})")
        prec, rec, _ = precision_recall_curve(y, s)
        ax2.plot(rec, prec, color=c, lw=2.2,
                 label=f"{name} (AP {average_precision_score(y, s):.4f})")
    ax1.plot([0, 1], [0, 1], ls=(0, (5, 4)), color=_figstyle.GATE, lw=1, alpha=0.8)
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate (recall)")
    ax1.set_title("ROC curves")
    leg1 = ax1.legend(fontsize=9, loc="lower right")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision–Recall curves")
    leg2 = ax2.legend(fontsize=9, loc="lower left")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    for leg in (leg1, leg2):
        leg.get_frame().set_edgecolor("#d0d0d0")
        leg.get_frame().set_linewidth(0.8)
    fig.suptitle("Headline detector performance", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_roc_pr.png", dpi=200)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    log.info("=== TwiBot-20 ===")
    curves["TwiBot-20"] = twibot20_gt_lr(Path("Twibot-20-2"))
    log.info("=== Cresci-2015 ===")
    curves["Cresci-2015"] = cresci2015_fusion(Path("cresci-15-orig"))
    log.info("=== Cresci-2017 ===")
    curves["Cresci-2017"] = cresci2017_fusion(Path("cresci_2017_datasets_full"))
    plot(curves)
    log.info("wrote %s  (%.0fs)", OUT / "fig7_roc_pr.png", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
