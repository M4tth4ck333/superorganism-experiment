"""A2 — aggregate the multi-seed (0-9) Cresci TVT runs.

Produces, per dataset:

- mean ± std (and min/max) of test AUC / PR-AUC / Acc / F1 per cell
  (LR, LR drop-L, sybilscar {g_t, g_b, fusion} × {neutral, lr}) for
  Tables 5/6 of the thesis;
- paired bootstrap 95% CIs on the test-AUC difference for the
  load-bearing comparisons (fusion vs G_B-only, fusion vs G_T-only,
  both priors; LR vs LR drop-L), per seed and pooled across seeds;
- LR coefficient mean ± std across seeds (L-ablation table, A6).

Per-user test scores were persisted by the runners (``test_scores``);
test labels are recovered from the dataset label loaders.

Usage::

    .venv/bin/python3 -m two_graph_fusion.scripts.aggregate_multiseed \\
        --output two_graph_fusion/cache/multiseed_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

SEEDS = tuple(range(10))
N_BOOT = 10_000
CELLS = [
    ("lr", "-", "-"),
    ("lr", "-", "drop_l"),
    ("sybilscar", "g_t", "neutral"),
    ("sybilscar", "g_b", "neutral"),
    ("sybilscar", "fusion", "neutral"),
    ("sybilscar", "g_t", "lr"),
    ("sybilscar", "g_b", "lr"),
    ("sybilscar", "fusion", "lr"),
]
COMPARISONS = [
    ("sybilscar|fusion|neutral", "sybilscar|g_b|neutral"),
    ("sybilscar|fusion|neutral", "sybilscar|g_t|neutral"),
    ("sybilscar|fusion|lr", "sybilscar|g_b|lr"),
    ("sybilscar|fusion|lr", "sybilscar|g_t|lr"),
    ("lr|-|-", "lr|-|drop_l"),
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate multi-seed Cresci TVT artifacts (A2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cresci2017-dir", type=Path,
                   default=Path("two_graph_fusion/cache/cresci2017_tvt"))
    p.add_argument("--cresci2015-dir", type=Path,
                   default=Path("two_graph_fusion/cache/cresci2015_tvt"))
    p.add_argument("--cresci2017-root", type=Path, default=Path("cresci_2017_datasets_full"))
    p.add_argument("--cresci2015-root", type=Path, default=Path("cresci-15-orig"))
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--output", type=Path,
                   default=Path("two_graph_fusion/cache/multiseed_summary.json"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _load_runs(d: Path, pattern: str) -> dict[int, dict]:
    runs = {}
    for seed in SEEDS:
        path = d / pattern.format(seed=seed)
        if not path.exists():
            logger.warning("missing %s", path)
            continue
        with path.open() as fh:
            runs[seed] = json.load(fh)
    return runs


def _cell_stats(runs: dict[int, dict]) -> dict:
    out = {}
    for engine, graph, prior in CELLS:
        vals = {m: [] for m in ("test_auc", "test_pr_auc", "test_acc", "test_f1")}
        for seed in sorted(runs):
            for r in runs[seed]["results"]:
                if (r["engine"], r["graph"], r["prior"]) == (engine, graph, prior):
                    for m in vals:
                        vals[m].append(r[m])
                    break
        if not vals["test_auc"]:
            continue
        key = f"{engine}|{graph}|{prior}"
        out[key] = {
            "n_seeds": len(vals["test_auc"]),
            **{
                m: {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "per_seed": [float(x) for x in v],
                }
                for m, v in vals.items()
            },
        }
    return out


def _paired_bootstrap(
    y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> dict:
    """95% CI on AUC(a) − AUC(b) by paired resampling of test users."""
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yi = y[idx]
        if yi.min() == yi.max():           # degenerate resample
            diffs[i] = np.nan
            continue
        diffs[i] = roc_auc_score(yi, s_a[idx]) - roc_auc_score(yi, s_b[idx])
    diffs = diffs[~np.isnan(diffs)]
    point = float(roc_auc_score(y, s_a) - roc_auc_score(y, s_b))
    return {
        "delta_auc": point,
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "p_delta_le_0": float((diffs <= 0).mean()),
    }


def _comparisons(runs: dict[int, dict], labels: dict[str, str],
                 n_boot: int) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for cell_a, cell_b in COMPARISONS:
        per_seed = {}
        deltas = []
        for seed in sorted(runs):
            ts = runs[seed].get("test_scores", {})
            if cell_a not in ts or cell_b not in ts:
                continue
            users = sorted(ts[cell_a])
            y = np.array([1 if labels[u] == "bot" else 0 for u in users])
            s_a = np.array([ts[cell_a][u] for u in users])
            s_b = np.array([ts[cell_b][u] for u in users])
            res = _paired_bootstrap(y, s_a, s_b, n_boot, rng)
            per_seed[seed] = res
            deltas.append(res["delta_auc"])
        if not deltas:
            continue
        deltas = np.asarray(deltas)
        out[f"{cell_a} minus {cell_b}"] = {
            "per_seed": {str(s): r for s, r in per_seed.items()},
            "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "n_seeds_delta_pos": int((deltas > 0).sum()),
            "n_seeds_ci_excludes_0": int(sum(
                1 for r in per_seed.values() if r["ci_lo"] > 0 or r["ci_hi"] < 0
            )),
            "n_seeds": len(deltas),
        }
    return out


def _lr_coefficients(runs: dict[int, dict]) -> dict:
    out = {}
    for key in ("lr_prior", "lr_prior_drop_l"):
        rows = [runs[s][key] for s in sorted(runs) if key in runs[s]]
        if not rows:
            continue
        cols = rows[0]["feature_columns"]
        coefs = np.array([r["coefficients"] for r in rows])
        out[key] = {
            "feature_columns": cols,
            "coef_mean": [float(x) for x in coefs.mean(axis=0)],
            "coef_std": [float(x) for x in coefs.std(axis=0, ddof=1)]
            if len(rows) > 1 else [0.0] * len(cols),
            "intercept_mean": float(np.mean([r["intercept"] for r in rows])),
            "n_seeds": len(rows),
        }
    return out


def _dataset_summary(runs: dict[int, dict], labels: dict[str, str],
                     n_boot: int) -> dict:
    return {
        "n_seeds": len(runs),
        "cells": _cell_stats(runs),
        "auc_comparisons_paired_bootstrap": _comparisons(runs, labels, n_boot),
        "lr_coefficients": _lr_coefficients(runs),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from two_graph_fusion.datasets.cresci2015 import load_labels as labels_2015
    from two_graph_fusion.datasets.cresci2017 import load_labels as labels_2017

    summary = {}
    runs17 = _load_runs(args.cresci2017_dir, "results_default_linear_seed{seed}.json")
    if runs17:
        summary["cresci2017"] = _dataset_summary(
            runs17, labels_2017(args.cresci2017_root, include_optional=False),
            args.n_boot,
        )
    runs15 = _load_runs(args.cresci2015_dir, "results_gtfollow_linear_seed{seed}.json")
    if runs15:
        summary["cresci2015"] = _dataset_summary(
            runs15, labels_2015(args.cresci2015_root), args.n_boot,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("wrote %s", args.output)

    for ds, s in summary.items():
        print(f"\n=== {ds} ({s['n_seeds']} seeds) ===")
        print(f"{'cell':<28} {'AUC mean±std':>18} {'PR mean±std':>18} "
              f"{'Acc':>8} {'F1':>8}")
        for key, c in s["cells"].items():
            a, p = c["test_auc"], c["test_pr_auc"]
            print(f"{key:<28} {a['mean']:.4f} ± {a['std']:.4f}   "
                  f"{p['mean']:.4f} ± {p['std']:.4f}   "
                  f"{c['test_acc']['mean']:.4f}  {c['test_f1']['mean']:.4f}")
        print("\npaired-bootstrap AUC deltas (95% CI per seed):")
        for name, comp in s["auc_comparisons_paired_bootstrap"].items():
            print(f"  {name}: mean Δ={comp['mean_delta']:+.4f} ± {comp['std_delta']:.4f}  "
                  f"(Δ>0 in {comp['n_seeds_delta_pos']}/{comp['n_seeds']} seeds; "
                  f"CI excl. 0 in {comp['n_seeds_ci_excludes_0']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
