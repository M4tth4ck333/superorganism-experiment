"""Behavioral-feature 2D projection across the homophily spectrum.

The SEGCN-style analog (its Figs 3-4 t-SNE the *learned GNN embedding* per
model). FuSST has no learned embedding; its per-node representation is the
five-dim behavioral feature vector f(u). This figure projects f(u) to 2D and
colours by class, one panel per timestamped dataset, so the reader sees the
behavioral modality separate cleanly on the high-homophily anchor and collapse
into an overlapping blob on the negative control -- the behavioral-side mirror
of fig6_edge_breakdown (the structural side).

Each panel is annotated with a homogeneity score (SEGCN's metric): KMeans(2) is
fit on the *original five-dim* features and scored against the true labels with
sklearn's homogeneity_score in [0, 1]. It is computed on the feature space, not
the lossy 2D projection, so it reflects the representation itself.

    .venv/bin/python3 -m two_graph_fusion.scripts.make_feature_tsne
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import homogeneity_score, silhouette_score  # noqa: E402

OUT = Path("thesis_writeup/figures")
CACHE = Path("two_graph_fusion/cache")
FEATURE_COLS = ["B_G_z", "M_z", "C_24_humanlike_z", "D_shape_z", "L_z"]
SEED = 42
MAX_PER_PANEL = 6000  # cap for t-SNE speed / visual clarity
HUMAN_C, BOT_C = "#3b82f6", "#ef4444"


@dataclass
class Panel:
    name: str
    csv: Path
    regime: str  # short tag shown under the title


# Datasets with cached z-scored behavioral features. Cresci-2015 has no cached
# feature CSV yet; add it here once two_graph_fusion writes one.
PANELS = [
    Panel("Cresci-2017", CACHE / "cresci2017" / "features_zscored.csv",
          "high-homophily anchor"),
    Panel("TwiBot-22", CACHE / "m4_local_20k_tt_features_zscored.csv",
          "negative control"),
]


def _load(panel: Panel, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(panel.csv)
    df = df.dropna(subset=FEATURE_COLS + ["label"])
    if len(df) > MAX_PER_PANEL:
        # stratified down-sample so the class balance is preserved
        frac = MAX_PER_PANEL / len(df)
        parts = [g.sample(max(1, int(round(len(g) * frac))), random_state=SEED)
                 for _, g in df.groupby("label")]
        df = pd.concat(parts)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = (df["label"].astype(str).str.lower() == "bot").to_numpy(dtype=int)
    return X, y


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.0 * len(PANELS), 5.0))
    if len(PANELS) == 1:
        axes = [axes]

    for ax, panel in zip(axes, PANELS):
        X, y = _load(panel, rng)

        # homogeneity in the real five-dim feature space (SEGCN's metric)
        km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(X)
        homog = homogeneity_score(y, km.labels_)
        sil = silhouette_score(X, y)  # class separation in feature space

        perplexity = float(min(30, max(5, (len(X) - 1) / 3)))
        emb = TSNE(n_components=2, init="pca", perplexity=perplexity,
                   random_state=SEED).fit_transform(X)

        for cls, color, marker, label in [
            (0, HUMAN_C, "o", "human"),
            (1, BOT_C, "x", "bot"),
        ]:
            m = y == cls
            ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.45, c=color,
                       marker=marker, linewidths=0.5,
                       label=f"{label} (n={int(m.sum())})")
        ax.set_title(f"{panel.name}\n({panel.regime})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.03, 0.97,
                f"homogeneity = {homog:.3f}\nsilhouette = {sil:+.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        print(f"{panel.name}: n={len(X)} homogeneity={homog:.4f} "
              f"silhouette={sil:.4f} perplexity={perplexity:.1f}")

    fig.tight_layout()
    out = OUT / "fig9_feature_tsne.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
