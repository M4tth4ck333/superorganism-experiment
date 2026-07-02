"""Generate thesis figures 1-6 (SybilSCAR only; no SybilHP).

All numbers are the final reported values from RESULTS.md (M8-M14) / the cache
JSONs. Writes PNGs to thesis_writeup/figures/.

    .venv/bin/python3 -m two_graph_fusion.scripts.make_thesis_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from two_graph_fusion.scripts import _figstyle

OUT = Path("thesis_writeup/figures")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def fig1_lift_spectrum() -> None:
    # G_T lift per dataset. TwiBot-22 is cohort-dependent (Table 2/4 footnote):
    # +0.0382 on the evaluated n_min=25 qualifying cohort (the cohort behind the
    # Table 4 AUCs); the stricter n_min=50 cohort is +0.0020, and the raw
    # full-graph aggregate (+0.0719, M9) is shown as a hatched bar because it is a
    # majority-class artefact (bot-bot edges 0.7% of the graph).
    data = [("TwiBot-22", 0.0382), ("TwiBot-20", 0.0251), ("Twitter-270k", 0.2658),
            ("Cresci-2015", 0.4906), ("Cresci-2017", 0.4709)]
    names = [d[0] for d in data]
    lifts = [d[1] for d in data]
    tw22 = _figstyle.ACCENT  # TwiBot-22 is not one of the four legend datasets
    hue = {**_figstyle.DATASET, "TwiBot-22": tw22}  # consistent per-dataset colour
    halo = _figstyle.halo(1.8)
    # single-column figure: compact figsize with column-legible fonts
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    _figstyle.clean(ax)
    x = np.arange(len(names))
    w = 0.66
    # shade the sub-gate zone so the failing datasets read at a glance
    ax.axhspan(-0.05, 0.05, color="#f2f2f2", zorder=0)
    ax.bar(x, lifts, w, color=[hue[n] for n in names], edgecolor=_figstyle.EDGE,
           linewidth=0.5, zorder=3)
    # Full-graph aggregate companion bar for TwiBot-22 (majority-class artefact).
    ax.bar([0.5], [0.0719], 0.30, facecolor="none", edgecolor=tw22,
           hatch="////", lw=1.0, zorder=3)
    ax.annotate("+0.072\naggregate\n(artefact)", xy=(0.5, 0.0719),
                xytext=(1.28, 0.40), fontsize=5.5, color=_figstyle.TXT,
                ha="center", va="center", path_effects=halo,
                arrowprops=dict(arrowstyle="->", color=_figstyle.SPINE, lw=0.6))
    ax.axhline(0.05, ls=(0, (4, 3)), c="#c0392b", lw=1.6, zorder=5)
    ax.text(-0.95, 0.05, "gate 0.05", fontsize=6.5, color="#c0392b",
            ha="right", va="center", style="italic", path_effects=halo,
            clip_on=False)
    for xi, v in zip(x, lifts):
        # nudge the sub-gate TwiBot labels clear of the gate line / shaded zone
        dy = 0.035 if names[xi] == "TwiBot-20" else (0.020 if names[xi] == "TwiBot-22" else 0.015)
        ax.text(xi, v + dy, f"{v:+.3f}", ha="center", fontsize=6,
                color=_figstyle.TXT, path_effects=halo)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=7)
    ax.set_ylabel("Homophily lift", fontsize=8)
    ax.set_title("Homophily-lift diagnostic ($G_T$)", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(-0.05, 0.58)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_lift_spectrum.png", dpi=200)
    plt.close(fig)


def fig2_lift_vs_auc() -> None:
    """Single panel: homophily lift vs graph-only propagation AUC under the
    neutral prior, for every dataset-graph point that has a neutral-prior run.
    Shows lift is necessary but not sufficient: the sub-gate graph sits near
    chance (low lift -> low AUC), while two high-lift graphs still score low
    because they are coverage-limited -- their largest connected component
    covers few accounts, so propagation reaches almost nobody. Colour = dataset
    (one hue each, matching the legend), shape = view (circle G_T / square G_B),
    so each dataset's behavioural graph is the square. Cresci AUCs = 10-seed
    means; Cresci-2015 interaction = seed-0 run."""
    from matplotlib.lines import Line2D

    DS_HUE = _figstyle.DATASET  # shared colourblind-safe palette, one hue/dataset
    halo = _figstyle.halo()
    TXT, MUTE = _figstyle.TXT, _figstyle.MUTE
    # (dataset, view, lift, AUC)
    pts = [
        ("TwiBot-20",    "G_T", 0.0251, 0.6089),
        ("Twitter-270k", "G_T", 0.2658, 0.8298),
        ("Cresci-2015",  "G_B", 0.3908, 0.9531),
        ("Cresci-2015",  "G_T", 0.4906, 0.8346),   # follow
        ("Cresci-2015",  "G_T", 0.4969, 0.7051),   # interaction (retweet+reply)
        ("Cresci-2017",  "G_B", 0.4572, 0.9926),
        ("Cresci-2017",  "G_T", 0.4709, 0.6890),
    ]
    mk = lambda v: "o" if v == "G_T" else "s"  # shape carries the view

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    _figstyle.clean(ax)

    for ds, view, x, y in pts:
        ax.scatter(x, y, s=145, marker=mk(view), facecolors=DS_HUE[ds],
                   edgecolors=_figstyle.EDGE, linewidths=0.9, zorder=3)

    # The two Cresci-2015 G_T circles share colour and shape, so they are the
    # only points that need disambiguating; everything else the legend covers.
    ax.annotate("follow", (0.4906, 0.8346), xytext=(0.513, 0.834),
                fontsize=9, color=TXT, ha="left", va="center", path_effects=halo)
    ax.annotate("interaction", (0.4969, 0.7051), xytext=(0.513, 0.712),
                fontsize=9, color=TXT, ha="left", va="center", path_effects=halo)

    # Coverage tags for the two high-lift / low-AUC points.
    ax.annotate("21% coverage", (0.4969, 0.7051), xytext=(0.513, 0.694),
                fontsize=8, color=MUTE, ha="left", va="center", path_effects=halo)
    ax.annotate("8% coverage", (0.4709, 0.6890), xytext=(0.471, 0.660),
                fontsize=8, color=MUTE, ha="center", va="top", path_effects=halo)

    # Single callout explaining why these high-lift points still score low.
    ax.annotate("high lift, low coverage:\npropagation reaches few accounts",
                xy=(0.464, 0.700), xytext=(0.305, 0.582),
                fontsize=8.5, color=TXT, ha="center", va="center",
                path_effects=halo,
                arrowprops=dict(arrowstyle="->", color=_figstyle.SPINE, lw=1.0,
                                connectionstyle="arc3,rad=0.18"))

    ax.axvline(0.05, ls=(0, (5, 4)), c=_figstyle.GATE, lw=1.1)
    ax.text(0.063, 0.517, "gate = 0.05", fontsize=8.5, color=MUTE, style="italic")

    ax.set_xlabel("Homophily lift")
    ax.set_ylabel("Graph-only propagation AUC (neutral prior)")
    ax.set_title("Lift is necessary but not sufficient", fontsize=12, pad=10)
    ax.set_ylim(0.5, 1.04)
    ax.set_xlim(-0.03, 0.66)

    ds_handles = [Line2D([], [], marker="o", ls="", mfc=DS_HUE[d], mec=_figstyle.EDGE,
                         ms=9, label=d) for d in DS_HUE]
    enc_handles = [
        Line2D([], [], marker="o", ls="", mfc="#bdbdbd", mec=_figstyle.EDGE, ms=9, label="$G_T$ (social)"),
        Line2D([], [], marker="s", ls="", mfc="#bdbdbd", mec=_figstyle.EDGE, ms=9, label="$G_B$ (behavioral)"),
    ]
    leg = fig.legend(handles=ds_handles + enc_handles, ncol=3, fontsize=9,
                     loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=True,
                     columnspacing=1.8, handletextpad=0.5)
    leg.get_frame().set_edgecolor("#d0d0d0")
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(OUT / "fig2_lift_vs_auc.png", dpi=200)
    plt.close(fig)


def fig3_twibot22_degree_core() -> None:
    k = [1, 2, 3, 5, 10, 20, 50, 100]
    lift = [0.0719, 0.0124, -0.0008, -0.0084, -0.0095, 0.0016, 0.0382, 0.0465]
    # assortativity (subset with both numbers)
    ka = [1, 2, 5, 10, 50, 100]
    base = [11.7, 7.5, 6.1, 6.0, 8.8, 9.1]
    botbot = [4.8, 3.9, 3.3, 2.9, 1.9, 1.8]
    tw = _figstyle.DATASET["TwiBot-20"]  # TwiBot-family hue (vermillion)
    halo = _figstyle.halo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.0))
    _figstyle.clean(a1)
    _figstyle.clean(a2)
    a1.plot(k, lift, "o-", color=tw, lw=1.8, mec=_figstyle.EDGE, mew=0.7)
    a1.axhline(0.05, ls=(0, (5, 4)), c=_figstyle.GATE, lw=1.1)
    a1.axhline(0.0, ls=":", c=_figstyle.SPINE, lw=1)
    a1.text(60, 0.056, "gate = 0.05", fontsize=8, color=_figstyle.MUTE,
            style="italic", path_effects=halo)
    # the only above-gate point is the full graph (k=1), a majority-class artefact
    a1.annotate("full-graph aggregate:\nmajority-class artefact", xy=(1, 0.0719),
                xytext=(1.8, 0.063), fontsize=7.5, color=_figstyle.TXT,
                ha="left", va="center", path_effects=halo,
                arrowprops=dict(arrowstyle="->", color=_figstyle.SPINE, lw=0.8))
    a1.set_xscale("log")
    a1.set_xlabel("degree threshold  (keep nodes with deg ≥ k)")
    a1.set_ylabel("homophily lift")
    a1.set_title("(a) Lift clears the gate only on the full graph")
    a2.plot(ka, base, "o-", color=_figstyle.MUTE, lw=1.8, mec=_figstyle.EDGE,
            mew=0.7, label="bot base rate")
    a2.plot(ka, botbot, "s-", color=tw, lw=1.8, mec=_figstyle.EDGE, mew=0.7,
            label="bot→bot edge share")
    a2.fill_between(ka, botbot, base, color=tw, alpha=0.12)
    a2.set_xscale("log")
    a2.set_xlabel("degree threshold k")
    a2.set_ylabel("percent (%)")
    a2.set_title("(b) Bots are heterophilic at every degree")
    a2.legend(fontsize=8)
    fig.suptitle("TwiBot-22 follow graph: no propagatable bot signal", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_twibot22_degree_core.png", dpi=200)
    plt.close(fig)


def fig4_cresci_config_bars() -> None:
    """Both panels: mean ± std over the ten stratified splits (seeds 0-9),
    canonical linear rule (cache/multiseed_summary.json)."""
    from matplotlib.lines import Line2D
    halo = _figstyle.halo()
    # Bars encode the prior (not the dataset); two high-contrast palette hues.
    C_NEU, C_LR = _figstyle.DATASET["Cresci-2017"], _figstyle.DATASET["Cresci-2015"]
    ekw = dict(ecolor=_figstyle.MUTE, lw=1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.0), sharey=True)
    _figstyle.clean(a1); _figstyle.clean(a2)
    cfgs = ["$G_T$", "$G_B$", "Fusion"]
    x = np.arange(len(cfgs)); w = 0.38

    def panel(ax, neu, neu_sd, lr, lr_sd, lr_only, title):
        # 10-seed means ± std (multiseed_summary.json); dashed line is LR-only.
        ax.bar(x - w / 2, neu, w, yerr=neu_sd, capsize=3, error_kw=ekw,
               color=C_NEU, edgecolor=_figstyle.EDGE, linewidth=0.5)
        ax.bar(x + w / 2, lr, w, yerr=lr_sd, capsize=3, error_kw=ekw,
               color=C_LR, edgecolor=_figstyle.EDGE, linewidth=0.5)
        ax.axhline(lr_only, ls=(0, (5, 4)), c=_figstyle.GATE, lw=1.1)
        ax.set_xticks(x); ax.set_xticklabels(cfgs)
        ax.set_title(title)
        for xi, (n, l) in enumerate(zip(neu, lr)):
            ax.text(xi - w / 2, n + 0.013, f"{n:.3f}", ha="center", fontsize=7,
                    color=_figstyle.TXT, path_effects=halo)
            ax.text(xi + w / 2, l + 0.013, f"{l:.3f}", ha="center", fontsize=7,
                    color=_figstyle.TXT, path_effects=halo)

    panel(a1, [0.6890, 0.9926, 0.9947], [0.0076, 0.0017, 0.0012],
          [0.9757, 0.9785, 0.9875], [0.0037, 0.0036, 0.0024], 0.9559,
          "Cresci-2017 (lift +0.4709)")
    a1.set_ylabel("test ROC-AUC (mean ± std, 10 splits)")
    a1.set_ylim(0.6, 1.04)
    panel(a2, [0.8346, 0.9531, 0.9769], [0.0102, 0.0068, 0.0039],
          [0.9779, 0.9651, 0.9779], [0.0052, 0.0074, 0.0051], 0.9647,
          "Cresci-2015, follow graph (lift +0.4906)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_NEU, ec=_figstyle.EDGE, label="neutral prior"),
        plt.Rectangle((0, 0), 1, 1, fc=C_LR, ec=_figstyle.EDGE, label="LR prior"),
        Line2D([], [], color=_figstyle.GATE, ls=(0, (5, 4)), lw=1.1,
               label="LR-only (feature prior)"),
    ]
    leg = a1.legend(handles=handles, fontsize=8, loc="lower right")
    leg.get_frame().set_edgecolor("#d0d0d0"); leg.get_frame().set_linewidth(0.8)
    fig.suptitle("Fusion $\\geq G_B \\gg G_T$ where homophily exists", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_cresci_config_bars.png", dpi=200)
    plt.close(fig)


def fig4_cresci_config_bars_split() -> None:
    """Same data as fig4_cresci_config_bars, but each dataset as a standalone
    figure (own x/y axes, own legend). Saves new files; does not overwrite the
    combined fig4."""
    from matplotlib.lines import Line2D
    halo = _figstyle.halo()
    C_NEU, C_LR = _figstyle.DATASET["Cresci-2017"], _figstyle.DATASET["Cresci-2015"]
    ekw = dict(ecolor=_figstyle.MUTE, lw=1)
    cfgs = ["$G_T$", "$G_B$", "Fusion"]
    x = np.arange(len(cfgs)); w = 0.38

    def make(fname, neu, neu_sd, lr, lr_sd, lr_only, title):
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        _figstyle.clean(ax)
        ax.bar(x - w / 2, neu, w, yerr=neu_sd, capsize=3, error_kw=ekw,
               color=C_NEU, edgecolor=_figstyle.EDGE, linewidth=0.5)
        ax.bar(x + w / 2, lr, w, yerr=lr_sd, capsize=3, error_kw=ekw,
               color=C_LR, edgecolor=_figstyle.EDGE, linewidth=0.5)
        ax.axhline(lr_only, ls=(0, (5, 4)), c=_figstyle.GATE, lw=1.1)
        ax.set_xticks(x); ax.set_xticklabels(cfgs)
        ax.set_ylabel("test ROC-AUC (mean ± std, 10 splits)")
        ax.set_ylim(0.6, 1.04)
        ax.set_title(title)
        for xi, (n, l) in enumerate(zip(neu, lr)):
            ax.text(xi - w / 2, n + 0.013, f"{n:.3f}", ha="center", fontsize=7,
                    color=_figstyle.TXT, path_effects=halo)
            ax.text(xi + w / 2, l + 0.013, f"{l:.3f}", ha="center", fontsize=7,
                    color=_figstyle.TXT, path_effects=halo)
        handles = [
            plt.Rectangle((0, 0), 1, 1, fc=C_NEU, ec=_figstyle.EDGE, label="neutral prior"),
            plt.Rectangle((0, 0), 1, 1, fc=C_LR, ec=_figstyle.EDGE, label="LR prior"),
            Line2D([], [], color=_figstyle.GATE, ls=(0, (5, 4)), lw=1.1,
                   label="LR-only (feature prior)"),
        ]
        leg = ax.legend(handles=handles, fontsize=8, loc="lower right")
        leg.get_frame().set_edgecolor("#d0d0d0"); leg.get_frame().set_linewidth(0.8)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=200)
        plt.close(fig)

    make("fig4a_cresci2017_config_bars.png",
         [0.6890, 0.9926, 0.9947], [0.0076, 0.0017, 0.0012],
         [0.9757, 0.9785, 0.9875], [0.0037, 0.0036, 0.0024], 0.9559,
         "Cresci-2017 (lift +0.4709)")
    make("fig4b_cresci2015_config_bars.png",
         [0.8346, 0.9531, 0.9769], [0.0102, 0.0068, 0.0039],
         [0.9779, 0.9651, 0.9779], [0.0052, 0.0074, 0.0051], 0.9647,
         "Cresci-2015, follow graph (lift +0.4906)")


def fig5_coverage_explainer() -> None:
    """Connected scatter of LCC coverage (x) vs neutral-prior graph-only AUC (y).
    All five Cresci dataset-graph points clear the lift gate, so coverage is what
    separates them. Within each dataset the points are joined in coverage order,
    and both tracks climb monotonically: the more of the graph propagation reaches,
    the higher the AUC, despite near-identical edge homophily. Cresci-2017 G_B
    covers ~7x the accounts of its G_T (0.5362 / 0.0759) and gains +0.30 AUC.
    Cresci-2017: LCC from the deterministic seed-0 artifact, AUC = 10-seed
    neutral-prior means (multiseed_summary.json)."""
    C17, C15 = _figstyle.DATASET["Cresci-2017"], _figstyle.DATASET["Cresci-2015"]
    halo = _figstyle.halo()
    # per dataset, sorted by coverage: (edge-type tag, coverage, auc, marker)
    c17 = [("retweet+reply", 0.0759, 0.6890, "o"),   # G_T
           (None,           0.5362, 0.9926, "s")]    # G_B
    c15 = [("interaction",   0.2118, 0.7051, "o"),   # G_T
           ("follow",        0.3411, 0.8346, "o"),   # G_T
           (None,            0.9899, 0.9531, "s")]    # G_B

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    _figstyle.clean(ax)
    for grp, col in [(c17, C17), (c15, C15)]:
        xs = [p[1] for p in grp]
        ys = [p[2] for p in grp]
        ax.plot(xs, ys, "-", color=col, lw=1.5, alpha=0.45, zorder=2)
        for _, x, y, m in grp:
            ax.scatter(x, y, s=95, marker=m, color=col, edgecolor=_figstyle.EDGE,
                       linewidths=0.9, zorder=3)

    # edge-type tags for the three G_T points, hand-placed to avoid collisions
    ax.annotate("retweet+reply", (0.0759, 0.6890), fontsize=8, color=_figstyle.TXT,
                ha="left", va="top", xytext=(0.095, 0.668), path_effects=halo)
    ax.annotate("interaction", (0.2118, 0.7051), fontsize=8, color=_figstyle.TXT,
                ha="center", va="bottom", xytext=(0.215, 0.726), path_effects=halo)
    ax.annotate("follow", (0.3411, 0.8346), fontsize=8, color=_figstyle.TXT,
                ha="left", va="center", xytext=(0.368, 0.835), path_effects=halo)

    # headline callout on the Cresci-2017 track (the connector line is the path)
    ax.text(0.31, 0.905, "~7× coverage → +0.30 AUC", fontsize=8.5,
            color=_figstyle.TXT, ha="center", style="italic", path_effects=halo)

    ax.set_xlabel("largest-connected-component fraction (coverage)")
    ax.set_ylabel("graph-only propagation AUC (neutral prior)")
    ax.set_title("All five points clear the lift gate;\n"
                 "coverage decides how much of it propagation can use")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.6, 1.04)
    handles = [
        plt.Line2D([], [], marker="o", ls="-", color=C17, mec=_figstyle.EDGE, label="Cresci-2017"),
        plt.Line2D([], [], marker="o", ls="-", color=C15, mec=_figstyle.EDGE, label="Cresci-2015"),
        plt.Line2D([], [], marker="o", ls="", color="#bdbdbd", mec=_figstyle.EDGE, label="$G_T$ (social)"),
        plt.Line2D([], [], marker="s", ls="", color="#bdbdbd", mec=_figstyle.EDGE, label="$G_B$ (behavioral)"),
    ]
    leg = ax.legend(handles=handles, fontsize=8, loc="lower right")
    leg.get_frame().set_edgecolor("#d0d0d0")
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_coverage_explainer.png", dpi=200)
    plt.close(fig)


def fig6_edge_breakdown() -> None:
    # (dataset/graph, bot-bot, human-human, cross)
    # TwiBot-22 row = full 693,761-node follow graph (M9 aggregate counts).
    rows = [
        ("TwiBot-22\n(follow, full graph)", 22717, 2831761, 446547),
        ("TwiBot-20\n($G_T$)", 4081, 4611, 7657),
        ("Twitter-270k\n($G_T$)", 1802557, 3774823, 1241121),
        ("Cresci-2015\n($G_B$)", 4880, 6099, 1305),
        ("Cresci-2017\n($G_T$)", 2010, 855, 34),
    ]
    names = [r[0] for r in rows]
    arr = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
    frac = arr / arr.sum(axis=1, keepdims=True)
    # Segments encode edge type (not dataset): vermillion bot-bot, blue
    # human-human, neutral grey cross-class.
    C_BB, C_HH, C_CR = _figstyle.DATASET["TwiBot-20"], _figstyle.DATASET["Cresci-2017"], "#bdbdbd"
    fig, ax = plt.subplots(figsize=(8.6, 3.1))
    _figstyle.clean(ax)
    x = np.arange(len(names))
    bb, hh, cr = frac[:, 0], frac[:, 1], frac[:, 2]
    ax.bar(x, bb, label="bot–bot", color=C_BB, edgecolor="white", linewidth=0.7)
    ax.bar(x, hh, bottom=bb, label="human–human", color=C_HH, edgecolor="white", linewidth=0.7)
    ax.bar(x, cr, bottom=bb + hh, label="cross-class", color=C_CR, edgecolor="white", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of labelled edges")
    ax.set_title("Edge composition: heterophilic (TwiBot) → homophilic (Cresci/270k)")
    leg = ax.legend(fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    leg.get_frame().set_edgecolor("#d0d0d0"); leg.get_frame().set_linewidth(0.8)
    for xi in range(len(names)):
        if bb[xi] > 0.03:
            ax.text(xi, bb[xi] / 2, f"{bb[xi]*100:.0f}%", ha="center",
                    fontsize=7.5, color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_edge_breakdown.png", dpi=200,
                bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.close(fig)


def fig8_twitter270k_comparison() -> None:
    """Twitter-270k engine validation: published structural algorithms + our run.

    SybilSCAR is a published algorithm (Wang et al. 2017), NOT a contribution of
    this thesis — here it is the adopted engine, and our *run* of it (0.8298,
    canonical linear rule) sits within the range of published SybilSCAR results
    (0.802 SybilGAT run / 0.861 SybilHP paper), validating the implementation. All
    AUC, same 269,640-node graph.
    Sources: SybilGAT thesis (SybilGAT L2/L4/L8, SybilRank, SybilBelief, SybilSCAR).
    """
    # (method, AUC, tier)  tier in {"our_run", "gnn", "graph"}
    rows = [
        ("SybilBelief", 0.526, "graph"),
        ("SybilRank", 0.738, "graph"),
        ("SybilGAT (L8)", 0.797, "gnn"),
        ("SybilSCAR\n(SybilGAT's run)", 0.802, "graph"),
        ("SybilGAT (L2)", 0.825, "gnn"),
        ("SybilGAT (L4)", 0.849, "gnn"),
        ("SybilSCAR\n(our run)", 0.8298, "our_run"),
    ]
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    aucs = [r[1] for r in rows]
    halo = _figstyle.halo()
    # Colour encodes the method tier; "our run" gets the Twitter-270k hue.
    cmap = {"our_run": _figstyle.DATASET["Twitter-270k"], "gnn": _figstyle.ACCENT,
            "graph": "#bdbdbd"}
    colors = [cmap[r[2]] for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    _figstyle.clean(ax)
    ax.grid(axis="y", visible=False)  # horizontal bars: keep only the value grid
    bars = ax.barh(names, aucs, color=colors, edgecolor=_figstyle.EDGE, linewidth=0.6)
    for b, r in zip(bars, rows):
        if r[2] == "our_run":
            b.set_linewidth(1.6)  # emphasise our run of the engine
    for b, v in zip(bars, aucs):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8, color=_figstyle.TXT, path_effects=halo)
    ax.axvline(0.5, ls=":", c=_figstyle.SPINE, lw=1)
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0.45, 0.92)
    ax.set_title("Twitter-270k engine validation (structure-only, no fusion)",
                 fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=cmap[t], ec=_figstyle.EDGE)
               for t in ("our_run", "gnn", "graph")]
    leg = ax.legend(handles, ["SybilSCAR (our run of the engine)",
                              "GNN (SybilGAT, published)",
                              "classic graph baselines (published)"],
                    fontsize=8, loc="lower right")
    leg.get_frame().set_edgecolor("#d0d0d0"); leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_twitter270k_comparison.png", dpi=200)
    plt.close(fig)


def main() -> int:
    fig8_twitter270k_comparison()
    fig1_lift_spectrum()
    fig2_lift_vs_auc()
    fig3_twibot22_degree_core()
    fig4_cresci_config_bars()
    fig4_cresci_config_bars_split()
    fig5_coverage_explainer()
    fig6_edge_breakdown()
    print("wrote 6 figures to", OUT)
    for p in sorted(OUT.glob("fig*.png")):
        print("  ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
