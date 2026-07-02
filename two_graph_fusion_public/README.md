# Two-Graph Fusion for Sybil Detection

Reference implementation accompanying the thesis. The method fuses a **trust
graph** `G_T` (the platform follow or interaction graph) with a **behavioural
graph** `G_B` (a mutual k-NN graph over five platform-agnostic behavioural
features) and runs belief propagation (SybilSCAR and SybilHP variants) over the
combined structure, optionally seeded with a logistic-regression prior.

This repository contains the complete code to reproduce every table and figure
in the thesis across four labelled datasets (Cresci-2015, Cresci-2017,
TwiBot-20, TwiBot-22) plus the Twitter-270k engine-validation benchmark.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`numpy`, `pandas`, `scipy`, `scikit-learn`, `networkx` and `matplotlib` cover
every dataset except the TwiBot-22 at-scale pipeline, which additionally needs
`ijson`, `pyarrow` and `faiss-cpu` (all listed in `requirements.txt`).

All commands below are run from the repository root and invoke modules with
`python -m two_graph_fusion.scripts.<name>`.

## Repository layout

```
two_graph_fusion/
  features/       five behavioural features + pooling pipeline
  graphs/         G_B (mutual k-NN), G_T (follow / interaction), F6 overlap diagnostic
  datasets/       loaders for Cresci-2015/2017, TwiBot-20/22, Twitter-270k, samplers
  propagation/    SybilSCAR + SybilHP engines, LR prior, train/val/test harness
  analysis/       F5 residual / scatter plotting
  scripts/        command-line drivers (one per dataset + figures + diagnostics)
  tests/          unit tests
```

## Datasets

The raw datasets are third-party and are **not** redistributed here. Download
each from its original source and place it at the path the runners expect (or
override with the `--root` / `--cresci-root` / `--twibot22-root` flag):

| Dataset       | Expected path (default)              | Source |
| ------------- | ------------------------------------ | ------ |
| Cresci-2015   | `cresci-15-orig/`                    | Cresci et al., "Fame for Sale" release |
| Cresci-2017   | `cresci_2017_datasets_full/`         | Cresci et al. 2017 release |
| TwiBot-20     | `Twibot-20-2/`                       | Feng et al., TwiBot-20 |
| TwiBot-22     | `twibot22/`                          | Feng et al., TwiBot-22 (`label.csv`, `split.csv`, `edge.csv`, `tweet_0..8.json`) |
| Twitter-270k  | pass `--root <path>`                 | SybilHP / SybilGAT benchmark (figshare) |

## Reproducing the results

### Cresci-2017 and Cresci-2015 (behavioural `G_B` available)

Single-seed sweep over `{G_T, G_B, fusion} x {neutral, LR prior}`, both engines,
hyperparameters tuned on validation, metrics reported on test:

```bash
python -m two_graph_fusion.scripts.run_cresci2017_tvt \
    --cresci-root cresci_2017_datasets_full --seed 0

python -m two_graph_fusion.scripts.run_cresci2015_tvt \
    --cresci-root cresci-15-orig --gt follow --seed 0
```

### Multi-seed aggregation (thesis Tables 5/6 and appendix A2/A6)

Run each Cresci sweep for seeds 0 through 9, then aggregate:

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  python -m two_graph_fusion.scripts.run_cresci2017_tvt --seed $s
  python -m two_graph_fusion.scripts.run_cresci2015_tvt --gt follow --seed $s
done

python -m two_graph_fusion.scripts.aggregate_multiseed \
    --output two_graph_fusion/cache/multiseed_summary.json
```

Aggregation reads the per-seed `results_*_seed{0..9}.json` files written by the
runners into `two_graph_fusion/cache/cresci{2015,2017}_tvt/` and reports mean,
standard deviation and paired-bootstrap confidence intervals per cell.

### TwiBot-20 (follow graph only; no per-tweet timestamps)

```bash
python -m two_graph_fusion.scripts.run_twibot20_eval \
    --root Twibot-20-2 --output-dir two_graph_fusion/cache/twibot20
```

### TwiBot-22 (at scale)

The full pipeline (tweet-shard extraction, feature computation, `G_B` build,
follow-graph assembly, official train/test evaluation, F6 diagnostic) is
resumable via stage markers:

```bash
python -m two_graph_fusion.scripts.run_at_scale \
    --twibot22-root twibot22 \
    --work-dir two_graph_fusion/cache/at_scale \
    --user-scope train_test \
    --stages all
```

The individual TwiBot-22 stages are also runnable on their own for a smaller
subset: `cache_follow_edges`, `build_connected_subset`, `extract_twibot22_subset`,
`compute_features_subset`, `build_gb`, `f6_diagnostic`, `homophily_report`.

### Twitter-270k engine validation and directionality

Structure-only directed follow graph with benign/sybil seed lists (no tweets or
features), used to validate the propagation engines against published SybilHP /
SybilGAT numbers:

```bash
python -m two_graph_fusion.scripts.run_twitter270k --root <path-to-twitter-270k>
```

### Supporting analyses reported in the thesis

```bash
# Interaction-graph G_T experiment
python -m two_graph_fusion.scripts.run_interaction_experiment

# Follow-graph homophily gate (TwiBot-20) and degree-core homophily (TwiBot-22)
python -m two_graph_fusion.scripts.twibot20_gate --root Twibot-20-2
python -m two_graph_fusion.scripts.degree_core_homophily

# Train-only vs all-label homophily lift (appendix A4)
python -m two_graph_fusion.scripts.train_only_lift

# F6 overlap diagnostic for the Cresci datasets (appendix A9)
python -m two_graph_fusion.scripts.f6_cresci --dataset cresci2015
```

## Figures

After the runs above have populated `two_graph_fusion/cache/`:

```bash
python -m two_graph_fusion.scripts.make_thesis_figures   # figures 1-6
python -m two_graph_fusion.scripts.make_roc_pr           # figure 7 (ROC / PR)
python -m two_graph_fusion.scripts.make_feature_tsne     # behavioural t-SNE
```

## Tests

```bash
python -m unittest discover -s two_graph_fusion/tests
```

## Behavioural features

The per-user vector is `f(u) = [B_G, M, C_24, D_shape, L]`, z-scored across the
population in `features.pipeline.compute_behavioral_features`.

| Symbol       | Module                         | Range     |
| ------------ | ------------------------------ | --------- |
| `B_G(u)`     | `features.burstiness`          | `[-1, 1]` |
| `M(u)`       | `features.memory`              | `[-1, 1]` |
| `C_24(u)`    | `features.circadian`           | real      |
| `D_shape(u)` | `features.shape_divergence`    | `[0, 1]`  |
| `L(u)`       | `features.activity_duration`   | `[0, 1]`  |

All timestamps are treated as Unix seconds (UTC); per-user event times are
sorted and de-duplicated before inter-arrival differencing.
