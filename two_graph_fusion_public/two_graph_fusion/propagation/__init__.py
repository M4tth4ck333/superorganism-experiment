"""Propagation pipelines.

- ``sybilscar``: SybilSCAR update rule + multi-relational extension.
- ``sybilhp``: directed adaptive-homophily LBP (SybilHP) + fusion engine.
- ``prior``: supervised LR producing ``pi_b(u)`` on ``f(u)``.
- ``evaluation``: stratified K-fold + official-split harnesses.
- ``tvt_evaluation``: train/val/test harness comparing both engines.
"""

from two_graph_fusion.propagation.evaluation import (
    FoldResult,
    OfficialSplitResult,
    dump_results_json,
    official_result_to_dict,
    run_kfold_evaluation,
    run_official_split_evaluation,
    summarise_with_stderr,
)
from two_graph_fusion.propagation.prior import (
    DEFAULT_FEATURES,
    LRPriorResult,
    fit_predict_lr_prior,
)
from two_graph_fusion.propagation.sybilhp import (
    SybilHPParams,
    SybilHPResult,
    run_sybilhp,
    sybilhp_priors_from_honest,
)
from two_graph_fusion.propagation.sybilscar import (
    DEFAULT_MAX_ITERS,
    DEFAULT_SEED_HI,
    DEFAULT_SEED_LO,
    DEFAULT_TOL,
    DEFAULT_W,
    PropagationResult,
    Relation,
    evaluation_priors,
    p_honest_to_dict,
    relation_from_graph,
    run_propagation,
    seed_priors_from_labels,
)
from two_graph_fusion.propagation.tvt_evaluation import (
    ConfigResult,
    Splits,
    make_stratified_splits,
    run_tvt_evaluation,
    splits_from_column,
)

__all__ = [
    "ConfigResult",
    "DEFAULT_FEATURES",
    "DEFAULT_MAX_ITERS",
    "DEFAULT_SEED_HI",
    "DEFAULT_SEED_LO",
    "DEFAULT_TOL",
    "DEFAULT_W",
    "FoldResult",
    "OfficialSplitResult",
    "LRPriorResult",
    "PropagationResult",
    "Relation",
    "Splits",
    "SybilHPParams",
    "SybilHPResult",
    "dump_results_json",
    "evaluation_priors",
    "official_result_to_dict",
    "fit_predict_lr_prior",
    "make_stratified_splits",
    "p_honest_to_dict",
    "relation_from_graph",
    "run_kfold_evaluation",
    "run_official_split_evaluation",
    "run_propagation",
    "run_sybilhp",
    "run_tvt_evaluation",
    "seed_priors_from_labels",
    "splits_from_column",
    "summarise_with_stderr",
    "sybilhp_priors_from_honest",
]
