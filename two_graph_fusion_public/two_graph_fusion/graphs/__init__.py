"""Graph construction and diagnostics for the two-graph fusion pivot.

- ``behavioral_graph``: build ``G_B`` from the z-scored feature vector
  via mutual k-NN (section 5 of the pivot plan).
- ``trust_graph``: stream the TwiBot-22 ``edge.csv`` and assemble the
  user-user follow graph ``G_T``.
- ``diagnostics``: F6 overlap statistics (edge / triangle Jaccard,
  spectral cosine) plus standard graph summaries (largest connected
  component, degree distribution).
"""

from two_graph_fusion.graphs.behavioral_graph import (
    BehavioralGraphBuildResult,
    build_mutual_knn_graph,
    build_mutual_knn_graph_auto,
    feature_matrix_for_qualifying_users,
)
from two_graph_fusion.graphs.homophily import (
    HomophilyReport,
    chance_baseline,
    edge_homophily,
    format_homophily_row,
    homophily_from_dataframe,
)
from two_graph_fusion.graphs.diagnostics import (
    F6Report,
    acceptance_verdict,
    degree_summary,
    edge_jaccard,
    f6_report,
    graph_summary,
    largest_connected_component_fraction,
    restrict_to_common_nodes,
    spectral_overlap,
    triangle_jaccard,
)
from two_graph_fusion.graphs.trust_graph import (
    DEFAULT_FOLLOW_RELATIONS,
    TrustGraphBuildResult,
    build_trust_graph,
)
from two_graph_fusion.graphs.interaction_graph import (
    DEFAULT_INTERACTION_RELATIONS,
    InteractionGraphBuildResult,
    build_interaction_gt,
)

__all__ = [
    "HomophilyReport",
    "BehavioralGraphBuildResult",
    "DEFAULT_FOLLOW_RELATIONS",
    "DEFAULT_INTERACTION_RELATIONS",
    "F6Report",
    "InteractionGraphBuildResult",
    "TrustGraphBuildResult",
    "acceptance_verdict",
    "build_interaction_gt",
    "build_mutual_knn_graph",
    "build_mutual_knn_graph_auto",
    "build_trust_graph",
    "degree_summary",
    "edge_jaccard",
    "f6_report",
    "feature_matrix_for_qualifying_users",
    "chance_baseline",
    "edge_homophily",
    "format_homophily_row",
    "graph_summary",
    "homophily_from_dataframe",
    "largest_connected_component_fraction",
    "restrict_to_common_nodes",
    "spectral_overlap",
    "triangle_jaccard",
]
