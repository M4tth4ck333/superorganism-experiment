"""Five platform-agnostic behavioral features from section 4 of the pivot plan.

Each feature is implemented in its own module so that the math stays close
to the documentation and so the unit tests can target a single feature at a
time. ``pipeline.compute_behavioral_features`` is the public entry point.
"""

from two_graph_fusion.features.activity_duration import activity_duration
from two_graph_fusion.features.analysis import (
    FeatureSeparation,
    feature_separation,
    feature_separation_table,
)
from two_graph_fusion.features.burstiness import burstiness
from two_graph_fusion.features.calibration import (
    calibrate_circadian_humanlike,
    refit_d_shape_with_reference,
)
from two_graph_fusion.features.circadian import (
    CircadianConfig,
    circadian_zscore,
    humanlike,
)
from two_graph_fusion.features.memory import memory_coefficient
from two_graph_fusion.features.pipeline import (
    BehavioralFeatures,
    PipelineConfig,
    RAW_FEATURE_COLUMNS,
    ZSCORE_FEATURE_COLUMNS,
    compute_behavioral_features,
)
from two_graph_fusion.features.shape_divergence import (
    ShapeBins,
    fit_reference_bins,
    shape_divergence,
)

__all__ = [
    "BehavioralFeatures",
    "CircadianConfig",
    "FeatureSeparation",
    "PipelineConfig",
    "RAW_FEATURE_COLUMNS",
    "ZSCORE_FEATURE_COLUMNS",
    "ShapeBins",
    "activity_duration",
    "burstiness",
    "calibrate_circadian_humanlike",
    "circadian_zscore",
    "compute_behavioral_features",
    "feature_separation",
    "feature_separation_table",
    "fit_reference_bins",
    "humanlike",
    "memory_coefficient",
    "refit_d_shape_with_reference",
    "shape_divergence",
]
