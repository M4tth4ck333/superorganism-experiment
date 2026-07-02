"""Two-graph fusion for sybil detection.

This package is a clean-room implementation of the pivot described in
``thesis_writeup/pivot_two_graph_fusion.md``. It deliberately does **not**
share code with the existing GitHub-specific ``main.py`` / ``experiments.py``
pipeline.

Layout
------

- ``features``: the five platform-agnostic behavioral features defined in
  section 4 of the pivot plan.
- ``datasets``: dataset-specific loaders. Currently only TwiBot-22.
- ``scripts``: thin command-line drivers that wire the pieces together.
- ``tests``: unit tests for the feature implementations.

The behavioral feature pipeline is the building block on top of which the
behavioral graph ``G_B`` (section 5) and the multi-relational SybilSCAR
algorithm (variant B, section 6.2) are built in later milestones.
"""

from two_graph_fusion.features.pipeline import (
    BehavioralFeatures,
    compute_behavioral_features,
)

__all__ = ["BehavioralFeatures", "compute_behavioral_features"]
