"""Activity duration feature ``L(u)``.

Definition
----------

For a user observed over a dataset window ``[T_start, T_end]``, with their
first event at ``t_first`` and last event at ``t_last``,

    L(u) = (t_last - t_first) / (T_end - T_start)

Range and interpretation
------------------------

- ``L = 0``  user appears in only a single instant (e.g. one tweet).
- ``L = 1``  user is active across the whole observation window.

Known confound
--------------

``L`` correlates with platform bans: banned accounts have ``t_last`` well
before ``T_end`` and therefore lower ``L``. Section 12.1 of the pivot plan
calls this out explicitly and requires an ablation that runs all experiments
with and without ``L``. The feature is included so that the ablation is
possible; downstream code should never assume ``L`` is causal.

References
----------
- Section 4.5 and 12.1 of ``thesis_writeup/pivot_two_graph_fusion.md``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def activity_duration(
    timestamps: npt.ArrayLike,
    window_start: float,
    window_end: float,
) -> float:
    """Compute the activity-duration feature ``L(u)``.

    Args:
        timestamps: 1-D iterable of event timestamps (Unix seconds or any
            consistent monotonic unit).
        window_start: Lower bound of the observation window. Must be in the
            same units as ``timestamps`` and ``<= window_end``.
        window_end: Upper bound of the observation window.

    Returns:
        ``L(u)`` in ``[0, 1]``. Returns ``float('nan')`` if the window is
        degenerate (``window_end <= window_start``) or if the user has no
        events.

    Notes:
        Values are clipped to ``[0, 1]`` even when a user's earliest or
        latest event falls outside ``[window_start, window_end]``. Such
        out-of-window timestamps usually indicate a dataset extraction bug
        and the clip prevents NaNs from propagating downstream.
    """
    t = np.asarray(timestamps, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"timestamps must be 1-D, got shape {t.shape}")
    if t.size == 0:
        return math.nan

    window = float(window_end) - float(window_start)
    if window <= 0.0:
        return math.nan

    span = float(t.max()) - float(t.min())
    return float(max(0.0, min(1.0, span / window)))
