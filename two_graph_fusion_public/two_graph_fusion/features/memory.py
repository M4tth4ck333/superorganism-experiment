"""Memory coefficient (lag-1 IAT autocorrelation).

The memory coefficient ``M`` measures whether long IATs are followed by long
IATs (positive memory) or whether successive IATs are uncorrelated
(``M = 0``).

Definition
----------

For an IAT sequence ``tau_1, ..., tau_{n-1}``,

    M = Pearson_correlation(tau_i, tau_{i+1})   for i = 1, ..., n-2

Range and interpretation
------------------------

- ``M > 0``  long IATs cluster together (human-like bursts and lulls).
- ``M = 0``  Poisson process (no temporal memory).
- ``M < 0``  alternating short/long IATs (a regular bot might look like this
  if it interleaves quick replies with long sleeps).

There are no hyperparameters. Together with ``B_G`` this places a user in the
"Goh-Barabasi (M, B_G) plane".

References
----------
- Goh, K.-I. and Barabasi, A.-L. (2008). "Burstiness and memory in complex
  systems." Europhysics Letters, 81(4): 48002.
- Section 4.2 of ``thesis_writeup/pivot_two_graph_fusion.md``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def memory_coefficient(iats: npt.ArrayLike) -> float:
    """Compute the lag-1 Pearson autocorrelation of an IAT sequence.

    Args:
        iats: 1-D iterable of inter-arrival times.

    Returns:
        ``M`` in ``[-1, 1]``. Returns ``float('nan')`` if there are fewer
        than three IATs (two pairs are required to define correlation) or if
        either branch of the pair has zero variance (making correlation
        undefined).
    """
    tau = np.asarray(iats, dtype=np.float64)
    if tau.ndim != 1:
        raise ValueError(f"iats must be 1-D, got shape {tau.shape}")
    if tau.size < 3:
        return math.nan

    x = tau[:-1]
    y = tau[1:]
    if x.std(ddof=0) == 0.0 or y.std(ddof=0) == 0.0:
        return math.nan

    return float(np.corrcoef(x, y)[0, 1])
