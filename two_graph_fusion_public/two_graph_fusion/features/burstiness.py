"""Burstiness coefficient (Goh & Barabasi 2008).

The burstiness coefficient ``B_G`` summarises the second-vs-first moment ratio
of the inter-arrival-time (IAT) distribution of a user's events.

Definition
----------

Given IATs ``tau_1, ..., tau_{n-1}`` (positive real numbers), let

    mu    = mean(tau)
    sigma = std(tau)            (population standard deviation)

then

    B_G = (sigma - mu) / (sigma + mu)

Range and interpretation
------------------------

- ``B_G = -1``  perfectly periodic (cron-like, sigma = 0).
- ``B_G =  0``  Poisson process (sigma = mu).
- ``B_G = +1``  extremely bursty (sigma >> mu).

There are no hyperparameters.

References
----------
- Goh, K.-I. and Barabasi, A.-L. (2008). "Burstiness and memory in complex
  systems." Europhysics Letters, 81(4): 48002.
- Section 4.1 of ``thesis_writeup/pivot_two_graph_fusion.md``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def burstiness(iats: npt.ArrayLike) -> float:
    """Compute the Goh-Barabasi burstiness coefficient ``B_G``.

    Args:
        iats: 1-D iterable of strictly non-negative inter-arrival times.
            At least two values are required for a meaningful estimate.

    Returns:
        ``B_G`` in ``[-1, 1]``. If fewer than two IATs are provided, or all
        IATs are zero (which would mean simultaneous events), returns
        ``float('nan')`` so callers can decide how to handle missing values.
    """
    tau = np.asarray(iats, dtype=np.float64)
    if tau.ndim != 1:
        raise ValueError(f"iats must be 1-D, got shape {tau.shape}")
    if tau.size < 2:
        return math.nan
    if np.any(tau < 0):
        raise ValueError("IATs must be non-negative")

    mu = float(tau.mean())
    sigma = float(tau.std(ddof=0))
    denom = sigma + mu
    if denom <= 0.0:
        return math.nan
    return (sigma - mu) / denom
