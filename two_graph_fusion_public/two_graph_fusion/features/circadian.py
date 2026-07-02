"""Circadian z-score and human-likeness mapping ``C_24(u)``.

The circadian feature measures how strongly a user's activity is concentrated
at a 24-hour period, relative to a uniform-within-day null model.

Definition
----------

Let ``H[0..23]`` be the histogram of event hour-of-day counts for user ``u``
(``H[k]`` = number of events whose ``created_at`` falls in hour ``k`` UTC).
The 24-hour spectral power is the squared magnitude of the first non-DC bin
of the DFT of ``H``:

    P_24(u) = |DFT(H)[1]|^2

Under the null hypothesis "events are placed uniformly across the day"
(no diurnal preference), we approximate the distribution of ``P_24`` by
permutation: draw ``N`` event hours uniformly from ``[0, 24)``, build a
histogram, recompute the 24-hour power, repeat ``N_perm`` times. Then

    C_24(u) = (P_24(u) - mean(null)) / std(null)

Human-likeness mapping
----------------------

Section 4.3 of the pivot plan describes a Gaussian-centred remapping that
peaks at moderate values. Milestone-1 results (see ``RESULTS.md`` RED-3)
showed that on TwiBot-22 the Gaussian is the wrong functional form: bots
concentrate on the *lower* side of the honest median, not symmetrically
around it. Two mappings are therefore implemented:

- ``"sigmoid"`` (default): monotone increasing in ``C_24``. High z scores
  near 1.0, low z near 0.0. Centred at ``target_z`` with steepness
  ``1 / sigma_z``.
- ``"gaussian"``: the original section-4.3 Gaussian. Kept for sensitivity
  analyses and reproducibility of the spec.

Both mappings use the same ``(target_z, sigma_z)`` calibration pair, which
should be derived from labeled-honest users via
:func:`two_graph_fusion.features.calibration.calibrate_circadian_humanlike`.
The defaults (``target_z=2.0``, ``sigma_z=1.0``) are only intended to keep
the pipeline runnable on unlabelled inputs.

Hyperparameters
---------------

- ``n_perm``: number of permutations for the shuffle null. Default ``100``.
  Cosmetic; the z-score is stable past ~50 permutations.
- ``target_z``, ``sigma_z``: parameters of the chosen mapping (see above).
- ``mapping``: ``"sigmoid"`` or ``"gaussian"``. Default ``"sigmoid"``.

References
----------
- Section 4.3 of ``thesis_writeup/pivot_two_graph_fusion.md``.
- ``two_graph_fusion/RESULTS.md`` RED-3 for the empirical evidence that the
  Gaussian is the wrong default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

HOURS_PER_DAY = 24

HumanlikeMapping = Literal["sigmoid", "gaussian"]


@dataclass(frozen=True)
class CircadianConfig:
    """Configuration for the circadian feature.

    Attributes:
        n_perm: Number of permutations to draw for the shuffle null.
        target_z: Centre of the human-likeness mapping.
        sigma_z: Width / scale of the human-likeness mapping.
        mapping: Which functional form to use. ``"sigmoid"`` is monotone
            and is the default; ``"gaussian"`` is the section-4.3 form
            kept for sensitivity analyses.
        seed: Optional seed for reproducible permutations.
    """

    n_perm: int = 100
    target_z: float = 2.0
    sigma_z: float = 1.0
    mapping: HumanlikeMapping = "sigmoid"
    seed: int | None = None


def _hour_of_day(timestamps: np.ndarray) -> np.ndarray:
    """Convert Unix-second timestamps to integer hours-of-day in [0, 23].

    Uses UTC so the feature is comparable across users in different
    timezones; if a per-user local timezone is available, the caller should
    convert before invoking this function.
    """
    return np.mod(np.floor_divide(timestamps.astype(np.int64), 3600), HOURS_PER_DAY)


def _power_24h(hist: np.ndarray) -> float:
    """Spectral power at the 24-hour period (first non-DC DFT bin)."""
    dft = np.fft.rfft(hist)
    return float(np.abs(dft[1]) ** 2)


def circadian_zscore(
    timestamps: npt.ArrayLike,
    config: CircadianConfig = CircadianConfig(),
) -> float:
    """Compute the circadian z-score ``C_24(u)``.

    Args:
        timestamps: 1-D iterable of Unix-second event timestamps.
        config: Hyperparameter bundle. See ``CircadianConfig``.

    Returns:
        Real-valued z-score. A larger value indicates stronger 24-hour
        rhythmicity than chance. Returns ``float('nan')`` if there are fewer
        than ``HOURS_PER_DAY`` events (the spectrum is unreliable below this
        threshold).
    """
    t = np.asarray(timestamps, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"timestamps must be 1-D, got shape {t.shape}")
    n = t.size
    if n < HOURS_PER_DAY:
        return math.nan

    obs_hours = _hour_of_day(t)
    obs_hist = np.bincount(obs_hours, minlength=HOURS_PER_DAY).astype(np.float64)
    p_obs = _power_24h(obs_hist)

    rng = np.random.default_rng(config.seed)
    null_powers = np.empty(config.n_perm, dtype=np.float64)
    for i in range(config.n_perm):
        sampled = rng.integers(0, HOURS_PER_DAY, size=n)
        hist = np.bincount(sampled, minlength=HOURS_PER_DAY).astype(np.float64)
        null_powers[i] = _power_24h(hist)

    mu = float(null_powers.mean())
    sigma = float(null_powers.std(ddof=0))
    if sigma <= 0.0:
        return math.nan
    return (p_obs - mu) / sigma


def humanlike(z: float, config: CircadianConfig = CircadianConfig()) -> float:
    """Map ``C_24`` z-score to a ``[0, 1]`` human-likeness score.

    Dispatches to either the sigmoid (default, monotone) or Gaussian
    (section-4.3 spec, symmetric) form based on ``config.mapping``.

    Args:
        z: ``C_24(u)`` from :func:`circadian_zscore`.
        config: Configuration bundle. ``mapping`` selects the form;
            ``target_z`` and ``sigma_z`` parameterise it.

    Returns:
        Value in ``[0, 1]``. ``NaN`` propagates.
    """
    if math.isnan(z):
        return math.nan
    if config.mapping == "sigmoid":
        return _humanlike_sigmoid(z, config.target_z, config.sigma_z)
    if config.mapping == "gaussian":
        return _humanlike_gaussian(z, config.target_z, config.sigma_z)
    raise ValueError(f"unknown circadian mapping: {config.mapping!r}")


def _humanlike_sigmoid(z: float, target_z: float, sigma_z: float) -> float:
    """Monotone increasing sigmoid centred at ``target_z``.

    ``humanlike(target_z) = 0.5``. Higher z maps closer to 1.0, lower z
    closer to 0.0. ``sigma_z`` controls the slope (smaller -> sharper).

    Uses the numerically stable branch of the logistic so that very large
    or very small ``z - target_z`` values saturate at ``1.0`` / ``0.0``
    rather than overflowing :func:`math.exp`.
    """
    scale = max(sigma_z, 1e-9)
    x = (z - target_z) / scale
    if x >= 0.0:
        return float(1.0 / (1.0 + math.exp(-x))) if x < 700.0 else 1.0
    e = math.exp(x) if x > -700.0 else 0.0
    return float(e / (1.0 + e))


def _humanlike_gaussian(z: float, target_z: float, sigma_z: float) -> float:
    """Symmetric Gaussian centred at ``target_z`` (the section-4.3 form)."""
    scale = max(sigma_z, 1e-9)
    return float(math.exp(-((z - target_z) ** 2) / (2.0 * scale ** 2)))
