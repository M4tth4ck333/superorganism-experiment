"""IAT shape-divergence feature ``D_shape(u)``.

The shape-divergence feature compares the user's IAT histogram against a
population reference IAT histogram. It penalises users whose IAT shape
differs from the reference (in particular, attackers using a uniform or
exponential IAT generator that doesn't match the human reference) while
ignoring the *level* of activity, which other features already capture.

Definition
----------

Given a user's IAT pmf ``p_u`` and a reference pmf ``p_ref`` over the same
bin edges,

    m  = 0.5 * (p_u + p_ref)
    JS = 0.5 * KL(p_u || m) + 0.5 * KL(p_ref || m)
    D_shape(u) = 1 - JS / log(2)

``JS`` is the Jensen-Shannon divergence with natural-log base. The maximum
``JS`` between any two distributions equals ``log(2)``, so ``D_shape`` lies
in ``[0, 1]``: ``1`` for identical shapes, ``0`` for maximum divergence.

Binning
-------

The pivot plan calls for Freedman-Diaconis bin edges on the *pooled*
reference IAT distribution (parameter-free):

    h_FD = 2 * IQR(tau_ref) / n_ref^(1/3)
    n_bins = ceil((max(tau_ref) - min(tau_ref)) / h_FD)

To keep the discretisation defensible when reference IATs span many orders
of magnitude (a typical situation for tweet IATs), we always bin in
``log10(tau + 1)`` space. The bin edges are computed once from the reference
and re-used for every user; this guarantees ``p_u`` and ``p_ref`` share a
common support.

References
----------
- Lin, J. (1991). "Divergence measures based on the Shannon entropy."
- Freedman, D. and Diaconis, P. (1981). "On the histogram as a density
  estimator: L_2 theory."
- Section 4.4 of ``thesis_writeup/pivot_two_graph_fusion.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

LOG2 = math.log(2.0)


@dataclass(frozen=True)
class ShapeBins:
    """Pre-computed bin edges and reference pmf for ``D_shape``.

    Attributes:
        edges: ``(n_bins + 1,)`` monotonically increasing array of bin edges
            on the ``log10(tau + 1)`` scale.
        p_ref: ``(n_bins,)`` reference pmf, already normalised and floored
            with a small epsilon so KL is well defined.
    """

    edges: np.ndarray
    p_ref: np.ndarray


def _freedman_diaconis_bins(values: np.ndarray) -> np.ndarray:
    """Return Freedman-Diaconis bin edges, with a safety fallback.

    Falls back to 20 equal-width bins if FD produces a degenerate or empty
    range (e.g. all reference IATs equal the same value).
    """
    if values.size < 2:
        raise ValueError("Need at least two reference values to bin")

    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    n = values.size
    h = 2.0 * iqr / (n ** (1.0 / 3.0)) if iqr > 0 else 0.0

    lo, hi = float(values.min()), float(values.max())
    if h <= 0.0 or not math.isfinite(h) or hi <= lo:
        n_bins = 20
    else:
        n_bins = max(8, int(math.ceil((hi - lo) / h)))

    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, n_bins + 1)


def _to_log_scale(iats: np.ndarray) -> np.ndarray:
    """Map raw IATs (seconds) to ``log10(tau + 1)``."""
    return np.log10(np.maximum(iats, 0.0) + 1.0)


def fit_reference_bins(
    reference_iats: npt.ArrayLike,
    epsilon: float = 1e-9,
) -> ShapeBins:
    """Fit bin edges and the reference pmf from a pooled IAT corpus.

    Args:
        reference_iats: 1-D iterable of IATs (seconds) pooled across the
            reference population (e.g. labeled honest users or the entire
            unlabelled subset).
        epsilon: Small positive floor added to every reference bin so the KL
            divergence stays finite. Choose ``epsilon << 1 / n_bins``.

    Returns:
        ``ShapeBins`` capturing the edges and reference pmf.

    Raises:
        ValueError: If fewer than ten reference IATs are provided. Below
            that threshold FD is meaningless.
    """
    ref = np.asarray(reference_iats, dtype=np.float64)
    if ref.ndim != 1:
        raise ValueError(f"reference_iats must be 1-D, got shape {ref.shape}")
    if ref.size < 10:
        raise ValueError(
            f"Need at least 10 reference IATs to fit shape bins, got {ref.size}"
        )

    log_ref = _to_log_scale(ref)
    edges = _freedman_diaconis_bins(log_ref)
    counts, _ = np.histogram(log_ref, bins=edges)
    p_ref = counts.astype(np.float64) + epsilon
    p_ref /= p_ref.sum()
    return ShapeBins(edges=edges, p_ref=p_ref)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with both ``p`` and ``q`` expected to be strictly positive."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def shape_divergence(
    iats: npt.ArrayLike,
    bins: ShapeBins,
    epsilon: float = 1e-9,
) -> float:
    """Compute ``D_shape(u) = 1 - JS(p_u || p_ref) / log(2)``.

    Args:
        iats: 1-D iterable of user IATs (seconds).
        bins: Pre-fitted reference bins from :func:`fit_reference_bins`.
        epsilon: Small positive floor on ``p_u``; should match the value
            passed to :func:`fit_reference_bins`.

    Returns:
        ``D_shape`` in ``[0, 1]``. ``1`` means user matches the reference;
        ``0`` means maximally divergent. Returns ``float('nan')`` if the
        user has fewer than 5 IATs (not enough samples to estimate a shape).
    """
    tau = np.asarray(iats, dtype=np.float64)
    if tau.ndim != 1:
        raise ValueError(f"iats must be 1-D, got shape {tau.shape}")
    if tau.size < 5:
        return math.nan

    log_tau = _to_log_scale(tau)
    counts, _ = np.histogram(log_tau, bins=bins.edges)
    if counts.sum() == 0:
        return math.nan
    p_u = counts.astype(np.float64) + epsilon
    p_u /= p_u.sum()

    m = 0.5 * (p_u + bins.p_ref)
    js = 0.5 * _kl(p_u, m) + 0.5 * _kl(bins.p_ref, m)
    return float(max(0.0, min(1.0, 1.0 - js / LOG2)))
