"""Multi-relational SybilSCAR propagation.

Provides two update rules, both in single- and multi-relational form
(the multi-relational form — variant B from section 6.2 of the pivot
plan — just adds one extra ``2 * w_g * sum_{v in N_g(u)} (p^t(v) - 0.5)``
term per additional relation):

1. ``linearized=True`` — the **canonical, published SybilSCAR-D local
   rule** (Wang, Jia, Gong, TIFS 2018), for faithful reproduction and
   apples-to-apples baselines::

       p^{t+1}(u) = clip( q(u)
                          + sum_g 2 * w_g * sum_{v in N_g(u)} (p^t(v) - 0.5),
                          0, 1 )

   Here ``q(u)`` is the **prior probability** of being honest (0.5 for
   unlabelled, ``pi_b`` for an LR prior, ``seed_lo``/``seed_hi`` for
   seeds) and enters additively; the posterior is clamped to ``[0, 1]``.
   This is linear and matches the paper, but **diverges** (needs clamping
   and a very small ``w``) on graphs with very high-degree hubs.

2. ``linearized=False`` (default) — a **logistic / sigmoid-stabilised
   variant** of the same rule::

       p^{t+1}(u) = sigmoid(  q(u)
                            + sum_g 2 * w_g * sum_{v in N_g(u)} (p^t(v) - 0.5) )

   where ``q(u) = logit(prior(u))`` is the log-odds of the prior. The
   sigmoid acts as a soft clamp so the iteration is unconditionally
   bounded in ``(0, 1)`` — robust on hub-heavy graphs (e.g. Twitter-270k,
   max degree ~22.8k) where the linear rule blows up. This is **not**
   bit-for-bit the published rule; use ``linearized=True`` to reproduce
   published numbers.

In both rules ``p^t(u)`` is the posterior probability that ``u`` is
**honest**; we follow Wang et al.'s convention and compute AUC against
the bot label as ``1 - p(u)``. ``w_g`` is the per-relation homophily
strength; ``N_g(u)`` is ``u``'s neighbourhood in relation ``g``.

Convergence is checked by ``max-abs`` change between consecutive
iterations. Complexity ``O((|E_1| + ... + |E_R|) * T)`` either way.

Notes on numerical stability
----------------------------

- Priors are clipped to ``[eps, 1 - eps]`` (``eps = 1e-6``) before
  ``logit`` (logistic rule) to avoid infinities.
- The sigmoid is applied via :func:`scipy.special.expit`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Hashable

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.special import expit, logit

logger = logging.getLogger(__name__)

DEFAULT_SEED_LO: float = 0.02
DEFAULT_SEED_HI: float = 0.98
DEFAULT_W: float = 0.5
DEFAULT_MAX_ITERS: int = 50
DEFAULT_TOL: float = 1e-5
_EPS: float = 1e-6


@dataclass(frozen=True)
class Relation:
    """One graph layer in the multi-relational SybilSCAR formulation.

    Attributes:
        name: Human-readable label (e.g. ``"trust"`` or ``"behavioral"``).
        adjacency: Symmetric sparse adjacency matrix in CSR form.
        weight: ``w_g`` in the update rule.
    """

    name: str
    adjacency: sparse.csr_matrix
    weight: float = DEFAULT_W


@dataclass(frozen=True)
class PropagationResult:
    """Output bundle from :func:`run_propagation`.

    Attributes:
        node_order: Ordering used internally for the adjacency matrices
            and the returned probability vector.
        p_honest: Posterior ``p(u)`` (probability of being honest) at
            convergence, indexed by ``node_order``.
        n_iters: Number of iterations actually run.
        max_delta: Final ``max_u |p^{t+1}(u) - p^t(u)|``.
        converged: True iff ``max_delta < tol`` before ``max_iters``.
    """

    node_order: list[Hashable]
    p_honest: np.ndarray
    n_iters: int
    max_delta: float
    converged: bool


def _adjacency_for_graph(
    graph: nx.Graph,
    node_order: list[Hashable],
) -> sparse.csr_matrix:
    """Return the symmetric adjacency for ``graph`` over ``node_order``.

    Edge weights are ignored (SybilSCAR operates on binary
    adjacency); use a weighted variant via the ``Relation.weight`` knob.
    """
    n = len(node_order)
    idx = {nd: i for i, nd in enumerate(node_order)}
    rows: list[int] = []
    cols: list[int] = []
    for u, v in graph.edges():
        i = idx.get(u)
        j = idx.get(v)
        if i is None or j is None or i == j:
            continue
        rows.append(i)
        cols.append(j)
        rows.append(j)
        cols.append(i)
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def relation_from_graph(
    name: str,
    graph: nx.Graph,
    node_order: list[Hashable],
    weight: float = DEFAULT_W,
) -> Relation:
    """Build a :class:`Relation` from a NetworkX graph + a fixed node order."""
    return Relation(
        name=name,
        adjacency=_adjacency_for_graph(graph, node_order),
        weight=weight,
    )


def _prepare_q_and_init(
    node_order: list[Hashable],
    priors: dict[Hashable, float],
    default_prior: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(q, p0)`` arrays for the propagation.

    Args:
        node_order: Node ordering.
        priors: Per-node prior probability of being honest. Missing
            entries default to ``default_prior``.
        default_prior: Fallback prior for nodes not in ``priors``.

    Returns:
        ``q`` (the log-odds of the prior) and ``p0`` (the initial
        probability vector), both shape ``(n,)``.
    """
    n = len(node_order)
    p0 = np.full(n, float(default_prior), dtype=np.float64)
    for i, nd in enumerate(node_order):
        if nd in priors:
            p0[i] = float(priors[nd])
    p0 = np.clip(p0, _EPS, 1.0 - _EPS)
    q = logit(p0)
    return q, p0.copy()


def run_propagation(
    relations: Sequence[Relation],
    node_order: list[Hashable],
    priors: dict[Hashable, float],
    default_prior: float = 0.5,
    max_iters: int = DEFAULT_MAX_ITERS,
    tol: float = DEFAULT_TOL,
    linearized: bool = False,
) -> PropagationResult:
    """Run multi-relational SybilSCAR until convergence.

    Args:
        relations: One or more :class:`Relation`. With a single
            relation this is identical to single-graph SybilSCAR.
        node_order: Node ordering used by every relation's adjacency
            matrix and by the returned ``p_honest``.
        priors: Per-node prior ``P(honest)``. Nodes not listed default
            to ``default_prior``.
        default_prior: Fallback prior for unlabeled nodes
            (typically ``0.5``).
        max_iters: Upper bound on iteration count.
        tol: Convergence tolerance on ``max_u |p^{t+1} - p^t|``.
        linearized: If ``True``, use the canonical published SybilSCAR-D
            rule — prior enters additively as a probability and the
            posterior is clamped to ``[0, 1]`` each iteration. If
            ``False`` (default), use the logistic/sigmoid-stabilised
            variant (prior as log-odds, sigmoid squash). See the module
            docstring for both forms.

    Returns:
        :class:`PropagationResult`.

    Raises:
        ValueError: If any relation's adjacency does not match
            ``len(node_order)``.
    """
    if not relations:
        raise ValueError("at least one relation is required")
    n = len(node_order)
    for r in relations:
        if r.adjacency.shape != (n, n):
            raise ValueError(
                f"relation {r.name!r} adjacency shape {r.adjacency.shape} "
                f"!= ({n}, {n})"
            )

    q_logit, p = _prepare_q_and_init(node_order, priors, default_prior)
    # Canonical rule keeps the prior as a probability (additive); the
    # logistic rule keeps it as log-odds.
    q = p.copy() if linearized else q_logit
    p_prev = p.copy()

    max_delta = float("inf")
    converged = False
    for it in range(1, max_iters + 1):
        update = q.copy()
        for r in relations:
            # adjacency @ (p - 0.5) returns sum_{v in N(u)} (p_v - 0.5)
            update += 2.0 * r.weight * (r.adjacency @ (p_prev - 0.5))
        p = np.clip(update, 0.0, 1.0) if linearized else expit(update)
        max_delta = float(np.max(np.abs(p - p_prev)))
        if max_delta < tol:
            converged = True
            logger.info(
                "propagation converged at iter %d (max_delta=%.3e)", it, max_delta
            )
            return PropagationResult(
                node_order=list(node_order),
                p_honest=p,
                n_iters=it,
                max_delta=max_delta,
                converged=True,
            )
        p_prev = p

    logger.info(
        "propagation hit max_iters=%d (max_delta=%.3e)", max_iters, max_delta
    )
    return PropagationResult(
        node_order=list(node_order),
        p_honest=p,
        n_iters=max_iters,
        max_delta=max_delta,
        converged=False,
    )


def seed_priors_from_labels(
    labels: dict[Hashable, str],
    honest_label: str,
    bot_label: str,
    seed_lo: float = DEFAULT_SEED_LO,
    seed_hi: float = DEFAULT_SEED_HI,
) -> dict[Hashable, float]:
    """Convert a node -> label dict into a node -> prior dict.

    Honest labels become ``seed_hi`` (defaults 0.98), bot labels become
    ``seed_lo`` (defaults 0.02). Nodes with neither label are omitted.
    """
    out: dict[Hashable, float] = {}
    for nd, lab in labels.items():
        if lab == honest_label:
            out[nd] = float(seed_hi)
        elif lab == bot_label:
            out[nd] = float(seed_lo)
    return out


def evaluation_priors(
    labels: dict[Hashable, str],
    honest_label: str,
    bot_label: str,
    pi_b_honest: dict[Hashable, float],
    seed_lo: float = DEFAULT_SEED_LO,
    seed_hi: float = DEFAULT_SEED_HI,
    fallback: float = 0.5,
) -> dict[Hashable, float]:
    """Build per-node priors for an evaluation run.

    Seeds (whose labels are 'visible' to the propagation) get
    ``seed_lo``/``seed_hi``. Eval nodes (labels withheld) get
    ``pi_b_honest[u]`` if available else ``fallback``.

    Args:
        labels: Visible labels for seed nodes only. Eval nodes should
            not appear here.
        honest_label: String used for the honest class.
        bot_label: String used for the bot class.
        pi_b_honest: Predicted honest probability from a side
            classifier (e.g. the LR prior). Indexed by node id.
        seed_lo: Prior for bot seeds.
        seed_hi: Prior for honest seeds.
        fallback: Fallback prior when ``pi_b_honest`` is missing for a
            node.

    Returns:
        Per-node prior dict consumable by :func:`run_propagation`.
    """
    out: dict[Hashable, float] = {}
    for nd, lab in labels.items():
        if lab == honest_label:
            out[nd] = float(seed_hi)
        elif lab == bot_label:
            out[nd] = float(seed_lo)
    for nd, pi in pi_b_honest.items():
        if nd in out:
            continue
        out[nd] = float(pi)
    return out


def p_honest_to_dict(result: PropagationResult) -> dict[Hashable, float]:
    """Convenience: zip ``result.node_order`` and ``result.p_honest`` into a dict."""
    return {nd: float(p) for nd, p in zip(result.node_order, result.p_honest)}
