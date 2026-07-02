"""SybilHP: directed adaptive-homophily Loopy Belief Propagation.

Faithful re-implementation of SybilHP (Lu, Gong, Li, Liu, Liu — *Appl.
Sci.* 2023, 13, 5341) and its extension to multi-layer fusion with a
behavioral similarity graph.

Where :mod:`two_graph_fusion.propagation.sybilscar` uses the *linearised*
SybilSCAR local rule, SybilHP runs the full binary sum-product LBP with
**direction-sensitive edge potentials** and two **adaptive homophily
estimators** that are refreshed every iteration.  This is the engine we
use as both our strongest lightweight-graph baseline and the propagation
core we extend with the behavioral layer (see
``thesis_writeup/milestones_M10_M14.md``).

State convention
----------------

Each node ``u`` carries a binary label ``x_u in {-1, +1}`` with
``-1 = benign`` and ``+1 = sybil`` (the paper's convention).  Internally
state ``0`` indexes benign and state ``1`` indexes sybil.  Priors are
supplied as ``q_u = P(x_u = sybil)``; the returned posterior is
``p_honest(u) = P(x_u = benign)`` to match the SybilSCAR scoring helpers.

Edge potentials (Lu et al. Eqs. 10–13)
--------------------------------------

For a message sent from ``s`` (sender) to ``r`` (receiver), the 2×2
potential ``Psi[x_s, x_r]`` depends on the *follow* relationship:

- **Bidirectional** (mutual follow, or any behavioral edge), Eq. 12 —
  symmetric, uses both endpoints' adaptive estimators.
- **Sender is the tail** (``s`` follows ``r``), Eq. 10 — a benign tail's
  homophily is scaled by ``c_s_b`` (its discernment); a sybil tail is
  uncertain (fixed ``ws_tail``).
- **Sender is the head** (``r`` follows ``s``), Eq. 11 — a sybil head's
  homophily is scaled by ``c_s_s`` (its enticement); a benign head is
  uncertain about its followers (fixed ``wb_head``).

Adaptive estimators (Eqs. 7, 9), recomputed each iteration from the
current posteriors:

- ``c_u_b = mean_{v in N_out(u)} P(x_v = benign)`` — fraction of the
  accounts ``u`` follows that look benign (its discernment).
- ``c_u_s = mean_{v in N_in(u)} P(x_v = sybil)`` — fraction of ``u``'s
  followers that look sybil (its enticement, inverted).

Bidirectional neighbours count toward both ``N_in`` and ``N_out``.  Nodes
with an empty neighbourhood default to ``c = 1`` (estimator inert).

Complexity is ``O(iter * |E_messages|)`` like SybilSCAR; everything is
vectorised over the directed message-edge array, so a few-thousand-node
graph converges in well under a second.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Hashable

import numpy as np
from scipy import sparse
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

# Defaults from Lu et al. Table 2 (tuned on their directed Twitter graph).
DEFAULT_W_BI: float = 0.99
DEFAULT_WB_TAIL: float = 0.97
DEFAULT_WS_TAIL: float = 0.77
DEFAULT_WB_HEAD: float = 0.73
DEFAULT_WS_HEAD: float = 0.95
DEFAULT_W_BEHAVIORAL: float = 0.90

DEFAULT_MAX_ITERS: int = 50
DEFAULT_TOL: float = 1e-5
_EPS: float = 1e-9

# Message-edge types.
_BIDIR = 0   # symmetric potential (Eq. 12)
_TAIL = 1    # sender is the tail / follower (Eq. 10)
_HEAD = 2    # sender is the head / followee (Eq. 11)


@dataclass(frozen=True)
class SybilHPParams:
    """Five initial homophily-strength parameters for the trust layer."""

    w_bi: float = DEFAULT_W_BI          # bidirectional (mutual follow)
    wb_tail: float = DEFAULT_WB_TAIL    # benign tail / follower
    ws_tail: float = DEFAULT_WS_TAIL    # sybil tail / follower
    wb_head: float = DEFAULT_WB_HEAD    # benign head / followee
    ws_head: float = DEFAULT_WS_HEAD    # sybil head / followee


@dataclass(frozen=True)
class SybilHPResult:
    """Output bundle from :func:`run_sybilhp`."""

    node_order: list[Hashable]
    p_honest: np.ndarray         # P(benign) per node, indexed by node_order
    n_iters: int
    max_delta: float
    converged: bool


# ---------------------------------------------------------------------------
# Message-edge graph construction
# ---------------------------------------------------------------------------

@dataclass
class _MessageGraph:
    """Internal directed message-edge representation."""

    n: int
    snd: np.ndarray              # (E,) sender node index
    rcv: np.ndarray              # (E,) receiver node index
    rev: np.ndarray              # (E,) index of the reverse message edge
    etype: np.ndarray            # (E,) one of _BIDIR/_TAIL/_HEAD
    w_bi: np.ndarray             # (E,) bidirectional homophily (used when etype==_BIDIR)
    wb_tail: np.ndarray          # (E,) used when etype==_TAIL
    ws_tail: np.ndarray
    wb_head: np.ndarray          # (E,) used when etype==_HEAD
    ws_head: np.ndarray
    a_out: sparse.csr_matrix     # (n,n) 0/1 out-neighbour adjacency for c_b
    a_in: sparse.csr_matrix      # (n,n) 0/1 in-neighbour adjacency for c_s


def _build_message_graph(
    node_order: list[Hashable],
    trust_edges: Sequence[tuple[Hashable, Hashable]],
    behavioral_edges: Sequence[tuple[Hashable, Hashable]],
    params: SybilHPParams,
    w_behavioral: float,
) -> _MessageGraph:
    """Assemble the directed message-edge arrays from the two layers.

    Each undirected graph relationship contributes a *mutually-reverse
    pair* of directed message edges (LBP passes messages both ways). A
    pair carrying both a trust edge and a behavioral edge yields two
    independent factor pairs (4 message edges) — the correct LBP
    treatment of two factors over the same nodes.
    """
    n = len(node_order)

    # Map each layer's (hashable, hashable) edges to deduplicated directed
    # index arrays. The one Python pass per layer is simple int assignment
    # (no set/list-of-tuple churn), so it scales to millions of edges.
    idx = {nd: i for i, nd in enumerate(node_order)}

    def _to_directed(edges: Sequence[tuple[Hashable, Hashable]]) -> tuple[np.ndarray, np.ndarray]:
        if not edges:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        u = np.empty(len(edges), np.int64)
        v = np.empty(len(edges), np.int64)
        k = 0
        for a, b in edges:
            ia = idx.get(a)
            ib = idx.get(b)
            if ia is None or ib is None or ia == ib:
                continue
            u[k] = ia
            v[k] = ib
            k += 1
        if k == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        key = np.unique(u[:k] * np.int64(n) + v[:k])   # dedupe directed edges
        return key // n, key % n

    tu, tv = _to_directed(trust_edges)        # directed trust edges
    bu, bv = _to_directed(behavioral_edges)   # directed behavioral edges (symmetrised below)

    # ---- Trust layer: classify each undirected pair by reciprocity ----
    ca = cb = np.empty(0, np.int64)
    et_ab = et_ba = np.empty(0, np.int64)
    if tu.size:
        a = np.minimum(tu, tv)
        b = np.maximum(tu, tv)
        ckey = np.unique(a * np.int64(n) + b)          # unique undirected pairs (a<b)
        ca, cb = ckey // n, ckey % n
        dset = tu * np.int64(n) + tv                    # directed edge keys (already unique)
        ab = np.isin(ca * np.int64(n) + cb, dset)       # a->b present
        ba = np.isin(cb * np.int64(n) + ca, dset)       # b->a present
        mutual, only_ab, only_ba = ab & ba, ab & ~ba, ba & ~ab
        # a->b message etype: BIDIR if mutual; sender a is TAIL if a follows b
        # (only_ab); sender a is HEAD if b follows a (only_ba). b->a is the mirror.
        et_ab = np.where(mutual, _BIDIR, np.where(only_ab, _TAIL, _HEAD))
        et_ba = np.where(mutual, _BIDIR, np.where(only_ab, _HEAD, _TAIL))

    # ---- Message edges (block layout: a->b first, then the b->a reverses) ----
    P, Q = ca.size, bu.size
    t_snd = np.concatenate([ca, cb])
    t_rcv = np.concatenate([cb, ca])
    t_etype = np.concatenate([et_ab, et_ba])
    b_snd = np.concatenate([bu, bv])
    b_rcv = np.concatenate([bv, bu])
    b_etype = np.full(2 * Q, _BIDIR, np.int64)

    snd = np.concatenate([t_snd, b_snd]).astype(np.int64)
    rcv = np.concatenate([t_rcv, b_rcv]).astype(np.int64)
    etype = np.concatenate([t_etype, b_etype]).astype(np.int64)
    # reverse links: within each layer the i-th forward message pairs with the
    # i-th backward message (offset by the layer's pair count).
    rev = np.concatenate([
        np.arange(P) + P, np.arange(P),
        2 * P + np.arange(Q) + Q, 2 * P + np.arange(Q),
    ]).astype(np.int64)

    E = snd.size
    w_bi = np.empty(E, np.float64)
    wb_tail = np.empty(E, np.float64)
    ws_tail = np.empty(E, np.float64)
    wb_head = np.empty(E, np.float64)
    ws_head = np.empty(E, np.float64)
    # Trust message edges carry all five params (the potential selects by
    # etype); behavioral edges use only the bidirectional strength.
    w_bi[:2 * P], w_bi[2 * P:] = params.w_bi, w_behavioral
    wb_tail[:2 * P], wb_tail[2 * P:] = params.wb_tail, 0.0
    ws_tail[:2 * P], ws_tail[2 * P:] = params.ws_tail, 0.0
    wb_head[:2 * P], wb_head[2 * P:] = params.wb_head, 0.0
    ws_head[:2 * P], ws_head[2 * P:] = params.ws_head, 0.0

    # ---- Adjacencies for the adaptive estimators ----
    # N_out(u) = accounts u follows; N_in(u) = accounts following u. A directed
    # trust edge u->v puts v in N_out(u) and u in N_in(v); behavioral edges are
    # symmetric (both endpoints in each other's N_out and N_in).
    out_rows = np.concatenate([tu, bu, bv])
    out_cols = np.concatenate([tv, bv, bu])
    in_rows = np.concatenate([tv, bu, bv])
    in_cols = np.concatenate([tu, bv, bu])
    a_out = _binary_csr(out_rows, out_cols, n)
    a_in = _binary_csr(in_rows, in_cols, n)

    return _MessageGraph(
        n=n, snd=snd, rcv=rcv, rev=rev, etype=etype,
        w_bi=w_bi, wb_tail=wb_tail, ws_tail=ws_tail,
        wb_head=wb_head, ws_head=ws_head, a_out=a_out, a_in=a_in,
    )


def _binary_csr(rows: np.ndarray, cols: np.ndarray, n: int) -> sparse.csr_matrix:
    """0/1 CSR adjacency with an entry per distinct ``(row, col)`` pair."""
    if rows.size == 0:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    m = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, cols)), shape=(n, n)
    )
    m.data[:] = 1.0   # collapse summed duplicates back to binary
    return m


# ---------------------------------------------------------------------------
# Adaptive estimators + edge potentials
# ---------------------------------------------------------------------------

def _adaptive_estimators(
    mg: _MessageGraph,
    p_sybil: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(c_b, c_s)`` per node from the current posteriors.

    ``c_b[u] = mean_{v in N_out(u)} P(benign_v)``,
    ``c_s[u] = mean_{v in N_in(u)} P(sybil_v)``.  Empty neighbourhoods
    default to ``1.0`` so the estimator leaves the base strength unchanged.
    """
    p_benign = 1.0 - p_sybil
    deg_out = np.asarray(mg.a_out.sum(axis=1)).ravel()
    deg_in = np.asarray(mg.a_in.sum(axis=1)).ravel()
    sum_b = mg.a_out @ p_benign
    sum_s = mg.a_in @ p_sybil
    c_b = np.ones(mg.n, dtype=np.float64)
    c_s = np.ones(mg.n, dtype=np.float64)
    nz_out = deg_out > 0
    nz_in = deg_in > 0
    c_b[nz_out] = sum_b[nz_out] / deg_out[nz_out]
    c_s[nz_in] = sum_s[nz_in] / deg_in[nz_in]
    return c_b, c_s


def _edge_potentials(
    mg: _MessageGraph,
    c_b: np.ndarray,
    c_s: np.ndarray,
) -> np.ndarray:
    """Build the ``(E, 2, 2)`` potential tensor ``Psi[e, x_s, x_r]``.

    State index ``0`` = benign, ``1`` = sybil.
    """
    E = mg.snd.shape[0]
    psi = np.empty((E, 2, 2), dtype=np.float64)

    csb, css = c_b[mg.snd], c_s[mg.snd]   # sender estimators
    crb, crs = c_b[mg.rcv], c_s[mg.rcv]   # receiver estimators

    is_bi = mg.etype == _BIDIR
    is_tail = mg.etype == _TAIL
    is_head = mg.etype == _HEAD

    # --- Bidirectional (Eq. 12) ---
    w = mg.w_bi
    psi[is_bi, 0, 0] = 0.5 * w[is_bi] * (csb[is_bi] + crb[is_bi])
    psi[is_bi, 0, 1] = 1.0 - 0.5 * w[is_bi] * (csb[is_bi] + crs[is_bi])
    psi[is_bi, 1, 1] = 0.5 * w[is_bi] * (css[is_bi] + crs[is_bi])
    psi[is_bi, 1, 0] = 1.0 - 0.5 * w[is_bi] * (css[is_bi] + crb[is_bi])

    # --- Sender is tail (Eq. 10): benign-tail row scaled by c_s_b ---
    wbt, wst = mg.wb_tail, mg.ws_tail
    psi[is_tail, 0, 0] = wbt[is_tail] * csb[is_tail]
    psi[is_tail, 0, 1] = 1.0 - wbt[is_tail] * csb[is_tail]
    psi[is_tail, 1, 1] = wst[is_tail]
    psi[is_tail, 1, 0] = 1.0 - wst[is_tail]

    # --- Sender is head (Eq. 11): sybil-head row scaled by c_s_s ---
    wbh, wsh = mg.wb_head, mg.ws_head
    psi[is_head, 0, 0] = wbh[is_head]
    psi[is_head, 1, 0] = 1.0 - wbh[is_head]
    psi[is_head, 1, 1] = wsh[is_head] * css[is_head]
    psi[is_head, 0, 1] = 1.0 - wsh[is_head] * css[is_head]

    np.clip(psi, _EPS, 1.0, out=psi)
    return psi


# ---------------------------------------------------------------------------
# LBP driver
# ---------------------------------------------------------------------------

def run_sybilhp(
    node_order: list[Hashable],
    sybil_prior: dict[Hashable, float],
    trust_edges: Sequence[tuple[Hashable, Hashable]] = (),
    behavioral_edges: Sequence[tuple[Hashable, Hashable]] = (),
    params: SybilHPParams = SybilHPParams(),
    w_behavioral: float = DEFAULT_W_BEHAVIORAL,
    default_prior: float = 0.5,
    max_iters: int = DEFAULT_MAX_ITERS,
    tol: float = DEFAULT_TOL,
    damping: float = 0.0,
    adaptive: bool = True,
) -> SybilHPResult:
    """Run SybilHP belief propagation over a trust layer + behavioral layer.

    Args:
        node_order: Node ordering for the returned posterior vector.
        sybil_prior: Per-node ``q_u = P(sybil)``. Missing nodes use
            ``default_prior``.
        trust_edges: Directed ``(tail, head)`` follow/trust edges. Mutual
            pairs are treated as bidirectional.
        behavioral_edges: Undirected behavioral-similarity pairs
            (bidirectional potentials).
        params: Five trust-layer homophily-strength parameters.
        w_behavioral: Bidirectional homophily strength for the behavioral
            layer.
        default_prior: ``q`` for nodes absent from ``sybil_prior``
            (typically 0.5).
        max_iters: Iteration cap.
        tol: Convergence tolerance on ``max_u |P(sybil_u)^{t+1} - P^t|``.
        damping: Optional log-message damping in ``[0, 1)`` to stabilise
            oscillation (0 = plain sum-product, as in the paper).

    Returns:
        :class:`SybilHPResult` with ``p_honest = P(benign)``.
    """
    n = len(node_order)
    mg = _build_message_graph(
        node_order, trust_edges, behavioral_edges, params, w_behavioral
    )

    # Node potential phi[u] = [P(benign), P(sybil)] from the prior.
    q = np.full(n, float(default_prior), dtype=np.float64)
    for i, nd in enumerate(node_order):
        if nd in sybil_prior:
            q[i] = float(sybil_prior[nd])
    q = np.clip(q, _EPS, 1.0 - _EPS)
    log_phi = np.column_stack([np.log1p(-q), np.log(q)])  # (n, 2)

    E = mg.snd.shape[0]
    if E == 0:
        # No edges: posterior == prior.
        p_sybil = q.copy()
        return SybilHPResult(
            node_order=list(node_order),
            p_honest=1.0 - p_sybil,
            n_iters=0,
            max_delta=0.0,
            converged=True,
        )

    # Messages in log-domain, uniform init (normalised).
    log_m = np.full((E, 2), -np.log(2.0), dtype=np.float64)

    # Receiver incidence matrix R (n x E): R[rcv[e], e] = 1. The belief
    # aggregation `sum of incoming messages per node` is then R @ log_m,
    # a fast sparse matmul (equivalent to np.add.at but scales to millions
    # of edges, where the unbuffered scatter does not).
    _R = sparse.csr_matrix(
        (np.ones(E, dtype=np.float64), (mg.rcv, np.arange(E, dtype=np.int64))),
        shape=(n, E),
    )

    def _node_log_belief(log_m_arr: np.ndarray) -> np.ndarray:
        return log_phi + _R @ log_m_arr

    def _posterior_sybil(log_m_arr: np.ndarray) -> np.ndarray:
        lb = _node_log_belief(log_m_arr)
        lb = lb - logsumexp(lb, axis=1, keepdims=True)
        return np.exp(lb[:, 1])

    p_sybil = _posterior_sybil(log_m)
    max_delta = float("inf")
    converged = False
    last_iter = 0

    ones = np.ones(mg.n, dtype=np.float64)
    for it in range(1, max_iters + 1):
        last_iter = it
        if adaptive:
            c_b, c_s = _adaptive_estimators(mg, p_sybil)
        else:
            c_b, c_s = ones, ones   # SybilHP_basic: fixed homophily strengths
        psi = _edge_potentials(mg, c_b, c_s)
        log_psi = np.log(psi)  # (E, 2, 2)

        log_bel = _node_log_belief(log_m)               # (n, 2)
        cavity = log_bel[mg.snd] - log_m[mg.rev]        # (E, 2) over sender state

        # m_new[e, x_r] = sum_{x_s} exp(cavity[e, x_s]) * psi[e, x_s, x_r]
        # Work in log-domain: logaddexp over x_s of (cavity[:,x_s] + log_psi[:,x_s,x_r]).
        cav = cavity[:, :, None]                        # (E, 2, 1)
        log_unnorm = logsumexp(cav + log_psi, axis=1)   # (E, 2)
        log_m_new = log_unnorm - logsumexp(log_unnorm, axis=1, keepdims=True)

        if damping > 0.0:
            log_m_new = damping * log_m + (1.0 - damping) * log_m_new
            log_m_new -= logsumexp(log_m_new, axis=1, keepdims=True)

        log_m = log_m_new
        p_new = _posterior_sybil(log_m)
        max_delta = float(np.max(np.abs(p_new - p_sybil)))
        p_sybil = p_new
        if max_delta < tol:
            converged = True
            logger.info("sybilhp converged at iter %d (max_delta=%.3e)", it, max_delta)
            break

    if not converged:
        logger.info("sybilhp hit max_iters=%d (max_delta=%.3e)", max_iters, max_delta)

    return SybilHPResult(
        node_order=list(node_order),
        p_honest=1.0 - p_sybil,
        n_iters=last_iter,
        max_delta=max_delta,
        converged=converged,
    )


def sybilhp_priors_from_honest(
    honest_priors: dict[Hashable, float],
) -> dict[Hashable, float]:
    """Convert a ``P(honest)`` prior dict to SybilHP's ``q = P(sybil)`` dict."""
    return {nd: float(1.0 - p) for nd, p in honest_priors.items()}


def p_honest_to_dict(result: SybilHPResult) -> dict[Hashable, float]:
    """Zip ``result.node_order`` with ``result.p_honest`` into a dict."""
    return {nd: float(p) for nd, p in zip(result.node_order, result.p_honest)}
