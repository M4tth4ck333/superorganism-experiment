"""LTR Community Thread — runs a single IPv8 peer that joins the LTR MAB
community distributed across all current network peers.

Each running app instance is *one* peer.  When the user clicks RUN, this
peer starts its local query-loop, gossips statistics with whoever else is
on the network, and emits live snapshots of its *own* arms/rewards so the
GUI can display the local experiment status.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

_BENCH_DIR = Path(__file__).parent / "ltr-benchmarking"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

# Shared file used by every running app instance to advertise its IPv8
# (host, port) so peers on the same machine can discover each other without
# depending on the public Dispersy bootstrap server (which is unreliable
# behind NAT, especially for two peers on the same host).
_PEER_REGISTRY_PATH = _BENCH_DIR / ".peer_registry.json"


def _read_peer_registry() -> dict:
    try:
        return json.loads(_PEER_REGISTRY_PATH.read_text())
    except Exception:
        return {}


def _write_peer_registry(registry: dict) -> None:
    try:
        _PEER_REGISTRY_PATH.write_text(json.dumps(registry))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Local-peer GUI state  (mirrors _GUIState from ltr_thread.py but single-peer)
# ---------------------------------------------------------------------------

class _LocalPeerState:
    """Tracks the state of *this* peer and emits GUI snapshots."""

    def __init__(self, snapshot_signal: Signal, log_signal: Signal):
        self._snap_sig = snapshot_signal
        self._log_sig = log_signal

        self.community = None          # LTRMABCommunity instance for this peer
        self.current_round = 0
        self.phase = "idle"
        self.config: dict = {}
        self.oracle: dict = {}
        self.round_history: list = []
        self.t0 = time.time()

    # ------------------------------------------------------------------ events

    def event(self, msg: str, kind: str = "info") -> None:
        entry = {"t": round(time.time() - self.t0, 2), "kind": kind, "msg": msg}
        self._log_sig.emit(entry)
        self._emit_snapshot()

    def _emit_snapshot(self) -> None:
        self._snap_sig.emit(self._build_snapshot())

    def _build_snapshot(self) -> dict:
        peer_data = self._peer_data()
        return {
            "round": self.current_round,
            "phase": self.phase,
            "config": self.config,
            "oracle": self.oracle,
            "elapsed": round(time.time() - self.t0, 1),
            "peer": peer_data,
            "round_history": list(self.round_history),
            "network_peers": self._network_peer_count(),
        }

    def _peer_data(self) -> dict:
        c = self.community
        if c is None:
            return {}
        stats = c.bandit.get_stats()
        q = max(c.queries_processed, 1)
        return {
            "id": c.peer_id,
            "queries": c.queries_processed,
            "active": sorted(c.active_models),
            "excluded": sorted(c.excluded_models),
            "best": c.bandit.get_best_arm() if c.bandit.total_pulls > 0 else None,
            "scores": {str(k): round(v / q, 4) for k, v in c.cumulative_scores.items()},
            "arms": {
                name: {
                    "pulls": s["pulls"],
                    "reward": round(c._get_mean_reward(s), 4),
                    "status": "excluded" if name in c.excluded_models else "active",
                }
                for name, s in stats.items()
            },
        }

    def _network_peer_count(self) -> int:
        if self.community is None:
            return 0
        return len(self.community.get_peers())

    # ------------------------------------------------------------------ hooks

    class _HookedList(list):
        def __init__(self, owner: "_LocalPeerState"):
            super().__init__()
            self._owner = owner

        def append(self, item):
            super().append(item)
            self._owner._emit_snapshot()

    def install_hooked_list(self) -> None:
        self.round_history = self._HookedList(self)


# ---------------------------------------------------------------------------
# QThread
# ---------------------------------------------------------------------------

class LTRCommunityThread(QThread):
    """Runs a single LTR MAB IPv8 peer in a background thread.

    The peer joins whatever other peers are on the network (same bootstrap),
    runs the bandit query loop, gossiping between rounds, and emits live
    snapshots of the *local* peer's state only.

    Signals (Thread → GUI):
        started_ok()          models loaded, peer up, about to start rounds
        snapshot(dict)        local-peer snapshot after every event / round
        log_event(dict)       single log entry {t, kind, msg}
        finished_ok()         all rounds done
        error(str)            fatal error
    """

    started_ok  = Signal()
    snapshot    = Signal(dict)
    log_event   = Signal(dict)
    finished_ok = Signal()
    error       = Signal(str)

    COMMUNITY_ID = b"superorg-ltr-exp-v1\x00"  # 20 bytes

    def __init__(
        self,
        dataset_id: str,
        algorithm: str,
        metric: str = "ndcg",
        num_rounds: int = 5,
        queries_per_round: int = 100,
        gossip_enabled: bool = True,
        hotswap_round: int = 0,
        peer_port: int = 0,          # 0 → pick a free port
        key_path: Optional[str] = None,
        bootstrap_addresses: Optional[list[tuple[str, int]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.dataset_id        = dataset_id
        self.algorithm         = algorithm
        self.metric            = metric
        self.num_rounds        = num_rounds
        self.queries_per_round = queries_per_round
        self.gossip_enabled    = gossip_enabled
        self.hotswap_round     = hotswap_round
        self.peer_port         = peer_port
        self.key_path          = key_path or str(
            _BENCH_DIR / f"peer_community_{os.getpid()}.pem"
        )

        self.bootstrap_addresses = bootstrap_addresses or []

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None

    # ------------------------------------------------------------------ QThread

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self._loop.close()

    def stop(self) -> None:
        if self._loop and not self._loop.is_closed() and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)

    # ------------------------------------------------------------------ async core

    async def _run(self) -> None:
        self._stop_event = asyncio.Event()

        import numpy as np
        import local_experiment as exp
        from local_experiment import (
            LTRMABCommunity,
            BASE_PORT,
            GOSSIP_ROUNDS,
            GOSSIP_DELAY,
            PEER_DISCOVERY_WAIT,
        )
        from ipv8.configuration import (
            ConfigBuilder, Strategy, WalkerDefinition, default_bootstrap_defs,
        )
        from ipv8_service import IPv8
        from datasets import get_dataset

        state = _LocalPeerState(self.snapshot, self.log_event)
        state.install_hooked_list()

        # ── Load dataset ──────────────────────────────────────────────────
        state.event(f"Loading dataset '{self.dataset_id}'…", "info")
        dataset = get_dataset(self.dataset_id, _BENCH_DIR / "data")
        X_test, y_test, _, groups = dataset.load_test()
        query_boundaries: list[tuple[int, int]] = []
        start = 0
        for g in groups:
            query_boundaries.append((start, start + g))
            start += g

        # ── Load models ───────────────────────────────────────────────────
        state.event("Loading models…", "info")
        models = exp.load_experiment_models(self.dataset_id)

        if not models:
            self.error.emit("No models found for dataset " + self.dataset_id)
            return

        # ── Precompute scores ─────────────────────────────────────────────
        state.event("Precomputing model scores…", "info")
        model_scores = exp.precompute_model_scores(
            models, X_test, y_test, query_boundaries,
            k_values=[1, 5, 10], metric=self.metric,
        )
        oracle = {
            name: round(
                sum(model_scores[name][10]) / max(len(model_scores[name][10]), 1), 4
            )
            for name in models
        }

        hotswap_model_name = None
        for name in models:
            if "xgboost" in name.lower() or "xgb" in name.lower():
                hotswap_model_name = name
                break

        initial_model_names = [n for n in models if n != hotswap_model_name]

        # ── Shared community state ────────────────────────────────────────
        shared = {
            "models": models,
            "initial_model_names": initial_model_names,
            "model_scores": model_scores,
            "query_boundaries": query_boundaries,
            "num_queries": len(query_boundaries),
            "algorithm": self.algorithm,
            "metric": self.metric,
        }
        LTRMABCommunity.set_state(shared)
        LTRMABCommunity._peer_counter = 0

      
        LTRMABCommunity.community_id = LTRCommunityThread.COMMUNITY_ID
        port = self.peer_port if self.peer_port else (BASE_PORT + os.getpid() % 1000)

        state.config = {
            "dataset": self.dataset_id,
            "algorithm": self.algorithm,
            "metric": self.metric,
            "num_rounds": self.num_rounds,
            "queries_per_round": self.queries_per_round,
            "gossip_enabled": self.gossip_enabled,
        }
        state.oracle = oracle

        # ── Start IPv8 peer ───────────────────────────────────────────────
        state.event(f"Starting peer on port {port}…", "info")
        builder = ConfigBuilder().clear_keys().clear_overlays()
        os.makedirs(Path(self.key_path).parent, exist_ok=True)
        builder.add_key("my peer", "medium", self.key_path)
        builder.set_port(port)

        from ipv8.configuration import BootstrapperDefinition, Bootstrapper
        from ipv8.configuration import DISPERSY_BOOTSTRAPPER
        bootstrap_defs = list(default_bootstrap_defs)
        if self.bootstrap_addresses:
            extra_init = dict(DISPERSY_BOOTSTRAPPER["init"])
            extra_init = {
                "ip_addresses": self.bootstrap_addresses,
                "dns_addresses": [],
            }
            bootstrap_defs.append(
                BootstrapperDefinition(Bootstrapper.DispersyBootstrapper, extra_init)
            )

        builder.add_overlay(
            "LTRMABCommunity", "my peer",
            [WalkerDefinition(Strategy.RandomWalk, 10, {"timeout": 3.0})],
            bootstrap_defs, {}, [("started",)],
        )

        ipv8 = IPv8(
            builder.finalize(),
            extra_communities={"LTRMABCommunity": LTRMABCommunity},
        )
        await ipv8.start()

        try:
            community: LTRMABCommunity = ipv8.get_overlay(LTRMABCommunity)
            state.community = community
            community.on_event = state.event

            state.event(f"Peer started (id={community.peer_id}). Discovering network…", "info")

            my_pid = str(os.getpid())
            registry = _read_peer_registry()
            known_peers = [
                tuple(addr) for pid, addr in registry.items() if pid != my_pid
            ]
            registry[my_pid] = ["127.0.0.1", port]
            _write_peer_registry(registry)

            for addr in known_peers:
                host, peer_port = addr[0], int(addr[1])
                state.event(f"Walking to known local peer at {host}:{peer_port}", "info")
                try:
                    community.walk_to((host, peer_port))
                except Exception as exc:
                    state.event(f"walk_to({host}:{peer_port}) failed: {exc}", "info")

            # Wait for peer discovery
            DISCOVERY_WAIT = 10  # seconds
            for elapsed_s in range(1, DISCOVERY_WAIT + 1):
                if self._stop_event.is_set():
                    return
                await asyncio.sleep(1)
                n = len(community.get_peers())
                state.event(
                    f"Discovery {elapsed_s}/{DISCOVERY_WAIT}s — {n} network peer(s) found so far",
                    "info",
                )
                if n > 0 and elapsed_s >= 5:
                    break

            if self._stop_event.is_set():
                return

            n_final = len(community.get_peers())
            state.event(
                f"Discovery complete — {n_final} network peer(s) connected. Starting rounds.",
                "info",
            )

            self.started_ok.emit()

            # ── Query / gossip loop ───────────────────────────────────────
            rng = np.random.default_rng()
            total_queries = len(query_boundaries)

            for round_num in range(1, self.num_rounds + 1):
                if self._stop_event.is_set():
                    break
                state.current_round = round_num
                state.event(f"Round {round_num}/{self.num_rounds} started", "round")

                community.reset_round_stats()

                # --- Query phase ---
                state.phase = "querying"
                state.event(f"Querying ({self.queries_per_round} queries)…", "info")
                replace = self.queries_per_round > total_queries
                query_indices = rng.choice(
                    total_queries, size=self.queries_per_round, replace=replace
                )
                for qi in query_indices:
                    community.process_query(int(qi))
                await asyncio.sleep(0.01)
                state._emit_snapshot()

                # --- Hot-swap ---
                if (
                    self.hotswap_round > 0
                    and round_num == self.hotswap_round
                    and hotswap_model_name
                ):
                    state.event(
                        f"HOT-SWAP: proposing {hotswap_model_name}", "round"
                    )
                    await community.propose_model(hotswap_model_name)
                    await asyncio.sleep(0.5)

                # --- Survival / exclusion ---
                state.phase = "survival"
                excluded_this_round = community.check_exclusions(round_num)
                for model_name in excluded_this_round:
                    lcb, ucb = community.bandit.confidence_bounds(model_name)
                    reason = f"UCB={ucb:.3f} < best_LCB"
                    if self.gossip_enabled:
                        await community.broadcast_exclusion(model_name, round_num, reason)
                    state.event(f"ARM EXCLUDED: {model_name} ({reason})", "exclusion")
                    await asyncio.sleep(0.05)

                # --- Record round snapshot (aggregated from this peer only) ---
                stats = community.bandit.get_stats()
                arm_pulls = {n: s["pulls"] for n, s in stats.items()}
                arm_mean_reward = {
                    n: round(community._get_mean_reward(s), 4) for n, s in stats.items()
                }
                cumulative_reward = community.cumulative_scores.get(10, 0.0)
                best_oracle_score = max(oracle.values()) if oracle else 0.0
                oracle_cumulative = best_oracle_score * community.queries_processed

                prev_arms = (
                    set(state.round_history[-1]["arm_pulls"].keys())
                    if state.round_history else set()
                )
                round_snapshot = {
                    "round": round_num,
                    "arm_pulls": arm_pulls,
                    "arm_mean_reward": arm_mean_reward,
                    "cumulative_reward": round(cumulative_reward, 4),
                    "oracle_cumulative": round(oracle_cumulative, 4),
                    "new_arms": [n for n in arm_pulls if n not in prev_arms],
                }
                state.round_history.append(round_snapshot)  # triggers snapshot via hook

                best = community.bandit.get_best_arm()
                state.event(
                    f"Round {round_num} done · best={best} · "
                    f"active={len(community.active_models)} · "
                    f"excluded={len(community.excluded_models)}",
                    "round",
                )

                # --- Gossip phase (end of round) ---
                if self.gossip_enabled:
                    state.phase = "gossiping"
                    for gr in range(1, GOSSIP_ROUNDS + 1):
                        n = await community.send_gossip()
                        if n > 0:
                            state.event(
                                f"Gossip round {gr}/{GOSSIP_ROUNDS}: sent to {n} peer(s)",
                                "gossip",
                            )
                        await asyncio.sleep(GOSSIP_DELAY)
                else:
                    state.event("Gossip skipped (disabled)", "info")

            if not self._stop_event.is_set():
                state.event("Experiment complete", "round")
                self.finished_ok.emit()

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            state.phase = "finished"
            state._emit_snapshot()
            try:
                await ipv8.stop()
            except Exception:
                pass
            try:
                reg = _read_peer_registry()
                reg.pop(str(os.getpid()), None)
                _write_peer_registry(reg)
            except Exception:
                pass
