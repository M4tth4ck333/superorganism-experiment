"""Multi-Armed Bandit for ranking model selection.

Uses a G-Counter CRDT data structure per arm: each arm stores a per-originator
table so gossip merges are idempotent and evidence is never double-counted.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import json
from datetime import datetime, timezone


def _derive_rng(seed: int | None, *tags) -> np.random.Generator:
    """Build a per-site deterministic Generator from a master seed + tags.

    When `seed` is None, returns a fresh non-deterministic Generator so the
    non-seeded code path keeps its previous behaviour. When seeded, the tags
    (e.g. peer_id, purpose) namespace the stream so distinct call sites don't
    share draws — critical because a single master seed is reused across
    every RNG in the experiment.
    """
    if seed is None:
        return np.random.default_rng()

    spawn_key = tuple(abs(hash(t)) % (2**32) for t in tags)
    ss = np.random.SeedSequence(entropy=seed, spawn_key=spawn_key)
    return np.random.default_rng(ss)


class RankingModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class OriginatorEntry:
    """One peer's locally-observed evidence for a single arm."""
    pulls: int = 0
    total_reward: float = 0.0  # UCB1: cumulative reward; Thompson: unused (α/β used instead)
    alpha: float = 1.0         # Thompson only: prior starts at 1
    beta: float = 1.0          # Thompson only: prior starts at 1
    # Local-only freshness timestamp (owner's tick counter at the time this
    # entry was last observed to make progress). Not serialized over gossip —
    # each peer applies TTL pruning against its own local clock.
    last_seen: int = 0


# Flat stats shape returned by get_stats() — unchanged so callers don't break
@dataclass
class ArmStats:
    """Aggregated statistics for a single arm (model)."""
    name: str
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0


def _crdt_merge_tables(
    local: dict[str, OriginatorEntry],
    remote: dict[str, OriginatorEntry],
    current_tick: int = 0,
) -> None:
    """G-Counter CRDT merge: for each originator take the entry with more pulls.

    Modifies `local` in-place. Stamps `last_seen = current_tick` on any entry
    whose evidence advances — used by TTL pruning to evict originators whose
    peers have left the network.
    """
    for origin, remote_entry in remote.items():
        if origin not in local or remote_entry.pulls > local[origin].pulls:
            local[origin] = OriginatorEntry(
                pulls=remote_entry.pulls,
                total_reward=remote_entry.total_reward,
                alpha=remote_entry.alpha,
                beta=remote_entry.beta,
                last_seen=current_tick,
            )


# How much of the remote mean reward to keep when adopting a new arm via
# gossip.  E.g. 0.10 means the arm starts at 10% of the gossipped mean,
# so the peer must explore it locally before it can compete with proven arms.
GOSSIP_REWARD_FRACTION = 0.10

# Number of virtual pulls to seed the new arm with.  Kept small so UCB's
# exploration bonus (or Thompson's wide posterior) kicks in quickly.
GOSSIP_SEED_PULLS = 2


def _discount_remote_table(
    remote_table: dict[str, OriginatorEntry],
) -> dict[str, OriginatorEntry]:
    """Replace a remote originator table with a single low-evidence entry
    that preserves only a small hint of the gossipped quality.
    """
    # Compute the remote arm's aggregate mean reward
    total_pulls = sum(e.pulls for e in remote_table.values())
    total_reward = sum(e.total_reward for e in remote_table.values())
    total_alpha = 1.0 + sum(e.alpha - 1.0 for e in remote_table.values())
    total_beta = 1.0 + sum(e.beta - 1.0 for e in remote_table.values())

    if total_pulls > 0:
        mean_reward = total_reward / total_pulls
    else:
        mean_reward = 0.0

    # Seed with a few pulls at a fraction of the remote mean
    seeded_reward = mean_reward * GOSSIP_REWARD_FRACTION * GOSSIP_SEED_PULLS

    # Thompson: nudge α/β slightly away from the flat prior
    if total_alpha + total_beta > 2.0:
        remote_mean_ab = total_alpha / (total_alpha + total_beta)
    else:
        remote_mean_ab = 0.5
    hint_strength = GOSSIP_SEED_PULLS * GOSSIP_REWARD_FRACTION
    seeded_alpha = 1.0 + remote_mean_ab * hint_strength
    seeded_beta = 1.0 + (1.0 - remote_mean_ab) * hint_strength

    # Collapse all remote originators into a single synthetic entry
    return {
        "__gossip_seed__": OriginatorEntry(
            pulls=GOSSIP_SEED_PULLS,
            total_reward=seeded_reward,
            alpha=seeded_alpha,
            beta=seeded_beta,
        )
    }


class UCB1:
    """Upper Confidence Bound algorithm using per-originator CRDT tables."""

    def __init__(self, arm_names: list[str], c: float = 2.0, peer_id: str = "local", seed: int | None = None):
        self.c = c
        self.peer_id = peer_id
        # Local gossip-tick counter used as the "clock" for TTL pruning.
        self.tick_counter = 0
        # arm → {originator_id → OriginatorEntry}
        self.tables: dict[str, dict[str, OriginatorEntry]] = {
            name: {peer_id: OriginatorEntry()} for name in arm_names
        }
        self.total_pulls = 0
        # Dedicated RNG for tiebreaking so ties don't depend on dict order.
        self._rng = _derive_rng(seed, peer_id, "ucb1")

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate(self, arm: str) -> tuple[float, int]:
        """Return (total_reward, total_pulls) aggregated across all originators."""
        R = sum(e.total_reward for e in self.tables[arm].values())
        n = sum(e.pulls for e in self.tables[arm].values())
        return R, n

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def arms(self) -> dict[str, ArmStats]:
        """Flat ArmStats view — keeps external callers working unchanged."""
        result = {}
        for name in self.tables:
            R, n = self._aggregate(name)
            stats = ArmStats(name=name, pulls=n, total_reward=R)
            result[name] = stats
        return result

    def select_arm(self, active: set[str] | None = None) -> str:
        """Select arm using UCB1 with log(N+1)/(n+1) formula.

        Iteration is over a sorted candidate list and ties are broken by the
        bandit's dedicated RNG so two runs with the same seed make the same
        choice regardless of set/dict insertion order.
        """
        candidates = sorted(active if active is not None else set(self.tables))

        # Force-explore untested arms first (sorted order is deterministic).
        for name in candidates:
            _, n = self._aggregate(name)
            if n == 0:
                return name

        best_ucb = -float("inf")
        best_arms: list[str] = []
        for name in candidates:
            R, n = self._aggregate(name)
            exploration = self.c * np.sqrt(np.log(self.total_pulls + 1) / (n + 1))
            ucb = R / n + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_arms = [name]
            elif ucb == best_ucb:
                best_arms.append(name)

        if len(best_arms) == 1:
            return best_arms[0]
        return best_arms[int(self._rng.integers(len(best_arms)))]

    def update(self, arm_name: str, reward: float) -> None:
        """Update own originator entry for the selected arm."""
        entry = self.tables[arm_name][self.peer_id]
        entry.pulls += 1
        entry.total_reward += reward
        entry.last_seen = self.tick_counter
        self.total_pulls += 1

    def crdt_merge(self, arm: str, remote_table: dict[str, OriginatorEntry]) -> None:
        """Merge a remote originator table into the local one for `arm`."""
        _crdt_merge_tables(self.tables[arm], remote_table, self.tick_counter)
        # Recompute total_pulls from scratch to stay consistent
        self.total_pulls = sum(
            sum(e.pulls for e in t.values()) for t in self.tables.values()
        )

    def add_arm(self, name: str, remote_table: dict[str, OriginatorEntry] | None = None) -> None:
        """Add a new arm, optionally seeded with a discounted remote table."""
        if name in self.tables:
            return
        self.tables[name] = {self.peer_id: OriginatorEntry()}
        if remote_table:
            _crdt_merge_tables(
                self.tables[name],
                _discount_remote_table(remote_table),
                self.tick_counter,
            )
            self.total_pulls = sum(
                sum(e.pulls for e in t.values()) for t in self.tables.values()
            )

    def prune_stale_originators(self, ttl_ticks: int) -> list[tuple[str, str]]:
        """Drop originator entries not refreshed within `ttl_ticks` ticks.

        The local peer's own entry is never pruned. Returns the list of
        (arm, originator_id) pairs that were evicted so callers can log it.
        """
        if ttl_ticks <= 0:
            return []
        cutoff = self.tick_counter - ttl_ticks
        evicted: list[tuple[str, str]] = []
        for arm, table in self.tables.items():
            for origin in list(table):
                if origin == self.peer_id:
                    continue
                if table[origin].last_seen < cutoff:
                    del table[origin]
                    evicted.append((arm, origin))
        if evicted:
            self.total_pulls = sum(
                sum(e.pulls for e in t.values()) for t in self.tables.values()
            )
        return evicted

    def get_best_arm(self) -> str:
        """Return arm with highest empirical mean reward."""
        return max(
            self.tables,
            key=lambda name: (lambda R, n: R / n if n > 0 else 0.0)(*self._aggregate(name)),
        )

    def confidence_bounds(self, arm: str) -> tuple[float, float]:
        """Return (lcb, ucb) using Hoeffding bound: mean ± sqrt(log(N+1) / (2n))."""
        R, n = self._aggregate(arm)
        if n == 0:
            return 0.0, 1.0
        mean = R / n
        half_width = np.sqrt(np.log(self.total_pulls + 1) / (2 * n))
        return max(0.0, mean - half_width), min(1.0, mean + half_width)

    def get_stats(self) -> dict:
        """Return flat stats dict compatible with existing callers."""
        result = {}
        for name in self.tables:
            R, n = self._aggregate(name)
            result[name] = {
                "pulls": n,
                "total_reward": R,
                "mean_reward": R / n if n > 0 else 0.0,
            }
        return result

    def get_originator_tables(self) -> dict[str, dict[str, dict]]:
        """Serialisable form of all originator tables for gossip."""
        return {
            arm: {
                origin: {
                    "pulls": e.pulls,
                    "total_reward": e.total_reward,
                }
                for origin, e in table.items()
            }
            for arm, table in self.tables.items()
        }


class ThompsonSampling:
    """Thompson Sampling using per-originator CRDT tables (continuous updates)."""

    def __init__(self, arm_names: list[str], peer_id: str = "local", seed: int | None = None):
        self.peer_id = peer_id
        # Local gossip-tick counter used as the "clock" for TTL pruning.
        self.tick_counter = 0
        # arm → {originator_id → OriginatorEntry}
        self.tables: dict[str, dict[str, OriginatorEntry]] = {
            name: {peer_id: OriginatorEntry(alpha=1.0, beta=1.0, pulls=0)}
            for name in arm_names
        }
        self.total_pulls = 0
        # Dedicated RNG for Beta posterior sampling.
        self._rng = _derive_rng(seed, peer_id, "thompson")

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate(self, arm: str) -> tuple[float, float, int]:
        """Return (alpha, beta, pulls) aggregated across all originators.

        alpha = 1 + Σ(α_i - 1),  beta = 1 + Σ(β_i - 1)
        The prior is counted once for the aggregate, not once per originator.
        """
        alpha = 1.0 + sum(e.alpha - 1.0 for e in self.tables[arm].values())
        beta = 1.0 + sum(e.beta - 1.0 for e in self.tables[arm].values())
        n = sum(e.pulls for e in self.tables[arm].values())
        return alpha, beta, n

    # ------------------------------------------------------------------
    # Public interface — mirrors UCB1 shape so local_experiment.py is uniform
    # ------------------------------------------------------------------

    @property
    def arms(self) -> dict[str, dict]:
        """Flat dict view keyed by arm name — keeps external callers working."""
        result = {}
        for name in self.tables:
            alpha, beta, n = self._aggregate(name)
            result[name] = {"alpha": alpha, "beta": beta, "pulls": n}
        return result

    def select_arm(self, active: set[str] | None = None) -> str:
        """Select arm by sampling from aggregated Beta posterior.

        Candidates are iterated in sorted order and Beta draws come from the
        bandit's dedicated RNG so the same seed yields the same arm every run.
        """
        candidates = sorted(active if active is not None else set(self.tables))

        # Force-explore untested arms first (sorted order is deterministic).
        for name in candidates:
            _, _, n = self._aggregate(name)
            if n == 0:
                return name

        best_name = candidates[0]
        best_sample = -1.0
        for name in candidates:
            alpha, beta, _ = self._aggregate(name)
            sample = float(self._rng.beta(alpha, beta))
            if sample > best_sample:
                best_sample = sample
                best_name = name

        return best_name

    def update(self, arm_name: str, reward: float) -> None:
        """Continuous Bayesian update on own originator entry."""
        entry = self.tables[arm_name][self.peer_id]
        entry.alpha += reward
        entry.beta += 1.0 - reward
        entry.pulls += 1
        entry.last_seen = self.tick_counter
        self.total_pulls += 1

    def crdt_merge(self, arm: str, remote_table: dict[str, OriginatorEntry]) -> None:
        """Merge a remote originator table into the local one for `arm`."""
        _crdt_merge_tables(self.tables[arm], remote_table, self.tick_counter)
        self.total_pulls = sum(
            sum(e.pulls for e in t.values()) for t in self.tables.values()
        )

    def add_arm(self, name: str, remote_table: dict[str, OriginatorEntry] | None = None) -> None:
        """Add a new arm, optionally seeded with a discounted remote table."""
        if name in self.tables:
            return
        self.tables[name] = {self.peer_id: OriginatorEntry(alpha=1.0, beta=1.0, pulls=0)}
        if remote_table:
            _crdt_merge_tables(
                self.tables[name],
                _discount_remote_table(remote_table),
                self.tick_counter,
            )
            self.total_pulls = sum(
                sum(e.pulls for e in t.values()) for t in self.tables.values()
            )

    def prune_stale_originators(self, ttl_ticks: int) -> list[tuple[str, str]]:
        """Drop originator entries not refreshed within `ttl_ticks` ticks.

        The local peer's own entry is never pruned. Returns the list of
        (arm, originator_id) pairs that were evicted so callers can log it.
        """
        if ttl_ticks <= 0:
            return []
        cutoff = self.tick_counter - ttl_ticks
        evicted: list[tuple[str, str]] = []
        for arm, table in self.tables.items():
            for origin in list(table):
                if origin == self.peer_id:
                    continue
                if table[origin].last_seen < cutoff:
                    del table[origin]
                    evicted.append((arm, origin))
        if evicted:
            self.total_pulls = sum(
                sum(e.pulls for e in t.values()) for t in self.tables.values()
            )
        return evicted

    def get_best_arm(self) -> str:
        """Return arm with highest posterior mean."""
        return max(
            self.tables,
            key=lambda name: (lambda a, b, _: a / (a + b))(*self._aggregate(name)),
        )

    def confidence_bounds(self, arm: str) -> tuple[float, float]:
        """Return (lcb, ucb) using Hoeffding bound: mean ± sqrt(log(N+1) / (2n))."""
        alpha, beta, n = self._aggregate(arm)
        if n == 0:
            return 0.0, 1.0
        mean = alpha / (alpha + beta)
        half_width = np.sqrt(np.log(self.total_pulls + 1) / (2 * n))
        return max(0.0, mean - half_width), min(1.0, mean + half_width)

    def get_stats(self) -> dict:
        """Return flat stats dict compatible with existing callers."""
        result = {}
        for name in self.tables:
            alpha, beta, n = self._aggregate(name)
            result[name] = {
                "pulls": n,
                "alpha": alpha,
                "beta": beta,
                "expected_reward": alpha / (alpha + beta),
            }
        return result

    def get_originator_tables(self) -> dict[str, dict[str, dict]]:
        """Serialisable form of all originator tables for gossip."""
        return {
            arm: {
                origin: {
                    "pulls": e.pulls,
                    "alpha": e.alpha,
                    "beta": e.beta,
                }
                for origin, e in table.items()
            }
            for arm, table in self.tables.items()
        }


@dataclass
class SimulationResult:
    """Results from a MAB simulation."""
    algorithm: str
    total_rounds: int
    cumulative_reward: float
    cumulative_regret: float
    arm_stats: dict
    reward_history: list[float] = field(default_factory=list)
    regret_history: list[float] = field(default_factory=list)
    selection_history: list[str] = field(default_factory=list)

    def save(self, path: Path | str):
        path = Path(path)
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "algorithm": self.algorithm,
            "total_rounds": self.total_rounds,
            "cumulative_reward": self.cumulative_reward,
            "cumulative_regret": self.cumulative_regret,
            "arm_stats": self.arm_stats,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class ModelBandit:
    """MAB wrapper for ranking models."""

    def __init__(
        self,
        models: dict[str, RankingModel],
        algorithm: str = "ucb1",
        c: float = 2.0,
        peer_id: str = "local",
    ):
        self.models = models
        arm_names = list(models.keys())

        if algorithm == "ucb1":
            self.bandit = UCB1(arm_names, c=c, peer_id=peer_id)
        elif algorithm == "thompson":
            self.bandit = ThompsonSampling(arm_names, peer_id=peer_id)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.algorithm = algorithm

    def select_and_rank(self, X: np.ndarray) -> tuple[str, np.ndarray]:
        """Select a model and return rankings."""
        arm = self.bandit.select_arm()
        scores = self.models[arm].predict(X)
        return arm, scores

    def update(self, arm: str, reward: float):
        """Update bandit with observed reward."""
        self.bandit.update(arm, reward)

    def get_stats(self) -> dict:
        return self.bandit.get_stats()


def simulate_bandit(
    models: dict[str, RankingModel],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    algorithm: str = "ucb1",
    c: float = 2.0,
) -> SimulationResult:
    """
    Simulate MAB model selection on test queries.

    Reward: NDCG@1 (continuous) for Thompson; binary hit@1 kept for UCB1
    to match original simulate_bandit behaviour.
    """
    bandit = ModelBandit(models, algorithm=algorithm, c=c)

    model_scores = {name: model.predict(X) for name, model in models.items()}

    query_rewards = {name: [] for name in models}
    start_idx = 0
    for group_size in groups:
        end_idx = start_idx + group_size
        y_q = y[start_idx:end_idx]

        for name in models:
            scores_q = model_scores[name][start_idx:end_idx]
            top_idx = np.argmax(scores_q)
            reward = 1.0 if y_q[top_idx] > 0 else 0.0
            query_rewards[name].append(reward)

        start_idx = end_idx

    n_queries = len(groups)
    optimal_rewards = [max(query_rewards[name][i] for name in models) for i in range(n_queries)]

    cumulative_reward = 0.0
    cumulative_regret = 0.0
    reward_history = []
    regret_history = []
    selection_history = []

    for i in range(n_queries):
        arm = bandit.bandit.select_arm()
        reward = query_rewards[arm][i]
        optimal = optimal_rewards[i]
        regret = optimal - reward

        bandit.update(arm, reward)

        cumulative_reward += reward
        cumulative_regret += regret
        reward_history.append(cumulative_reward)
        regret_history.append(cumulative_regret)
        selection_history.append(arm)

    return SimulationResult(
        algorithm=algorithm,
        total_rounds=n_queries,
        cumulative_reward=cumulative_reward,
        cumulative_regret=cumulative_regret,
        arm_stats=bandit.get_stats(),
        reward_history=reward_history,
        regret_history=regret_history,
        selection_history=selection_history,
    )


if __name__ == "__main__":
    from ltr_evaluator import LETORDataset, load_model

    # Load models
    models_dir = Path(__file__).parent / "models"
    model_paths = [
        ("XGBoost", models_dir / "tribler_xgboost.json"),
        ("LightGBM", models_dir / "tribler_lightgbm.txt"),
        ("PDGD", models_dir / "pdgd_ranker.npy"),
    ]

    models = {}
    for name, path in model_paths:
        model, _ = load_model(path)
        models[name] = model

    # Load test data
    data_dir = Path(__file__).parent / "data" / "tribler_data" / "tribler_data" / "_normalized"
    dataset = LETORDataset(data_dir)
    X_test, y_test, _, groups_test = dataset.load_test()

    print(f"Running simulation on {len(groups_test)} queries...")
    print(f"Models: {list(models.keys())}")
    print()

    for algo in ["ucb1", "thompson"]:
        result = simulate_bandit(models, X_test, y_test, groups_test, algorithm=algo)

        print(f"=== {algo.upper()} ===")
        print(f"Total reward: {result.cumulative_reward:.0f} / {result.total_rounds}")
        print(f"Total regret: {result.cumulative_regret:.0f}")
        print(f"Reward rate: {result.cumulative_reward / result.total_rounds:.2%}")
        print("Arm stats:")
        for name, stats in result.arm_stats.items():
            pulls = stats["pulls"]
            mr = stats.get("mean_reward", stats.get("expected_reward", 0))
            print(f"  {name}: {pulls} pulls, {mr:.2%} reward")
        print()
