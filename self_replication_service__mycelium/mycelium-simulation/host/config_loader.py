"""YAML config loader."""

import os
import yaml

DEFAULTS = {
    "economy": {
        "seed_capital": 100000,
        "rent_per_cycle": 1000,
        "min_reserve_cycles": 3,
        "spawn_threshold_cycles": 10,
        "inheritance_ratio": 0.4,
        "caution_variance": 0.0,
        "caution_heritability": 0.8,
        "caution_mutation": 0.1,
    },
    "faucet": {
        "rate": 500,
        "distribution": "random_living",
        "enabled": True,
        "growth_rate": 0,   # satoshis added to rate per tick; 0 = no growth
        "noise": 0.0,       # 0.0 = pure proportional, 1.0 = uniform; only used by wealth_proportional
    },
    "simulation": {
        "tick_interval": 30,
        "max_ticks": 2880,
        "max_nodes": 1000,
        "node_prefix": "myc-",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _validate(cfg: dict) -> None:
    eco = cfg["economy"]
    if eco["seed_capital"] <= 0:
        raise ValueError("seed_capital must be positive")
    if eco["rent_per_cycle"] <= 0:
        raise ValueError("rent_per_cycle must be positive")
    if not (0 < eco["inheritance_ratio"] < 1):
        raise ValueError("inheritance_ratio must be between 0 and 1 (exclusive)")
    if eco["min_reserve_cycles"] < 1:
        raise ValueError("min_reserve_cycles must be >= 1")
    if eco["spawn_threshold_cycles"] <= eco["min_reserve_cycles"]:
        raise ValueError("spawn_threshold_cycles must exceed min_reserve_cycles")
    if not (0 <= eco["caution_variance"] <= 1):
        raise ValueError("caution_variance must be between 0 and 1")
    if not (0 <= eco["caution_heritability"] <= 1):
        raise ValueError("caution_heritability must be between 0 and 1")
    if eco["caution_mutation"] < 0:
        raise ValueError("caution_mutation must be non-negative")

    sim = cfg["simulation"]
    if sim["tick_interval"] < 0:
        raise ValueError("tick_interval must be non-negative (0 = run at full speed)")
    if sim["max_ticks"] <= 0:
        raise ValueError("max_ticks must be positive")
    if sim["max_nodes"] <= 0:
        raise ValueError("max_nodes must be positive")

    faucet = cfg["faucet"]
    if faucet["rate"] < 0:
        raise ValueError("faucet rate must be non-negative")
    if faucet.get("growth_rate", 0) < 0:
        raise ValueError("faucet growth_rate must be non-negative")
    if not (0.0 <= faucet.get("noise", 0.0) <= 1.0):
        raise ValueError("faucet noise must be between 0.0 and 1.0")
    if faucet["distribution"] not in (
        "random_living", "poorest", "round_robin", "wealth_proportional"
    ):
        raise ValueError(
            "faucet distribution must be random_living, poorest, round_robin, or wealth_proportional"
        )


def load_config(path: str | None = None) -> dict:
    if path and os.path.isfile(path):
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
    else:
        user_cfg = {}

    cfg = _deep_merge(DEFAULTS, user_cfg)
    _validate(cfg)
    return cfg
