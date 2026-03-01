#!/usr/bin/env python3
"""Mycelium economic agent."""


def compute_failsafe_target(my_balance: int, peers: dict, rent: int, spawn_threshold: int, my_node_id: str = "") -> str | None:
    """Choose the peer where donating all our funds has maximum impact."""
    candidates = []
    for peer_id, peer_bal in peers.items():
        if peer_id == my_node_id:
            continue
        if peer_bal <= 0:
            continue

        peer_after = peer_bal + my_balance
        spawn_amount = rent * spawn_threshold

        if peer_bal < spawn_amount and peer_after >= spawn_amount:
            headroom = (peer_after - spawn_amount) // rent
            score = 3000 + headroom
        elif peer_bal < spawn_amount:
            cycles_gained = my_balance // rent
            score = 2000 + cycles_gained
        else:
            cycles_gained = my_balance // rent
            score = 1000 + cycles_gained

        candidates.append((score, peer_id))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def _compute_spawn(balance: int, config: dict) -> int | None:
    rent = config["rent_per_cycle"]
    min_reserve = config["min_reserve_cycles"]
    spawn_threshold = config["spawn_threshold_cycles"]
    inheritance_ratio = config["inheritance_ratio"]

    if balance < rent * spawn_threshold:
        return None

    reserve = rent * min_reserve
    excess = balance - reserve
    if excess <= 0:
        return None

    child_share = int(excess * inheritance_ratio)
    if child_share < rent * 2:
        return None

    return child_share


def decide(node_id: str, balance: int, peers: dict, config: dict) -> dict:
    rent = config["rent_per_cycle"]
    cycles_affordable = balance // rent if rent > 0 else float("inf")

    if cycles_affordable < 1:
        target = compute_failsafe_target(
            balance, peers, rent, config["spawn_threshold_cycles"], node_id
        )
        return {"action": "failsafe", "target": target}

    spawn = _compute_spawn(balance, config)
    if spawn:
        return {"action": "spawn", "inheritance": spawn}

    return {"action": "none"}
