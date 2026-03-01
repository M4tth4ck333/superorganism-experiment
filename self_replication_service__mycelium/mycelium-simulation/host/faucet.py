"""Periodic BTC injection into living nodes."""

import random

from host.ledger import Ledger


class Faucet:

    def __init__(self, ledger: Ledger, rate: int, distribution: str, enabled: bool,
                 growth_rate: int = 0, noise: float = 0.0):
        self._ledger = ledger
        self._rate = rate
        self._distribution = distribution
        self._enabled = enabled
        self._growth_rate = growth_rate
        self._noise = noise          # only used by wealth_proportional
        self._round_robin_idx = 0

    def _effective_rate(self, tick: int) -> int:
        return self._rate + self._growth_rate * tick

    def drip(self, tick: int = 0) -> list[tuple[str, int]]:
        effective_rate = self._effective_rate(tick)
        if not self._enabled or effective_rate <= 0:
            return []

        living = self._ledger.living_nodes()
        if not living:
            return []

        if self._distribution == "random_living":
            target = random.choice(living)
            self._ledger.inject_faucet(target, effective_rate)
            return [(target, effective_rate)]

        elif self._distribution == "poorest":
            balances = self._ledger.all_balances()
            target = min(living, key=lambda nid: balances.get(nid, 0))
            self._ledger.inject_faucet(target, effective_rate)
            return [(target, effective_rate)]

        elif self._distribution == "round_robin":
            self._round_robin_idx = self._round_robin_idx % len(living)
            target = sorted(living)[self._round_robin_idx]
            self._round_robin_idx += 1
            self._ledger.inject_faucet(target, effective_rate)
            return [(target, effective_rate)]

        elif self._distribution == "wealth_proportional":
            balances = self._ledger.all_balances()
            living_balances = {nid: balances[nid] for nid in living}
            total_wealth = sum(living_balances.values())  # > 0 guaranteed by living_nodes()
            n = len(living)

            shares: dict[str, int] = {}
            for nid, bal in living_balances.items():
                proportional = (effective_rate * bal) // total_wealth
                uniform = effective_rate // n
                noise_k = int(self._noise * 1000)
                shares[nid] = (proportional * (1000 - noise_k) + uniform * noise_k) // 1000

            remainder = effective_rate - sum(shares.values())
            richest = max(living, key=lambda nid: (living_balances[nid], nid))
            shares[richest] += remainder

            result = []
            for nid, amount in shares.items():
                if amount > 0:
                    self._ledger.inject_faucet(nid, amount)
                    result.append((nid, amount))
            return result

        else:
            target = random.choice(living)
            self._ledger.inject_faucet(target, effective_rate)
            return [(target, effective_rate)]
