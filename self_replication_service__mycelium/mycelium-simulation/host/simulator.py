#!/usr/bin/env python3
"""Main simulator: tick loop and lifecycle orchestration."""

import argparse
import logging
import random
import signal
import sys
import threading

import agent.agent as agent_module
from host.config_loader import load_config
from host.event_log import EventLog
from host.faucet import Faucet
from host.ledger import Ledger
from host.node_manager import NodeManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("simulator")

SPAWNS_PER_TICK = 5


class Simulator:
    def __init__(self, config: dict):
        self.config = config
        self.eco = config["economy"]
        self.sim = config["simulation"]

        self.ledger = Ledger(self.eco["seed_capital"])
        self.event_log = EventLog()
        self.faucet = Faucet(
            self.ledger,
            config["faucet"]["rate"],
            config["faucet"]["distribution"],
            config["faucet"]["enabled"],
            growth_rate=config["faucet"]["growth_rate"],
            noise=config["faucet"]["noise"],
        )

        self.manager = NodeManager(config)

        self._node_counter = 0
        self._node_traits: dict[str, dict] = {}
        self._tick = 0
        self._stop = threading.Event()

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"{self.config['simulation']['node_prefix']}{self._node_counter:04d}"

    def _assign_caution(self, node_id: str, parent_id: str = "") -> None:
        variance = self.eco.get("caution_variance", 0.0)
        if variance <= 0:
            self._node_traits[node_id] = {"caution": 0.0}
            return

        parent_traits = self._node_traits.get(parent_id)
        if parent_traits and parent_id:
            heritability = self.eco.get("caution_heritability", 0.8)
            mutation_std = self.eco.get("caution_mutation", 0.1)
            parent_caution = parent_traits["caution"]
            fresh = random.uniform(0, variance)
            caution = heritability * parent_caution + (1 - heritability) * fresh
            caution += random.gauss(0, mutation_std)
        else:
            caution = random.uniform(0, variance)

        caution = max(0.0, min(caution, variance))
        self._node_traits[node_id] = {"caution": caution}

    def _spawn_node(self, initial_balance: int, parent_id: str = "") -> str | None:
        living = self.ledger.living_nodes()
        if len(living) >= self.sim["max_nodes"]:
            log.warning("Max nodes (%d) reached, rejecting spawn", self.sim["max_nodes"])
            return None

        node_id = self._next_node_id()
        self.ledger.create_node(node_id, initial_balance)

        if not self.manager.create(node_id):
            self.ledger.remove_node(node_id)
            return None

        if not self.manager.start(node_id):
            self.manager.destroy(node_id)
            self.ledger.remove_node(node_id)
            return None

        self._assign_caution(node_id, parent_id)
        caution = self._node_traits[node_id]["caution"]
        self.event_log.log(
            self._tick, "spawn", node_id,
            f"balance={initial_balance} parent={parent_id} caution={caution:.3f}",
        )
        log.info("Spawned %s (balance=%d, parent=%s, caution=%.3f)", node_id, initial_balance, parent_id or "genesis", caution)
        return node_id

    def _kill_node(self, node_id: str, reason: str = "bankrupt") -> None:
        remaining = self.ledger.balance(node_id)
        if remaining > 0:
            peers = [n for n in self.ledger.living_nodes() if n != node_id]
            if peers:
                recipient = random.choice(peers)
                self.ledger.transfer(node_id, recipient, remaining)
                self.event_log.log(
                    self._tick, "inheritance", recipient,
                    f"from={node_id} amount={remaining} reason={reason}_sweep",
                )
                log.info("Swept %d sat from %s to %s on death", remaining, node_id, recipient)
        self.ledger.remove_node(node_id)
        self._node_traits.pop(node_id, None)
        self.manager.destroy(node_id)
        self.event_log.log(self._tick, "death", node_id, f"reason={reason} remaining={remaining}")
        log.info("Killed %s (reason=%s, remaining=%d)", node_id, reason, remaining)

    def _deduct_rent(self) -> list[str]:
        rent = self.eco["rent_per_cycle"]
        bankrupt = []
        for node_id in self.ledger.living_nodes():
            bal = self.ledger.balance(node_id)
            if bal < rent:
                bankrupt.append(node_id)
            else:
                self.ledger.deduct_rent(node_id, rent)
                self.event_log.log(
                    self._tick, "rent", node_id,
                    f"amount={rent} remaining={self.ledger.balance(node_id)}",
                )
        return bankrupt

    def _agent_decide_phase(self) -> list[tuple[str, dict]]:
        living = list(self.ledger.living_nodes())
        random.shuffle(living)
        balances = self.ledger.all_balances()
        peers = {n: balances[n] for n in living}
        decisions = []
        for node_id in living:
            bal = self.ledger.balance(node_id)
            traits = self._node_traits.get(node_id, {})
            caution = traits.get("caution", 0.0)
            if caution > 0:
                node_eco = dict(self.eco)
                node_eco["min_reserve_cycles"] = self.eco["min_reserve_cycles"] * (1 + caution)
                node_eco["spawn_threshold_cycles"] = self.eco["spawn_threshold_cycles"] * (1 + caution)
            else:
                node_eco = self.eco
            decision = agent_module.decide(node_id, bal, peers, node_eco)
            decisions.append((node_id, decision))
        return decisions

    def _process_decisions(self, decisions: list[tuple[str, dict]]) -> None:
        failsafe_nodes = []
        spawn_count = 0

        for node_id, decision in decisions:
            action = decision.get("action", "none")

            if action == "failsafe":
                target = decision.get("target")
                bal = self.ledger.balance(node_id)
                if target and bal > 0:
                    self.ledger.transfer(node_id, target, bal)
                    self.event_log.log(
                        self._tick, "inheritance", target,
                        f"from={node_id} amount={bal} reason=failsafe",
                    )
                    log.info("Failsafe: %s donated %d to %s", node_id, bal, target)
                failsafe_nodes.append(node_id)

            elif action == "spawn" and spawn_count < SPAWNS_PER_TICK:
                inheritance = decision.get("inheritance", 0)
                parent_bal = self.ledger.balance(node_id)
                if parent_bal < inheritance:
                    log.warning(
                        "Spawn rejected: %s can't afford %d (has %d)",
                        node_id, inheritance, parent_bal,
                    )
                    continue
                child_id = self._spawn_node(0, node_id)
                if child_id:
                    self.ledger.transfer(node_id, child_id, inheritance)
                    self.event_log.log(
                        self._tick, "inheritance", child_id,
                        f"from={node_id} amount={inheritance}",
                    )
                spawn_count += 1

        for node_id in failsafe_nodes:
            if node_id in self.ledger.all_balances():
                self._kill_node(node_id, reason="failsafe")

    def _check_conservation(self) -> None:
        ok, msg = self.ledger.check_conservation()
        if not ok:
            log.error("CONSERVATION VIOLATION: %s", msg)
            self.event_log.log(self._tick, "error", "", msg)
        else:
            log.debug(msg)

    def tick(self) -> dict:
        self._tick += 1
        log.info("=== Tick %d ===", self._tick)

        bankrupt = self._deduct_rent()

        for node_id in bankrupt:
            self._kill_node(node_id, reason="bankrupt")

        for node_id, amount in self.faucet.drip(tick=self._tick):
            self.event_log.log(self._tick, "faucet", node_id, f"amount={amount}")

        decisions = self._agent_decide_phase()
        self._process_decisions(decisions)

        self._check_conservation()
        living = self.ledger.living_nodes()
        balances = self.ledger.all_balances()
        total_bal = sum(balances.get(n, 0) for n in living)
        stats = {
            "tick": self._tick,
            "living": len(living),
            "total_balance": total_bal,
            "total_faucet": self.ledger.total_faucet,
            "total_rent": self.ledger.total_rent_collected,
        }
        self.event_log.log(
            self._tick, "summary", "",
            f"living={stats['living']} total_bal={total_bal}",
        )

        if self.eco.get("caution_variance", 0) > 0 and living:
            caution_vals = [
                self._node_traits[n]["caution"]
                for n in living if n in self._node_traits
            ]
            if caution_vals:
                avg_c = sum(caution_vals) / len(caution_vals)
                min_c = min(caution_vals)
                max_c = max(caution_vals)
                self.event_log.log(
                    self._tick, "trait_summary", "",
                    f"avg_caution={avg_c:.4f} min_caution={min_c:.4f} max_caution={max_c:.4f}",
                )

        log.info("Living: %d", stats["living"])
        return stats

    def run(self) -> None:
        self._spawn_node(self.eco["seed_capital"])
        self.event_log.log(0, "genesis", "", f"seed_capital={self.eco['seed_capital']}")

        try:
            while self._tick < self.sim["max_ticks"] and not self._stop.is_set():
                stats = self.tick()

                if stats["living"] == 0:
                    log.info("All nodes dead. Simulation over at tick %d.", self._tick)
                    break

                if self.sim["tick_interval"] > 0:
                    self._stop.wait(self.sim["tick_interval"])
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        log.info("Shutting down...")
        self._stop.set()
        self.manager.cleanup_all()
        self.event_log.close()
        log.info("Done. %d ticks completed.", self._tick)


def main():
    parser = argparse.ArgumentParser(description="Mycelium Economic Simulation")
    parser.add_argument("-c", "--config", default="config/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)

    log.info("Seed capital: %d sat", config["economy"]["seed_capital"])
    log.info("Rent/cycle: %d sat", config["economy"]["rent_per_cycle"])
    log.info("Faucet: %d sat/tick (%s)", config["faucet"]["rate"], config["faucet"]["distribution"])
    log.info("Ticks: %d @ %ds interval", config["simulation"]["max_ticks"], config["simulation"]["tick_interval"])

    sim = Simulator(config)

    def handle_signal(signum, frame):
        log.info("Signal %d received, stopping...", signum)
        sim._stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    sim.run()


if __name__ == "__main__":
    main()
