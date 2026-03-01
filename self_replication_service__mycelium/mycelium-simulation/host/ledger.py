"""In-memory ledger with conservation invariant."""

import threading


class Ledger:
    """Tracks satoshi balances for all nodes."""

    def __init__(self, seed_capital: int):
        self._lock = threading.Lock()
        self._balances: dict[str, int] = {}
        self._seed_capital = seed_capital
        self._total_faucet = 0
        self._total_rent_collected = 0

    # --- Queries ---

    def balance(self, node_id: str) -> int:
        with self._lock:
            return self._balances.get(node_id, 0)

    def living_nodes(self) -> list[str]:
        with self._lock:
            return [nid for nid, bal in self._balances.items() if bal > 0]

    def all_balances(self) -> dict[str, int]:
        with self._lock:
            return dict(self._balances)

    @property
    def total_faucet(self) -> int:
        with self._lock:
            return self._total_faucet

    @property
    def total_rent_collected(self) -> int:
        with self._lock:
            return self._total_rent_collected

    @property
    def seed_capital(self) -> int:
        return self._seed_capital

    # --- Mutations ---

    def create_node(self, node_id: str, initial_balance: int) -> None:
        with self._lock:
            if node_id in self._balances:
                raise ValueError(f"Node {node_id} already exists")
            self._balances[node_id] = initial_balance

    def deduct_rent(self, node_id: str, amount: int) -> int:
        with self._lock:
            bal = self._balances.get(node_id)
            if bal is None:
                raise KeyError(f"Node {node_id} not in ledger")
            if bal < amount:
                raise ValueError(f"Node {node_id} has {bal} but rent is {amount}")
            self._balances[node_id] -= amount
            self._total_rent_collected += amount
            return self._balances[node_id]

    def inject_faucet(self, node_id: str, amount: int) -> None:
        with self._lock:
            if node_id not in self._balances:
                raise KeyError(f"Node {node_id} not in ledger")
            self._balances[node_id] += amount
            self._total_faucet += amount

    def transfer(self, from_id: str, to_id: str, amount: int) -> None:
        with self._lock:
            if from_id not in self._balances:
                raise KeyError(f"Node {from_id} not in ledger")
            if to_id not in self._balances:
                raise KeyError(f"Node {to_id} not in ledger")
            if self._balances[from_id] < amount:
                raise ValueError(f"Node {from_id} has {self._balances[from_id]} but needs {amount}")
            self._balances[from_id] -= amount
            self._balances[to_id] += amount

    def remove_node(self, node_id: str) -> int:
        with self._lock:
            bal = self._balances.pop(node_id, 0)
            return bal

    # --- Conservation ---

    def check_conservation(self) -> tuple[bool, str]:
        with self._lock:
            total_balances = sum(self._balances.values())
            lhs = self._seed_capital + self._total_faucet
            rhs = total_balances + self._total_rent_collected
            ok = lhs == rhs
            msg = (
                f"Conservation {'OK' if ok else 'VIOLATED'}: "
                f"seed({self._seed_capital}) + faucet({self._total_faucet}) = {lhs} | "
                f"balances({total_balances}) + rent({self._total_rent_collected}) = {rhs}"
            )
            return ok, msg
