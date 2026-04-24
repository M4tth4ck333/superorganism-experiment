"""
Persistent key-value state backed by SQLite.

Survives os._exit(42) restarts thanks to WAL mode with immediate commits.
"""

import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from config import Config
from utils import setup_logger

logger = setup_logger(__name__, log_file=Config.LOG_DIR / "orchestrator.log", level=Config.LOG_LEVEL)


class NodePersistentState:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    # ── generic get / set / delete ──────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM state WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self._conn.commit()

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM state WHERE key = ?", (key,))
        self._conn.commit()

    # ── caution trait ───────────────────────────────────────────

    def get_caution_trait(self) -> float:
        return float(self.get("caution_trait", 0.5))

    def set_caution_trait(self, value: float) -> None:
        self.set("caution_trait", value)

    # ── spawn guard ─────────────────────────────────────────────

    def is_spawn_in_progress(self) -> bool:
        return bool(self.get("spawn_in_progress", False))

    def get_spawn_id(self) -> str:
        return str(self.get("spawn_id", ""))

    def get_spawn_started_at(self) -> float:
        return float(self.get("spawn_started_at", 0) or 0)

    def mark_spawn_started(self, spawn_id: str) -> None:
        self.set("spawn_in_progress", True)
        self.set("spawn_id", spawn_id)
        self.set("spawn_started_at", time.time())

    def mark_spawn_completed(self, success: bool, child_btc_address: str = "") -> None:
        spawn_id = self.get_spawn_id()
        self.set("spawn_in_progress", False)
        if success:
            history = self.get("spawn_history", [])
            history.append({
                "spawn_id": spawn_id,
                "child_btc_address": child_btc_address,
                "started_at": self.get("spawn_started_at", 0),
                "completed_at": time.time(),
                "success": True,
            })
            self.set("spawn_history", history)
            if spawn_id:
                shutil.rmtree(
                    Config.DATA_DIR / "spawn" / spawn_id,
                    ignore_errors=True,
                )
        self.delete("spawn_id")
        self.delete("spawn_started_at")
        self.delete("spawn_identity")
        self.delete("spawn_vps_info")

    # ── failsafe guard ──────────────────────────────────────────

    def is_failsafe_in_progress(self) -> bool:
        return bool(self.get("failsafe_in_progress", False))

    def mark_failsafe_started(self) -> None:
        self.set("failsafe_in_progress", True)

    def mark_failsafe_completed(self) -> None:
        self.set("failsafe_in_progress", False)


# ── module-level singleton ──────────────────────────────────────

_instance: Optional[NodePersistentState] = None


def init(db_path: Path) -> NodePersistentState:
    global _instance
    _instance = NodePersistentState(db_path)
    logger.info("Persistent state initialized at %s", db_path)
    return _instance


def get() -> Optional[NodePersistentState]:
    return _instance
