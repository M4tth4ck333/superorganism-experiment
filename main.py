from __future__ import annotations

import atexit
import json
import logging.config
import pathlib
import sys
from logging.handlers import QueueHandler
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent / "torrent_health_and_investment"))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from config import DATA_PATH
from democracy.models.issue import Issue
from democracy.models.issue_vote import IssueVote
from democracy.models.person import Person
from democracy.models.solution import Solution
from democracy.models.solution_vote import SolutionVote
from democracy.network.ipv8_thread import IPv8Thread
from democracy.storage.json_store import JSONStore
from healthchecker.db import init_db
from healthchecker.health_thread import TorrentHealthThread
from ui.app import Application
from ui.common.fonts import load_application_fonts


def setup_logging() -> None:
    config_file = pathlib.Path(__file__).parent / "logging_config.json"

    with open(config_file, "r", encoding="utf-8") as f:
        logging_config = json.load(f)

    _ensure_log_directories_exist(logging_config, base_dir=config_file.parent)

    logging.config.dictConfig(logging_config)

    queue_listeners = []

    for handler_name in logging_config.get("handlers", {}):
        handler = logging.getHandlerByName(handler_name)

        if isinstance(handler, QueueHandler):
            listener = getattr(handler, "listener", None)

            if listener is not None:
                listener.start()
                queue_listeners.append(listener)

    if queue_listeners:
        atexit.register(_stop_queue_listeners, queue_listeners)


def _ensure_log_directories_exist(logging_config: dict, base_dir: pathlib.Path) -> None:
    """
    Create parent directories for all file-based logging handlers.

    Relative log file paths are resolved relative to the logging config file. The log file
    itself does not need to be created manually; the file handler creates it when logging
    starts.

    :param logging_config: The logging configuration dictionary.
    :param base_dir: The directory relative paths should be resolved against.
    :returns: None.
    """
    handlers = logging_config.get("handlers", {})

    for handler_config in handlers.values():
        filename = handler_config.get("filename")

        if filename is None:
            continue

        log_file = pathlib.Path(filename)

        if not log_file.is_absolute():
            log_file = base_dir / log_file
            handler_config["filename"] = str(log_file)

        log_file.parent.mkdir(parents=True, exist_ok=True)


def _stop_queue_listeners(queue_listeners: list) -> None:
    for listener in queue_listeners:
        listener.stop()


# -----------------------------
# App entrypoint
# -----------------------------
def main() -> None:
    # --- Logging ---
    setup_logging()

    # --- Session user ---
    user = Person()  # Person generates a random ID by default

    # --- Data stores ---
    base_path = Path(DATA_PATH) / str(user.id)

    issue_store = JSONStore[Issue](
        path=base_path / "issues.json",
        model_factory=Issue.from_dict,
        dictify=lambda i: i.to_dict(),
    )

    solution_store = JSONStore[Solution](
        path=base_path / "solutions.json",
        model_factory=Solution.from_dict,
        dictify=lambda s: s.to_dict(),
    )

    issue_vote_store = JSONStore[IssueVote](
        path=base_path / "issue_votes.json",
        model_factory=IssueVote.from_dict,
        dictify=lambda iv: iv.to_dict(),
    )

    solution_vote_store = JSONStore[SolutionVote](
        path=base_path / "solution_votes.json",
        model_factory=SolutionVote.from_dict,
        dictify=lambda sv: sv.to_dict(),
    )

    # --- UI creation (main thread) ---
    app = QApplication(sys.argv)
    load_application_fonts()

    # --- Torrent health ---
    init_db()
    KEY_FILE = str(
        Path(__file__).parent / "torrent_health_and_investment" / "liberation_key.pem"
    )
    health_thread = TorrentHealthThread(key_file=KEY_FILE)
    health_thread.error.connect(lambda msg: print("Health error:", msg))
    health_thread.startedOk.connect(lambda: print("Health thread started"))
    health_thread.start()

    worker: Optional[IPv8Thread] = None

    def broadcast_new_issue(i: Issue) -> None:
        if worker is not None:
            worker.broadcastIssue.emit(i)

    def broadcast_new_issue_vote(iv: IssueVote) -> None:
        if worker is not None:
            worker.broadcastIssueVote.emit(iv)

    def broadcast_new_solution(s: Solution) -> None:
        if worker is not None:
            worker.broadcastSolution.emit(s)

    def broadcast_new_solution_vote(sv: SolutionVote) -> None:
        if worker is not None:
            worker.broadcastSolutionVote.emit(sv)

    window = Application(
        user,
        issue_store,
        issue_vote_store,
        solution_store,
        solution_vote_store,
        broadcast_new_issue,
        broadcast_new_issue_vote,
        broadcast_new_solution,
        broadcast_new_solution_vote,
        health_thread,
    )

    # Start IPv8 in QThread
    worker = IPv8Thread(
        user.id, issue_store, issue_vote_store, solution_store, solution_vote_store
    )
    worker.dataChanged.connect(
        window.schedule_refresh, type=Qt.ConnectionType.QueuedConnection
    )
    worker.error.connect(
        lambda msg: print("IPv8 error:", msg), type=Qt.ConnectionType.QueuedConnection
    )
    worker.startedOk.connect(
        lambda: print("IPv8 started"), type=Qt.ConnectionType.QueuedConnection
    )
    worker.start()

    # --- Run UI ---
    try:
        window.show()
        sys.exit(app.exec())
    finally:
        # Stop the background loop when the application exits
        health_thread.stop()
        health_thread.wait(1000)
        if worker is not None:
            worker.stop()
            worker.wait(1000)


if __name__ == "__main__":
    main()
