"""Append-only CSV event logger."""

import csv
import os
import time


class EventLog:

    HEADER = ["timestamp", "tick", "event", "node_id", "detail"]

    def __init__(self, path: str = "data/events.csv"):
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._file.flush()

    def log(self, tick: int, event: str, node_id: str = "", detail: str = "") -> None:
        self._writer.writerow([time.time(), tick, event, node_id, detail])
        self._file.flush()

    def close(self) -> None:
        self._file.close()
