"""In-memory node registry."""

import logging

log = logging.getLogger(__name__)


class NodeManager:
    """Tracks living nodes as a simple set."""

    def __init__(self, config: dict):
        self._containers: set[str] = set()

    def create(self, node_id: str) -> bool:
        if node_id in self._containers:
            return False
        self._containers.add(node_id)
        return True

    def start(self, node_id: str, api_base: str = "") -> bool:
        return node_id in self._containers

    def destroy(self, node_id: str) -> bool:
        if node_id not in self._containers:
            return False
        self._containers.discard(node_id)
        return True

    def cleanup_all(self) -> None:
        self._containers.clear()

    def list_containers(self) -> set[str]:
        return set(self._containers)
