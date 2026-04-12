"""
Failsafe pipeline stub.

Triggered by the decision loop when runway drops below FAILSAFE_TRIGGER_DAYS.
Selects the most needy live peer and transfers the node's BTC balance to it,
then lets the VPS expire naturally.
"""

import logging
from typing import List, Optional

from modules.node_monitor import NodeState
from modules.peer_registry import PeerInfo

logger = logging.getLogger(__name__)


def select_best_peer(peers: List[PeerInfo]) -> Optional[PeerInfo]:
    """Return the most needy peer (lowest BTC balance)."""
    if not peers:
        return None
    return min(peers, key=lambda p: p.btc_balance_sat)


async def execute_failsafe(node_state: NodeState, peers: List[PeerInfo]) -> None:
    """Transfer this node's BTC balance to the best available peer."""
    best = select_best_peer(peers)
    if best is None:
        logger.error("[FAILSAFE] No peers available for failsafe")
        return
    target_address = best.btc_address

    # TODO 11: sign and broadcast BTC transfer to target_address
    logger.info(
        "[FAILSAFE STUB] would transfer %d sat to %s — not implemented",
        node_state.btc_balance_sat, target_address,
    )
