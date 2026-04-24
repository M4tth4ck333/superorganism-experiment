"""Sends inheritance BTC to the child and marks spawn complete in persistent state."""

import asyncio

from config import Config
from utils import setup_logger
from ..core import state as state_module
from ..core import wallet as wallet_module
from ..monitoring.node_monitor import NodeState
from ..orchestration.spawn_thresholds import compute_child_share
from .errors import SpawnError
from .spawn_identity import ChildIdentity

logger = setup_logger(__name__, log_file=Config.LOG_DIR / "orchestrator.log", level=Config.LOG_LEVEL)


async def transfer_inheritance(
    identity: ChildIdentity,
    node_state: NodeState,
) -> str:
    """Send inheritance BTC to child. Returns txid. Leaves spawn_in_progress=True on failure for retry."""
    ps = state_module.get()
    wallet = wallet_module.get_wallet()
    if ps is None:
        raise SpawnError("transfer", "NodePersistentState not initialised")
    if wallet is None:
        raise SpawnError("transfer", "SpendingWallet not initialised")

    # Re-read balance immediately before sizing the share — NodeMonitor's snapshot in node_state
    # is stale by up to MONITOR_INTERVAL AND spawn_identity already spent ~TOPUP_TARGET_DAYS of
    # runway funding SporeStack. On recovery the snapshot is even more stale.
    parent_balance_sat = await asyncio.to_thread(wallet.get_balance_satoshis)
    child_share_sat = compute_child_share(parent_balance_sat)
    logger.info(
        "Transferring inheritance: spawn_id=%s parent_balance=%d sat amount=%d sat to=%s",
        identity.spawn_id, parent_balance_sat, child_share_sat, identity.btc_address,
    )

    try:
        txid = await asyncio.to_thread(
            wallet.send, identity.btc_address, child_share_sat
        )
    except Exception as e:
        raise SpawnError(
            "transfer",
            f"Inheritance transfer failed for {identity.spawn_id}: {e}",
        ) from e

    ps.mark_spawn_completed(success=True, child_btc_address=identity.btc_address)
    logger.info(
        "Spawn complete: spawn_id=%s txid=%s amount=%d sat",
        identity.spawn_id, txid, child_share_sat,
    )
    return txid
