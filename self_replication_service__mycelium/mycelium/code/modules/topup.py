"""
Ensures the SporeStack account balance covers TOPUP_TARGET_DAYS of burn.
Called by the decision loop when runway drops below TOPUP_TRIGGER_DAYS.
"""

import logging

from config import Config
from modules.node_monitor import NodeState

logger = logging.getLogger(__name__)


async def topup_sporestack(node_state: NodeState) -> None:
    """
    Ensure SporeStack account balance covers TOPUP_TARGET_DAYS of burn.

    TODO: calculate shortfall in cents, convert BTC from wallet to SporeStack tokens
    via SporeStack payment API, then verify balance updated.
    """
    if node_state.burn_rate_cents_per_day is None or node_state.burn_rate_cents_per_day == 0:
        logger.warning("[TOPUP] Burn rate unknown, cannot compute topup amount")
        return

    target_cents = node_state.burn_rate_cents_per_day * Config.TOPUP_TARGET_DAYS
    current_cents = node_state.sporestack_balance_cents or 0
    shortfall_cents = target_cents - current_cents

    if shortfall_cents <= 0:
        logger.info("[TOPUP] SporeStack balance already sufficient")
        return

    # TODO: convert shortfall_cents worth of BTC to SporeStack tokens
    logger.info(
        "[TOPUP STUB] would top up %d cents ($%.2f) — not implemented",
        shortfall_cents, shortfall_cents / 100,
    )
