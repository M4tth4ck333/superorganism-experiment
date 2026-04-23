from config import Config
from utils import setup_logger
from ..core import state as state_module
from ..monitoring.node_monitor import NodeState
from .spawn_identity import prepare_child_identity, SpawnIdentityError
from .spawn_provision import provision_child_vps, SpawnProvisionError
from .spawn_deploy import deploy_child_code, boot_child_orchestrator, SpawnDeployError
from .spawn_transfer import transfer_inheritance, SpawnTransferError

logger = setup_logger(__name__, log_file=Config.LOG_DIR / "orchestrator.log", level=Config.LOG_LEVEL)


async def spawn_child(node_state: NodeState, caution_trait: float, child_token: str) -> None:
    """Orchestrate the full spawn pipeline. On failure, leave spawn_in_progress=True for retry."""
    ps = state_module.get()
    ssh_deployer = None
    logger.info("=== Spawn pipeline start: child_token=%s ===", child_token)
    try:
        logger.info("[1/5] Preparing child identity: child_token=%s", child_token)
        identity = await prepare_child_identity(child_token, node_state)

        logger.info("[2/5] Provisioning child VPS: child_token=%s", child_token)
        vps_info = await provision_child_vps(identity)

        logger.info(
            "[3/5] Deploying child code: child_token=%s host=%s",
            child_token, vps_info.host,
        )
        ssh_deployer = await deploy_child_code(identity, vps_info)

        logger.info("[4/5] Booting child orchestrator: child_token=%s", child_token)
        child_caution = await boot_child_orchestrator(ssh_deployer, identity, caution_trait)

        logger.info("[5/5] Transferring inheritance BTC: child_token=%s", child_token)
        txid = await transfer_inheritance(identity, node_state)

        ps.delete("spawn_vps_info")

        logger.info(
            "=== Spawn complete: child_token=%s child_btc=%s txid=%s caution=%.3f ===",
            child_token, identity.btc_address, txid, child_caution,
        )
    except (SpawnIdentityError, SpawnProvisionError, SpawnDeployError, SpawnTransferError) as e:
        logger.error(
            "Spawn pipeline failed: child_token=%s — %s. "
            "spawn_in_progress=True preserved; will retry on next restart.",
            child_token, e,
        )
    except Exception:
        logger.exception(
            "Unexpected error in spawn pipeline: child_token=%s. "
            "spawn_in_progress=True preserved; will retry on next restart.",
            child_token,
        )
    finally:
        if ssh_deployer is not None:
            ssh_deployer.disconnect()
