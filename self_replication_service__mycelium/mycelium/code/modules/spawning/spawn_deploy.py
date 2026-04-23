"""
SSHes into the provisioned child VPS and lays down the code + content the child
needs before 10.6 writes secrets and boots the orchestrator: installs deps,
configures the firewall, sparse-clones the monorepo, and uploads the CC
video-ID list + YouTube cookies.

Returns the connected SSHDeployer so 10.6 can reuse the same session; 10.9
owns the final disconnect. On any failure, disconnects and raises
SpawnDeployError; deployer.spawn_child catches it and leaves
spawn_in_progress=True so the whole spawn retries from scratch on restart.
"""

import asyncio

from config import Config
from utils import setup_logger
from .spawn_identity import ChildIdentity
from .spawn_provision import ChildVpsInfo
from .ssh_deployer import SSHDeployer

logger = setup_logger(__name__, log_file=Config.LOG_DIR / "orchestrator.log", level=Config.LOG_LEVEL)


class SpawnDeployError(Exception):
    """Any failure during child code/content deployment."""
    pass


async def deploy_child_code(
    identity: ChildIdentity,
    vps_info: ChildVpsInfo,
) -> SSHDeployer:
    """SSH into the child VPS and install code + content files.

    Returns the connected SSHDeployer so 10.6 can reuse it to write secrets
    and start the orchestrator. Caller (10.9) owns disconnect.
    """
    logger.info(
        "Deploying code to child VPS: child_token=%s host=%s:%d",
        identity.child_token, vps_info.host, vps_info.ssh_port,
    )

    deployer = SSHDeployer(ssh_key_path=vps_info.ssh_key_path)

    try:
        await asyncio.to_thread(
            deployer.connect, vps_info.host, port=vps_info.ssh_port
        )
        await asyncio.to_thread(deployer.install_dependencies)
        await asyncio.to_thread(deployer.setup_firewall)
        await asyncio.to_thread(deployer.deploy_mycelium)
        await asyncio.to_thread(
            deployer.deploy_video_ids, str(Config.VIDEO_IDS_FILE)
        )
        await asyncio.to_thread(
            deployer.deploy_cookies, str(Config.COOKIES_FILE)
        )
    except Exception as e:
        deployer.disconnect()
        raise SpawnDeployError(
            f"Child code deployment failed for {identity.child_token}: {e}"
        ) from e

    logger.info(
        "Child code deployed: child_token=%s host=%s",
        identity.child_token, vps_info.host,
    )
    return deployer
