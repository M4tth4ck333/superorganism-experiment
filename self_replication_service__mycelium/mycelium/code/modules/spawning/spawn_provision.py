"""Provisions a child VPS via SporeStack and persists connection metadata for SSH deploy."""

import asyncio
from dataclasses import dataclass

from config import Config
from utils import setup_logger
from ..core import state as state_module
from ..monitoring import sporestack_client
from .errors import SpawnError
from .spawn_identity import ChildIdentity

logger = setup_logger(__name__, log_file=Config.LOG_DIR / "orchestrator.log", level=Config.LOG_LEVEL)


@dataclass
class ChildVpsInfo:
    spawn_id: str
    machine_id: str
    host: str           # picked from ipv4/ipv6 per bootstrap logic
    ipv4: str           # raw field kept for diagnostics
    ipv6: str           # raw field kept for diagnostics
    ssh_port: int
    ssh_key_path: str   # carried through from identity for deploy convenience


async def provision_child_vps(identity: ChildIdentity) -> ChildVpsInfo:
    ps = state_module.get()
    if ps is None:
        raise SpawnError("provision", "persistent state not initialised")

    logger.info(
        "Provisioning VPS for %s (hostname=%s, provider=%s, flavor=%s, region=%s)",
        identity.spawn_id, identity.spawn_id,
        Config.VPS_PROVIDER, Config.VPS_FLAVOR, Config.VPS_REGION,
    )

    machine_id = await asyncio.to_thread(
        sporestack_client.launch_server,
        identity.sporestack_token,
        identity.ssh_public_key,
        hostname=identity.spawn_id,
    )
    if not machine_id:
        raise SpawnError("provision", "launch_server returned no machine_id")

    server = await asyncio.to_thread(
        sporestack_client.wait_for_server_ready,
        identity.sporestack_token,
        machine_id,
    )
    if server is None:
        raise SpawnError(
            "provision",
            f"machine {machine_id} not ready within timeout",
        )

    ipv4 = server.get("ipv4") or ""
    ipv6 = server.get("ipv6") or ""
    has_ipv4 = bool(ipv4) and ipv4 != "0.0.0.0"
    has_ipv6 = bool(ipv6) and ipv6 not in ("", "::")
    host = ipv4 if has_ipv4 else (ipv6 if has_ipv6 else None)
    if host is None:
        raise SpawnError(
            "provision",
            f"machine {machine_id} has no usable IPv4/IPv6",
        )
    ssh_port = int(server.get("ssh_port", 22))

    vps_info = ChildVpsInfo(
        spawn_id=identity.spawn_id,
        machine_id=machine_id,
        host=host,
        ipv4=ipv4,
        ipv6=ipv6,
        ssh_port=ssh_port,
        ssh_key_path=identity.ssh_private_key_path,
    )

    ps.set("spawn_vps_info", {
        "spawn_id": vps_info.spawn_id,
        "machine_id": vps_info.machine_id,
        "host": vps_info.host,
        "ipv4": vps_info.ipv4,
        "ipv6": vps_info.ipv6,
        "ssh_port": vps_info.ssh_port,
        "ssh_key_path": vps_info.ssh_key_path,
    })

    logger.info(
        "Child VPS ready: spawn_id=%s machine_id=%s host=%s:%d",
        identity.spawn_id, machine_id, host, ssh_port,
    )

    return vps_info
