"""Provisions a child VPS via SporeStack and persists connection metadata for SSH deploy.

Wraps `launch_server` with a write-ahead `spawn_vps_intent` record. On resume,
if intent exists but `spawn_vps_info` doesn't, query `get_servers(token)`: any
server found under our token is by definition the one we just launched
(the token is single-use by this spawn), so we adopt it instead of launching
a second VPS that would silently consume the same budget.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

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


def _pick_host(server: dict) -> tuple[str, str, Optional[str]]:
    """Apply the bootstrap's ipv4/ipv6 selection → (ipv4, ipv6, host-or-None)."""
    ipv4 = server.get("ipv4") or ""
    ipv6 = server.get("ipv6") or ""
    has_ipv4 = bool(ipv4) and ipv4 != "0.0.0.0"
    has_ipv6 = bool(ipv6) and ipv6 not in ("", "::")
    host = ipv4 if has_ipv4 else (ipv6 if has_ipv6 else None)
    return ipv4, ipv6, host


def _adopt_orphan_server(sporestack_token: str) -> Optional[dict]:
    """Look for a server already launched under this token (from a prior crashed launch).

    The token is allocated per-spawn, so any server under it is ours. If multiple
    exist (a diagnostic — should not happen on a single-restart path), log CRITICAL
    and pick the most recently created one so the pipeline can proceed.
    """
    servers = sporestack_client.get_servers(sporestack_token)
    if not servers:
        return None
    if len(servers) > 1:
        ids = [s.get("machine_id") for s in servers]
        logger.critical(
            "Multiple servers (%d) found under spawn token — orphaned from a prior failure. "
            "machine_ids=%s. Proceeding with the most recently created one; operator must "
            "manually reconcile the others (SporeStack delete does NOT refund).",
            len(servers), ids,
        )

    def _created_at(s):
        return s.get("created_at") or s.get("created") or 0
    return sorted(servers, key=_created_at)[-1]


async def provision_child_vps(identity: ChildIdentity) -> ChildVpsInfo:
    ps = state_module.get()
    if ps is None:
        raise SpawnError("provision", "persistent state not initialised")

    # Resume path: intent exists but no vps_info → a crash might have left a
    # live server behind. Reconcile via the SporeStack API before re-launching.
    intent = ps.get("spawn_vps_intent")
    server: Optional[dict] = None
    if (
        intent
        and intent.get("spawn_id") == identity.spawn_id
        and intent.get("sporestack_token") == identity.sporestack_token
    ):
        logger.info(
            "Resume: spawn_vps_intent found — checking SporeStack for orphaned server",
        )
        server = await asyncio.to_thread(_adopt_orphan_server, identity.sporestack_token)
        if server is not None:
            machine_id = server.get("machine_id")
            logger.info(
                "Resume: adopting orphaned server machine_id=%s — skipping re-launch",
                machine_id,
            )

    if server is None:
        logger.info(
            "Provisioning VPS for %s (hostname=%s, provider=%s, flavor=%s, region=%s)",
            identity.spawn_id, identity.spawn_id,
            Config.VPS_PROVIDER, Config.VPS_FLAVOR, Config.VPS_REGION,
        )

        ps.set("spawn_vps_intent", {
            "spawn_id": identity.spawn_id,
            "sporestack_token": identity.sporestack_token,
        })

        machine_id = await asyncio.to_thread(
            sporestack_client.launch_server,
            identity.sporestack_token,
            identity.ssh_public_key,
            hostname=identity.spawn_id,
        )
        if not machine_id:
            raise SpawnError("provision", "launch_server returned no machine_id")
    else:
        machine_id = server.get("machine_id")
        if not machine_id:
            raise SpawnError("provision", "adopted server has no machine_id")

    # Whether fresh or adopted, wait for the machine to reach a usable state.
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

    ipv4, ipv6, host = _pick_host(server)
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
