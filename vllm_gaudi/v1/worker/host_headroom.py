# SPDX-License-Identifier: Apache-2.0
"""Host memory headroom check shared by the EngineCore sleep guard and the engine reconfigure hook.

Kept free of HPU imports so the engine process can use it without loading the device stack.
"""
import math
import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

SKIP_HOST_GUARD_ENV = "VLLM_GAUDI_SKIP_SLEEP_HOST_GUARD"
HOST_RESERVE_ENV = "VLLM_GAUDI_SLEEP_HOST_RESERVE_MB"
_DEFAULT_HOST_RESERVE_MB = 1024
# Moving weights to CPU measured at 1.001x the device-resident bytes on Gaudi 3.
_REQUIRED_OVERHEAD = 0.01
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC = Path("/proc")
_CGROUP_V1_UNLIMITED = 1 << 60


def host_guard_disabled() -> bool:
    return os.environ.get(SKIP_HOST_GUARD_ENV, "0") == "1"


def _host_reserve_bytes() -> int:
    try:
        return max(0, int(os.environ.get(HOST_RESERVE_ENV, _DEFAULT_HOST_RESERVE_MB))) * 2**20
    except ValueError:
        return _DEFAULT_HOST_RESERVE_MB * 2**20


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return None


def _read_stat(path: Path) -> dict[str, int]:
    """Parse ``key value`` files such as ``memory.stat`` and ``/proc/meminfo`` (kB suffix kept as kB)."""
    stat: dict[str, int] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return stat
    for line in lines:
        fields = line.replace(":", " ").split()
        if len(fields) >= 2 and fields[1].isdigit():
            stat[fields[0]] = int(fields[1])
    return stat


def _own_cgroup_dirs(mount: Path, controller: str | None, proc: Path) -> list[Path]:
    """This process's cgroup directory under ``mount``, followed by its ancestors up to ``mount``.

    Inside a cgroup namespace ``mount`` is already the process's cgroup and ancestors are not
    visible, so the list is just ``[mount]``.
    """
    relative = ""
    try:
        for line in (proc / "self" / "cgroup").read_text().splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if (hierarchy == "0" and controller is None) or (controller in controllers.split(",")):
                relative = path.strip().lstrip("/")
    except (OSError, ValueError):
        pass
    leaf = mount / relative if relative and (mount / relative).is_dir() else mount
    dirs = [leaf]
    while dirs[-1] != mount and mount in dirs[-1].parents:
        dirs.append(dirs[-1].parent)
    return dirs


def _cgroup_v2_levels(dirs: list[Path]) -> list[tuple[int | None, int | None]]:
    levels = []
    for cgroup in dirs:
        try:
            raw_limit = (cgroup / "memory.max").read_text().strip()
        except OSError:
            continue
        usage = _read_int(cgroup / "memory.current")
        if raw_limit == "max" or usage is None:
            memory = None
        else:
            stat = _read_stat(cgroup / "memory.stat")
            reclaimable = max(0, stat.get("file", 0) - stat.get("shmem", 0))
            memory = max(0, int(raw_limit) - (usage - reclaimable))
        swap: int | None = None
        try:
            raw_swap = (cgroup / "memory.swap.max").read_text().strip()
            if raw_swap != "max":
                swap = max(0, int(raw_swap) - (_read_int(cgroup / "memory.swap.current") or 0))
        except (OSError, ValueError):
            pass
        levels.append((memory, swap))
    return levels


def _cgroup_v1_levels(dirs: list[Path]) -> list[tuple[int | None, int | None]]:
    levels = []
    for cgroup in dirs:
        limit = _read_int(cgroup / "memory.limit_in_bytes")
        usage = _read_int(cgroup / "memory.usage_in_bytes")
        if limit is None or usage is None:
            continue
        stat = _read_stat(cgroup / "memory.stat")
        # Includes the limits of ancestors that are not visible from inside the container.
        limit = min(limit, stat.get("hierarchical_memory_limit", limit))
        reclaimable = max(0, stat.get("total_cache", 0) - stat.get("total_shmem", 0))
        memory = None if limit >= _CGROUP_V1_UNLIMITED else max(0, limit - (usage - reclaimable))
        swap: int | None = None
        if _read_int(cgroup / "memory.swappiness") == 0:
            swap = 0
        else:
            memsw_limit = _read_int(cgroup / "memory.memsw.limit_in_bytes")
            memsw_usage = _read_int(cgroup / "memory.memsw.usage_in_bytes")
            if memsw_limit is not None and memsw_usage is not None:
                memsw_limit = min(memsw_limit, stat.get("hierarchical_memsw_limit", memsw_limit))
                if memsw_limit < _CGROUP_V1_UNLIMITED:
                    total = max(0, memsw_limit - (memsw_usage - reclaimable))
                    memory = total if memory is None else min(memory, total)
                    swap = total - memory
        levels.append((memory, swap))
    return levels


def host_memory_headroom_bytes(cgroup_root: Path = _CGROUP_ROOT, proc: Path = _PROC) -> int | None:
    """Bytes this process can still allocate on the host before an OOM kill, or None if unknown.

    The smallest of the memory limits of this cgroup and its visible ancestors (cgroup v2, or
    v1 with ``hierarchical_memory_limit``) and of the host's ``MemAvailable``, plus the swap
    those levels allow. Page cache (e.g. mmapped safetensors) is reclaimable, so it is not
    counted as used. Shared memory (tmpfs, /dev/shm) is accounted as page cache but cannot be
    reclaimed without swap, so it stays counted.
    """
    if (cgroup_root / "cgroup.controllers").exists():
        levels = _cgroup_v2_levels(_own_cgroup_dirs(cgroup_root, None, proc))
    else:
        mount = cgroup_root / "memory"
        levels = _cgroup_v1_levels(_own_cgroup_dirs(mount, "memory", proc)) if mount.is_dir() else []
    meminfo = _read_stat(proc / "meminfo")
    host_swap = meminfo.get("SwapFree", 0) * 1024
    memory_limits = [memory for memory, _ in levels if memory is not None]
    if "MemAvailable" in meminfo:
        memory_limits.append(meminfo["MemAvailable"] * 1024)
    if not memory_limits:
        return None
    swap = min([host_swap] + [swap for _, swap in levels if swap is not None])
    return min(memory_limits) + swap


def host_identity(cgroup_root: Path = _CGROUP_ROOT, proc: Path = _PROC) -> str:
    """Key shared by the workers that draw on the same host memory.

    Hostnames can repeat across pods or machines, so the key adds the kernel boot id and the
    inode of this process's memory cgroup, which separates containers with their own limits.
    """
    try:
        boot_id = (proc / "sys" / "kernel" / "random" / "boot_id").read_text().strip()
    except OSError:
        boot_id = ""
    v1_mount = cgroup_root / "memory"
    if (cgroup_root / "cgroup.controllers").exists() or not v1_mount.is_dir():
        cgroup = _own_cgroup_dirs(cgroup_root, None, proc)[0]
    else:
        cgroup = _own_cgroup_dirs(v1_mount, "memory", proc)[0]
    try:
        inode = str(cgroup.stat().st_ino)
    except OSError:
        inode = ""
    return f"{socket.gethostname()}/{boot_id}/{inode}"


def _host_shortfall(worker_reports: Any) -> str | None:
    """Describe the first host whose workers would not fit in its headroom, or None.

    Each report is ``{"host", "host_id", "required_bytes", "headroom_bytes"}`` from
    ``HPUWorker.check_sleep_host_headroom``. Workers on one host share its memory and move
    their shards to CPU concurrently, so requirements are summed per host and compared
    with the smallest headroom reported on that host. Missing or malformed reports pass.
    """
    if not isinstance(worker_reports, (list, tuple)):
        return None
    per_host: dict[Any, tuple[Any, int, int | None]] = {}
    for report in worker_reports:
        if not isinstance(report, dict):
            continue
        key = report.get("host_id", report.get("host"))
        _, required, headroom = per_host.get(key, (None, 0, None))
        reported_required = report.get("required_bytes")
        if isinstance(reported_required, int):
            required += reported_required
        reported = report.get("headroom_bytes")
        if isinstance(reported, int):
            headroom = reported if headroom is None else min(headroom, reported)
        per_host[key] = (report.get("host"), required, headroom)
    reserve = _host_reserve_bytes()
    for host, required, headroom in per_host.values():
        if required <= 0 or headroom is None:
            continue
        needed = math.ceil(required * (1 + _REQUIRED_OVERHEAD)) + reserve
        if headroom < needed:
            return (f"insufficient host memory to move the model to CPU on {host or 'host'} "
                    f"(required={required / 2**30:.1f}GiB plus {_REQUIRED_OVERHEAD:.0%} and "
                    f"{reserve / 2**30:.1f}GiB reserve, headroom={headroom / 2**30:.1f}GiB)")
    return None


def raise_if_reports_exceed_host_headroom(worker_reports: Any, *, action: str, outcome: str) -> None:
    shortfall = _host_shortfall(worker_reports)
    if shortfall is not None:
        raise RuntimeError(f"{action} aborted: {shortfall}; {outcome}. Set {SKIP_HOST_GUARD_ENV}=1 to bypass "
                           f"or {HOST_RESERVE_ENV} to change the reserve.")


def collect_host_headroom_reports(engine_core: Any, trim: bool) -> list[Any]:
    """Every worker's ``check_sleep_host_headroom`` report, as seen by this engine.

    With ``external_launcher`` each rank runs its own engine and ``collective_rpc`` only
    reaches the local worker, so the reports are all-gathered over the world group to give
    every rank the same decision. Every rank calls ``sleep`` together in that mode.
    """
    reports = list(engine_core.collective_rpc("check_sleep_host_headroom", kwargs={"trim": trim}) or [])
    parallel_config = getattr(getattr(engine_core, "vllm_config", None), "parallel_config", None)
    if getattr(parallel_config, "distributed_executor_backend", None) != "external_launcher":
        return reports
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_world_group
    world = get_world_group()
    if world.world_size <= 1:
        return reports
    gathered: list[Any] = [None] * world.world_size
    dist.all_gather_object(gathered, reports, group=world.cpu_group)
    return [report for rank_reports in gathered for report in rank_reports or []]


def guard_host_headroom(collect: Callable[[bool], Any], *, action: str, outcome: str) -> None:
    """Raise before moving the model to CPU when some host cannot hold it.

    The first round skips the garbage collection and allocator trim; they only run, followed by
    a second round, when the first one falls short. If the second round yields no reports, the
    first round's shortfall stands.
    """
    if host_guard_disabled():
        return
    first = collect(False)
    if _host_shortfall(first) is None:
        return
    second = collect(True)
    if not isinstance(second, (list, tuple)) or not any(isinstance(report, dict) for report in second):
        second = first
    raise_if_reports_exceed_host_headroom(second, action=action, outcome=outcome)
