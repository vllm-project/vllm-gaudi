###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Optional CPU/NUMA affinity of an HPU worker process.

Ray workers start without any CPU affinity, so the host Python of a rank may
run on a different NUMA node than its card. ``VLLM_HPU_BIND_NUMA`` pins the
process (and every thread created afterwards, as they inherit the mask) to:

* ``1``    - all CPUs of the NUMA node the card is attached to,
* ``core`` - a disjoint slice of the physical cores of that node (hyperthread
  siblings included); the node's cores are split evenly among the cards
  attached to it, so ranks do not compete for the same cores,
* ``0``    - (default) no binding; the affinity is only logged.

Call it AFTER the device is acquired: the card is the one whose ``/dev/accel``
node the process holds open (``/proc/self/fd`` -> ``/sys/class/accel/accelN/device``
-> PCI address). Ray workers all see the node-wide ``HABANA_VISIBLE_MODULES``
(the platform sets RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES), so that env and
``HL_VISIBLE_PCI_ADDRESS`` are only a fallback, used when they name exactly one card; the log names the source.
Only mode ``core`` resizes the torch thread pool (to the physical cores of the rank's slice).
The mask is applied to every thread of the process. Nothing here raises: on any
failure the process stays unbound.
"""

import contextlib
import glob
import os
import re

import torch

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

_SYSFS = "/sys"
_HABANA_VENDOR = "0x1da3"
_PROC = "/proc"
_DEV = "/dev"
_ACCEL_NODE = re.compile(r"^/dev/(?:accel/)?(accel_controlD|accel|hl_controlD|hl)(\d+)$")


def bind_mode() -> str:
    value = os.environ.get("VLLM_HPU_BIND_NUMA", "0").strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return "0"
    if value in ("1", "true", "on", "yes"):
        return "1"
    if value == "core":
        return "core"
    logger.warning("HPU affinity: unknown VLLM_HPU_BIND_NUMA=%r (expected 0, 1 or core), not binding", value)
    return "0"


def parse_cpu_list(text: str) -> set[int]:
    """Parses a sysfs cpulist such as ``0-3,8,10-11``."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def format_cpu_list(cpus) -> str:
    """Compact form of a CPU set, e.g. ``0-27,112-139``."""
    ordered = sorted(cpus)
    if not ordered:
        return "-"
    ranges = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu != prev + 1:
            ranges.append((start, prev))
            start = cpu
        prev = cpu
    ranges.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _habana_cards(sysfs: str) -> dict[str, int | None]:
    """Maps PCI address -> module id for all Habana cards of the host."""
    cards: dict[str, int | None] = {}
    for dev in sorted(glob.glob(os.path.join(sysfs, "bus/pci/devices/*"))):
        if _read(os.path.join(dev, "vendor")) != _HABANA_VENDOR:
            continue
        pci = os.path.basename(dev)
        cards[pci] = None
        for attr in glob.glob(os.path.join(dev, "accel/accel*/module_id")) + [os.path.join(dev, "module_id")]:
            text = _read(attr)
            if text is not None:
                with contextlib.suppress(ValueError):
                    cards[pci] = int(text, 0)
                break
    return cards


def _device_pci(index: str, prefix: str, sysfs: str) -> str | None:
    cls = "habanalabs" if prefix.startswith("hl") else "accel"
    link = os.path.join(sysfs, "class", cls, f"{prefix.replace('_controlD', '')}{index}", "device")
    if not os.path.exists(link):
        return None
    return os.path.basename(os.path.realpath(link))


def find_opened_card_pci(sysfs: str = _SYSFS, proc: str = _PROC, quiet: bool = False) -> str | None:
    """PCI address of the card whose device node this process holds open, or None."""
    found: list[str] = []
    try:
        fds = os.listdir(os.path.join(proc, "self/fd"))
    except OSError:
        return None
    for fd in fds:
        try:
            target = os.readlink(os.path.join(proc, "self/fd", fd))
        except OSError:
            continue
        match = _ACCEL_NODE.match(target)
        if match:
            pci = _device_pci(match.group(2), match.group(1), sysfs)
            if pci and pci not in found:
                found.append(pci)
    if len(found) > 1:
        note = logger.debug if quiet else logger.warning
        note("HPU affinity: process holds several Habana devices open (%s), not using them", ",".join(found))
        return None
    return found[0] if found else None


def find_env_card_pci(sysfs: str = _SYSFS, environ=None, quiet: bool = False) -> str | None:
    """The card named by the environment, only when it names exactly one.

    Under Ray every worker sees the node-wide HABANA_VISIBLE_MODULES / HL_VISIBLE_PCI_ADDRESS list, from which the
    card of a rank cannot be told, so a list with several entries identifies nothing.
    """
    env = os.environ if environ is None else environ
    modules = [m.strip() for m in env.get("HABANA_VISIBLE_MODULES", "").split(",") if m.strip()]
    if len(modules) == 1:
        with contextlib.suppress(ValueError):
            module_id = int(modules[0])
            for pci, mid in _habana_cards(sysfs).items():
                if mid == module_id:
                    return pci
        note = logger.debug if quiet else logger.warning
        note("HPU affinity: module id %r from HABANA_VISIBLE_MODULES not found in sysfs", modules[0])
    pcis = [p.strip() for p in env.get("HL_VISIBLE_PCI_ADDRESS", "").split(",") if p.strip()]
    return pcis[0] if len(pcis) == 1 else None


def find_card_pci(sysfs: str = _SYSFS, proc: str = _PROC, environ=None, quiet: bool = False) -> tuple[str | None, str]:
    """Returns (pci, source) where source is 'opened-device', 'env' or 'none'.

    ``quiet`` (binding disabled) turns the diagnostics into DEBUG: they are only interesting when we bind.
    """
    note = logger.debug if quiet else logger.warning
    opened = find_opened_card_pci(sysfs, proc, quiet)
    guess = None
    try:
        guess = find_env_card_pci(sysfs, environ, quiet)
    except Exception as exc:
        logger.debug("HPU affinity: env based card lookup failed (%s: %s)", type(exc).__name__, exc)
    if opened:
        if guess and guess != opened:
            note("HPU affinity: env says card %s but the process holds %s open, using the opened device", guess, opened)
        return opened, "opened-device"
    if guess:
        note(
            "HPU affinity: no open Habana device node found, card %s taken from the environment "
            "(HABANA_VISIBLE_MODULES / HL_VISIBLE_PCI_ADDRESS with a single entry)", guess)
        return guess, "env"
    return None, "none"


def _numa_node(pci: str, sysfs: str) -> int | None:
    text = _read(os.path.join(sysfs, "bus/pci/devices", pci, "numa_node"))
    try:
        node = int(text) if text is not None else -1
    except ValueError:
        return None
    return node if node >= 0 else None


def _node_cpus(pci: str, node: int | None, sysfs: str) -> set[int]:
    text = _read(os.path.join(sysfs, "bus/pci/devices", pci, "local_cpulist"))
    if not text and node is not None:
        text = _read(os.path.join(sysfs, "devices/system/node", f"node{node}", "cpulist"))
    return parse_cpu_list(text) if text else set()


def _physical_cores(cpus: set[int], sysfs: str) -> list[list[int]]:
    """Groups ``cpus`` into physical cores (thread siblings), ordered by first cpu."""
    seen: set[int] = set()
    cores: list[list[int]] = []
    for cpu in sorted(cpus):
        if cpu in seen:
            continue
        text = _read(os.path.join(sysfs, f"devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"))
        siblings = (parse_cpu_list(text) if text else {cpu}) & cpus
        siblings.add(cpu)
        seen |= siblings
        cores.append(sorted(siblings))
    return cores


_DEVICE_NODE = re.compile(r"^(accel|hl)(\d+)$")


def _visible_cards(sysfs: str, dev: str | None = None) -> list[str]:
    """PCI addresses of the cards whose device nodes exist in this container; all host cards if none can be seen."""
    dev = _DEV if dev is None else dev
    found: list[str] = []
    for directory in (os.path.join(dev, "accel"), dev):
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in names:
            match = _DEVICE_NODE.match(name)
            pci = _device_pci(match.group(2), match.group(1), sysfs) if match else None
            if pci and pci not in found:
                found.append(pci)
    return found or list(_habana_cards(sysfs))


def _core_slice(pci: str, node: int | None, cpus: set[int], sysfs: str, dev: str | None = None) -> set[int]:
    """The share of the physical cores of ``cpus`` (node CPUs the pod may use) that belongs to this card.

    The cards visible in the container that share the NUMA node split the cores evenly (in PCI order). With fewer
    cores than cards the cores are shared round-robin, so no card ends up with an empty set.
    """
    peers = [p for p in _visible_cards(sysfs, dev) if _numa_node(p, sysfs) == node]
    if pci not in peers:
        peers.append(pci)
    peers.sort()
    cores = _physical_cores(cpus, sysfs)
    n, k = len(peers), peers.index(pci)
    if len(cores) < n:
        logger.warning("HPU affinity: only %d physical cores for %d cards on NUMA node %s, cores are shared",
                       len(cores), n, node)
        chosen = [cores[k % len(cores)]]
    else:
        chosen = cores[k * len(cores) // n:(k + 1) * len(cores) // n]
    return {cpu for core in chosen for cpu in core}


def _pin_all_threads(mask: set[int], proc: str) -> tuple[int, int]:
    """sched_setaffinity pins one thread only: apply the mask to every tid. Returns (pinned, total)."""
    tids = [int(t) for t in os.listdir(os.path.join(proc, "self/task")) if t.isdigit()]
    pinned = 0
    for tid in tids:
        try:
            os.sched_setaffinity(tid, mask)
            pinned += 1
        except Exception:
            pass
    return pinned, len(tids)


def apply_affinity(rank: int, local_rank: int, sysfs: str = _SYSFS, proc: str = _PROC, dev: str | None = None) -> None:
    """Logs (and, if enabled, sets) the CPU affinity of this process. Never raises.

    Call after the HPU device is acquired, so the opened device node identifies the card.
    """
    mode = "0"
    pci: str | None = None
    source = "none"
    node: int | None = None
    before: set[int] = set()
    threads = ""
    try:
        mode = bind_mode()
        before = set(os.sched_getaffinity(0))
        target: set[int] | None = None
        pci, source = find_card_pci(sysfs, proc, quiet=mode == "0")
        if pci is None:
            if mode != "0":
                logger.warning("HPU affinity: card of local_rank=%s not identified, not binding", local_rank)
                mode = "0"
        else:
            node = _numa_node(pci, sysfs)
            if mode != "0":
                allowed = _node_cpus(pci, node, sysfs) & before
                target = _core_slice(pci, node, allowed, sysfs, dev) if mode == "core" and allowed else allowed
                if not target:
                    logger.warning("HPU affinity: no allowed CPU on the card's NUMA node (%s), not binding", node)
                    target = None
                    mode = "0"
        if target:
            pinned, total = _pin_all_threads(target, proc)
            threads = f" threads_pinned={pinned}/{total}"
            if pinned == 0:
                mode = "0"
            elif mode == "core" and "OMP_NUM_THREADS" not in os.environ:
                # only the per-card slice sizes the pool; with 'node' binding the ranks share the cores, so the
                # torch default is kept. Never raise the thread count.
                with contextlib.suppress(Exception):
                    torch.set_num_threads(max(1, min(len(_physical_cores(target, sysfs)), torch.get_num_threads())))
    except Exception as exc:
        logger.warning("HPU affinity: binding failed (%s: %s), continuing unbound", type(exc).__name__, exc)
        mode = "0"
    try:
        now = set(os.sched_getaffinity(0))
    except Exception:
        now = before
    try:
        torch_threads = torch.get_num_threads()
    except Exception:
        torch_threads = -1
    logger.info(
        "HPU affinity: rank=%s local_rank=%s card_pci=%s (source: %s) numa_node=%s mode=%s cpus=%s "
        "(before: %d cpus)%s torch_threads=%s", rank, local_rank, pci, source, node, mode, format_cpu_list(now),
        len(before), threads, torch_threads)
