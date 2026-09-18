"""pytest plugin: run upstream's async-scheduler suite against HPUAsyncScheduler.

vllm-gaudi overrides ``AsyncScheduler``, so upstream's own suite for that class
never exercises the override, because upstream cannot import a plugin scheduler.
This plugin rebinds the name, which turns that suite into an executable
specification for the override: it asserts that the delivered token stream
equals the sampled one and that positions stay consistent, neither of which a
serving benchmark can see.

Substitution happens in ``pytest_configure``, before collection, because the
upstream tests bind ``AsyncScheduler`` with ``from ... import`` at module scope;
patching the defining module after those imports would silently do nothing.
``pytest_collection_finish`` then re-verifies every binding, because a
substitution that quietly failed would report upstream's own green result as
ours.

Usage (see ``run_upstream_async_scheduler_gate`` in
``ci_e2e_discoverable_tests.sh``)::

    PYTHONPATH=<this dir> pytest -p upstream_async_sched_plugin \\
        --confcutdir <vllm>/tests/v1/core <vllm>/tests/v1/core/test_async_scheduler.py
"""
import sys

_MARK = "[hpu-async-sched-gate]"


def _guard_class():
    from vllm_gaudi.v1.core.sched.hpu_async_scheduler import HPUAsyncScheduler
    return HPUAsyncScheduler


def pytest_configure(config):
    """Rebind AsyncScheduler to HPUAsyncScheduler before any test is imported."""
    from vllm.v1.core.sched import async_scheduler

    async_scheduler.AsyncScheduler = _guard_class()
    print(f"{_MARK} AsyncScheduler <- HPUAsyncScheduler")


def pytest_collection_finish(session):
    """Prove the substitution reached the names the tests actually call.

    Rather than guess module names, which depend on how pytest rooted the test
    package, scan every module imported during collection for an
    ``AsyncScheduler`` binding and require each one to resolve to the override.
    Abort if any does not, or if none exists at all: a gate with nothing
    substituted is void either way. The override's own module is exempt, since it
    holds the base class it subclasses.
    """
    guard = _guard_class()
    bindings, wrong = [], []
    for name, module in list(sys.modules.items()):
        if name == guard.__module__:
            continue
        bound = getattr(module, "AsyncScheduler", None)
        if not isinstance(bound, type):
            continue
        bindings.append(name)
        if bound is not guard:
            wrong.append(f"{name} -> {bound.__name__}")

    if not bindings:
        raise SystemExit(f"{_MARK} SUBSTITUTION FAILED, gate is void: no module holds an "
                         "AsyncScheduler binding, so nothing was substituted")
    if wrong:
        raise SystemExit(f"{_MARK} SUBSTITUTION FAILED, gate is void: " + "; ".join(wrong))
    print(f"{_MARK} verified {len(bindings)} AsyncScheduler binding(s) resolve to "
          f"HPUAsyncScheduler: {', '.join(sorted(bindings))}")
