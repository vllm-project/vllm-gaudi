# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from vllm.model_executor.custom_op import CustomOp


@contextlib.contextmanager
def temporary_op_registry_oot():
    """
    Contextmanager which allows to temporarly modify the op registry content.
    It clears current op_registry_oot and restors its content on exit.
    It is usefull for testing purposes, e.g. to deregister hpu version
    of the op. (Because when running tests, if registration happened in one
    of them, then it is still valid in every other test).
    """
    old_registry = CustomOp.op_registry_oot
    CustomOp.op_registry_oot = {}
    try:
        yield
    finally:
        CustomOp.op_registry_oot = old_registry


def register_op(base_cls, oot_cls):
    """
    Manual registration of the oot op. It should be used
    within temporary_op_registry_oot context manager.
    """
    CustomOp.op_registry_oot[base_cls.__name__] = oot_cls
