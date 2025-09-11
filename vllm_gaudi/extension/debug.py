###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from vllm_gaudi.extension.logger import logger


def init_debug_logger(area: str):
    """ Initialize logging function if area is currently debugged """
    from vllm_gaudi.extension.runtime import get_config
    dbg_flags: list[str] = get_config().VLLM_DEBUG or []
    if area in dbg_flags:

        def enabled_dbg_logger(msg):
            logger().warning(f'[{area}] {msg}')

        return enabled_dbg_logger
    return None
