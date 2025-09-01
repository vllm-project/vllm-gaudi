# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.core.sched.scheduler import Scheduler
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


class HPUScheduler(Scheduler):

    def __init__(self, *args, **kwargs):
        logger.info("Using HPUScheduler")
        super().__init__(*args, **kwargs)
