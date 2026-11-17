# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from https://github.com/vllm-project/vllm/blob/b46907cce0/vllm/v1/spec_decode/eagle.py
# (removed upstream in aac1b2a4f3, vllm-project/vllm#63373). Local changes: imports retargeted.
import torch

from vllm.config import VllmConfig
from vllm_gaudi.v1._vendored.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


class EagleProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
