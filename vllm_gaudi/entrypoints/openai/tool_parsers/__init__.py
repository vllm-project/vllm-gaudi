# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-tree tool-call parsers bundled with vllm-gaudi.

Importing this package registers every HPU tool parser with vLLM's
``ToolParserManager`` (see ``vllm_gaudi.register_tool_parsers``), so they can be
selected with ``--tool-call-parser <name>`` without a ``--tool-parser-plugin``.
"""

from vllm_gaudi.entrypoints.openai.tool_parsers.minimax_m3 import (
    MinimaxM3PyToolParser, )

__all__ = ["MinimaxM3PyToolParser"]
