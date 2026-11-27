"""Speculative-decoding method groups the HPU model runner dispatches on."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig

# The methods vLLM's SpeculativeConfig.use_eagle() matched before vLLM PR #68798 removed it. The HPU runner serves all
# of them with HpuEagleProposer and reserves block space for their draft tokens.
EAGLE_STYLE_METHODS = frozenset({"eagle", "eagle3", "mtp", "dflash", "dspark"})


def uses_eagle_proposer(speculative_config: "SpeculativeConfig") -> bool:
    """Tell whether speculative decoding runs an EAGLE-style drafter on HPU.

    Args:
        speculative_config: The engine's speculative decoding config.

    Returns:
        True if the method drafts from the target model's hidden states with the HPU EAGLE proposer.
    """
    return speculative_config.method in EAGLE_STYLE_METHODS
