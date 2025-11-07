# SPDX-License-Identifier: Apache-2.0

from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.spec_decode.eagle import EagleProposer


class HpuEagleProposer (EagleProposer):
    def propose(
            self,
            target_token_ids,
            target_positions,
            target_hidden_states,
            last_token_indices,
            common_attn_metadata,
    ):
        if self.method == "eagle3":
            assert isinstance(self.model.model, Eagle3LlamaForCausalLM)
            target_hidden_states = \
                self.model.model.combine_hidden_states(
                    target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        # htorch.core.mark_step()
        ret_hidden_states = self.model(
            input_ids=target_token_ids,
            positions=target_positions,
            hidden_states=target_hidden_states,
            inputs_embeds=None,
            attn_metadata=common_attn_metadata,
        )
        # htorch.core.mark_step()
        if self.method in ("deepseek_mtp", "ernie_mtp"):
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
        last_hidden_states = last_hidden_states.view(-1,
                                                     last_hidden_states.shape[
                                                         -1])
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        draft_token_ids = logits.argmax(dim=-1)
        return draft_token_ids, hidden_states
