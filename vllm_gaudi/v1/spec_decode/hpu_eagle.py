# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class HpuEagleProposer(EagleProposer):

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

        ret_hidden_states = self.model(
            input_ids=target_token_ids,
            positions=target_positions,
            hidden_states=target_hidden_states,
            inputs_embeds=None,
            attn_metadata=common_attn_metadata,
        )

        if self.method in ("deepseek_mtp", "ernie_mtp"):
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
        last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[-1])
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        draft_token_ids = logits.argmax(dim=-1)
        return draft_token_ids

    def prepare_inputs(
        self,
        common_attn_metadata,
        spec_decode_metadata: SpecDecodeMetadata,
        sampled_token_ids: list[list[int]],
    ):
        assert spec_decode_metadata is not None
        num_draft_tokens = \
            spec_decode_metadata.num_draft_tokens
        max_num_draft_tokens = max(num_draft_tokens)

        num_picked_token_indices = []
        last_token_indices = []
        starting_index = 0
        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0 for i, n in enumerate(num_draft_tokens)
        ]
        for i, n in enumerate(num_draft_tokens):
            r = num_rejected_tokens[i]
            step = max_num_draft_tokens + 1
            for j in range(step):
                if j == n - r:
                    last_token_indices.append(starting_index + j)
                if j < n + 1 - r:
                    num_picked_token_indices.append(starting_index + j)
                else:
                    num_picked_token_indices.append(-1)
            starting_index += step
        hidden_states_indices = torch.tensor(num_picked_token_indices, device=self.device)
        last_token_indices = torch.tensor(last_token_indices, device=self.device)
        return common_attn_metadata, hidden_states_indices, last_token_indices
