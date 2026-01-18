# SPDX-License-Identifier: Apache-2.0
import itertools

import torch
from vllm_gaudi.utils import async_h2d_copy
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class HpuEagleProposer(EagleProposer):

    def propose(
        self,
        # [virtual_batch_size, seq_len]
        target_token_ids,
        # [virtual_batch_size, seq_len]
        target_positions,
        # [virtual_batch_size, seq_len, hidden_size]
        target_hidden_states,
        # [batch_size]
        last_token_indices,
        common_attn_metadata,
        # [num_seq, total_blocks]
        block_table_cpu_tensor,
        model_runner,
    ):
        # For decode, the virtual batch_size is batch size * num_tokens
        # and the seq_len is always 1
        batch_size = last_token_indices.shape[0]

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

        # All MTP related method names are now unified to "mtp"
        if self.method == "mtp":
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
        last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[-1])
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_token_ids = logits.argmax(dim=-1)
            return draft_token_ids.view(-1, 1)

        # [num_tokens, 1]
        target_positions = target_positions.view(-1)
        # [batch_size]
        positions = target_positions[last_token_indices]
        if self.method == "mtp":
            hidden_states = target_hidden_states.view(-1, target_hidden_states.shape[-1])
        else:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # [batch_size, hidden_size]
        hidden_states = hidden_states[last_token_indices]

        # The first draft tokens
        draft_token_ids = logits.argmax(dim=-1)
        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        # Positions used by prepare_attn_metadata needs to be cpu because
        # compile only mode for warmup will not do any real computations
        target_positions_cpu = target_positions.cpu()
        positions_cpu = target_positions_cpu[last_token_indices.cpu()]

        # Decode 1 token each time
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            # [batch_size]
            input_ids = draft_token_ids_list[-1].int()

            positions += 1
            exceeds_max_model_len = positions >= self.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

            # Prepare the attn metadata
            positions_cpu += 1
            attn_metadata = self.prepare_attn_metadata(block_table_cpu_tensor, positions_cpu, model_runner)

            # [batch_size, 1]
            input_ids = input_ids.view(-1, 1)
            # [batch_size, 1]
            input_positions = clamped_positions.view(-1, 1)
            # [batch_size, 1, hidden_size]
            input_hidden_states = hidden_states.view(-1, 1, hidden_states.shape[-1])
            inputs_embeds = None

            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=input_positions,
                hidden_states=input_hidden_states,
                inputs_embeds=inputs_embeds,
                attn_metadata=attn_metadata,
            )
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

            # The shape of the returned hidden_states and last_hidden_states:
            # [batch_size, 1, hidden_size]
            # viewed to: [batch_size, hidden_size]
            last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[-1])
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size])
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
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

    def prepare_attn_metadata(
            self,
            # [num_seq, total_blocks]
            block_table_cpu_tensor,
            # CPU tensor: [batch_size]
            positions,
            model_runner):
        # Prepare attn metadata on CPU. (Improve for pure HPU based attn metadata preparation)
        block_size = model_runner.block_size
        batch_size = positions.shape[0]
        exceeds_max_model_len = positions >= self.max_model_len
        clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

        # Note: block_table_cpu_tensor doesn't include the padding
        # which might smaller than the (padded) batch_size
        num_seq = block_table_cpu_tensor.shape[0]

        # Prepare block tables list
        # block_tables_list is a nested list of shape [num_seq, num_blocks]
        # num_blocks should include the slots needed for the current token
        # positions are the context lengths, and we need +1 for num_blocks
        num_blocks = torch.ceil((positions + 1) / block_size).int()
        num_blocks = num_blocks[:num_seq].tolist()
        block_tables_list = []
        for i, n in enumerate(num_blocks):
            seq_block_table = block_table_cpu_tensor[i, :n].tolist()
            assert len(seq_block_table) == n
            block_tables_list.append(seq_block_table)
        # Needs to be resolved by defragmenter
        block_tables_list = model_runner.defragmenter.resolve_all(block_tables_list)

        # Compute slot mapping in [batch_size, 1] shape
        clamped_positions = clamped_positions.view(-1, 1)
        block_numbers = clamped_positions // block_size

        # Limit with num_seq because block_table_cpu_tensor is in the shape [num_seq, x]
        block_numbers = block_numbers.to(torch.int64)[:num_seq]
        block_ids = torch.ones((batch_size, 1), dtype=torch.int32) * model_runner._PAD_BLOCK_ID
        block_ids[:num_seq] = block_table_cpu_tensor.gather(dim=1, index=block_numbers)
        # Needs to be resolved by defragmenter
        block_ids.apply_(model_runner.defragmenter.resolve)

        # Calculate the slot mapping and fill with padding
        slot_mapping = block_ids * block_size + clamped_positions % block_size
        dummy_slots = itertools.cycle(range(model_runner._PAD_SLOT_ID, model_runner._PAD_SLOT_ID + block_size))
        slot_mapping[num_seq:].apply_(lambda _, ds=dummy_slots: next(ds))
        # Slot mapping needs to be int64 (long) type
        slot_mapping = slot_mapping.to(torch.int64)

        block_list, block_groups, block_usage = \
            model_runner.get_habana_paged_attn_buffers(
                block_tables_list,
                slot_mapping.tolist(),
                batch_size
            )

        block_list_device = async_h2d_copy(block_list, device=self.device)
        block_usage_device = async_h2d_copy(block_usage, device=self.device)
        block_groups_device = async_h2d_copy(block_groups, device=self.device)
        slot_mapping_device = async_h2d_copy(slot_mapping, device=self.device)

        common_attn_metadata = HPUAttentionMetadataV1.make_decode_metadata(
            block_list=block_list_device,
            block_usage=block_usage_device,
            block_groups=block_groups_device,
            input_positions=None,
            slot_mapping=slot_mapping_device,
            block_size=block_size,
            window_block_list=None,
            window_block_usage=None,
            window_block_groups=None,
        )

        return common_attn_metadata
