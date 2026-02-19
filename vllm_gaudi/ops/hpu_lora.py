import torch
import torch.nn.functional as F
from vllm.lora.layers import VocabParallelEmbeddingWithLoRA
from vllm.lora.layers import LogitsProcessorWithLoRA
from vllm.model_executor.layers.vocab_parallel_embedding import (VocabParallelEmbedding)
from vllm.lora import layers
from vllm.platforms import current_platform
from typing import Optional


class HPUVocabParallelEmbeddingWithLoRA(VocabParallelEmbeddingWithLoRA):

    @property
    def quant_method(self):
        """Delegate quant_method access to the base layer."""
        return self.base_layer.quant_method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NB: Don't use torch.narrow here. torch.narrow triggers some
        # Dynamic Shape specialization in torch.compile
        # flatten to get num_tokens since HPU uses 2d input layout
        # reshape indices_1, indices_0 to match shape of input
        num_tokens = x.view(-1).shape[0]
        indices_1 = self.punica_wrapper._embeddings_indices[1][:num_tokens].view_as(x)

        full_lora_a_embeddings = F.embedding(
            x + indices_1,
            self.lora_a_stacked_2d,
        )
        full_output = self.base_layer.forward(x)

        full_output_org = full_output
        if full_output.ndim == 3:
            full_output = full_output.view(full_output.shape[0] * full_output.shape[1], -1)
        if full_lora_a_embeddings.ndim == 3:
            full_lora_a_embeddings = full_lora_a_embeddings.view(
                full_lora_a_embeddings.shape[0] * full_lora_a_embeddings.shape[1],
                -1,
            )

        lora_output: Optional[torch.Tensor] = self.punica_wrapper.add_lora_embedding(full_output,
                                                                                     full_lora_a_embeddings,
                                                                                     self.lora_b_stacked,
                                                                                     add_input=True)

        if not current_platform.can_update_inplace():
            full_output = lora_output

        return full_output.view_as(full_output_org)


class HPULogitsProcessorWithLoRA(LogitsProcessorWithLoRA):

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head, hidden_states)
        if embedding_bias is not None:
            logits += embedding_bias

        # Gather logits for TP
        logits = self.base_layer._gather_logits(logits)

        if logits is None:
            return None

        if self.sharded_to_full_mapping_gpu is not None:
            # Reindex full logits tensor to ensure 1:1 mapping between
            # index and token_id
            # Example for:
            #   org_vocab_size = 4
            #   added_vocab_size = 2
            #   pad_to_size = 8
            #   tp_size = 2

            # indices:  [0, 1, 2,  3, 4, 5, 6,  7]
            # token_id: [0, 1, 4, -1, 2, 3, 5, -1]

            # Therefore, the mapping is expected to be:
            # [0, 1, 4, 6, 2, 3, 5, 7] so that when we reindex,
            # we get:
            # indices:  [0, 1, 2, 3, 4, 5,  6,  7]
            # token_id: [0, 1, 2, 3, 4, 5, -1, -1]
            logits = logits[:, self.sharded_to_full_mapping_gpu]

        lora_output: Optional[torch.Tensor] = self.punica_wrapper.add_lora_logits(logits, hidden_states,
                                                                                  self.lora_a_stacked,
                                                                                  self.lora_b_stacked, 1.0)

        if not current_platform.can_update_inplace():
            logits = lora_output

        # Remove paddings in vocab (if any).
        logits = logits[:, :self.base_layer.vocab_size]
        return logits


# refer to https://github.com/vllm-project/vllm/pull/21923 for more details
# on why this patching is needed.
layers.VocabParallelEmbeddingWithLoRA = HPUVocabParallelEmbeddingWithLoRA
layers.LogitsProcessorWithLoRA = HPULogitsProcessorWithLoRA
