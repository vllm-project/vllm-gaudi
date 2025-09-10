import torch
import torch.nn.functional as F
from vllm.lora.layers import VocabParallelEmbeddingWithLoRA
from vllm.lora import layers
from vllm.platforms import current_platform
from typing import Optional


class HPUVocabParallelEmbeddingWithLoRA(VocabParallelEmbeddingWithLoRA):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        added_tokens_mask = torch.where(x > self.base_layer.org_vocab_size - 1, 1, 0)

        # NB: Don't use torch.narrow here. torch.narrow triggers some
        # Dynamic Shape specialization in torch.compile
        # flatten to get num_tokens since HPU uses 2d input layout
        # reshape indices_1, indices_0 to match shape of input
        num_tokens = x.view(-1).shape[0]
        indices_1 = self.punica_wrapper._embeddings_indices[1][:num_tokens].view_as(x)
        indices_0 = self.punica_wrapper._embeddings_indices[0][:num_tokens].view_as(x)

        full_lora_a_embeddings = F.embedding(
            x + indices_1,
            self.lora_a_stacked_2d,
        )
        full_output = self.base_layer.forward(x + (indices_0 * added_tokens_mask))

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


# refer to https://github.com/vllm-project/vllm/pull/21923 for more details
# on why this patching is needed.
layers.VocabParallelEmbeddingWithLoRA = HPUVocabParallelEmbeddingWithLoRA
