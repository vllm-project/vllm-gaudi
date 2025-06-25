import torch
import torch.nn.functional as F
from vllm.model_executor.custom_op import CustomOp


@CustomOp.register_oot(name='VocabParallelEmbeddingWithLoRA')
class HPUVocabParallelEmbeddingWithLoRA:

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        # x need to reshaped into 2d as batch is there
        # can be removed on moving to flat tensors
        shape = x.shape
        x = x.view(shape[0] * shape[1])

        added_tokens_mask = torch.where(x > self.base_layer.org_vocab_size - 1,
                                        1, 0)
        embeddings_indices = torch.narrow(
            self.punica_wrapper._embeddings_indices, 1, 0, x.size(0))

        indices = embeddings_indices[1]
        full_lora_a_embeddings = F.embedding(
            x + indices,
            self.lora_a_stacked_2d,
        )
        indices = embeddings_indices[0]
        full_output = self.base_layer.forward(x +
                                              (indices * added_tokens_mask))

        full_output_org = full_output
        if full_output.ndim == 3:
            full_output = full_output.view(
                full_output.shape[0] * full_output.shape[1], -1)
        if full_lora_a_embeddings.ndim == 3:
            full_lora_a_embeddings = full_lora_a_embeddings.view(
                full_lora_a_embeddings.shape[0] *
                full_lora_a_embeddings.shape[1],
                -1,
            )
        self.punica_wrapper.add_lora_embedding(full_output,
                                               full_lora_a_embeddings,
                                               self.lora_b_stacked,
                                               add_input=True)
        # can be removed on moving to flat tensors
        full_output_org = full_output_org.view(shape[0], shape[1],
                                               full_output_org.shape[1])
        return full_output.view_as(full_output_org)
