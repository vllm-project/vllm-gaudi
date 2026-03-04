import torch
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.bert import TOKEN_TYPE_SHIFT, BertForSequenceClassification


def patched_BertForSequenceClassification_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if token_type_ids is not None:
        assert self.bert.config.vocab_size < (1 << TOKEN_TYPE_SHIFT)
        assert input_ids is not None

    return self.bert(
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
        intermediate_tensors=intermediate_tensors,
    )


BertForSequenceClassification.forward = patched_BertForSequenceClassification_forward
