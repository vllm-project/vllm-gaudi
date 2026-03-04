import torch
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.bert import TOKEN_TYPE_SHIFT
from vllm.model_executor.models.roberta import RobertaForSequenceClassification, replace_roberta_positions


def patched_RobertaForSequenceClassification_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    replace_roberta_positions(input_ids=input_ids, position_ids=positions, padding_idx=self.padding_idx)
    if token_type_ids is not None:
        assert self.roberta.config.vocab_size < (1 << TOKEN_TYPE_SHIFT)
        assert input_ids is not None

    return self.roberta(
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
        intermediate_tensors=intermediate_tensors,
    )


RobertaForSequenceClassification.forward = patched_RobertaForSequenceClassification_forward
