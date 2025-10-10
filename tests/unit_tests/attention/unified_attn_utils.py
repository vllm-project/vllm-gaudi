from vllm_gaudi.extension.unified import (create_attention_metadata)
import torch
from vllm_gaudi.extension.runtime import finalize_config
import vllm_gaudi.extension.environment as environment


def get_unified_attn_metadata(vllm_config, common_attn_metadata, batch_spec, query_dtype, device):
    environment.set_vllm_config(vllm_config)
    finalize_config()
    block_size = vllm_config.cache_config.block_size
    block_table = common_attn_metadata.block_table_tensor.cpu()
    num_computed_tokens = common_attn_metadata.num_computed_tokens_cpu
    num_scheduled_tokens = torch.tensor(batch_spec.query_lens)
    # NOTE(kzawora): nasty hack - use num_computed_tokens as prompt_len for decodes (qlen == 1), use seq_len otherwise
    num_prompt_tokens = torch.tensor([
        (nct if ql == 1 else sl)
        for ql, sl, nct in zip(batch_spec.query_lens, batch_spec.seq_lens, num_computed_tokens.tolist())
    ])
    attn_metadata, _ = create_attention_metadata(num_computed_tokens, num_scheduled_tokens, num_prompt_tokens,
                                                 block_table, block_size, query_dtype)
    return attn_metadata
