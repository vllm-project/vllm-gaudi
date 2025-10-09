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
    import pdb
    pdb.set_trace()
    # TODO: fix this
    num_prompt_tokens = common_attn_metadata.seq_lens_cpu - common_attn_metadata.num_computed_tokens_cpu
    return create_attention_metadata(num_computed_tokens, num_scheduled_tokens, num_prompt_tokens, block_table,
                                     block_size, query_dtype)
