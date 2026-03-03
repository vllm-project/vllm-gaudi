from dataclasses import dataclass  
import torch  
from vllm.v1.attention.backend import AttentionBackend, AttentionImpl  
from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum  
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1  
  
@register_backend(AttentionBackendEnum.CUSTOM, "GDN_ATTN")  
class GDNAttentionBackend(AttentionBackend):  
    @staticmethod  
    def get_name() -> str:  
        return "GDN_ATTN"  
  
    @staticmethod  
    def get_impl_cls() -> type["AttentionImpl"]:  
        return GDNAttentionImpl  
  
    @staticmethod  
    def get_metadata_cls() -> type["AttentionMetadata"]:  
        return GDNAttentionMetadata  
  
@dataclass  
class GDNAttentionMetadata(HPUAttentionMetadataV1):  
    # GDN-specific fields  
    num_prefills: int  
    num_prefill_tokens: int  
    num_decodes: int  
    num_decode_tokens: int  
    num_spec_decodes: int  
    num_spec_decode_tokens: int  
    num_actual_tokens: int  
  
    has_initial_state: torch.Tensor | None = None  
    spec_query_start_loc: torch.Tensor | None = None  
    non_spec_query_start_loc: torch.Tensor | None = None  
    spec_state_indices_tensor: torch.Tensor | None = None  
    non_spec_state_indices_tensor: torch.Tensor | None = None  
    spec_sequence_masks: torch.Tensor | None = None  
    spec_token_indx: torch.Tensor | None = None  
    non_spec_token_indx: torch.Tensor | None = None  
    num_accepted_tokens: torch.Tensor | None = None  
    nums_dict: dict | None = None  
    batch_ptr: torch.Tensor | None = None  
    token_chunk_offset_ptr: torch.Tensor | None = None 
    
@classmethod  
def make_gdn_prefill_metadata(cls,   
                             num_prefills: int,  
                             num_prefill_tokens: int,  
                             num_decodes: int = 0,  
                             num_decode_tokens: int = 0,  
                             num_spec_decodes: int = 0,  
                             num_spec_decode_tokens: int = 0,  
                             num_actual_tokens: int = None,  
                             # Tensor fields - only add if provided  
                             has_initial_state: torch.Tensor | None = None,  
                             spec_query_start_loc: torch.Tensor | None = None,  
                             non_spec_query_start_loc: torch.Tensor | None = None,  
                             spec_state_indices_tensor: torch.Tensor | None = None,  
                             non_spec_state_indices_tensor: torch.Tensor | None = None,  
                             spec_sequence_masks: torch.Tensor | None = None,  
                             spec_token_indx: torch.Tensor | None = None,  
                             non_spec_token_indx: torch.Tensor | None = None,  
                             num_accepted_tokens: torch.Tensor | None = None,  
                             batch_ptr: torch.Tensor | None = None,  
                             token_chunk_offset_ptr: torch.Tensor | None = None,  
                             # NEVER include - dict type incompatible with HPUGraphs  
                             # nums_dict: dict | None = None,  
                             **kwargs):  
    """  
    Create GDN attention metadata with proper tensor field handling.  
      
    Only tensor fields are added to metadata_dict.update() to ensure  
    HPUGraph compatibility. Scalar fields are handled separately.  
    """  
    # Create base metadata first  
    base_metadata = super().make_prefill_metadata(**kwargs)  
      
    # Convert to dict and add GDN fields  
    metadata_dict = base_metadata.__dict__.copy()  
      
    # Add scalar count fields (these may need conversion to tensors for HPUGraphs)  
    metadata_dict.update({  
        'num_prefills': num_prefills,  
        'num_prefill_tokens': num_prefill_tokens,  
        'num_decodes': num_decodes,  
        'num_decode_tokens': num_decode_tokens,  
        'num_spec_decodes': num_spec_decodes,  
        'num_spec_decode_tokens': num_spec_decode_tokens,  
        'num_actual_tokens': num_actual_tokens or num_prefill_tokens,  
    })  
      
    # Tensor field check - only add if not None and is actually a tensor  
    tensor_fields = {  
        'has_initial_state': has_initial_state,  
        'spec_query_start_loc': spec_query_start_loc,  
        'non_spec_query_start_loc': non_spec_query_start_loc,  
        'spec_state_indices_tensor': spec_state_indices_tensor,  
        'non_spec_state_indices_tensor': non_spec_state_indices_tensor,  
        'spec_sequence_masks': spec_sequence_masks,  
        'spec_token_indx': spec_token_indx,  
        'non_spec_token_indx': non_spec_token_indx,  
        'num_accepted_tokens': num_accepted_tokens,  
        'batch_ptr': batch_ptr,  
        'token_chunk_offset_ptr': token_chunk_offset_ptr,  
    }  
      
    # Only add tensor fields that are provided and are actually tensors  
    for field_name, field_value in tensor_fields.items():  
        if field_value is not None:  
            if torch.is_tensor(field_value):  
                metadata_dict[field_name] = field_value  
            else:  
                # Log warning if non-tensor value is provided for a tensor field  
                import warnings  
                warnings.warn(f"Field {field_name} expects torch.Tensor but got {type(field_value)}")  
      
    # NEVER add nums_dict - dict type breaks HPUGraph hashing  
    # See: trim_attn_metadata warnings about primitive types [1](#3-0)   
      
    return cls(**metadata_dict)

@classmethod  
def make_gdn_decode_metadata(cls,  
                            num_prefills: int = 0,  
                            num_prefill_tokens: int = 0,  
                            num_decodes: int,  
                            num_decode_tokens: int,  
                            num_spec_decodes: int = 0,  
                            num_spec_decode_tokens: int = 0,  
                            num_actual_tokens: int = None,  
                            # Tensor fields for decode phase  
                            has_initial_state: torch.Tensor | None = None,  
                            spec_query_start_loc: torch.Tensor | None = None,  
                            non_spec_query_start_loc: torch.Tensor | None = None,  
                            spec_state_indices_tensor: torch.Tensor | None = None,  
                            non_spec_state_indices_tensor: torch.Tensor | None = None,  
                            spec_sequence_masks: torch.Tensor | None = None,  
                            spec_token_indx: torch.Tensor | None = None,  
                            non_spec_token_indx: torch.Tensor | None = None,  
                            num_accepted_tokens: torch.Tensor | None = None,  
                            batch_ptr: torch.Tensor | None = None,  
                            token_chunk_offset_ptr: torch.Tensor | None = None,  
                            **kwargs):  
    """  
    Create GDN attention metadata for decode phase with tensor field checking.  
    """  
    # Create base metadata first  
    base_metadata = super().make_decode_metadata(**kwargs)  
      
    # Convert to dict and add GDN fields  
    metadata_dict = base_metadata.__dict__.copy()  
      
    # Add scalar count fields  
    metadata_dict.update({  
        'num_prefills': num_prefills,  
        'num_prefill_tokens': num_prefill_tokens,  
        'num_decodes': num_decodes,  
        'num_decode_tokens': num_decode_tokens,  
        'num_spec_decodes': num_spec_decodes,  
        'num_spec_decode_tokens': num_spec_decode_tokens,  
        'num_actual_tokens': num_actual_tokens or num_decode_tokens,  
    })  
      
    # Same tensor field check as prefill  
    tensor_fields = {  
        'has_initial_state': has_initial_state,  
        'spec_query_start_loc': spec_query_start_loc,  
        'non_spec_query_start_loc': non_spec_query_start_loc,  
        'spec_state_indices_tensor': spec_state_indices_tensor,  
        'non_spec_state_indices_tensor': non_spec_state_indices_tensor,  
        'spec_sequence_masks': spec_sequence_masks,  
        'spec_token_indx': spec_token_indx,  
        'non_spec_token_indx': non_spec_token_indx,  
        'num_accepted_tokens': num_accepted_tokens,  
        'batch_ptr': batch_ptr,  
        'token_chunk_offset_ptr': token_chunk_offset_ptr,  
    }  
      
    for field_name, field_value in tensor_fields.items():  
        if field_value is not None and torch.is_tensor(field_value):  
            metadata_dict[field_name] = field_value  
      
    return cls(**metadata_dict)