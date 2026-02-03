from typing import Union
import os

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import get_tp_group

@RowParallelLinear.register_oot
class HPURowParallelLinear(RowParallelLinear):
    """HPU-optimized RowParallelLinear implementation.
    
    This implementation provides chunked computation for overlapping
    compute and communication on HPU devices.
    """

    def __init__(self, *args, **kwargs):
        """Initialize HPURowParallelLinear with chunking support.
        
        The number of chunks can be configured via the VLLM_ROW_PARALLEL_CHUNKS
        environment variable. Default is 1 chunk (disabled).
        
        The token threshold for enabling chunking can be configured via
        VLLM_ROW_PARALLEL_CHUNK_THRESHOLD. Default is 8192 tokens.
        """
        super().__init__(*args, **kwargs)
        # Check for chunking configuration via environment variable
        self.num_chunks = int(os.environ.get("VLLM_ROW_PARALLEL_CHUNKS", "1"))
        if self.num_chunks < 1:
            raise ValueError(f"VLLM_ROW_PARALLEL_CHUNKS must be >= 1, got {self.num_chunks}")

        # Check for chunk threshold configuration
        self.chunk_threshold = int(os.environ.get("VLLM_ROW_PARALLEL_CHUNK_THRESHOLD", "8192"))
        if self.chunk_threshold < 1:
            raise ValueError(f"VLLM_ROW_PARALLEL_CHUNK_THRESHOLD must be >= 1, got {self.chunk_threshold}")

    def forward_oot(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Parameter]]:
        """Forward pass with HPU-specific optimizations.
        
        Args:
            input_: Input tensor to process
            
        Returns:
            Output tensor, or tuple of (output, bias) if skip_bias_add is True
        """
        
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()

        assert self.quant_method is not None
        
        # Determine total tokens for chunking decision
        input_shape = input_parallel.shape
        # For 3D input [batch, seq, hidden], total_tokens = batch * seq
        # For 2D input [tokens, hidden], total_tokens = tokens
        if input_parallel.ndim == 3:
            batch_size, seq_len, _ = input_parallel.shape
            total_tokens = batch_size * seq_len
        else:
            total_tokens = input_shape[0]
        
        # Check if we should use chunking
        # Don't chunk for inputs below threshold as there's no overlap benefit
        should_chunk = (self.num_chunks > 1 and 
                       self.reduce_results and 
                       self.tp_size > 1 and
                       total_tokens >= self.chunk_threshold)
        
        # Chunked computation for overlapping compute and communication
        if should_chunk:
            torch._dynamo.graph_break()
            
            # Determine if input is 3D [batch, seq, hidden] or 2D [batch*seq, hidden]
            is_3d = input_parallel.ndim == 3
            
            if is_3d:
                # Input is [batch, seq, hidden]
                # For multi-token sequences (prompts): chunk along sequence dimension
                # For single-token batches (decodes): chunk along batch dimension
                batch_size, seq_len, hidden_dim = input_parallel.shape
                if seq_len > 1:
                    # Chunk along sequence dimension for prompts
                    chunk_dim = 1
                    total_tokens = seq_len
                else:
                    # Chunk along batch dimension for batched decodes
                    chunk_dim = 0
                    total_tokens = batch_size
                output = torch.empty(
                    batch_size,
                    seq_len,
                    self.output_size_per_partition, 
                    dtype=input_parallel.dtype,
                    device=input_parallel.device
                )
            else:
                # Input is [batch*seq, hidden], chunk along batch dimension
                total_tokens, hidden_dim = input_parallel.shape
                chunk_dim = 0
                output = torch.empty(
                    total_tokens, 
                    self.output_size_per_partition, 
                    dtype=input_parallel.dtype,
                    device=input_parallel.device
                )
            
            chunk_size = (total_tokens + self.num_chunks - 1) // self.num_chunks
            
            # Lists to store chunks and handles
            output_chunks = []
            handles = []
            chunk_ranges = []
            
            # Phase 1: Compute all chunks and start all-reduces
            for i in range(self.num_chunks):
                torch._dynamo.graph_break()
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_tokens)
                
                # Skip empty chunks
                if start_idx >= end_idx:
                    continue
                
                # Slice input along the appropriate dimension
                if is_3d:
                    if chunk_dim == 1:
                        # Chunking along sequence dimension
                        input_chunk = input_parallel[:, start_idx:end_idx, :]
                    else:
                        # Chunking along batch dimension
                        input_chunk = input_parallel[start_idx:end_idx, :, :]
                else:
                    input_chunk = input_parallel[start_idx:end_idx]
                
                # Compute on chunk (no bias - will be added after all-reduce)
                output_chunk = self.quant_method.apply(self, input_chunk, bias=None)
                torch._dynamo.graph_break()
                
                # Start async all-reduce for this chunk
                handle = torch.distributed.all_reduce(
                    output_chunk, group=get_tp_group().device_group, async_op=True
                )
                
                # Store chunk, handle, and range info
                output_chunks.append(output_chunk)
                handles.append(handle)
                chunk_ranges.append((start_idx, end_idx))
            
            # Phase 2: Wait for all handles and combine outputs
            for handle in handles:
                handle.wait()
            
            torch._dynamo.graph_break()
            
            # Copy all chunks to output
            for chunk, (start_idx, end_idx) in zip(output_chunks, chunk_ranges):
                if is_3d:
                    if chunk_dim == 1:
                        # Chunked along sequence dimension
                        output[:, start_idx:end_idx, :] = chunk
                    else:
                        # Chunked along batch dimension
                        output[start_idx:end_idx, :, :] = chunk
                else:
                    output[start_idx:end_idx] = chunk
            
            torch._dynamo.graph_break()
            
            # Apply bias after all-reduce (only on rank 0 if not skip_bias_add)
            if self.bias is not None and self.tp_rank == 0 and not self.skip_bias_add:
                output = output + self.bias
        else:
            # Original single-shot computation
            bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
            output_parallel = self.quant_method.apply(self, input_parallel, bias_)

            if self.reduce_results and self.tp_size > 1:
                output = tensor_model_parallel_all_reduce(output_parallel)
            else:
                output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias

