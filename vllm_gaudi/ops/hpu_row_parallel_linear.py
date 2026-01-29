from typing import Union
import os
import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
import logging

logger = logging.getLogger(__name__)

@RowParallelLinear.register_oot
class HPURowParallelLinear(RowParallelLinear):
    """HPU-optimized RowParallelLinear implementation.
    
    This implementation provides chunked computation for overlapping
    compute and communication on HPU devices.
    """

    def __init__(self, *args, **kwargs):
        """Initialize HPURowParallelLinear with chunking support.
        
        The number of chunks can be configured via the VLLM_ROW_PARALLEL_CHUNKS
        environment variable. Default is 4 chunks.
        """
        super().__init__(*args, **kwargs)
        # Check for chunking configuration via environment variable
        self.num_chunks = int(os.environ.get("VLLM_ROW_PARALLEL_CHUNKS", "4"))
        logger.info(f"HPURowParallelLinear initialized with num_chunks={self.num_chunks}, "
                   f"tp_size={self.tp_size}, reduce_results={self.reduce_results}")

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
        logger.info(f"HPURowParallelLinear.forward_oot called with input shape={input_.shape}, "
                   f"num_chunks={self.num_chunks}, tp_size={self.tp_size}, "
                   f"reduce_results={self.reduce_results}")
        
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()

        assert self.quant_method is not None
        
        # Log the shape and chunking decision criteria
        input_shape = input_parallel.shape
        # Check if we should use chunking
        # For 2D input [tokens, hidden], check if tokens > 1
        # For 3D input [batch, seq, hidden], check if seq > 1
        should_chunk = (self.num_chunks > 1 and 
                       self.reduce_results and 
                       self.tp_size > 1)
        
        # Additional check: for 2D input, shape is [tokens, hidden], check tokens
        # For 3D input, shape is [batch, seq, hidden], check seq
        if should_chunk and len(input_shape) >= 2:
            if len(input_shape) == 2:
                should_chunk = input_shape[0] > 1  # tokens dimension
            else:  # 3D
                should_chunk = input_shape[1] > 1  # sequence dimension
        else:
            should_chunk = False
            
        logger.info(f"Chunking decision: num_chunks={self.num_chunks}, "
                   f"reduce_results={self.reduce_results}, tp_size={self.tp_size}, "
                   f"input_shape={input_shape}, should_chunk={should_chunk}")
        
        # Chunked computation for overlapping compute and communication
        if should_chunk:
            logger.info(f"Using chunked computation with {self.num_chunks} chunks")
            torch._dynamo.graph_break()
            
            # Determine if input is 3D [batch, seq, hidden] or 2D [batch*seq, hidden]
            is_3d = input_parallel.ndim == 3
            
            if is_3d:
                # Input is [batch, seq, hidden], chunk along sequence dimension
                batch_size, seq_len, hidden_dim = input_parallel.shape
                chunk_dim = 1  # sequence dimension
                total_tokens = seq_len
            else:
                # Input is [batch*seq, hidden], chunk along batch dimension
                total_tokens, hidden_dim = input_parallel.shape
                chunk_dim = 0
            
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
                    input_chunk = input_parallel[:, start_idx:end_idx, :]
                else:
                    input_chunk = input_parallel[start_idx:end_idx]
                
                # Compute on chunk (bias only on first chunk for rank 0)
                bias_ = self.bias if (i == 0 and self.tp_rank == 0 and not self.skip_bias_add) else None
                output_chunk = self.quant_method.apply(self, input_chunk, bias_)
                torch._dynamo.graph_break()
                
                # Start async all-reduce for this chunk
                handle = torch.distributed.all_reduce(
                    output_chunk, group=torch.distributed.group.WORLD, async_op=True
                )
                
                # Store chunk, handle, and range info
                output_chunks.append(output_chunk)
                handles.append(handle)
                chunk_ranges.append((start_idx, end_idx))
            
            # Phase 2: Wait for all handles and combine outputs
            for handle in handles:
                handle.wait()
            
            torch._dynamo.graph_break()
            
            # Pre-allocate output buffer matching input shape
            if is_3d:
                output = torch.empty(
                    batch_size,
                    seq_len,
                    self.output_size_per_partition, 
                    dtype=input_parallel.dtype,
                    device=input_parallel.device
                )
            else:
                output = torch.empty(
                    total_tokens, 
                    self.output_size_per_partition, 
                    dtype=input_parallel.dtype,
                    device=input_parallel.device
                )
            
            # Copy all chunks to output
            for chunk, (start_idx, end_idx) in zip(output_chunks, chunk_ranges):
                if is_3d:
                    output[:, start_idx:end_idx, :] = chunk
                else:
                    output[start_idx:end_idx] = chunk
            
            torch._dynamo.graph_break()
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


# Log registration
logger.info("HPURowParallelLinear registered as OOT custom op for RowParallelLinear")
