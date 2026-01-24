# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Selector module to switch between PyTorch and Triton implementations
of Mamba operations based on environment variable.

Set VLLM_MAMBA_USE_PYTORCH=1 to use PyTorch implementations.
Default (unset or 0) uses optimized Triton implementations.
"""

import os
from typing import Optional

import torch

# Check environment variable
_USE_PYTORCH = os.environ.get("VLLM_MAMBA_USE_PYTORCH", "0") == "1"
_USE_SELECTIVE_STATE_UPDATE_REF = os.environ.get("VLLM_MAMBA_USE_SELECTIVE_STATE_UPDATE_REF_PT", "1") == "1" #selective_state_update_ref

def _use_pytorch_runtime():
    """Check at runtime whether to use PyTorch implementation.  
    This allows torch.compile to respect the environment variable."""
    return os.environ.get("VLLM_MAMBA_USE_PYTORCH", "0") == "1"

def use_pytorch_ops() -> bool:
    """Returns True if PyTorch implementations should be used."""
    return _USE_PYTORCH

def use_pytorch_selective_state_update_ref() -> bool:
    return _USE_SELECTIVE_STATE_UPDATE_REF

# Disable torch.compile for selector functions so they execute dynamically
# This ensures the environment variable is checked at runtime, not compile time
try:
    import torch._dynamo
    disable_compile = torch._dynamo.disable
except (ImportError, AttributeError):
    # Fallback if torch._dynamo is not available
    def disable_compile(fn):
        return fn

@disable_compile
def get_selective_state_update_impl():
    """
    Returns the selective state update implementation.
    
    PyTorch version signature:
        selective_state_update_ref(state, x, dt, A, B, C, D=None, z=None, 
                                   dt_bias=None, dt_softplus=False)
        Returns: output tensor
        
    """
    # Import both implementations
    from .pytorch_implementation import selective_state_update_ref
    
    # Create wrapped PyTorch version
    pytorch_wrapped = _wrap_selective_state_update_ref(selective_state_update_ref)
    
    # Return a runtime dispatcher
    @disable_compile
    def dispatcher(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False,
                   state_batch_indices=None, dst_state_batch_indices=None, out=None):
        return pytorch_wrapped(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus,
                               state_batch_indices, dst_state_batch_indices, out)

    return dispatcher

def _wrap_selective_state_update_ref(selective_state_update_ref_fn):
    """Wrapper to adapt PyTorch selective_state_update_ref to match Triton API."""
    def wrapped(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False,
                state_batch_indices=None, dst_state_batch_indices=None, out=None):
        # PyTorch ref version doesn't support the batch indices parameters
        # These are used in Triton for selective state updates with batching
        if state_batch_indices is not None or dst_state_batch_indices is not None:
            # Triton uses state_batch_indices to select which state slots to read from
            # and dst_state_batch_indices to select which state slots to write to
            # The PyTorch version doesn't support this, so we need to handle it manually
            
            # When indices are provided, we need to:
            # 1. Select the appropriate state slices based on state_batch_indices
            # 2. Run the update on those slices
            # 3. Write back to the appropriate locations based on dst_state_batch_indices
            
            if state_batch_indices is None:
                state_batch_indices = torch.arange(x.shape[0], device=x.device)
            if dst_state_batch_indices is None:
                dst_state_batch_indices = state_batch_indices
            
            # Select state slices for reading
            selected_state = state[state_batch_indices].clone()
            
            # Run the update
            result = selective_state_update_ref_fn(
                selected_state, x, dt, A, B, C,
                D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus
            )
            
            # Write back the updated states
            state[dst_state_batch_indices] = selected_state
            
            # Handle output
            if out is not None:
                out.copy_(result)
                return out
            else:
                return result
        else:
            # No batch indices, use the simple path
            result = selective_state_update_ref_fn(
                state, x, dt, A, B, C,
                D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus
            )
            
            # If out is provided, copy result into it (to match Triton's in-place behavior)
            if out is not None:
                out.copy_(result)
                return out
            else:
                return result
    
    return wrapped
