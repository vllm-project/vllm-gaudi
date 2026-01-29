"""Test Llama 70B-like inference with only 2 layers using TP4 and BF16."""
import os
import tempfile
import shutil
import argparse
import torch
from transformers import LlamaConfig, AutoTokenizer
from vllm import LLM, SamplingParams

# Import habana_frameworks to ensure HPU is available
try:
    import habana_frameworks.torch as htorch
    HPU_AVAILABLE = True
except ImportError:
    HPU_AVAILABLE = False
    print("WARNING: habana_frameworks not available, HPU may not be used")


def test_llama_70b_2layer_tp4_bf16(
    max_tokens: int = 16, warmup: bool = True, prompt_len: int = 8192, profile: bool = False
):
    """
    Test inference with Llama 70B configuration but only 2 layers.
    Creates a config with 70B dimensions but only 2 layers, copies tokenizer from real model.
    
    Args:
        max_tokens: Number of tokens to generate
        warmup: Whether to run warmup with same prompt length
        prompt_len: Approximate prompt length in tokens (default 8192)
    """
    # Set environment variable for chunking before LLM initialization
    if "VLLM_ROW_PARALLEL_CHUNKS" not in os.environ:
        os.environ["VLLM_ROW_PARALLEL_CHUNKS"] = "4"
        print(f"Set VLLM_ROW_PARALLEL_CHUNKS={os.environ['VLLM_ROW_PARALLEL_CHUNKS']}")
    else:
        print(f"VLLM_ROW_PARALLEL_CHUNKS already set to {os.environ['VLLM_ROW_PARALLEL_CHUNKS']}")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="llama_2layer_test_")
    
    # Verify HPU is available
    if HPU_AVAILABLE:
        print(f"HPU available: {torch.hpu.is_available()}")
        if torch.hpu.is_available():
            print(f"HPU device count: {torch.hpu.device_count()}")
    
    try:
        # Create Llama 70B config with only 2 layers
        config = LlamaConfig(
            vocab_size=128256,  # Llama 3.3 vocab size
            hidden_size=8192,
            intermediate_size=28672,
            num_hidden_layers=4,  # Only 4 layers
            num_attention_heads=64,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            rms_norm_eps=1e-5,
            torch_dtype="bfloat16",
        )
        config.save_pretrained(temp_dir)
        
        # Copy tokenizer from real model
        tokenizer_path = "/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.save_pretrained(temp_dir)
        
        # Initialize LLM with TP4 and BF16, using dummy weights
        # Set max_num_batched_tokens higher than prompt+decode to handle chunking
        # Set num_gpu_blocks_override slightly larger than max decode to prevent
        # scheduler from forcing KV cache movement causing recompilations
        llm = LLM(
            model=temp_dir,
            enable_prefix_caching=False,
            tensor_parallel_size=4,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=True,
            load_format="dummy",  # Initialize with random weights, no checkpoint needed
            max_num_batched_tokens=prompt_len + max_tokens + 1024,  # Extra buffer for chunking
            num_gpu_blocks_override=((prompt_len + max_tokens + 1024)//128 + 10),  # Slightly larger than max decode to avoid recompilations. This is used because defrag=True fails so we avoid defragmentation limiting kv cache
        )
        
    
        # Create prompt with approximately prompt_len tokens
        # "Hello, how are you?" is ~5 tokens, so repeat it prompt_len/5 times
        repeat_count = prompt_len // 5
        prompt_text = "Hello, how are you?" * repeat_count
        prompts = [prompt_text]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        
        # Warmup: Run with same prompt length to warm up the correct buckets
        if warmup:
            print(f"Running warmup with ~{prompt_len} token prompt...")
            llm.generate(prompts, sampling_params)
            print("Warmup complete.")
        
        # Run actual inference
        
        print(f"Running inference with max_tokens={max_tokens}...")
        prompts = [prompt_text+'xxxx']
        outputs = llm.generate(prompts, sampling_params)
        prompts = [prompt_text+'xxxxxxxxxxxxx']
        if profile:
            llm.start_profile()
        outputs = llm.generate(prompts, sampling_params)
        if profile:
            llm.stop_profile()
        # Verify output
        assert len(outputs) == 1
        assert len(outputs[0].outputs) > 0
        print(f"Generated {len(outputs[0].outputs[0].token_ids)} tokens")
        print(f"Output: {outputs[0].outputs[0].text[:100]}...")
        print("Test passed!")
    
    finally:
        # Cleanup
        
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=2, help="Number of tokens to generate")
    parser.add_argument("--prompt-len", type=int, default=8192, help="Approximate prompt length in tokens")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup phase")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    args = parser.parse_args()
    
    test_llama_70b_2layer_tp4_bf16(
        max_tokens=args.max_tokens, warmup=not args.no_warmup, prompt_len=args.prompt_len, profile=args.profile
    )
