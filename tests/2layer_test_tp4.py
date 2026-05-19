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
    import habana_frameworks.torch as htorch  # noqa: F401
    HPU_AVAILABLE = True
except ImportError:
    HPU_AVAILABLE = False
    print("WARNING: habana_frameworks not available, HPU may not be used")


def test_llama_70b_2layer_tp4_bf16(max_tokens: int = 16,
                                   warmup: bool = True,
                                   prompt_len: int = 8192,
                                   profile: bool = False,
                                   num_decodes: int = 1):
    """
    Test inference with Llama 70B configuration but only 2 layers.
    Creates a config with 70B dimensions but only 2 layers, copies tokenizer from real model.
    
    Args:
        max_tokens: Number of tokens to generate per decode
        warmup: Whether to run warmup with same prompt length
        prompt_len: Approximate prompt length in tokens (default 8192)
        profile: Whether to enable profiling
        num_decodes: Number of decode iterations to run (default 1)
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
        # Set max_num_batched_tokens to handle the full batch
        # Set num_gpu_blocks_override slightly larger than max decode to prevent
        # scheduler from forcing KV cache movement causing recompilations
        max_tokens_per_seq = prompt_len + max_tokens + 100  # Small buffer per sequence
        llm = LLM(
            model=temp_dir,
            max_num_seqs=num_decodes,
            enable_prefix_caching=False,
            tensor_parallel_size=4,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=True,
            load_format="dummy",  # Initialize with random weights, no checkpoint needed
            max_num_batched_tokens=num_decodes * max_tokens_per_seq,  # Enough for full batch
            num_gpu_blocks_override=((num_decodes * max_tokens_per_seq) // 128 +
                                     10),  # Slightly larger than max decode to avoid recompilations.
            # defrag=True fails so we avoid defragmentation limiting kv cache
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

        # Run decode iterations
        print(f"Running batched decodes with batch_size={num_decodes}, max_tokens={max_tokens}...")

        # Get tokenizer to create actual token IDs
        from vllm.inputs import TokensPrompt

        # Tokenize the base prompt once
        tokenizer = AutoTokenizer.from_pretrained(temp_dir)
        base_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

        # Create a list of TokensPrompt with slightly different tokens for each request
        # to avoid caching effects while maintaining same prompt length
        decode_prompts = []
        for i in range(num_decodes):
            # Add a few different tokens at the end to make each prompt unique
            tokens = base_tokens + tokenizer.encode('x' * (i + 1), add_special_tokens=False)
            decode_prompts.append(TokensPrompt(prompt_token_ids=tokens))

        outputs = llm.generate(decode_prompts, sampling_params, use_tqdm=False)

        if profile:
            llm.start_profile()

        # Single batched generate call with all prompts
        outputs = llm.generate(decode_prompts, sampling_params, use_tqdm=False)

        if profile:
            llm.stop_profile()

        # Verify outputs
        assert len(outputs) == num_decodes
        for i, output in enumerate(outputs):
            assert len(output.outputs) > 0
            print(f"Prompt {i+1}/{num_decodes}: Input={len(output.prompt_token_ids)} tokens, "
                  f"Generated {len(output.outputs[0].token_ids)} tokens")

        print(f"First output: {outputs[0].outputs[0].text[:100]}...")
        print("Test passed!")

    finally:
        # Cleanup

        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=2, help="Number of tokens to generate per request")
    parser.add_argument("--prompt-len", type=int, default=8192, help="Approximate prompt length in tokens")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup phase")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--num-decodes",
                        type=int,
                        default=1,
                        help="Batch size for decode (number of concurrent requests)")
    args = parser.parse_args()

    test_llama_70b_2layer_tp4_bf16(max_tokens=args.max_tokens,
                                   warmup=not args.no_warmup,
                                   prompt_len=args.prompt_len,
                                   profile=args.profile,
                                   num_decodes=args.num_decodes)
