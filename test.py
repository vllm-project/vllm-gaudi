#!/usr/bin/env python3  
"""  
Run Qwen3.5 in language-only or text+image mode based on arguments.  
"""  
import argparse  
import re
import sys  
from vllm import LLM, SamplingParams  
from PIL import Image  
import requests  
from io import BytesIO  


def _strip_think_text(text: str) -> str:
    """
    Remove Qwen reasoning blocks from visible output.
    Matches the official vLLM Qwen3ReasoningParser behavior.
    
    The Qwen3 reasoning parser works as follows:
    1. Strip <think> if present in the generated output (old template style)
    2. Everything before </think> is reasoning (hidden)
    3. Everything after </think> is content (visible/scoreable)
    4. If no </think> present, everything is reasoning (truncated output)
    
    This returns only the CONTENT part (after </think>), which is what
    lm_eval uses for scoring.
    """
    start_token = "<think>"
    end_token = "</think>"
    
    # Strip <think> if present in the generated output
    # (newer templates put <think> in the prompt, so it usually won't appear)
    if start_token in text:
        text = text.partition(start_token)[2]
    
    # Find the end token
    if end_token not in text:
        # No </think> means output was truncated - everything is reasoning
        # Return empty string since there's no content yet
        return ""
    
    # Extract content (everything after </think>)
    reasoning, _, content = text.partition(end_token)
    
    # Return only the content part (what comes after </think>)
    return content.strip() if content else ""
  
  
def run_text_only(
    model_name: str,
    prompt: str,
    text_api: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    enable_reasoning: bool = False,  # Off by default
    enforce_eager: bool = False,
):
    """Run Qwen3.5 in language-only mode."""  
    mode_desc = "with reasoning" if enable_reasoning else "without reasoning"
    print(f"Running {model_name} in language-only mode ({mode_desc})...")  
      
    llm = LLM(  
        model=model_name,  
        trust_remote_code=True,
        language_model_only=True,  # Disables all multimodal modules
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    
    # Adjust max_tokens based on reasoning mode: 50 for direct, 500 for reasoning
    max_tokens = 500 if enable_reasoning else 50
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    # # Alternative: Balanced sampling parameters for natural, helpful responses
    # # For deterministic/factual use: temperature=0.0
    # # For creative use: temperature=0.9, top_p=0.95
    # sampling_params = SamplingParams(
    #     temperature=0.7,           # Balanced creativity
    #     top_p=0.9,                 # Nucleus sampling
    #     max_tokens=2048,           # Allow longer responses
    #     repetition_penalty=1.05,   # Discourage repetition
    #     stop=["</s>", "<|im_end|>"],  # Natural stop tokens
    # )

    # Get processor for chat template
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name)
    if enable_reasoning:
        system_content = "You are a helpful assistant. Think step-by-step and show your reasoning."
    else:
        system_content = (
            "You are a helpful assistant. Answer directly and concisely. "
            "Do not output <think> tags or hidden reasoning."
        )
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]

    # Apply chat template with enable_thinking parameter
    prompts = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_reasoning  # Control reasoning at template level
    )

    outputs = llm.generate(prompts, sampling_params)

    print(f"\n****Prompt****:\n{prompts}\n")
    for output in outputs:
        generated_text = output.outputs[0].text
        
        if enable_reasoning:
            # Show both raw and cleaned output when reasoning is enabled
            clean_text = _strip_think_text(generated_text)
            print(f"****Generated (with <think> tags)****:\n{generated_text}\n")
            print(f"=" * 80)
            print(f"****Generated (cleaned)****:\n{clean_text}\n")
        else:
            # Just show the output when reasoning is disabled
            print(f"****Generated****:\n{generated_text}\n")


def run_gsm8k(
    model_name: str,
    prompt: str,
    text_api: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    enforce_eager: bool = False,
):
    """Run Qwen3.5 with GSM8K lm_eval settings."""  
    print(f"Running {model_name} in GSM8K mode (matching lm_eval settings)...")  
      
    llm = LLM(  
        model=model_name,  
        trust_remote_code=True,
        language_model_only=True,  # Disables all multimodal modules
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        reasoning_parser='qwen3',  # CRITICAL: Enable Qwen3 reasoning parser at engine level
        seed=1234,  # Match lm_eval seed for reproducibility
        enforce_eager=enforce_eager,
    )  
      
    # Match lm_eval defaults for GSM8K
    # lm_eval uses: gen_kwargs: {'until': ['Question:', '</s>', '<|im_end|>'], 'do_sample': False, 'temperature': 0.0}
    # In vLLM, 'until' maps to 'stop' parameter
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding (deterministic) - matches do_sample=False
        max_tokens=16384,  # Set high, but generation will stop at 'stop' sequences
        stop=["Question:", "</s>", "<|im_end|>"],  # Match lm_eval 'until' parameter
    )

    # Get processor for chat template
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name)

    # Match lm_eval format: no system message for GSM8K
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Apply chat template
    prompts = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = llm.generate(prompts, sampling_params)

    print(f"\n****Prompt****:\n{prompts}\n")
    for output in outputs:
        generated_text = output.outputs[0].text
        clean_text = _strip_think_text(generated_text)

        # Debug info for reasoning parser behavior
        num_tokens = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason
        print(f"****Generated (Raw with <think> tags)****:")
        print(f"Tokens: {num_tokens}, Finish reason: {finish_reason}")
        print(f"{generated_text}\n")
        print(f"=" * 80)

        print(f"****Generated (Cleaned - like lm_eval reasoning_parser=qwen3)****:")
        print(f"This is what lm_eval would use for scoring:\n{clean_text}\n")

def run_text_image(
    model_name: str,
    prompt: str,
    image_url: str | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int = 512,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int = 512,
    gpu_memory_utilization: float = 0.5,
    enforce_eager: bool = False,
):
    """Run Qwen3.5 with text+image support."""  
    print(f"Running {model_name} with text+image support...")  
      
    llm = LLM(  
        model=model_name,  
        trust_remote_code=True,  
        limit_mm_per_prompt={"image": 1},  # Allow 1 image per prompt
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )  

    if not image_url:
        print("Error: Image mode requires a image URL")
        return

    try:
        from vllm.multimodal.utils import fetch_image
        image = fetch_image(image_url)
    except Exception as e:
        print(f"Error loading image: {e}")
        return


    # Load image  
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt },
            {"type": "image"},  # Placeholder for template formatting
        ],
    }]

    # Get processor for chat template
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name)

    # Apply chat template
    prompts = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    sampling_params = SamplingParams(max_tokens=300, temperature=0.7)
    outputs = llm.generate({
        "prompt": prompts,
        "multi_modal_data": {"image": image},
    }, sampling_params)

    print(f"\nPrompt:\n{prompts}\n")
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated: {generated_text}")


def run_text_video(
    model_name: str,
    prompt: str,
    video_url: str | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int = 4096,
    gpu_memory_utilization: float = 0.5,
    enforce_eager: bool = False,
):
    """Run Qwen3.5 with text+video support."""
    print(f"Running {model_name} with text+video {video_url} support with prompt: {prompt}...")
    
    # Use mm_processor_kwargs to limit video frames
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        limit_mm_per_prompt={"video": 1},  # Allow 1 video per prompt
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    
    # Load video
    if not video_url:
        print("Error: Video mode requires a video URL")
        return

    try:
        from vllm.multimodal.utils import fetch_video
        video = fetch_video(video_url)
    except Exception as e:
        print(f"Error loading video: {e}")
        return

    # Get processor for chat template
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name)

    # Create messages with video placeholder - similar to gemma3 pattern
    # Note: The placeholder is just for the chat template formatting
    # The actual video data is passed via multi_modal_data
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video"},  # Placeholder for template formatting
        ],
    }]

    # Apply chat template
    prompts = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"\nPrompt:\n{prompts}\n")
    sampling_params = SamplingParams(max_tokens=300, temperature=0.7)
    outputs = llm.generate(
        {"prompt": prompts, "multi_modal_data": {"video": video}},
        sampling_params
    )

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated: {generated_text}")

def main():
    parser = argparse.ArgumentParser(description="Run Qwen3.5 in language-only, text+image, text+video, or GSM8K mode")  
    parser.add_argument(  
        "--mode",  
        choices=["text", "image", "video", "gsm8k"],  
        required=True,  
        help="Mode: 'text' for language-only, 'image' for text+image, 'video' for text+video, 'gsm8k' for GSM8K evaluation"  
    )
    parser.add_argument(  
        "--model",  
        default="/software/data/pytorch/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307/",  
        help="Model name or path"  
    )
    parser.add_argument(  
        "--prompt",  
        default=None,  
        help="Text prompt"  
    )
    parser.add_argument(
        "--text-api",
        choices=["chat", "generate"],
        default="generate",
        help="Text mode API: 'chat' (recommended) or 'generate' for comparison",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for model loading",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Maximum model context length (lower reduces KV memory)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum number of concurrent sequences",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=512,
        help="Maximum total tokens per batch",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.75,
        help="Fraction of device memory vLLM may reserve for KV/cache planning",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable graph compilation and run in eager mode for stability debugging.",
    )
    parser.add_argument(  
        "--image-url",
        default="https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        help="Image URL (required for image mode)"  
    )
    parser.add_argument(
        "--video-url",
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
        help="Video URL (required for video mode)"
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable reasoning mode for text generation (uses <think> tags). Off by default."
    )
    parser.add_argument(
        "--use-reasoning-parser",
        action="store_true",
        help="Use Qwen3 reasoning parser (like lm_eval reasoning_parser=qwen3). "
             "Shows both reasoning and final answer separately."
    )
      
    args = parser.parse_args()
    
    # Set default prompt based on mode if not provided
    if args.prompt is None:
        if args.mode == "image":
            args.prompt = "What's in this image?"
        elif args.mode == "video":
            args.prompt = "What's in this video?"
        elif args.mode == "gsm8k":
            args.prompt = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
        else:
            args.prompt = "Hello, how are you?"  
    if args.mode == "text":  
        run_text_only(
            args.model,
            args.prompt,
            args.text_api,
            args.tensor_parallel_size,
            args.max_model_len,
            args.max_num_seqs,
            args.max_num_batched_tokens,
            args.gpu_memory_utilization,
            enable_reasoning=args.reasoning,  # Pass reasoning flag
            enforce_eager=args.enforce_eager,
        )
    elif args.mode == "gsm8k":  
        run_gsm8k(
            args.model,
            args.prompt,
            args.text_api,
            args.tensor_parallel_size,
            args.max_model_len,
            args.max_num_seqs,
            args.max_num_batched_tokens,
            args.gpu_memory_utilization,
            args.enforce_eager,
        )
    elif args.mode == "image":  
        if not args.image_url:  
            print("Error: --image-url is required for image mode")  
            sys.exit(1)  
        run_text_image(
            args.model,
            args.prompt,
            args.image_url,
            args.tensor_parallel_size,
            args.max_model_len,
            args.max_num_seqs,
            args.max_num_batched_tokens,
            args.gpu_memory_utilization,
            args.enforce_eager,
        )
    elif args.mode == "video":
        if not args.video_url:
            print("Error: --video-url is required for video mode")
            sys.exit(1)
        run_text_video(
            args.model,
            args.prompt,
            args.video_url,
            args.tensor_parallel_size,
            args.max_model_len,
            args.max_num_seqs,
            args.max_num_batched_tokens,
            args.gpu_memory_utilization,
            args.enforce_eager,
        )
  
  
if __name__ == "__main__":  
    main()
