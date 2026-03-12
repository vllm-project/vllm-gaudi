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
    """Remove Qwen reasoning blocks from visible output."""
    # Remove complete <think>...</think> blocks first.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # If generation was truncated mid-think, drop the dangling think suffix.
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()
  
  
def run_text_only(
    model_name: str,
    prompt: str,
    text_api: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
):
    """Run Qwen3.5 in language-only mode."""  
    print(f"Running {model_name} in language-only mode...")  
      
    llm = LLM(  
        model=model_name,  
        trust_remote_code=True,  #enforce_eager=True,
        language_model_only=True,  # Disables all multimodal modules
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization, #enforce_eager=True
    )  
      
    sampling_params = SamplingParams(max_tokens=50, temperature=0.0, top_p=1.0)

    if text_api == "chat":
        # Use chat-form input for instruct models to avoid prompt-format drift.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Reply in English only. "
                    "Do not output <think> tags or hidden reasoning."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        outputs = llm.chat(messages, sampling_params)
    elif text_api == "generate":
        direct_prompt = (
            "Answer directly and concisely. "
            "Do not output <think> tags or hidden reasoning.\n"
            f"User: {prompt}\nAssistant:"
        )
        outputs = llm.generate(direct_prompt, sampling_params)
    else:
        raise ValueError(f"Unsupported text_api: {text_api}")
      
    for output in outputs:  
        generated_text = _strip_think_text(output.outputs[0].text)
        print(f"Generated: {generated_text}")  
  
  
def run_text_image(
    model_name: str,
    prompt: str,
    image_url: str | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int = 512,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int = 512,
    gpu_memory_utilization: float = 0.5,
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
    parser = argparse.ArgumentParser(description="Run Qwen3.5 in language-only, text+image, or text+video mode")  
    parser.add_argument(  
        "--mode",  
        choices=["text", "image", "video"],  
        required=True,  
        help="Mode: 'text' for language-only, 'image' for text+image, 'video' for text+video"  
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
        "--image-url",
        default="https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        help="Image URL (required for image mode)"  
    )
    parser.add_argument(
        "--video-url",
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
        help="Video URL (required for video mode)"
    )  
      
    args = parser.parse_args()
    
    # Set default prompt based on mode if not provided
    if args.prompt is None:
        if args.mode == "image":
            args.prompt = "What's in this image?"
        elif args.mode == "video":
            args.prompt = "What's in this video?"
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
        )
  
  
if __name__ == "__main__":  
    main()
