#!/usr/bin/env python3  
"""  
Run Qwen3.5 in language-only or text+image mode based on arguments.  
"""  
import argparse  
import sys  
from vllm import LLM, SamplingParams  
from PIL import Image  
import requests  
from io import BytesIO  
  
  
def run_text_only(model_name: str, prompt: str):  
    """Run Qwen3.5 in language-only mode."""  
    print(f"Running {model_name} in language-only mode...")  
      
    llm = LLM(  
        model=model_name,  
        trust_remote_code=True,  
        language_model_only=True  # Disables all multimodal modules  
    )  
      
    sampling_params = SamplingParams(max_tokens=512, temperature=0.7)  
    outputs = llm.generate(prompt, sampling_params)  
      
    for output in outputs:  
        generated_text = output.outputs[0].text  
        print(f"Generated: {generated_text}")  
  
  
def run_text_image(model_name: str, prompt: str, image_url: str = None):  
    """Run Qwen3.5 with text+image support."""  
    print(f"Running {model_name} with text+image support...")  
      
    llm = LLM(  
        model=model_name,  
        trust_remote_code=True,  
        # Remove language_model_only to enable multimodal  
        limit_mm_per_prompt={"image": 1}  # Allow 1 image per prompt  
    )  
      
    # Load image  
    if image_url:  
        try:  
            response = requests.get(image_url)  
            image = Image.open(BytesIO(response.content))  
        except Exception as e:  
            print(f"Error loading image: {e}")  
            return  
    else:  
        print("Error: Image mode requires an image URL")  
        return  
      
    # Format prompt for multimodal input  
    formatted_prompt = f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"  
      
    sampling_params = SamplingParams(max_tokens=512, temperature=0.7)  
    outputs = llm.generate({  
        "prompt": formatted_prompt,  
        "multi_modal_data": {"image": image},  
    }, sampling_params)  
      
    for output in outputs:  
        generated_text = output.outputs[0].text  
        print(f"Generated: {generated_text}")  
  
  
def main():  
    parser = argparse.ArgumentParser(description="Run Qwen3.5 in language-only or text+image mode")  
    parser.add_argument(  
        "--mode",  
        choices=["text", "image"],  
        required=True,  
        help="Mode: 'text' for language-only, 'image' for text+image"  
    )  
    parser.add_argument(  
        "--model",  
        default="/software/data/pytorch/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307/",  
        help="Model name or path"  
    )  
    parser.add_argument(  
        "--prompt",  
        default="What's in this image?" if "--image" in sys.argv else "Hello, how are you?",  
        help="Text prompt"  
    )  
    parser.add_argument(  
        "--image-url",
        default="https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        help="Image URL (required for image mode)"  
    )  
      
    args = parser.parse_args()  
      
    if args.mode == "text":  
        run_text_only(args.model, args.prompt)  
    elif args.mode == "image":  
        if not args.image_url:  
            print("Error: --image-url is required for image mode")  
            sys.exit(1)  
        run_text_image(args.model, args.prompt, args.image_url)  
  
  
if __name__ == "__main__":  
    main()