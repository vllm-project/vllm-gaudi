from argparse import ArgumentParser
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset, ImageAssetName
from vllm.assets.video import VideoAsset
from vllm.multimodal.image import convert_image_mode
from dataclasses import asdict
from typing import Union, get_args
from PIL import Image
from dataclasses import dataclass
import yaml
import os
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


@dataclass
class PROMPT_DATA:
    _questions = {
        "image": [
            "What is the most prominent object in this image?", "Describe the scene in the image.",
            "What is the weather like in the image?", "Write a short poem about this image."
        ],
        "multi_image": [
            "Compare and contrast these images. What are the similarities and differences?",
            "Tell a story that connects all these images together.",
            "What common themes do you see across these images?",
            "Describe the progression or sequence shown in these images.", "Which image stands out the most and why?",
            "What emotions or moods are conveyed by these images collectively?"
        ],
        "video": ["Describe this video", "Which movie would you associate this video with?"]
    }

    def __post_init__(self):
        self._questions = self._questions

    def _load_single_image(self, source: str) -> Image.Image:
        """Load a single image"""
        if source == "default":
            return convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
        else:
            return convert_image_mode(Image.open(source), "RGB")

    def _load_video(self, source: str):
        """Load video data"""
        return VideoAsset(name="baby_reading" if source == "default" else source, num_frames=16).np_ndarrays

    def _load_multiple_images(self, source: Union[str, list[str]]) -> list[Image.Image]:
        images = []
        """Load multiple images from various sources"""
        if source == "default":
            # Get all available ImageAsset names from the Literal type
            available_assets = list(get_args(ImageAssetName))
            logger.info("Available ImageAssets: %(available_assets)s", {"available_assets": available_assets})

            # Load up to 6 different assets (or more if needed)
            target_count = 6
            loaded_count = 0
            for asset_name in available_assets:
                if loaded_count >= target_count:
                    break

                try:
                    img = ImageAsset(asset_name).pil_image
                    converted_img = convert_image_mode(img, "RGB")
                    images.append(converted_img)
                    loaded_count += 1
                    logger.info("Successfully loaded ImageAsset: %(asset_name)s (Size: %(size)s)",
                                dict(asset_name=asset_name, size=converted_img.size))
                except Exception as e:
                    logger.warning("Failed to load ImageAsset '%(asset_name)s': %(e)s", dict(asset_name=asset_name,
                                                                                             e=e))
                    continue

        elif isinstance(source, list):
            # Load from list of file paths
            for img_path in source:
                try:
                    img = Image.open(img_path)
                    images.append(convert_image_mode(img, "RGB"))
                except Exception as e:
                    logger.warning("Failed to load image %(img_path)s: %(e)s", dict(img_path=img_path, e=e))

        logger.info("Loaded %(num_images)s images for multi-image processing", {"num_images": len(images)})
        return images

    def _get_data(self, modality: str, source: str):
        """Get data based on modality"""
        if modality == "image":
            return self._load_single_image(source)
        elif modality == "multi_image":
            return self._load_multiple_images(source)
        elif modality == "video":
            return self._load_video(source)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def get_prompts(self,
                    model_name: str = "",
                    modality: str = "image",
                    media_source: str = "default",
                    num_prompts: int = 1,
                    num_images: int = 1,
                    skip_vision_data=False):

        # Handle multi-image modality
        if modality == "multi_image" or modality == "image":
            pholder = "<start_of_image>" * num_images if "gemma" in model_name.lower() else "<|image_pad|>" * num_images
        elif modality == "video":
            pholder = "<video>" if "gemma" in model_name.lower() else "<|video_pad|>"
        else:
            raise ValueError(f"Unsupported modality: {modality}."
                             " Supported modality: [image, video, multi_image]")

        questions = self._questions[modality]

        prompts = [("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n<|vision_start|>{pholder}<|vision_end|>"
                    f"{question}<|im_end|>\n"
                    "<|im_start|>assistant\n") for question in questions]

        data = self._get_data(modality, media_source)

        # For multi_image, data is a list of images
        if modality == "multi_image":
            inputs = [
                {
                    "prompt": prompts[i % len(prompts)],
                    "multi_modal_data": {
                        "image": data  # Pass list of images
                    },
                } if not skip_vision_data else {
                    "prompt": questions[i % len(questions)],
                } for i in range(num_prompts)
            ]
        else:
            inputs = [{
                "prompt": prompts[i % len(prompts)],
                "multi_modal_data": {
                    modality: data
                },
            } if not skip_vision_data else {
                "prompt": questions[i % len(questions)],
            } for i in range(num_prompts)]

        return inputs


def run_model(model_name: str, inputs: Union[dict, list[dict]], modality: str, **extra_engine_args):
    # Default mm_processor_kwargs
    passed_mm_processor_kwargs = extra_engine_args.get("mm_processor_kwargs", {})
    passed_mm_processor_kwargs.setdefault("min_pixels", 28 * 28)
    passed_mm_processor_kwargs.setdefault("max_pixels", 1280 * 28 * 28)
    passed_mm_processor_kwargs.setdefault("fps", 1)
    extra_engine_args.update({"mm_processor_kwargs": passed_mm_processor_kwargs})

    extra_engine_args.setdefault("max_model_len", 32768)
    extra_engine_args.setdefault("max_num_seqs", 5)

    # For multi-image, allow multiple images per prompt
    if modality == "multi_image":
        extra_engine_args.setdefault("limit_mm_per_prompt", {"image": 10})  # Allow up to 10 images
    else:
        extra_engine_args.setdefault("limit_mm_per_prompt", {modality: 1})

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,  # Increased for multi-image responses
    )

    engine_args = EngineArgs(model=model_name, **extra_engine_args)

    engine_args = asdict(engine_args)
    llm = LLM(**engine_args)

    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
        use_tqdm=False,  # Disable tqdm for CI tests
    )
    return outputs


def start_test(model_card_path: str):
    with open(model_card_path) as f:
        model_card = yaml.safe_load(f)

    model_name = model_card.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
    test_config = model_card.get("test_config", [])
    if not test_config:
        logger.warning("No test configurations found.")
        return

    for config in test_config:
        modality = "image"  # Ensure modality is always defined
        try:
            modality = config.get("modality", "image")
            extra_engine_args = config.get("extra_engine_args", {})
            input_data_config = config.get("input_data_config", {})
            num_prompts = input_data_config.get("num_prompts", 1)
            num_images = input_data_config.get("num_images", 6)
            media_source = input_data_config.get("media_source", "default")

            logger.info(
                "================================================\n"
                "Running test with configs:\n"
                "modality: %(modality)s\n"
                "input_data_config: %(input_data_config)s\n"
                "extra_engine_args: %(extra_engine_args)s\n"
                "================================================",
                dict(modality=modality, input_data_config=input_data_config, extra_engine_args=extra_engine_args))

            data = PROMPT_DATA()
            inputs = data.get_prompts(model_name=model_name,
                                      modality=modality,
                                      media_source=media_source,
                                      num_prompts=num_prompts,
                                      num_images=num_images)

            logger.info("*** Questions for modality %(modality)s: %(questions)s",
                        dict(modality=modality, questions=data._questions[modality]))
            responses = run_model(model_name, inputs, modality, **extra_engine_args)
            for response in responses:
                print(f"{response.outputs[0].text}")
                print("=" * 80)
        except Exception as e:
            logger.error("Error during test with modality %(modality)s: %(e)s", dict(modality=modality, e=e))
            raise


def main():
    parser = ArgumentParser()
    parser.add_argument("--model-card-path", required=True, help="Path to .yaml file describing model parameters")
    args = parser.parse_args()
    start_test(args.model_card_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import os
        import traceback
        print("An error occurred during generation:")
        traceback.print_exc()
        os._exit(1)
