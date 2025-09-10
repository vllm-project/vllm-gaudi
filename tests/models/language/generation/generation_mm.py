from argparse import ArgumentParser
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.multimodal.image import convert_image_mode
from dataclasses import asdict
from typing import Union
from PIL import Image
from dataclasses import dataclass
import yaml
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


@dataclass
class PROMPT_DATA:
    _questions = {
        "image": [
            "What is the most prominent object in this image?",
            "Describe the scene in the image.",
            "What is the weather like in the image?",
            "Write a short poem about this image."
        ],
        "video": [
            "Describe this video",
            "Which movie would you associate this video with?"
        ]
    }

    _data = {
        "image":
        lambda source: convert_image_mode(
            ImageAsset("cherry_blossom").pil_image
            if source == "default" else Image.open(source), "RGB"),
        "video":
        lambda source: VideoAsset(name="baby_reading"
                                  if source == "default" else source,
                                  num_frames=16).np_ndarrays
    }

    def __post_init__(self):
        self._questions = self._questions
        self._data = self._data

    def get_prompts(self,
                    modality: str = "image",
                    media_source: str = "default",
                    num_prompts: int = 1,
                    skip_vision_data=False):
        if modality == "image":
            pholder = "<|image_pad|>"
        elif modality == "video":
            pholder = "<|video_pad|>"
        else:
            raise ValueError(f"Unsupported modality: {modality}."
                             " Supported modality: [image, video]")
        questions = self._questions[modality]
        prompts = [
            ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
             f"<|im_start|>user\n<|vision_start|>{pholder}<|vision_end|>"
             f"{question}<|im_end|>\n"
             "<|im_start|>assistant\n") for question in questions
        ]

        data = self._data[modality](media_source)
        inputs = [{
            "prompt": prompts[i % len(prompts)],
            "multi_modal_data": {
                modality: data
            },
        } if not skip_vision_data else {
            "prompt": questions[i % len(questions)],
        } for i in range(num_prompts)]

        return inputs


def run_model(model_name: str, inputs: Union[dict, list[dict]], modality: str,
              **extra_engine_args):
    # Default mm_processor_kwargs
    # mm_processor_kwargs={
    #    "min_pixels": 28 * 28,
    #    "max_pixels": 1280 * 28 * 28,
    #    "fps": 1,
    # }
    passed_mm_processor_kwargs = extra_engine_args.get("mm_processor_kwargs",
                                                       {})
    passed_mm_processor_kwargs.setdefault("min_pixels", 28 * 28)
    passed_mm_processor_kwargs.setdefault("max_pixels", 1280 * 28 * 28)
    passed_mm_processor_kwargs.setdefault("fps", 1)
    extra_engine_args.update(
        {"mm_processor_kwargs": passed_mm_processor_kwargs})

    extra_engine_args.setdefault("max_model_len", 32768)
    extra_engine_args.setdefault("max_num_seqs", 5)
    extra_engine_args.setdefault("limit_mm_per_prompt", {modality: 1})

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
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
            media_source = input_data_config.get("media_source", "default")

            logger.info(
                "================================================\n"
                "Running test with configs:\n"
                "modality: %(modality)s\n"
                "input_data_config: %(input_data_config)s\n"
                "extra_engine_args: %(extra_engine_args)s\n"
                "================================================",
                dict(modality=modality,
                     input_data_config=input_data_config,
                     extra_engine_args=extra_engine_args))

            data = PROMPT_DATA()
            inputs = data.get_prompts(modality=modality,
                                      media_source=media_source,
                                      num_prompts=num_prompts)

            logger.info(
                "*** Questions for modality %(modality)s: %(questions)s",
                dict(modality=modality, questions=data._questions[modality]))
            responses = run_model(model_name, inputs, modality,
                                  **extra_engine_args)
            for response in responses:
                print(f"{response.outputs[0].text}")
                print("=" * 80)
        except Exception as e:
            logger.error("Error during test with modality %(modality)s: %(e)s",
                         dict(modality=modality, e=e))

            raise


def main():
    parser = ArgumentParser()
    parser.add_argument("--model-card-path",
                        required=True,
                        help="Path to .yaml file describing model parameters")
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
