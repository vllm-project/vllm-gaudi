# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="Qwen/Qwen3-8B", enforce_eager=False)
    return parser


def print_outputs(outputs):
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


def main(args):
    """
    Test script to actually instantiate HPUWorker and test sleep/wakeup functionality.
    This test creates a real HPUWorker instance and calls the methods.
    """
    llm = LLM(**args)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    outputs = llm.generate(prompts)
    print_outputs(outputs)

    for i in range(3):
        assert llm.llm_engine.is_sleeping() == False
        llm.sleep()
        assert llm.llm_engine.is_sleeping() == True
        llm.wake_up(["weights"])
        assert llm.llm_engine.is_sleeping() == True
        llm.wake_up(["kv_cache"])
        assert llm.llm_engine.is_sleeping() == False
        outputs = llm.generate(prompts)
        print_outputs(outputs)


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    try:
        main(args)
    except Exception:
        import traceback
        print("An error occurred during generation:")
        traceback.print_exc()
        os._exit(1)
