# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_gaudi.extension.profiler import HabanaMemoryProfiler


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct", enforce_eager=False)
    return parser


def print_outputs(outputs):
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


def assert_model_device(model_runner, targert_device):
    if model_runner:
        params_devices = list(set([p.device for p in model_runner.model.parameters()]))
        assert len(params_devices) == 1
        assert params_devices[0].type == targert_device


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

    multiproc = os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING")
    model_runner = None
    if multiproc == "0":
        model_runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner

    outputs = llm.generate(prompts)
    print_outputs(outputs)

    for i in range(3):
        with HabanaMemoryProfiler() as m:
            llm.sleep()
        assert m.consumed_device_memory < -60 * 1024 * 1024 * 1024  # check if more than 60GB was freed
        assert_model_device(model_runner, "cpu")

        with HabanaMemoryProfiler() as m:
            llm.wake_up(["weights", "kv_cache"])
        assert m.consumed_device_memory > 60 * 1024 * 1024 * 1024  # check if more than 60GB was allocated
        assert_model_device(model_runner, "hpu")

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
