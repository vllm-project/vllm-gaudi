# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)

    # Set defaults for pooling/embedding
    parser.set_defaults(
        model="intfloat/e5-mistral-7b-instruct",
        runner="pooling",
        enforce_eager=True,
    )
    return parser


def main(args: dict):
    # Create an LLM for embedding
    args = parser.parse_args()
    llm = LLM(**vars(args))

    # Run embedding
    outputs = llm.embed(PROMPTS)

    # Print outputs
    print("-" * 50)
    for prompt, output in zip(PROMPTS, outputs):
        embedding = output.outputs.embedding
        trimmed = (embedding[:16] + ["..."] if len(embedding) > 16 else embedding)
        print(f"Prompt: {prompt!r}\nEmbedding: {trimmed} "
              f"(length={len(embedding)})")
        print("-" * 50)


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
