###############################################################################
# Copyright (C) 2025
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import pytest
from argparse import Namespace
import vllm

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@pytest.mark.parametrize(
    "model",
    [
        "intfloat/e5-mistral-7b-instruct",
        # "ssmits/Qwen2-7B-Instruct-embed-base",
        # "BAAI/bge-multilingual-gemma2",
    ])
def test_embeddings(model):

    args = Namespace(model=model, runner="pooling", enforce_eager=True)
    llm = vllm.LLM(**vars(args))

    outputs = llm.embed(PROMPTS)

    assert isinstance(outputs, list)
    assert len(outputs) == len(PROMPTS)

    for out in outputs:
        emb = out.outputs.embedding
        assert isinstance(emb, list)
        assert all(isinstance(x, float) for x in emb)
