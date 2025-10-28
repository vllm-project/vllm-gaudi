import vllm
from vllm.entrypoints.llm import LLM

RUN_20B_MODEL = True  # Set to False to run the 120B model instead
MODEL_PATH = "lmsys/gpt-oss-20b-BF16"
MODEL_PATH_120 = "lmsys/gpt-oss-120b-BF16"
# reference https://github.com/huggingface/transformers/blob/68eb1a9a6353911f491b1c8139eb73d052a8e9b9/tests/models/gpt_oss/test_modeling_gpt_oss.py#L397
original_output = "Roses are red, violets are blue, I love you, and I love you too.\n\nRoses are red, vio"
# reference https://github.com/huggingface/transformers/blob/68eb1a9a6353911f491b1c8139eb73d052a8e9b9/tests/models/gpt_oss/test_modeling_gpt_oss.py#L462
original_output_120 = "Roses are red, violets are blue,\nI am a language model, not a human being"


def do_sample(llm: LLM, original_output: str, rtol: float, atol: float, max_num_seqs: int) -> list[str]:
    prompts = [
        "Roses are red, violets",
    ] * max_num_seqs

    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=20,
    )
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    assert prompts[0] + generated_texts[0] == original_output, "Generated text does not match the expected output."
    return generated_texts


expected_output = [
    "are blue, I love you, and I love you too.\n\nRoses are red, vio"  # noqa: E501
]


def _test_gpt_oss():
    """Main function that sets up and runs the prompt processing."""
    if RUN_20B_MODEL:
        llm = LLM(
            MODEL_PATH,
            max_num_seqs=8,
            dtype='bfloat16',
            enforce_eager=True,
            max_model_len=512,
            max_num_batched_tokens=2048,
            tensor_parallel_size=1,
        )
        generated_texts = do_sample(llm, original_output=original_output, rtol=1e-01, atol=1e-01, max_num_seqs=1)
    else:
        llm = LLM(
            MODEL_PATH_120,
            max_num_seqs=8,
            dtype='bfloat16',
            enforce_eager=False,
            max_model_len=512,
            max_num_batched_tokens=2048,
            tensor_parallel_size=4,
        )
        generated_texts = do_sample(llm, original_output=original_output_120, rtol=1e-01, atol=1e-01, max_num_seqs=1)
    assert generated_texts == expected_output


def test_gpt_oss_1x():
    _test_gpt_oss()
