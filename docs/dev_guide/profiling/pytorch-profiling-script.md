# PyTorch Profiling via Script

!!! note
    This method is part of end-to-end profiling and does not need to be performed separately if end-to-end profiling has already been completed.

To trace specific portions of code in standalone Python scripts using PyTorch tracing tools, follow these steps:

1. Set the output directory.

    ```bash
    export VLLM_TORCH_PROFILER_DIR=/tmp
    ```

2. Enable tracing in the script by instructing the LLM object to start and stop profiling.

    ```bash
    from vllm import LLM, SamplingParams
    llm = LLM(model="facebook/opt-125m")
    llm.start_profile() # Start profiling
    outputs = llm.generate(["San Francisco is a"])
    llm.stop_profile() # Stop profiling
    ```

Performing this procedure results in generating a `*.pt.trace.json.gz` file that can be opened using [Perfetto](https://perfetto.habana.ai).
