# Environment Variables

**Diagnostic and Profiling Knobs:**

- `VLLM_PROFILER_ENABLED`: if `true` - enables high-level profiler. Resulting JSON traces can be viewed at [perfetto.habana.ai](https://perfetto.habana.ai/#!/viewer). Disabled by default.
- `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION`: if `true` - logs graph compilations for each vLLM engine step, but only if any compilation occurs. It is highly recommended to use this in conjunction with `PT_HPU_METRICS_GC_DETAILS=1`.
  Disabled by default.
- `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL`: if `true` - logs graph compilations for every vLLM engine step, even if no compilation occurs. Disabled by default.
- `VLLM_HPU_LOG_STEP_CPU_FALLBACKS`: if `true` - logs CPU fallbacks for each vLLM engine step, but only if any fallback occurs. Disabled by default.
- `VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL`: if `true` - logs CPU fallbacks for each vLLM engine step, even if no fallback occur. Disabled by default.
- `VLLM_T_COMPILE_FULLGRAPH`: if `true` - PyTorch compile function raises an error if any graph breaks happen during compilation. This allows for the easy detection of existing graph breaks, which usually reduce performance. Disabled by default.
- `VLLM_T_COMPILE_DYNAMIC_SHAPES`: if `true` - PyTorch compiles graph with dynamic options set to None. It causes using dynamic shapes when needed.
- `VLLM_FULL_WARMUP`: if `true` - PyTorch assumes that warmup fully cover all possible tensor sizes and no compilation will occur afterwards. If compilation occurs after warmup, PyTorch will crash (with message like this: `Recompilation triggered with skip_guard_eval_unsafe stance. This usually means that you have not warmed up your model with enough inputs such that you can guarantee no more recompilations.`). If this happens, disable it. `false` by default.

**Performance Tuning Knobs:**

- `VLLM_GRAPH_RESERVED_MEM`: percentage of memory dedicated to HPUGraph capture. The default is `0.1`.
- `VLLM_EXPONENTIAL_BUCKETING`: if `true`, enables exponential bucket spacing instead of linear. The default is `true`.
- `VLLM_SKIP_WARMUP`: if `true`, warmup is skipped. The default is `false`.

!!! note
    If the model config reports a high `max_model_len`, set it to max `input_tokens+output_tokens` rounded up to a multiple of `block_size` as per actual requirements.

!!! tip
    When a deployed workload does not utilize the full context that a model can handle, it is good practice to limit the maximum values upfront based on the input and output token lengths that will be generated after serving the vLLM server.
    <br><br>**Example:**<br><br>Let's assume that we want to deploy text generation model Qwen2.5-1.5B, which has a defined `max_position_embeddings` of 131072 (our `max_model_len`). At the same time, we know that our workload pattern will not use the full context length because we expect a maximum input token size of 1K and predict generating a maximum of 2K tokens as output. In this case, starting the vLLM server to be ready for the full context length is unnecessary. Instead, we should limit it upfront to achieve faster service preparation and decrease warmup time. The recommended values in this example should be:
    > - `--max_model_len`: `3072` - the sum of input and output sequences (1+2)*1024.  
    > - `VLLM_PROMPT_SEQ_BUCKET_MAX`: `1024` - the maximum input token size that we expect to handle.

    - `VLLM_HANDLE_TOPK_DUPLICATES`: if ``true`` - handles duplicates outside top-k. The default is `false`.
    - `VLLM_CONFIG_HIDDEN_LAYERS`: configures how many hidden layers to run in a HPUGraph for model splitting among hidden layers when TP is 1.
        It helps to improve throughput by reducing inter-token latency limitations in some models. The default is `1`.

Additionally, there are HPU PyTorch Bridge environment variables impacting vLLM execution:

- `PT_HPU_LAZY_MODE`: if `0`, PyTorch Eager backend for Gaudi will be used. If `1`, PyTorch Lazy backend for Gaudi will be used. The default is `0`.

- `PT_HPU_ENABLE_LAZY_COLLECTIVES`: must be set to `true` for tensor parallel inference with HPU Graphs. The default is `true`.
- `PT_HPUGRAPH_DISABLE_TENSOR_CACHE`: must be set to `false` for LLaVA, qwen, and RoBERTa models. The default is `false`.
- `VLLM_PROMPT_USE_FLEX_ATTENTION`: enabled only for the Llama model, allowing usage of `torch.nn.attention.flex_attention` instead of FusedSDPA. Requires `VLLM_PROMPT_USE_FUSEDSDPA=0`. The default is `false`.

**Additional Performance Tuning Knobs - Linear Bucketing Strategy only:**

- `VLLM_{phase}_{dim}_BUCKET_{param}`: collection of 12 environment variables configuring ranges of bucketing mechanism (linear bucketing only).
  - `{phase}` is either `PROMPT` or `DECODE`
  - `{dim}` is either `BS`, `SEQ` or `BLOCK`
  - `{param}` is either `MIN`, `STEP` or `MAX`
  - Default values:
    - Prompt:
      - batch size min (`VLLM_PROMPT_BS_BUCKET_MIN`): `1`
      - batch size step (`VLLM_PROMPT_BS_BUCKET_STEP`): `32`
      - batch size max (`VLLM_PROMPT_BS_BUCKET_MAX`): `max_num_prefill_seqs`
      - sequence length min (`VLLM_PROMPT_SEQ_BUCKET_MIN`): `block_size`
      - sequence length step (`VLLM_PROMPT_SEQ_BUCKET_STEP`): `block_size`
      - sequence length max (`VLLM_PROMPT_SEQ_BUCKET_MAX`): `max_model_len`
      - sequence ctx min (`VLLM_PROMPT_CTX_BUCKET_MIN`): `0`
      - sequence ctx step (`VLLM_PROMPT_CTX_BUCKET_STEP`): `1`
      - sequence ctx max (`VLLM_PROMPT_CTX_BUCKET_MAX`): `(max_model_len - block_size) // block_size`
    - Decode:
      - batch size min (`VLLM_DECODE_BS_BUCKET_MIN`): `1`
      - batch size step (`VLLM_DECODE_BS_BUCKET_STEP`): `32`
      - batch size max (`VLLM_DECODE_BS_BUCKET_MAX`): `max_num_seqs`
      - block size min (`VLLM_DECODE_BLOCK_BUCKET_MIN`): `block_size`
      - block size step (`VLLM_DECODE_BLOCK_BUCKET_STEP`): `block_size`
      - block size max (`VLLM_DECODE_BLOCK_BUCKET_MAX`): `max_blocks`
