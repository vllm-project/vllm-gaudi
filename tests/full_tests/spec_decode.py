from vllm import LLM, SamplingParams

import os
import time
import argparse
import multiprocessing
import logging
from vllm.v1.metrics.reader import Counter, Vector

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(processName)s][%(asctime)s] %(message)s",
)

os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["VLLM_CONTIGUOUS_PA"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def time_generation(llm: LLM,
                    prompts: list[str],
                    sampling_params: SamplingParams,
                    num_spec_tokens=5):
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    # Warmup first
    logging.info("Warming up the model...")
    llm.generate(prompts, sampling_params)
    llm.generate(prompts, sampling_params)
    logging.info("Starting generation...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
    latency = end - start
    logging.info("Generation completed in %.2f seconds.", latency)
    # Print the outputs.
    ret = []
    acceptance_counts = [0] * (num_spec_tokens + 1)
    for output in outputs:
        generated_text = output.outputs[0].text
        ret.append(generated_text)

    try:
        metrics = llm.llm_engine.get_metrics()
    except Exception as e:
        logging.error("Error getting metrics: %s", e)
        return ret, latency, acceptance_counts, 0
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    accept_rate = num_accepted_tokens / num_draft_tokens \
        if num_draft_tokens > 0 else 0.0
    return ret, latency, acceptance_counts, accept_rate


def test_ngram(is_enable, args, prompts, sampling_params, task_key,
               result_queue):
    if not is_enable:
        llm = LLM(
            model="Qwen/Qwen3-4B",
            disable_log_stats=False,
        )
    else:
        llm = LLM(
            model="Qwen/Qwen3-4B",
            speculative_config={
                "method": "ngram",
                "prompt_lookup_max": 3,
                "num_speculative_tokens": args.num_spec_tokens,
            },
            disable_log_stats=False,
        )

    try:
        ret_spec_ngram, latency, acceptance_counts, acc_rate = time_generation(
            llm, prompts, sampling_params, args.num_spec_tokens)
    except Exception as e:
        logging.error("Error during generation: %s", e)
        ret_spec_ngram = []
        latency = 0
        acceptance_counts = [0] * (args.num_spec_tokens + 1)
        acc_rate = 0.0
    result_dict = {
        'ret_spec': ret_spec_ngram,
        'latency': latency,
        'acc_counts': acceptance_counts,
        'acc_rate': acc_rate,
    }
    result_queue.put((task_key, result_dict))


def test_draft_model(is_enable, args, prompts, sampling_params, task_key,
                     result_queue):
    if not is_enable:
        llm = LLM(
            model="Qwen/Qwen3-4B",
            disable_log_stats=False,
        )
    else:
        llm = LLM(
            model="Qwen/Qwen3-4B",
            speculative_config={
                "method": "draft_model",
                "model": "Qwen/Qwen3-0.6B",
                "num_speculative_tokens": args.num_spec_tokens,
            },
            disable_log_stats=False,
        )

    ret_spec_draft, latency, acceptance_counts, acc_rate_draft = \
        time_generation(llm, prompts, sampling_params, args.num_spec_tokens)
    result_dict = {
        'ret_spec': ret_spec_draft,
        'latency': latency,
        'acc_counts': acceptance_counts,
        'acc_rate': acc_rate_draft,
    }
    result_queue.put((task_key, result_dict))


def test_mtp_model(is_enable, args, prompts, sampling_params, task_key,
                   result_queue):
    if not is_enable:
        llm = LLM(
            model="Qwen/Qwen3-4B",
            disable_log_stats=False,
        )
    else:
        llm = LLM(
            model="Qwen/Qwen3-4B",
            speculative_config={
                "method": "deepseek_mtp",
                "model": "Qwen/Qwen3-0.6B",
                "num_speculative_tokens": args.num_spec_tokens,
            },
            disable_log_stats=False,
        )

    ret_spec_mtp, latency, acc_counts, acc_rate_mtp = time_generation(
        llm, prompts, sampling_params, args.num_spec_tokens)
    result_dict = {
        'ret_spec': ret_spec_mtp,
        'latency': latency,
        'acc_counts': acc_counts,
        'acc_rate': acc_rate_mtp,
    }
    result_queue.put((task_key, result_dict))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Test spec decode.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--osl", type=int, default=256)
    parser.add_argument("--num_spec_tokens",
                        type=int,
                        default=5,
                        help="Number of speculative tokens to generate.")
    parser.add_argument("--tasks",
                        type=list,
                        default=["ngram"],
                        help="Tasks to run the evaluation on.")
    parser.add_argument(
        "--run_base",
        action="store_true",
        help="Run the baseline tasks without speculative decoding.")
    # 'ngram', 'eagle', 'eagle3', 'medusa', 'mlp_speculator',
    # 'draft_model' or 'deepseek_mtp
    # V1 does not support draft_model yet.
    # MLP speculator => https://github.com/vllm-project/vllm/pull/21276
    args = parser.parse_args()

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "San Francisco is know for its",
        "Facebook was created in 2004 by",
        "Curious George is a",
        "Python 3.11 brings improvements to its",
    ]
    if args.batch_size < len(prompts):
        prompts = prompts[:args.batch_size]
    else:
        prompts = prompts * (args.batch_size // len(prompts)
                             ) + prompts[:args.batch_size % len(prompts)]

    sampling_params = SamplingParams(temperature=0,
                                     max_tokens=args.osl,
                                     ignore_eos=True)

    task_queue = {}
    result_queue = multiprocessing.Queue()
    for task in args.tasks:
        if task == "ngram":
            if args.run_base:
                task_queue['baseline_ngram'] = {
                    'proc':
                    multiprocessing.Process(
                        target=test_ngram,
                        args=(False, args, prompts, sampling_params,
                              'baseline_ngram', result_queue))
                }
            task_queue['spec_ngram'] = {
                'proc':
                multiprocessing.Process(target=test_ngram,
                                        args=(True, args, prompts,
                                              sampling_params, 'spec_ngram',
                                              result_queue))
            }
        elif task == "deepseek_mtp":
            if args.run_base:
                task_queue['baseline_mtp'] = {
                    'proc':
                    multiprocessing.Process(
                        target=test_mtp_model,
                        args=(False, args, prompts, sampling_params,
                              'baseline_mtp', result_queue))
                }
            task_queue['spec_mtp'] = {
                'proc':
                multiprocessing.Process(target=test_mtp_model,
                                        args=(True, args, prompts,
                                              sampling_params, 'spec_mtp',
                                              result_queue))
            }
        elif task == "draft_model":
            if args.run_base:
                task_queue['baseline_draft'] = {
                    'proc':
                    multiprocessing.Process(
                        target=test_draft_model,
                        args=(False, args, prompts, sampling_params,
                              'baseline_draft', result_queue))
                }
            task_queue['spec_draft'] = {
                'proc':
                multiprocessing.Process(target=test_draft_model,
                                        args=(True, args, prompts,
                                              sampling_params, 'spec_draft',
                                              result_queue))
            }

    try:
        for key, task in task_queue.items():
            logging.info(
                "=============== Starting task: %s ====================", key)
            task['proc'].start()
            task['proc'].join()
            logging.info(
                "=============== Task %s completed. ====================", key)
        for _ in range(len(task_queue)):
            key, result_data = result_queue.get()
            task_queue[key]['result'] = result_data
    except KeyboardInterrupt:
        logging.info("Interrupted by user, terminating processes...")
    finally:
        for key, proc in task_queue.items():
            print(f"================= {key} =================")
            print(f"ret_spec: {[text[:50]+'...' \
                    for text in proc['result']['ret_spec']]}")
            print(f"latency: {proc['result']['latency']}")
            print(f"acc_counts: {proc['result']['acc_counts']}")
            print(f"acc_rate: {proc['result']['acc_rate']}")
            print("=========================================")
            if proc['proc'].is_alive():
                proc['proc'].terminate()
                proc['proc'].join(timeout=2)
        logging.info("Benchmark finished.")
