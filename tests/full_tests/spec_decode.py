from vllm import LLM, SamplingParams

import os
import time
import argparse
import multiprocessing
import logging
from vllm.v1.metrics.reader import Counter, Vector
from typing import Optional
import lm_eval
from lm_eval.models.vllm_causallms import VLLM

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(processName)s][%(asctime)s] %(message)s",
)

os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["VLLM_CONTIGUOUS_PA"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def time_generation(llm: LLM,
                    prompts: Optional[list[str]],
                    sampling_params: SamplingParams,
                    num_spec_tokens=5,
                    num_warmups=1,
                    do_profile=False,
                    accuracy=None,
                    limit=None) -> dict:
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    # Warmup first
    accuracy_check = prompts is None
    ret = [""]
    if accuracy_check:
        task = "gsm8k"
        RTOL = 0.03
        FILTER = "exact_match,strict-match"
        accuracy = accuracy or 0.3
        start = time.time()
        results = lm_eval.simple_evaluate(model=llm, tasks=[task], limit=limit)
        end = time.time()
        latency = end - start
        try:
            measured_value = results["results"][task][FILTER]
        except KeyError as e:
            raise KeyError(f"Available metrics: {results['results']}") from e
        ret = [f"Task: {task} | Metric: {FILTER} | Expected: {accuracy} | Measured: {measured_value}"]
        metrics = llm.model.llm_engine.get_metrics()
        if accuracy > (measured_value + RTOL):
            raise AssertionError(f"Expected: {accuracy} |  Measured: {measured_value}")
    else:
        logging.info("Warming up the model...")
        for _ in range(num_warmups):
            llm.generate(prompts, sampling_params)
        logging.info("Starting generation...")
        ret = []
        if do_profile:
            llm.start_profile()
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end = time.time()
        latency = end - start
        if do_profile:
            llm.stop_profile()
        # Print the outputs.
        for output in outputs:
            generated_text = output.outputs[0].text
            ret.append(generated_text[:200])

        metrics = llm.llm_engine.get_metrics()
    logging.info("Generation completed in %.2f seconds.", latency)

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * (num_spec_tokens + 1)
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
    result_dict = {
        'ret_spec': ret,
        'latency': latency,
        'acc_counts': acceptance_counts,
        'acc_rate': accept_rate,
        'num_draft_tokens': num_draft_tokens,
        'num_drafts': num_drafts,
    }
    return result_dict


def create_error_result(e: Exception) -> dict:
    """Helper function to create a standardized error result dictionary."""
    return {
        'ret_spec': [],
        'latency': 0,
        'acc_counts': [],
        'acc_rate': 0,
        'num_draft_tokens': 0,
        'num_drafts': 0,
        'error': str(e)
    }


def test_ngram(is_enable, args, prompts, sampling_params, task_key, result_queue):
    VLLM_CLS = LLM if prompts is not None else VLLM
    kwargs = {"model":"Qwen/Qwen3-4B",} if prompts is not None \
        else {"pretrained":"Qwen/Qwen3-4B","batch_size":"16"}
    try:
        if not is_enable:
            llm = VLLM_CLS(
                **kwargs,
                disable_log_stats=False,
            )
        else:
            llm = VLLM_CLS(
                **kwargs,
                speculative_config={
                    "method": "ngram",
                    "prompt_lookup_max": 3,
                    "num_speculative_tokens": args.num_spec_tokens,
                },
                disable_log_stats=False,
            )

        result_dict = time_generation(llm,
                                      prompts,
                                      sampling_params,
                                      args.num_spec_tokens,
                                      args.num_warmups,
                                      args.do_profile,
                                      accuracy=args.accuracy_rate,
                                      limit=args.limit)
    except Exception as e:
        logging.exception("Task %s failed: %s", task_key, e)
        result_dict = create_error_result(e)

    result_queue.put((task_key, result_dict))


def test_eagle_model(is_enable, args, prompts, sampling_params, task_key, result_queue):
    VLLM_CLS = LLM if prompts is not None else VLLM
    kwargs = {"model":"meta-llama/Meta-Llama-3-8B-Instruct"} if prompts is not None \
        else {"pretrained":"meta-llama/Meta-Llama-3-8B-Instruct","batch_size":"16"}
    try:
        if not is_enable:
            llm = VLLM_CLS(
                **kwargs,
                disable_log_stats=False,
                enforce_eager=args.enforce_eager,
            )
        else:
            llm = VLLM_CLS(
                **kwargs,
                speculative_config={
                    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
                    "num_speculative_tokens": args.num_spec_tokens,
                },
                disable_log_stats=False,
                enforce_eager=args.enforce_eager,
            )

        result_dict = time_generation(llm,
                                      prompts,
                                      sampling_params,
                                      args.num_spec_tokens,
                                      args.num_warmups,
                                      args.do_profile,
                                      accuracy=args.accuracy_rate,
                                      limit=args.limit)
    except Exception as e:
        logging.exception("Task %s failed: %s", task_key, e)
        result_dict = create_error_result(e)
    result_queue.put((task_key, result_dict))


def test_eagle3_model(is_enable, args, prompts, sampling_params, task_key, result_queue):
    VLLM_CLS = LLM if prompts is not None else VLLM
    kwargs = {"model":"meta-llama/Meta-Llama-3-8B-Instruct",} if prompts is not None \
        else {"pretrained":"meta-llama/Meta-Llama-3-8B-Instruct","batch_size":"16"}
    try:
        if not is_enable:
            llm = VLLM_CLS(
                **kwargs,
                disable_log_stats=False,
                enforce_eager=args.enforce_eager,
            )
        else:
            llm = VLLM_CLS(
                **kwargs,
                speculative_config={
                    "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                    "num_speculative_tokens": args.num_spec_tokens,
                    "method": "eagle3",
                },
                disable_log_stats=False,
                enforce_eager=args.enforce_eager,
            )

        result_dict = time_generation(llm,
                                      prompts,
                                      sampling_params,
                                      args.num_spec_tokens,
                                      args.num_warmups,
                                      args.do_profile,
                                      accuracy=args.accuracy_rate,
                                      limit=args.limit)
    except Exception as e:
        logging.exception("Task %s failed: %s", task_key, e)
        result_dict = create_error_result(e)
    result_queue.put((task_key, result_dict))


def test_medusa_model(is_enable, args, prompts, sampling_params, task_key, result_queue):
    VLLM_CLS = LLM if prompts is not None else VLLM
    kwargs = {"model":"JackFram/llama-68m",} if prompts is not None \
        else {"pretrained":"JackFram/llama-68m",}
    try:
        if not is_enable:
            llm = VLLM_CLS(
                **kwargs,
                disable_log_stats=False,
                enforce_eager=args.enforce_eager,
            )
        else:
            llm = VLLM_CLS(
                **kwargs,
                speculative_config={
                    "model": "abhigoyal/vllm-medusa-llama-68m-random",
                    "num_speculative_tokens": args.num_spec_tokens,
                },
                disable_log_stats=False,
                enforce_eager=args.enforce_eager,
            )

        result_dict = time_generation(llm,
                                      prompts,
                                      sampling_params,
                                      args.num_spec_tokens,
                                      args.num_warmups,
                                      args.do_profile,
                                      accuracy=args.accuracy_rate,
                                      limit=args.limit)
    except Exception as e:
        logging.exception("Task %s failed: %s", task_key, e)
        result_dict = create_error_result(e)
    result_queue.put((task_key, result_dict))


def test_eaglemtp_model(is_enable, args, prompts, sampling_params, task_key, result_queue):
    VLLM_CLS = LLM if prompts is not None else VLLM
    kwargs = {"model":"eagle618/deepseek-v3-random",} if prompts is not None \
        else {"pretrained":"eagle618/deepseek-v3-random",}
    try:
        if not is_enable:
            llm = VLLM_CLS(
                **kwargs,
                disable_log_stats=False,
            )
        else:
            llm = VLLM_CLS(
                **kwargs,
                speculative_config={
                    "model": "eagle618/eagle-deepseek-v3-random",
                    "num_speculative_tokens": args.num_spec_tokens,
                },
                disable_log_stats=False,
            )

        result_dict = time_generation(llm,
                                      prompts,
                                      sampling_params,
                                      args.num_spec_tokens,
                                      args.num_warmups,
                                      args.do_profile,
                                      accuracy=args.accuracy_rate,
                                      limit=args.limit)
    except Exception as e:
        logging.exception("Task %s failed: %s", task_key, e)
        result_dict = create_error_result(e)
    result_queue.put((task_key, result_dict))


def test_mtp_model(is_enable, args, prompts, sampling_params, task_key, result_queue):
    VLLM_CLS = LLM if prompts is not None else VLLM
    kwargs = {"model":"/mnt/weka/data/pytorch/DeepSeek-R1",} if prompts is not None \
        else {"pretrained":"/mnt/weka/data/pytorch/DeepSeek-R1",}
    try:
        if not is_enable:
            llm = VLLM_CLS(
                **kwargs,
                tensor_parallel_size=8,
                enable_expert_parallel=True,
                disable_log_stats=False,
                trust_remote_code=True,
            )
        else:
            llm = VLLM_CLS(
                **kwargs,
                tensor_parallel_size=8,
                enable_expert_parallel=True,
                speculative_config={
                    "num_speculative_tokens": args.num_spec_tokens,
                    "method": "mtp",
                },
                disable_log_stats=False,
                trust_remote_code=True,
            )
            # llm = LLM(
            #     model="wemaster/deepseek_mtp_main_random_bf16",
            #     tensor_parallel_size=1,
            #     speculative_config={
            #         "num_speculative_tokens": 1,
            #     },
            #     trust_remote_code=True,
            #     disable_log_stats=False,
            #     max_model_len=4096,
            # )

        result_dict = time_generation(llm,
                                      prompts,
                                      sampling_params,
                                      args.num_spec_tokens,
                                      args.num_warmups,
                                      args.do_profile,
                                      accuracy=args.accuracy_rate,
                                      limit=args.limit)
    except Exception as e:
        logging.exception("Task %s failed: %s", task_key, e)
        result_dict = create_error_result(e)
    result_queue.put((task_key, result_dict))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Test spec decode.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--osl", type=int, default=50)
    parser.add_argument("--num_spec_tokens", type=int, default=1, help="Number of speculative tokens to generate.")
    parser.add_argument("--task", type=str, default="eagle", help="Tasks to run the evaluation on.")
    parser.add_argument("--run_base", action="store_true", help="Run the baseline tasks without speculative decoding.")
    parser.add_argument("--enforce_eager", action="store_true", help="Enforce eager execution for Eagle model.")
    parser.add_argument("--num_warmups", type=int, default=1, help="Number of warmup runs before timing.")
    parser.add_argument("--assert_accept_rate",
                        type=float,
                        default=0.0,
                        help="Assert that the acceptance rate is at least this value.")
    parser.add_argument("--do_profile", action="store_true", help="Enable profiling during generation.")
    parser.add_argument("--accuracy_rate",
                        type=float,
                        default=None,
                        help="Assert that the acceptance rate is at least this value.")
    parser.add_argument("--limit", type=int, default=64, help="Limit the number of samples for accuracy evaluation.")

    # 'ngram', 'eagle', 'eagle3', 'medusa', 'mlp_speculator',
    # 'draft_model' or 'deepseek_mtp
    # V1 does not support draft_model yet.
    # MLP speculator => https://github.com/vllm-project/vllm/pull/21276
    args = parser.parse_args()
    if args.do_profile:
        logging.info('Profiling is enabled. Results will be saved to '
                     './vllm_profile_spec_decode')
        os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile_spec_decode"

    sampling_params = SamplingParams(temperature=0, max_tokens=args.osl, ignore_eos=True)
    prompts: Optional[list[str]] = None
    if not args.accuracy_rate:
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
            prompts = prompts * (args.batch_size // len(prompts)) + prompts[:args.batch_size % len(prompts)]

    task_queue: dict[str, dict] = {}
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    task = args.task
    if task == "ngram":
        if args.run_base:
            task_queue['baseline_ngram'] = {
                'proc':
                multiprocessing.Process(target=test_ngram,
                                        args=(False, args, prompts, sampling_params, 'baseline_ngram', result_queue))
            }
        else:
            task_queue['spec_ngram'] = {
                'proc':
                multiprocessing.Process(target=test_ngram,
                                        args=(True, args, prompts, sampling_params, 'spec_ngram', result_queue))
            }
    elif task == "deepseek_eaglemtp":
        if args.run_base:
            task_queue['baseline_eaglemtp'] = {
                'proc':
                multiprocessing.Process(target=test_eaglemtp_model,
                                        args=(False, args, prompts, sampling_params, 'baseline_eaglemtp', result_queue))
            }
        task_queue['spec_eaglemtp'] = {
            'proc':
            multiprocessing.Process(target=test_eaglemtp_model,
                                    args=(True, args, prompts, sampling_params, 'spec_eaglemtp', result_queue))
        }
    elif task == "deepseek_mtp":
        if args.run_base:
            task_queue['baseline_mtp'] = {
                'proc':
                multiprocessing.Process(target=test_mtp_model,
                                        args=(False, args, prompts, sampling_params, 'baseline_mtp', result_queue))
            }
        task_queue['spec_mtp'] = {
            'proc':
            multiprocessing.Process(target=test_mtp_model,
                                    args=(True, args, prompts, sampling_params, 'spec_mtp', result_queue))
        }
    elif task == "eagle":
        if args.run_base:
            task_queue['baseline_eagle'] = {
                'proc':
                multiprocessing.Process(target=test_eagle_model,
                                        args=(False, args, prompts, sampling_params, 'baseline_eagle', result_queue))
            }
        task_queue['spec_eagle'] = {
            'proc':
            multiprocessing.Process(target=test_eagle_model,
                                    args=(True, args, prompts, sampling_params, 'spec_eagle', result_queue))
        }
    elif task == "eagle3":
        if args.run_base:
            task_queue['baseline_eagle3'] = {
                'proc':
                multiprocessing.Process(target=test_eagle3_model,
                                        args=(False, args, prompts, sampling_params, 'baseline_eagle3', result_queue))
            }
        task_queue['spec_eagle3'] = {
            'proc':
            multiprocessing.Process(target=test_eagle3_model,
                                    args=(True, args, prompts, sampling_params, 'spec_eagle3', result_queue))
        }
    elif task == "medusa":
        if args.run_base:
            task_queue['baseline_medusa'] = {
                'proc':
                multiprocessing.Process(target=test_medusa_model,
                                        args=(False, args, prompts, sampling_params, 'baseline_medusa', result_queue))
            }
        task_queue['spec_medusa'] = {
            'proc':
            multiprocessing.Process(target=test_medusa_model,
                                    args=(True, args, prompts, sampling_params, 'spec_medusa', result_queue))
        }
    else:
        raise ValueError(f"Unknown task: {task}")

    try:
        for key, task in task_queue.items():
            logging.info("=============== Starting task: %s ====================", key)
            task['proc'].start()
            task['proc'].join()
            logging.info("=============== Task %s completed. ====================", key)
        for _ in range(len(task_queue)):
            key, result_data = result_queue.get()
            task_queue[key]['result'] = result_data
    except KeyboardInterrupt:
        logging.info("Interrupted by user, terminating processes...")
    finally:
        for key, proc in task_queue.items():
            print(f"================= {key} =================")
            print(f"latency: {proc['result']['latency']}")
            print(f"acc_counts: {proc['result']['acc_counts']}")
            print(f"acc_rate: {proc['result']['acc_rate']}")
            print(f"num_draft_tokens: {proc['result']['num_draft_tokens']}")
            print(f"num_drafts: {proc['result']['num_drafts']}")
            if prompts:
                for prompt, text in zip(prompts, proc['result']['ret_spec']):
                    print("---")
                    print(f"Prompt: {prompt}")
                    print(f"Generated text: {text}'...'")
            else:
                print(f"accuracy check: {proc['result']['ret_spec'][0]}")
            print("=========================================")
            if proc['proc'].is_alive():
                proc['proc'].terminate()
                proc['proc'].join(timeout=2)
            if args.assert_accept_rate > 0 and 'spec' in key:
                assert proc['result']['acc_rate'] >= args.assert_accept_rate, \
                    f"Acceptance rate {proc['result']['acc_rate']} is lower" \
                    f"than the threshold {args.assert_accept_rate}"
        logging.info("Benchmark finished.")
