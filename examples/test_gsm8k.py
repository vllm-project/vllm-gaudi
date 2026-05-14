import os

from vllm import LLM, SamplingParams

os.environ["PT_HPU_QKV_SLICE_SEQ_LEN_THLD"] = "1"
os.environ["PT_HPU_SDPA_BR_FACTOR"] = "64"
os.environ["PT_HPU_SDPA_BC_FACTOR"] = "64"
os.environ["PT_HPU_SDPA_QKV_SLICE_MODE_FWD"] = "1"
os.environ["VLLM_FUSEDSDPA_SLIDE_THLD"] = "0"
os.environ["VLLM_USE_V1"] = "true"
os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["PT_HPU_LAZY_MODE"] = "1"


def main():
    os.environ["VLLM_SKIP_WARMUP"] = "true"
    prompts = [
        "Question: Dante needs half as many cups of flour to bake his chocolate cake "
        "as he needs eggs. If he uses 60 eggs in his recipe, calculate the total number "
        "of cups of flour and eggs that he uses altogether.\n"
        "Answer: Dante needs half as many cups of flour to bake his chocolate cake as he needs eggs, "
        "meaning he needs 60/2 = <<60/2=30>>30 cups of flour for his recipe.\n"
        "Altogether, Dante needs 30+60 = <<30+60=90>>90 eggs and cups of flour for his recipe.\n#### 90\n\n"
        "Question: Vincent’s washer broke so he had to go to the laundromat. On Wednesday he washed six "
        "loads of clothes. The next day he had time to wash double the number of loads he did the day before. "
        "On Friday he had a test and could only manage half of the loads he did on Thursday. On Saturday the "
        "laundromat closed at noon and he could only wash a third of the loads of laundry he did on Wednesday. "
        "How many loads of laundry had he washed that week?\n"
        "Answer: On Thursday he washed 6*2=<<6*2=12>>12 loads of laundry\n"
        "On Friday he washed 12/2=<<12/2=6>>6 loads of laundry\n"
        "On Saturday he washed 6/3=<<6/3=2>>2 loads of laundry\n"
        "In total he washed 6+12+6+2= <<6+12+6+2=26>>26 loads of laundry\n#### 26\n\n"
        "Question: Nick hides 10 chocolates in his closet. His brother Alix hides 3 times as many "
        "chocolates than Nick hides. Last night, their mom found and took 5 chocolates from Alix. "
        "How many more chocolates does Alix have than Nick?\n"
        "Answer: Alix has 3 * 10 = <<3*10=30>>30 chocolates.\n"
        "After his mom took 5 chocolates, Alix has now 30 - 5 = <<30-5=25>>25 chocolate.\n"
        "So Alix has 25 - 10 = <<25-10=15>>15 chocolates more than Nick.\n#### 15\n\n"
        "Question: Tyler has $100. If he buys 8 scissors for $5 each and 10 erasers for $4 each, "
        "how much of his money remains?\n"
        "Answer: 8 * $5 = $<<8*5=40>>40 is the price of 8 scissors.\n"
        "10 * $4 = $<<10*4=40>>40 is the price of 10 erasers.\n"
        "You can find the total price by adding the cost of the scissors and erasers: "
        "$40 + $40 = $<<40+40=80>>80\n"
        "$100 - $80 = $<<100-80=20>>20 is the remaining amount of his money.\n#### 20\n\n"
        "Question: Mary, Edna, and Lucy are athletes who train every Saturday. Mary ran 3/8 of a "
        "24-kilometer field on one Saturday. Edna ran 2/3 of the distance of Edna, and Lucy ran 5/6 "
        "of the distance of Edna. How many more kilometers should Lucy run to cover the same distance as Mary?\n"
        "Answer: Mary ran 3/8 x 24 = <<3/8*24=9>>9 kilometers.\n"
        "Edna ran 2/3 x 9 = <<2/3*9=6>>6 kilometers.\n"
        "Lucy ran 5/6 x 6 = <<5/6*6=5>>5 kilometers.\n"
        "Lucy should run 9 - 5 = <<9-5=4>>4 kilometers more.\n#### 4\n\n"
        "Question: Jeans makeup artist charges her $250 an hour.  She requires very expensive makeup for a "
        "movie she is in and it takes 6 hours to do each day and she needs it done 4 times a week.  The movie "
        "takes 5 weeks to finish.  After the movie is done the makeup artist gives Jean a 10% discount because "
        "of the amount of work done.  How much did Jean pay?\nAnswer:"
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=1024)
    model = "/mnt/disk3/HF_models/Step-3.5-Flash"
    # model = "/mnt/disk6/HF_models/Hunyuan-A13B-Instruct-FP8-G2"
    # model = "/mnt/weka/llm/DeepSeek-V2-Lite-Chat/"
    # model = "/mnt/weka/data/mlperf_models/Mixtral-8x7B-Instruct-v0.1"
    # model = "/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B/"
    kwargs = {"tensor_parallel_size": 8}
    # if os.path.basename(model) in ["Qwen3-30B-A3B", "DeepSeek-V2-Lite-Chat"]:
    kwargs["enable_expert_parallel"] = True
    llm = LLM(model=model, max_model_len=4096, trust_remote_code=True, **kwargs)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print()
        print(f"Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
