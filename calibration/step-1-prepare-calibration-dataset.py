###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
import argparse
import os

import pandas as pd
import transformers

os.environ["EXPERIMENTAL_WEIGHT_SHARING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "true"


def get_ds(args):
    print(f"Loading source dataset: {args.dataset}")
    ds = pd.read_pickle(args.dataset)

    if args.max_dataset_samples:
        ds = ds.sample(frac=1, random_state=42)
        ds = ds.head(args.max_dataset_samples)

    return ds


def load_chat_template(chat_template_path: str) -> str:

    with open(chat_template_path) as f:
        return f.read()


def main(args):

    calibration_ds = get_ds(args)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model,
            model_max_length=args.max_model_length,
            padding_side="left",
            use_fast=False,
        )
    except (OSError, ValueError, RuntimeError, ImportError):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model,
            model_max_length=args.max_model_length,
            padding_side="left",
            use_fast=True,
        )

    chat_template = load_chat_template(args.chat_template) if args.chat_template else None

    print("Creating calibration dataset...")
    inputs = []
    for _, row in calibration_ds.iterrows():
        question = row["question"]
        system_prompt = row["system_prompt"]
        if "mixtral" in args.model or "Mixtral" in args.model:
            tmp_conversation = [{"role": "user", "content": question}, {"role": "assistant", "content": system_prompt}]
        else:
            tmp_conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        try:
            tmp_input = tokenizer.apply_chat_template(tmp_conversation,
                                                      chat_template=chat_template,
                                                      tokenize=False,
                                                      truncation=True)
        except ValueError:
            # Case when given model don't need any chat-template and can process raw string without any system tokens,
            # e.g. facebook/opt-125m
            tmp_input = f"{system_prompt}. {question}"
        inputs.append(tmp_input)

    calibration_ds['input'] = inputs

    print("Saving calibration dataset...")
    calibration_ds.to_pickle(f"{args.output_name}-calibration-dataset.pkl")
    print("Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a calibration dataset for a model.")
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-o", "--output_name", type=str, required=True)
    parser.add_argument("--max-model-length", type=int, default=1024)
    parser.add_argument("--max-dataset-samples", type=int, default=0)
    parser.add_argument("--chat-template",
                        type=str,
                        default="",
                        help="If not provided, the default chat-template from the model will be used.")

    args = parser.parse_args()

    main(args)
