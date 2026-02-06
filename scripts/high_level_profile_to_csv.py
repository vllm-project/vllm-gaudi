import argparse
import json
import csv
import os
import re


def is_valid_record(record):
    return isinstance(record, dict) and all(key in record for key in ['pid', 'tid', 'name', 'ts'])


def json_to_csv(json_name):
    with open(json_name) as file:
        lines = file.readlines()
        if lines and lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1] + "]"
        records = json.loads("".join(lines))

    records = [record for record in records if is_valid_record(record)]
    if not records:
        return

    # Get warmup end timestamp to filter out warmup records
    warmup_record = next((record for record in records if record.get("name") == "warmup"), None)
    warmup_end_ts = warmup_record.get("ts") + warmup_record.get("dur") if warmup_record is not None else 0.0

    # Get block size
    block_size = next(
        (record.get("args", {}).get("const_block_size") for record in records if record.get("name") == "utils"), None)
    if block_size is None:
        print("[WARNING] Cannot find block size from utils record. Using the default value 128.")
        block_size = 128

    # Get use_graphs from the model forward records
    use_graphs_dict = {}  # {model_forward_name (str): use_graphs (bool)}
    for record in records:
        # skip warmup records
        time_stamp = record.get("ts")
        if warmup_record is not None and time_stamp < warmup_end_ts:
            continue

        record_name = record.get("name")
        # recode name is f'model_forward_bs{bs}_seq{seq_len}_ctx{ctx_blocks}_graphs{use_graphs}'
        # extract phase, bs, seq_len, ctx_blocks, use_graphs from record_name
        match = re.match(r'model_forward_bs(\d+)_seq(\d+)_ctx(\d+)_graphs([TF])', record_name)
        if match:
            bucket_bs, bucket_seq_len, bucket_ctx_blocks, use_graphs = match.groups()
            use_graphs = use_graphs == "T"
            model_forward_name = f"model_forward_bs{bucket_bs}_seq{bucket_seq_len}_ctx{bucket_ctx_blocks}"
            use_graphs_dict[model_forward_name] = use_graphs

    # Get shapes from the utils records
    out_records = []
    for record in records:
        # skip warmup records
        time_stamp = record.get("ts")
        if warmup_record is not None and time_stamp < warmup_end_ts:
            continue

        record_name = record.get("name")
        if record_name != "utils":
            continue

        for phase in ["decode", "prompt"]:
            if f"{phase}_bucket_batch_size" in record["args"]:
                bucket_batch_size = record["args"][f"{phase}_bucket_batch_size"]
                batch_size = record["args"][f"{phase}_batch_size"]
                bucket_seq_len = record["args"][f"{phase}_bucket_seq_len"]
                seq_len = record["args"][f"{phase}_seq_len"]
                bucket_ctx_blocks = record["args"][f"{phase}_bucket_ctx_blocks"]
                bucket_ctx_len = bucket_ctx_blocks * block_size
                ctx_len = record["args"][f"{phase}_ctx_len"]
                model_forward_name = f"model_forward_bs{bucket_batch_size}_seq{bucket_seq_len}_ctx{bucket_ctx_blocks}"
                use_graphs = use_graphs_dict.get(model_forward_name, False)
                out_record = {
                    "phase": phase,
                    "bucket_batch_size": bucket_batch_size,
                    "batch_size": batch_size,
                    "bucket_seq_len": bucket_seq_len,
                    "seq_len": seq_len,
                    "bucket_ctx_len": bucket_ctx_len,
                    "ctx_len": ctx_len,
                    "use_graphs": use_graphs,
                    "time_stamp": time_stamp,
                }
                out_records.append(out_record)
    if out_records:
        csv_name = json_name.replace(".json", ".csv")
        print(f"Saving CSV file to {csv_name}")
        out_keys = out_records[0].keys()
        with open(csv_name, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, out_keys)
            dict_writer.writeheader()
            dict_writer.writerows(out_records)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Convert JSON profiling results to CSV files.")
    arg_parser.add_argument("--input_path",
                            "-i",
                            type=str,
                            default=".",
                            help='''
        Path to the JSON high-level profiling file or a directory containing JSON files. 
        Default is current directory. 
        The script will process the specified JSON file or all JSON files in the specified directory 
        and its subdirectories and save the results to the corresponding CSV files.
        ''')
    args = arg_parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: The specified path {args.input_path} does not exist.")
        exit(1)

    if os.path.isfile(args.input_path):
        if args.input_path.endswith(".json"):
            print(f"Processing {args.input_path}")
            json_to_csv(args.input_path)
        else:
            print(f"Error: The specified file {args.input_path} is not a JSON file.")
            exit(1)
    else:
        for root, dirs, files in os.walk(args.input_path):
            for name in files:
                if name.endswith(".json"):
                    json_path = os.path.join(root, name)
                    print(f"Processing {json_path}")
                    json_to_csv(json_path)
