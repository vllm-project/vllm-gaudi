# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import os
from glob import glob

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 240.0  #torch.finfo(torch.float8_e4m3fn).max


def calc_maxabs_scale(xmaxabs, fullscale, backoff=1):
    scale = xmaxabs / (fullscale * backoff)
    return scale


def dynamic_quant_per_tensor(data, use_unit_quant=False):
    amax = torch.abs(data).max() + 1e-8
    scale = calc_maxabs_scale(amax, FP8_MAX, 1.0)
    if use_unit_quant:
        scale = torch.ones_like(scale)
    scale = scale.to(data.dtype)
    data_fp8 = data / scale.unsqueeze(-1)
    cliped_qtensor = torch.clamp(data_fp8, -FP8_MAX, FP8_MAX)
    cliped_qtensor_fp8 = cliped_qtensor.to(torch.float8_e4m3fn)
    return cliped_qtensor_fp8, scale.float()


def copy_other_files(input_path, output_path):
    import shutil

    for file in os.listdir(input_path):
        if file.endswith(".json") or \
            file.endswith(".txt") or \
            file.endswith(".py") or \
            file.endswith("jinja"):
            print(f"copying {file} to {output_path}")
            shutil.copyfile(
                os.path.join(input_path, file),
                os.path.join(output_path, file),
            )



def convert_files(input_path, output_path, input_scale_path, num_experts,
                  use_unit_quant):
    all_safetensors = glob(f"{input_path}/*.safetensors")
    # sort by file name
    all_safetensors.sort()
    model_list = {}

    with safe_open(input_scale_path, framework="pt",
                   device="cpu") as input_scale:
        for safetensors_path in all_safetensors:
            print(f"processing {safetensors_path}")
            tensors = {}
            with safe_open(safetensors_path, framework="pt",
                           device="cpu") as tensor_file:
                for k in tensor_file.keys():  # noqa: SIM118
                    tensor = tensor_file.get_tensor(k)
                    if len(tensor.shape) == 3 and tensor.size(0) == num_experts:
                        for idx in range(num_experts):
                            new_field = "experts." + str(idx)
                            weight_name = k.replace(".moe.", f".moe.{new_field}.", 1)
                            weight_fp32 = tensor[idx]
                            weight_fp8, weight_scale = dynamic_quant_per_tensor(
                                weight_fp32,
                                use_unit_quant=use_unit_quant,
                            )
                            weight_scale_name = weight_name + "_scale"
                            input_scale_name = weight_name.rstrip(
                                "weight") + "input_scale"
                            input_scale_tensor = input_scale.get_tensor(
                                input_scale_name).float() * 448.0 / 240.0
                            tensors.update({
                                input_scale_name:
                                input_scale_tensor
                            })
                            tensors.update(
                               {weight_name: weight_fp8})
                            tensors.update({
                                weight_scale_name:
                                weight_scale
                            })
                            model_list.update({
                                input_scale_name:
                                safetensors_path.split("/")[-1]
                            })
                            model_list.update({
                                weight_name:
                                safetensors_path.split("/")[-1]
                            })
                            model_list.update({
                                weight_scale_name:
                                safetensors_path.split("/")[-1]
                            })
                    else:
                        print(f"skip {k}.")
                        tensors.update({k: tensor})
                        model_list.update({k: safetensors_path.split("/")[-1]})
            new_tensor_path = safetensors_path.replace(input_path, output_path)
            save_file(tensors, new_tensor_path)
            print(f"saving to {new_tensor_path}")

    result = {"weight_map": model_list, "metadata": {}}
    out_json_path = output_path + "/model.safetensors.index.json"
    with open(out_json_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert tensors to float8 format.")
    parser.add_argument(
        "-i",
        "--input_path",
        default="/mnt/disk3/HF_models/Step-3.5-Flash",
        help="Path to the official model weights.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="/mnt/disk3/HF_models/Step-3.5-Flash-FP8-G2-skip",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "-s",
        "--input_scale_path",
        default="/mnt/disk6/HF_models/step3p5-input-scale.safetensors",
        help="Path to the output directory.",
    )
    parser.add_argument("-u",
                        "--unit_quant",
                        action="store_true",
                        help="Enable Unit FP8 Quant for the entire model")
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    input_scale_path = args.input_scale_path
    use_unit_quant = args.unit_quant

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    copy_other_files(input_path, output_path)
    convert_files(input_path, output_path, input_scale_path, 288,
                  use_unit_quant)
