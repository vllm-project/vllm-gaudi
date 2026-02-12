###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def find_measurement_path(measurement, measurements_dir_path, scales, group_size):
    measurment_card = "_" + measurement + "_" + str(group_size)
    for measurment_file in os.listdir(measurements_dir_path):
        filename = os.fsdecode(measurment_file)
        if (not filename.endswith(".json") or "_mod_list" in filename or measurment_card not in filename):
            continue
        if scales:
            if "MAXABS" in filename:
                return os.path.join(measurements_dir_path, measurment_file)
        else:
            if "MAXABS" not in filename:
                return os.path.join(measurements_dir_path, measurment_file)


def is_fused_moe_op(node_name):
    return ("moe" in node_name.lower() and ".w13_list" not in node_name and ".w2_list" not in node_name)


def is_moe_experts(node_name):
    return ("moe" in node_name.lower() and (".w13_list" in node_name or ".w2_list" in node_name))


def get_expert_id(node_name):
    parts = node_name.split(".")
    assert parts[-1].isdigit()
    expert_id = int(parts[-1])
    return expert_id


def get_expert_prefix(node_name):
    parts = node_name.split(".")
    assert parts[-1].isdigit()
    prefix = ".".join(parts[:-1])
    return prefix


def get_local_expert_num(data):
    expert_id = -1
    for mod_name in data:
        if is_moe_experts(mod_name):
            idx = get_expert_id(mod_name)
            expert_id = max(expert_id, idx)
    return expert_id + 1


def unify_measurements(measurement_group,
                       measurements_dir_path,
                       output_path,
                       groups_size,
                       groups_num,
                       group_index,
                       scales=False,
                       use_ep=False):
    measurements_paths = []
    group_name = ""

    # save all the jsons paths in the given measurement group
    for measurement in measurement_group:
        measurement_path = find_measurement_path(measurement, measurements_dir_path, scales, groups_size)
        if measurement_path is not None:
            measurements_paths.append(measurement_path)
        group_name += measurement

    if len(measurements_paths) == 0:
        print("Error: invalid measurement paths. No *.json files or no "
              "*mod_list.json files.")
        return

    # save all the jsons content in the given measurement group
    measurements_jsons = []
    for measurement_path in measurements_paths:
        with open(measurement_path) as f:
            js = json.load(f)
            measurements_jsons.append(js["Nodes"])
    # create a name for the unified json that will be created for this measurement group
    base_path = find_measurement_path(measurement_group[0], measurements_dir_path, scales, groups_size)
    old_suffix = "_" + measurement_group[0] + "_" + str(groups_size)
    new_suffix = "_" + str(group_index) + "_" + str(groups_num)
    unified_json_name = base_path.split("/")[-1].replace(old_suffix, new_suffix)
    unified_json_path = os.path.join(output_path, unified_json_name)

    # open a unified json file
    with open(measurements_paths[0]) as origin, open(unified_json_path, "w") as copy:
        copy.write(origin.read())
    with open(unified_json_path) as json_file:
        unified_json = json.load(json_file)
        unified_json["LocalRank"] = group_index if groups_num != 1 else -1

    moe_experts_data = {}
    # expert_num is original local_expert_num, it is used only when use_ep is True
    expert_num = get_local_expert_num(unified_json["Nodes"]) if use_ep else -1

    # iterate all unified json nodes
    for node_name, node_values in unified_json["Nodes"].items():
        max_inputs = node_values["inputs"]
        max_outputs = None
        if node_values.get("outputs") is not None:
            max_outputs = node_values["outputs"]
        max_weight = None
        if node_values.get("params") is not None and node_values["params"].get("weight") is not None:
            max_weight = node_values["params"]["weight"]

        # iterate over all the measurment group and take the maximum for each tensor and its channel
        if scales:
            for idx, measurement_json in enumerate(measurements_jsons):
                # for experts of moe, append results in all measurements
                if use_ep and is_moe_experts(node_name):
                    if node_name not in moe_experts_data:
                        moe_experts_data[node_name] = node_values
                    else:
                        prefix, local_expert_id = get_expert_prefix(node_name), get_expert_id(node_name)
                        # take original total_rank=8, total_expert_num=128,
                        # local_expert_num=16 and expert string.MoeOp.w13_list.11 on rank 3
                        # if target total_rank=4, then new local_expert_num=32,
                        # new expert is string.MoeOp.w13_list.27(16*1+11) on rank 1
                        new_node_name = ".".join((prefix, str(expert_num * idx + local_expert_id)))
                        assert new_node_name not in moe_experts_data
                        moe_experts_data[new_node_name] = measurement_json[node_name]
                    continue

                # for moe op, keep max of the first, retain rest from other measurements
                if use_ep and is_fused_moe_op(node_name) and idx > 0:
                    # input 0 of moe is hidden_states, we should get the max value
                    # across ranks during unification
                    # input 1 ~ local_expert_num is the intermidiate_amax of each
                    # expert, we should extend them during unification
                    max_inputs[0] = max(measurement_json[node_name]["inputs"][0], max_inputs[0])
                    max_inputs.extend(measurement_json[node_name]["inputs"][1:])
                else:
                    for i in range(0, len(max_inputs)):
                        max_inputs[i] = max(measurement_json[node_name]["inputs"][i], max_inputs[i])
                if max_outputs is not None:
                    max_outputs = max(measurement_json[node_name]["outputs"], max_outputs)
                if max_weight is not None:
                    max_weight = max(measurement_json[node_name]["params"]["weight"], max_weight)
        else:
            for idx, measurement_json in enumerate(measurements_jsons):
                # for experts of moe, append results in all measurements
                if use_ep and is_moe_experts(node_name):
                    if node_name not in moe_experts_data:
                        moe_experts_data[node_name] = node_values
                    else:
                        prefix, local_expert_id = get_expert_prefix(node_name), get_expert_id(node_name)
                        new_node_name = ".".join((prefix, str(expert_num * idx + local_expert_id)))
                        assert new_node_name not in moe_experts_data
                        moe_experts_data[new_node_name] = measurement_json[node_name]
                    continue

                for i in range(0, len(max_inputs)):
                    for j in range(0, len(max_inputs[i])):
                        max_inputs[i][j][0] = max(measurement_json[node_name]["inputs"][i][j][0], max_inputs[i][j][0])
                if max_outputs is not None:
                    if use_ep and is_fused_moe_op(node_name) and idx > 0:
                        max_outputs[0][0] = max(measurement_json[node_name]["outputs"][0][0], max_outputs[0][0])
                        max_outputs.extend(measurement_json[node_name]["outputs"][1:])
                    else:
                        for i in range(0, len(max_outputs)):
                            max_outputs[i][0] = max(measurement_json[node_name]["outputs"][i][0], max_outputs[i][0])
                if max_weight is not None:
                    for i in range(0, len(max_weight)):
                        max_weight[i][0] = max(measurement_json[node_name]["params"]["weight"][i][0], max_weight[i][0])

        # update the maximum in the unified json
        if scales:
            for i in range(0, len(max_inputs)):
                unified_json["Nodes"][node_name]["inputs"][i] = max_inputs[i]
            if max_outputs is not None:
                unified_json["Nodes"][node_name]["outputs"] = max_outputs
            if max_weight is not None:
                unified_json["Nodes"][node_name]["params"]["weight"] = max_weight
        else:
            for i in range(0, len(max_inputs)):
                for j in range(0, len(max_inputs[i])):
                    unified_json["Nodes"][node_name]["inputs"][i][j][0] = max_inputs[i][j][0]
            if max_outputs is not None:
                for i in range(0, len(max_outputs)):
                    unified_json["Nodes"][node_name]["outputs"][i][0] = max_outputs[i][0]
            if max_weight is not None:
                for i in range(0, len(max_weight)):
                    unified_json["Nodes"][node_name]["params"]["weight"][i][0] = max_weight[i][0]
    if use_ep:
        unified_json["Nodes"].update(moe_experts_data)
    global_rank = None
    local_rank = group_index if groups_num != 1 else -1
    mode = ""
    layers = {}
    with open(unified_json_path, "w") as json_file:
        json.dump(unified_json, json_file, indent=4)
    mode = unified_json["Mode"]
    nodes = unified_json["Nodes"]

    # create unified npz file from the unified json
    unified_npz_path = os.path.join(output_path, unified_json_name.replace(".json", ".npz"))
    for layer, dlayer in nodes.items():
        layers[layer] = {}
        layers[layer]["inputs"] = [np.array(x) for x in dlayer["inputs"]]
        if dlayer.get("outputs") is not None:
            layers[layer]["outputs"] = [np.array(x) for x in dlayer["outputs"]]
        if dlayer.get("params") is not None and dlayer["params"].get("weight") is not None:
            layers[layer]["params"] = {}
            layers[layer]["params"]["weight"] = np.array(dlayer["params"]["weight"])
    df = {"GlobalRank": global_rank, "LocalRank": local_rank, "Mode": mode, "Nodes": layers}
    with open(unified_npz_path, "w"):
        np.savez(unified_npz_path, df)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Run the measurements parser",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m",
                        "--measurements",
                        type=str,
                        help="path to the directory of the measurements that will be "
                        "unified")
    parser.add_argument("-r", "--rank", type=int, help="rank of unified measurements")
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=os.getcwd(),
        help="path to the directory where the unified measurements will be "
        "written",
    )
    parser.add_argument(
        "-u",
        "--use_expert_paral",
        action="store_true",
        help="unify original measurement results based on expert parallelism "
        "rules",
    )
    parser.add_argument(
        "-s",
        "--skip_unify_scales",
        action="store_true",
        help="skip the scale unification step.",
    )
    return parser.parse_args(args)


def prepare_group_list(measurements_path, rank):
    measure_files = glob.glob(os.path.join(measurements_path, "*_mod_list.json"))
    if len(measure_files) > 0:
        # take original rank=8 as an example, target file name: string_0_8_mod_list.json
        matched = re.match(r"^(\w+)_(\d+)_(\d+)_(\w+)_(\w+)\.json$", os.path.basename(measure_files[0]))
        if matched:
            total_rank = int(matched.group(3))
            assert (rank < total_rank) and (total_rank % rank) == 0, (
                f"Original total_rank {total_rank} should be larger than your "
                f"target rank {rank} and be divisible by it")
            group_size = total_rank // rank
            group_list = [[str(i * group_size + j) for j in range(group_size)] for i in range(rank)]
            print("Card grouping list >> {}".format(group_list))
            return group_list
        else:
            raise ValueError("Unrecognized file name!")
    else:
        raise ValueError("*_mod_list.json doesn't exist in {}".format(measurements_path))


def main(args):
    args = parse_args(args)
    output_path = args.out
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    measurements_path = args.measurements
    groups = prepare_group_list(measurements_path, args.rank)

    num_jsons_drange = 0
    num_jsons_scales = 0
    for path in os.listdir(measurements_path):
        if path.endswith(".json"):
            if "MAXABS" in path:
                num_jsons_scales += 1
            elif "mod_list" not in path:
                num_jsons_drange += 1
    assert (os.path.isdir(measurements_path) and (num_jsons_drange % len(groups)) == 0
            and (num_jsons_scales % len(groups)) == 0)

    for group_index, group in enumerate(groups):
        unify_measurements(group,
                           measurements_path,
                           output_path,
                           num_jsons_drange,
                           len(groups),
                           group_index,
                           scales=False,
                           use_ep=args.use_expert_paral)
        if not args.skip_unify_scales:
            unify_measurements(
                group,
                measurements_path,
                output_path,
                num_jsons_scales,
                len(groups),
                group_index,
                scales=True,
                use_ep=args.use_expert_paral,
            )

    print("finished measurement unifier script")


if __name__ == "__main__":
    main(sys.argv[1:])
