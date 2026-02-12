###############################################################################
# Copyright (C) 2025 Habana Labs, Ltd. an Intel Company
###############################################################################
import argparse
import json
import os
import sys

import numpy as np
import logging

# from loguru import logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_measurement_path(measurement, measurements_dir_path, group_size):
    measurement_card = "_" + measurement + "_" + str(group_size)
    for measurement_file in os.listdir(measurements_dir_path):
        filename = os.fsdecode(measurement_file)
        if (not filename.endswith(".json") or "_mod_list" in filename or measurement_card not in filename):
            continue
        if "MAXABS" not in filename:
            return os.path.join(measurements_dir_path, measurement_file)


def is_fused_moe_op(node_name):
    return ("moe" in node_name.lower() and ".w13_list" not in node_name and ".w2_list" not in node_name)


def is_moe_experts(node_name):
    # model.layers.3.mlp.experts.moe_op.w13_list.0
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


def expand_measurements(
    measurements_dir_path,
    output_path,
    local_rank,
    world_size,
):
    measurement_group = ["0"]
    # save all the jsons paths in the given measurement group
    groups_size = 1
    unified_measurement_index = "0"
    measurement_path = find_measurement_path(unified_measurement_index, measurements_dir_path, groups_size)
    measurements_jsons = []

    with open(measurement_path) as f:
        js = json.load(f)
        measurements_jsons.append(js["Nodes"])
    # New json file name
    new_json_name = (find_measurement_path(measurement_group[0], measurements_dir_path,
                                           groups_size).split("/")[-1].replace(
                                               "_" + measurement_group[0] + "_" + str(groups_size),
                                               "_" + str(local_rank) + "_" + str(world_size),
                                           ))
    logger.info(
        "Generating new json file: %s with local_rank %d and world_size %d",
        new_json_name,
        local_rank,
        world_size,
    )
    new_json_path = os.path.join(output_path, new_json_name)

    # Create a replica of the measurement json file
    with open(measurement_path) as origin, open(new_json_path, "w") as copy:
        copy.write(origin.read())
    with open(new_json_path) as json_file:
        new_json = json.load(json_file)
        new_json["LocalRank"] = local_rank

    expert_num = get_local_expert_num(new_json["Nodes"])
    total_experts = expert_num

    # Iterate all nodes
    for node_name, node_values in new_json["Nodes"].items():
        max_outputs = None
        if node_values.get("outputs") is not None:
            max_outputs = node_values["outputs"]

        # iterate over all the measurements and update the fused moe op with selected experts data

        for idx, measurement_json in enumerate(measurements_jsons):
            if is_fused_moe_op(node_name):
                node_res = measurement_json[node_name]["outputs"]
                node_res_output = node_res[0]
                node_res_experts_intermediate_amax = node_res[1:]
                num_intermediate_amax = len(node_res_experts_intermediate_amax)
                assert num_intermediate_amax == total_experts, (
                    f"the number of intermediate amax should be {total_experts}, but got {num_intermediate_amax}")
                ep_size = world_size
                ep_rank = local_rank
                num_local_experts = total_experts // ep_size
                expert_start_index = ep_rank * num_local_experts
                expert_end_index = expert_start_index + num_local_experts
                node_intermediate_amax = node_res_experts_intermediate_amax[expert_start_index:expert_end_index]
                assert len(node_intermediate_amax) == num_local_experts, (
                    f"len(node_intermediate_amax) should be {num_local_experts}, but got {len(node_intermediate_amax)}")
                max_outputs = [node_res_output, *node_intermediate_amax]
                logger.debug(
                    "Selecting %d outputs for %s "
                    "ep_rank %d with expert_start_index %d and expert_end_index %d",
                    len(max_outputs),
                    node_name,
                    ep_rank,
                    expert_start_index,
                    expert_end_index,
                )

        if max_outputs is not None and is_fused_moe_op(node_name):
            new_json["Nodes"][node_name]["outputs"] = max_outputs

    global_rank = None
    local_rank = local_rank
    mode = ""
    layers = {}
    with open(new_json_path, "w") as json_file:
        json.dump(new_json, json_file, indent=4)
    mode = new_json["Mode"]
    nodes = new_json["Nodes"]

    # create unified npz file from the new json
    unified_npz_path = os.path.join(output_path, new_json_name.replace(".json", ".npz"))
    for layer, dlayer in nodes.items():
        layers[layer] = {}
        layers[layer]["inputs"] = [np.array(x) for x in dlayer["inputs"]]
        if dlayer.get("outputs") is not None:
            layers[layer]["outputs"] = [np.array(x) for x in dlayer["outputs"]]
        if (dlayer.get("params") is not None and dlayer["params"].get("weight") is not None):
            layers[layer]["params"] = {}
            layers[layer]["params"]["weight"] = np.array(dlayer["params"]["weight"])
    df = {
        "GlobalRank": global_rank,
        "LocalRank": local_rank,
        "Mode": mode,
        "Nodes": layers,
    }
    with open(unified_npz_path, "w"):
        np.savez(unified_npz_path, df)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run the measurements parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--measurements",
        type=str,
        help="path to the directory of the measurements that have been unified",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=os.getcwd(),
        help="path to the directory where the expand measurements will be written",
    )
    parser.add_argument(
        "-w",
        "--target_world_size",
        type=int,
        help="The target number of ranks to expand the measurements to.",
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    output_path = args.out
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    measurements_path = args.measurements

    target_world_size = args.target_world_size
    for ep_rank in range(target_world_size):
        expand_measurements(
            measurements_dir_path=measurements_path,
            output_path=output_path,
            local_rank=ep_rank,
            world_size=target_world_size,
        )

    logger.info("finished expanding measurements for %d ranks", target_world_size)


if __name__ == "__main__":
    main(sys.argv[1:])
