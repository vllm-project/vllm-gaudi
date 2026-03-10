import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch


TENSOR_FIELDS = [
    "core_attn_out_non_spec",
    "mixed_qkv_non_spec_T",
    "mixed_qkv_non_spec",
    "last_recurrent_state",
    "input_hidden_states",
    "output_hidden_states",
]
INDEX_FIELD = "updated_state_indices"
META_FIELDS = [
    "layer_name",
    "layer_idx",
    "layer_type",
    "tp_rank",
    "step",
]


def natural_key(path: str):
    base = os.path.basename(path)
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", base)]


def list_pt_files(path: str) -> list[str]:
    if os.path.isfile(path) and path.endswith(".pt"):
        return [os.path.abspath(path)]

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    files = []
    abs_dir = os.path.abspath(path)
    for name in os.listdir(abs_dir):
        full = os.path.join(abs_dir, name)
        if os.path.isfile(full) and name.endswith(".pt"):
            files.append(full)
    return sorted(files, key=natural_key)


def load_payload(path: str) -> dict:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(data)}")
    return data


def payload_key(payload: dict):
    layer_id = payload.get("layer_name")
    if layer_id is None:
        layer_idx = payload.get("layer_idx", "<unknown_layer_idx>")
        layer_type = payload.get("layer_type", "<unknown_layer_type>")
        layer_id = f"decoder_layer_{layer_idx}.{layer_type}"

    return (
        layer_id,
        payload.get("tp_rank", "<unknown_tp_rank>"),
        payload.get("step", "<unknown_step>"),
    )


def meta_line(payload: dict) -> str:
    if payload.get("layer_name") is not None:
        return (
            f"layer_name={payload.get('layer_name')} "
            f"tp_rank={payload.get('tp_rank')} "
            f"step={payload.get('step')}"
        )

    return (
        f"layer_idx={payload.get('layer_idx')} "
        f"layer_type={payload.get('layer_type')} "
        f"tp_rank={payload.get('tp_rank')} "
        f"step={payload.get('step')}"
    )


def to_np_fp32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(dtype=torch.float32, device="cpu").numpy()


def angle(a: np.ndarray, b: np.ndarray) -> float:
    v1 = a.flatten()
    v2 = b.flatten()
    d = v1 - v2
    if np.min(d) == np.max(d) == 0:
        return 0.0

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")

    dot = np.dot(v1 / n1, v2 / n2)
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.arccos(dot))


def metric_line(a: np.ndarray, b: np.ndarray) -> str:
    af = a.flatten()
    bf = b.flatten()
    diff = af - bf

    nonzero = np.count_nonzero(af)
    maep = np.sum(
        np.abs(np.divide(diff, af, out=np.zeros_like(diff), where=af != 0))
    ) / nonzero if nonzero else 0.0

    l1_norm = np.linalg.norm(diff, 1) / len(af)
    inf_norm = np.linalg.norm(diff, np.inf)
    allclose = np.allclose(af, bf, atol=1e-3, rtol=1e-3)
    ang = angle(a, b)

    return (
        f"MAEP:{maep:.6f} L1_NORM:{l1_norm:.6f} INF_NORM:{inf_norm:.6f} "
        f"ANGLE:{ang:.6f} ALLCLOSE:{allclose}"
    )


def compare_index_field(left: dict, right: dict):
    l = left.get(INDEX_FIELD)
    r = right.get(INDEX_FIELD)

    if l is None and r is None:
        print(f"  {INDEX_FIELD}: both None")
        return
    if l is None or r is None:
        print(f"  {INDEX_FIELD}: mismatch (one is None)")
        return
    if not isinstance(l, torch.Tensor) or not isinstance(r, torch.Tensor):
        print(f"  {INDEX_FIELD}: mismatch (non-tensor value)")
        return

    same_shape = tuple(l.shape) == tuple(r.shape)
    same_value = same_shape and torch.equal(l.cpu(), r.cpu())
    print(
        f"  {INDEX_FIELD}: shape_l={tuple(l.shape)} shape_r={tuple(r.shape)} equal={same_value}"
    )


def compare_tensor_field(left: dict, right: dict, field: str):
    l = left.get(field)
    r = right.get(field)

    if l is None and r is None:
        print(f"  {field}: both None")
        return
    if l is None or r is None:
        print(f"  {field}: mismatch (one is None)")
        return
    if not isinstance(l, torch.Tensor) or not isinstance(r, torch.Tensor):
        print(f"  {field}: mismatch (non-tensor value)")
        return

    print(f"  {field}: shape_l={tuple(l.shape)} shape_r={tuple(r.shape)}")
    if tuple(l.shape) != tuple(r.shape):
        print(f"   shape mismatch (skip metrics) {l.shape=} {r.shape=}")
        if l.ndim == 2 and r.ndim == 2:
            rows = min(l.shape[0], r.shape[0])
            cols = min(l.shape[1], r.shape[1])
            l = l[:rows, :cols]
            r = r[:rows, :cols]
            print(f"{field} converting to {l.shape=} {r.shape=}") 
        elif l.ndim == 4 and l.shape[1] != r.shape[1]:
           v = l.shape[1]
           r = r[:,:v]
           print(f"{field} converting to {l.shape=} {r.shape=}")

    print("   " + metric_line(to_np_fp32(l), to_np_fp32(r)))


def compare_pair(l_path: str, l_payload: dict, r_path: str, r_payload: dict):
    print("=" * 100)
    print(f"left : {l_path}")
    print(f"right: {r_path}")
    print("meta : " + meta_line(l_payload))

    compare_index_field(l_payload, r_payload)
    for field in TENSOR_FIELDS:
        compare_tensor_field(l_payload, r_payload, field)


def compare_dirs(left_dir: str, right_dir: str):
    left_files = list_pt_files(left_dir)
    right_files = list_pt_files(right_dir)

    left_map = defaultdict(list)
    right_map = defaultdict(list)

    for p in left_files:
        payload = load_payload(p)
        left_map[payload_key(payload)].append((p, payload))

    for p in right_files:
        payload = load_payload(p)
        right_map[payload_key(payload)].append((p, payload))

    keys_left = set(left_map.keys())
    keys_right = set(right_map.keys())

    only_left = sorted(keys_left - keys_right)
    only_right = sorted(keys_right - keys_left)
    common = sorted(keys_left & keys_right)

    if only_left:
        print(f"Only in left ({len(only_left)}):")
        for k in only_left:
            print(" ", k)
    if only_right:
        print(f"Only in right ({len(only_right)}):")
        for k in only_right:
            print(" ", k)

    if not common:
        print("No common payload keys found by (layer_id, tp_rank, step).")
        return

    compared = 0
    for key in common:
        l_entries = sorted(left_map[key], key=lambda x: natural_key(x[0]))
        r_entries = sorted(right_map[key], key=lambda x: natural_key(x[0]))
        n = min(len(l_entries), len(r_entries))

        if len(l_entries) != len(r_entries):
            print(
                f"Warning key={key}: duplicate count differs "
                f"left={len(l_entries)} right={len(r_entries)}"
            )

        for i in range(n):
            compare_pair(l_entries[i][0], l_entries[i][1], r_entries[i][0], r_entries[i][1])
            compared += 1

    print("=" * 100)
    print(f"Compared {compared} matched payload pair(s).")


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python compare_qwen3next_debug_pt.py "
            "<left_pt_or_dir> <right_pt_or_dir>"
        )
        sys.exit(1)

    left = sys.argv[1]
    right = sys.argv[2]

    left_is_pt = os.path.isfile(left) and left.endswith(".pt")
    right_is_pt = os.path.isfile(right) and right.endswith(".pt")

    if left_is_pt and right_is_pt:
        left_payload = load_payload(left)
        right_payload = load_payload(right)
        compare_pair(left, left_payload, right, right_payload)
        return

    compare_dirs(left, right)


if __name__ == "__main__":
    main()
