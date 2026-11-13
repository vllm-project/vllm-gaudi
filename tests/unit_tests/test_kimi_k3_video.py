# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 video input: pure helpers, prompt construction, K3-only gating and the processor / model glue.

Runs without vllm: the vllm classes the video module builds on are replaced by small stubs.
"""
import importlib.util
import logging
import math
import sys
import types
from collections.abc import Mapping
from typing import Any
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / "vllm_gaudi"
UTILS_NAME = "vllm_gaudi.models.kimi_k3_video_utils"

# media_proc_cfg of the Kimi-K3 checkpoint (the keys the helpers read)
K3_CFG = {
    "patch_size": 14,
    "merge_kernel_size": 2,
    "in_patch_limit": 65536,
    "patch_limit_on_one_side": 512,
    "in_patch_limit_each_frame": 16384,
    "in_patch_limit_video": 655360,
    "temporal_merge_kernel_size": 4,
    "sample_fps": 8.0,
}
IBM_WIDTH, IBM_HEIGHT, IBM_FRAMES, IBM_FPS = 864, 480, 20, 2.0
RAW_PIXEL = 255
IBM_CHUNK_TOKENS = 558  # 868x504 padded -> 62 x 36 patches / 2x2 merge
IBM_VISION_TOKENS = 5 * IBM_CHUNK_TOKENS


def _navit_chunk_tokens(
    width: int,
    height: int,
    media_proc_cfg: Mapping[str, object],
) -> int:
    """Vision tokens of ONE video chunk, whatever its length (the tower pools the frames of a chunk over time).

    Re-implements the size arithmetic of ``navit_resize_video`` for a chunk so the count can be checked without vLLM;
    the ``in_patch_limit_video`` term never binds for 4-frame chunks with the K3 config and is not modelled.
    """
    patch_size = int(media_proc_cfg["patch_size"])  # type: ignore[call-overload]
    merge = int(media_proc_cfg["merge_kernel_size"])  # type: ignore[call-overload]
    per_frame_limit = media_proc_cfg.get("in_patch_limit_each_frame") or media_proc_cfg["in_patch_limit"]
    side_limit = int(media_proc_cfg["patch_limit_on_one_side"])  # type: ignore[call-overload]

    s1 = math.sqrt(per_frame_limit /
                   (max(1.0, width // patch_size) * max(1.0, height // patch_size)))  # type: ignore[operator]
    s2 = side_limit * patch_size / width
    s3 = side_limit * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = min(max(1, int(width * scale)), side_limit * patch_size)
    new_h = min(max(1, int(height * scale)), side_limit * patch_size)

    factor = merge * patch_size
    pad_w = (factor - new_w % factor) % factor
    pad_h = (factor - new_h % factor) % factor
    return ((new_h + pad_h) // factor) * ((new_w + pad_w) // factor)


def _video_vision_tokens(
    utils,
    width: int,
    height: int,
    num_frames: int,
    source_fps: float,
    media_proc_cfg: Mapping[str, object],
) -> int:
    """Total ``<|media_pad|>`` count of a video: planned chunks times tokens per chunk."""
    chunk_size = int(media_proc_cfg.get("temporal_merge_kernel_size",
                                        utils.DEFAULT_TEMPORAL_CHUNK_SIZE))  # type: ignore[call-overload]
    max_fps = float(media_proc_cfg.get("sample_fps", utils.DEFAULT_MAX_SAMPLE_FPS))  # type: ignore[arg-type]
    chunks, _ = utils.plan_video_chunks(num_frames, num_frames, source_fps, None, max_fps, chunk_size)
    return len(chunks) * _navit_chunk_tokens(width, height, media_proc_cfg)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def utils(monkeypatch):
    monkeypatch.delitem(sys.modules, UTILS_NAME, raising=False)
    mod = _load(UTILS_NAME, ROOT / "models" / "kimi_k3_video_utils.py")
    monkeypatch.setitem(sys.modules, UTILS_NAME, mod)
    return mod


# ---------------------------------------------------------------- pure helpers


def test_ibm_shape_is_five_chunks_of_558_tokens(utils):
    assert _navit_chunk_tokens(IBM_WIDTH, IBM_HEIGHT, K3_CFG) == IBM_CHUNK_TOKENS
    assert _video_vision_tokens(utils, IBM_WIDTH, IBM_HEIGHT, IBM_FRAMES, IBM_FPS, K3_CFG) == IBM_VISION_TOKENS == 2790


def test_sampling_keeps_every_frame_of_a_low_fps_video(utils):
    assert utils.sample_frame_indices(20, 20, 2.0, 8.0) == list(range(20))


def test_sampling_downsamples_to_the_sample_fps(utils):
    # 30 fps source, 3 s: 90 decoded frames -> 24 frames at 8 fps, evenly spread, first and last frame kept
    kept = utils.sample_frame_indices(90, 90, 30.0, 8.0)
    assert len(kept) == 24
    assert kept[0] == 0 and kept[-1] == 89
    assert kept == sorted(set(kept))


def test_sampling_never_exceeds_the_decoded_frames(utils):
    # the loader already dropped frames (fps cap): the plan cannot invent more than were decoded
    assert len(utils.sample_frame_indices(10, 300, 30.0, 8.0)) == 10


def test_single_frame_video(utils):
    assert utils.sample_frame_indices(1, 1, 24.0, 8.0) == [0]


@pytest.mark.parametrize("args", [(0, 1, 2.0, 8.0), (4, 4, 0.0, 8.0)])
def test_sampling_rejects_invalid_input(utils, args):
    with pytest.raises(ValueError):
        utils.sample_frame_indices(*args)


def test_chunking_by_four_with_a_short_tail(utils):
    chunks = utils.split_into_chunks(list(range(10)), 4)
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    with pytest.raises(ValueError):
        utils.split_into_chunks([0], 0)


@pytest.mark.parametrize("seconds,text", [(0, "00:00:00.000"), (2.0, "00:00:02.000"), (61.5, "00:01:01.500"),
                                          (3725.123, "01:02:05.123"), (-1, "00:00:00.000")])
def test_timestamp_format(utils, seconds, text):
    assert utils.format_timestamp(seconds) == text


def test_plan_uses_source_positions_for_start_times(utils):
    chunks, starts = utils.plan_video_chunks(20, 20, 2.0, list(range(20)))
    assert [len(c) for c in chunks] == [4] * 5
    assert starts == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_plan_without_source_positions_spreads_frames_over_the_duration(utils):
    _, starts = utils.plan_video_chunks(8, 8, 2.0, None)
    assert starts == [0.0, 2.0]


def test_prompt_has_one_timestamped_block_per_chunk(utils):
    prompt = utils.build_video_prompt([0.0, 2.0], [3, 2], "<P>")
    assert prompt == ("00:00:00.000<|media_begin|>video<|media_content|><P><P><P><|media_end|>"
                      "00:00:02.000<|media_begin|>video<|media_content|><P><P><|media_end|>")
    with pytest.raises(ValueError):
        utils.build_video_prompt([0.0], [1, 2], "<P>")


def test_prompt_pad_count_for_the_ibm_shape(utils):
    chunks, starts = utils.plan_video_chunks(IBM_FRAMES, IBM_FRAMES, IBM_FPS, None)
    prompt = utils.build_video_prompt(starts, [IBM_CHUNK_TOKENS] * len(chunks), "<|media_pad|>")
    assert prompt.count("<|media_pad|>") == IBM_VISION_TOKENS
    assert prompt.count("<|media_begin|>video<|media_content|>") == 5


# --------------------------------------------------------------- vllm stubs


class _Info:
    media_token = "<|media_pad|>"
    media_token_id = 7
    utils: Any = None

    def __init__(self):
        self.image_processor = types.SimpleNamespace(media_proc_cfg=K3_CFG, preprocess=self._preprocess)
        self.preprocess_calls = []
        self.device_normalize = True  # mm_device_do_normalize, on by default for Kimi-K3
        self.ctx = types.SimpleNamespace(model_config=types.SimpleNamespace(dtype=torch.bfloat16),
                                         tokenizer=None,
                                         get_merged_mm_kwargs=self._merge_mm_kwargs)

    def _get_expected_hidden_size(self):
        return None

    allow_missing_mm_embeddings = False

    def get_tokenizer(self):
        return "tokenizer"

    def media_tokens_calculator(self, media):
        frame = media["video_chunk"][0]
        return _navit_chunk_tokens(frame.size[0], frame.size[1], K3_CFG)

    def _merge_mm_kwargs(self, kwargs):
        # MultiModalConfig.merge_mm_processor_kwargs: device normalization forces both host flags off
        merged = dict(kwargs)
        if self.device_normalize:
            merged.update(do_rescale=False, do_normalize=False)
        return merged

    def _preprocess(self, medias, do_rescale, do_normalize, **kwargs):
        self.preprocess_calls.append(medias)
        self.preprocess_flags = (do_rescale, do_normalize)
        grids = []
        for media in medias:
            frame = media["video_chunk"][0]
            grids.append([len(media["video_chunk"]), 36, 62] if frame.size == (IBM_WIDTH, IBM_HEIGHT) else [1, 2, 2])
        grid = torch.tensor(grids)
        shape = (int(grid.prod(-1).sum()), 4)
        if not (do_rescale or do_normalize):
            # the fused processor with both host steps off returns the raw 8-bit patches
            return {"pixel_values": torch.full(shape, RAW_PIXEL, dtype=torch.uint8), "grid_thws": grid}
        # otherwise float32 patches: rescale to 0..1, normalize (mean = std = 0.5)
        pixels = torch.full(shape, float(RAW_PIXEL))
        if do_rescale:
            pixels = pixels / 255
        if do_normalize:
            pixels = (pixels - 0.5) / 0.5
        return {"pixel_values": pixels, "grid_thws": grid}


class _FieldConfig:

    def __init__(self, kind, modality, sizes=None, keep_on_cpu=False):
        self.kind, self.modality, self.sizes, self.keep_on_cpu = kind, modality, sizes, keep_on_cpu

    @classmethod
    def flat_from_sizes(cls, modality, sizes, dim=0, *, keep_on_cpu=False):
        return cls("flat", modality, sizes, keep_on_cpu)

    @classmethod
    def batched(cls, modality, *, keep_on_cpu=False):
        return cls("batched", modality, None, keep_on_cpu)


@pytest.fixture
def video(monkeypatch, utils):
    """The video module built against stub vllm classes. Returns a namespace with the module and its fakes."""
    ns = types.SimpleNamespace(utils=utils, registrations=[])
    _Info.utils = utils

    class BaseDummy:

        def get_dummy_text(self, mm_counts):
            return "<|kimi_image_placeholder|>" * mm_counts.get("image", 0)

        def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
            return {"image": ["img"] * mm_counts.get("image", 0)}

    class BaseProcessor:

        def __init__(self):
            self.info = _Info()

        def _call_hf_processor(self, hf_data, hf_kwargs):
            return {"pixel_values": torch.zeros(1, 4), "grid_thws": torch.tensor([[1, 2, 2]])}

        def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
            return {"pixel_values": _FieldConfig("flat", "image")}

        def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
            return ["image-update"]

    class K3Model:
        marked: list = []

        @classmethod
        def get_placeholder_str(cls, modality, i):
            if modality == "image":
                return "<|kimi_image_placeholder|>"
            raise ValueError(modality)

        def _mark_tower_model(self, vllm_config, modalities, *args, **kwargs):
            K3Model.marked.append(modalities)

        def embed_multimodal(self, **kwargs):
            return "image-path"

    class OtherModel(K3Model):  # stands for Kimi-K2.5/K2.6
        pass

    for attr in ("get_placeholder_str", "_mark_tower_model", "embed_multimodal"):
        setattr(OtherModel, attr, K3Model.__dict__[attr])

    class Registry:

        def register_processor(self, processor, *, info, dummy_inputs):

            def deco(cls):
                ns.registrations.append((cls, processor, info, dummy_inputs))
                return cls

            return deco

    def stub(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        mod.__path__ = []
        parts = name.split(".")
        for depth in range(1, len(parts)):  # parent packages, so that `import a.b.c` resolves
            parent = ".".join(parts[:depth])
            if parent not in sys.modules or not hasattr(sys.modules[parent], "__path__"):
                pkg = types.ModuleType(parent)
                pkg.__path__ = []
                monkeypatch.setitem(sys.modules, parent, pkg)
        monkeypatch.setitem(sys.modules, name, mod)
        if len(parts) > 1:
            setattr(sys.modules[".".join(parts[:-1])], parts[-1], mod)
        return mod

    class BatchFeature(dict):

        def __init__(self, data=None, **kwargs):
            super().__init__(data or {})

    class MultiModalDataParser:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class VideoProcessorItems:
        pass

    class PromptUpdateDetails:

        def __init__(self, full, token_id):
            self.full, self.token_id = full, token_id

        @classmethod
        def select_token_id(cls, full, token_id):
            return cls(full, token_id)

    class PromptReplacement:

        def __init__(self, modality, target, replacement):
            self.modality, self.target, self.replacement = modality, target, replacement

    renderer = stub("vllm.renderers.kimi_k3", _K3_MEDIA_IO_DEFAULTS={"image": {"image_mode": None}})
    amd_model = stub(
        "vllm.models.kimi_k3.amd.model",
        KimiK3ForConditionalGeneration=K3Model,
        vision_tower_forward=_stub_vision_tower_forward,
    )
    stub("transformers", BatchFeature=BatchFeature)
    stub("vllm.multimodal", MULTIMODAL_REGISTRY=Registry())
    stub("vllm.multimodal.inputs", MultiModalFieldConfig=_FieldConfig)
    stub("vllm.multimodal.parse", MultiModalDataParser=MultiModalDataParser, VideoProcessorItems=VideoProcessorItems)
    stub("vllm.multimodal.processing",
         PromptReplacement=PromptReplacement,
         PromptUpdateDetails=PromptUpdateDetails,
         cached_encode=lambda tok, text, add_special_tokens=False: ("enc", text))
    stub("vllm.models.kimi_k3.common.mm_preprocess",
         KimiK3DummyInputsBuilder=BaseDummy,
         KimiK3MultiModalProcessor=BaseProcessor,
         KimiK3ProcessingInfo=_Info)

    monkeypatch.delitem(sys.modules, "vllm_gaudi.models.kimi_k3_video", raising=False)
    ns.mod = _load("vllm_gaudi.models.kimi_k3_video", ROOT / "models" / "kimi_k3_video.py")
    monkeypatch.setitem(sys.modules, "vllm_gaudi.models.kimi_k3_video", ns.mod)
    ns.K3Model, ns.OtherModel, ns.renderer, ns.amd_model = K3Model, OtherModel, renderer, amd_model
    ns.PromptReplacement = PromptReplacement
    K3Model.marked.clear()
    return ns


def _stub_vision_tower_forward(tower, pixels, grids, *, use_data_parallel=False, **kw):
    """Row counts of the real vision_tower_forward for a [t, h, w] grid with a 2x2 merge, without the tower weights.

    The plain path pools a chunk over time (tpool_patch_merger): h*w//4 rows. The data-parallel path
    (run_dp_sharded_mrope_vision_model) sizes its outputs as t*h*w//4 and ignores the pooling.
    """
    merge_area = tower.merge_kernel_size[0] * tower.merge_kernel_size[1]
    return [
        torch.full((int((g[0] if use_data_parallel else 1) * g[1] * g[2] // merge_area), 3), float(i))
        for i, g in enumerate(grids)
    ]


class _Tower:
    merge_kernel_size = (2, 2)


def _ibm_video(utils):
    import numpy as np
    frames = np.zeros((IBM_FRAMES, IBM_HEIGHT, IBM_WIDTH, 3), dtype=np.uint8)
    metadata = {"total_num_frames": IBM_FRAMES, "fps": IBM_FPS, "frames_indices": list(range(IBM_FRAMES))}
    return frames, metadata


# -------------------------------------------------------------------- gating


def test_install_touches_only_the_k3_model(video):
    video.mod.install_kimi_k3_video()

    assert [r[0] for r in video.registrations] == [video.K3Model]
    assert video.K3Model.get_placeholder_str("video", 0) == "<|kimi_video_placeholder|>"
    assert video.K3Model.get_placeholder_str("image", 0) == "<|kimi_image_placeholder|>"
    assert video.renderer._K3_MEDIA_IO_DEFAULTS["video"] == {"num_frames": 256, "fps": 8.0}
    assert video.renderer._K3_MEDIA_IO_DEFAULTS["image"] == {"image_mode": None}


def test_other_models_keep_their_behaviour(video):
    """Negative check: a Kimi-K2.5/K2.6-like class (and any other model) gets no video support from the K3 patch."""
    video.mod.install_kimi_k3_video()

    other = video.OtherModel
    with pytest.raises(ValueError):
        other.get_placeholder_str("video", 0)
    assert other.get_placeholder_str("image", 0) == "<|kimi_image_placeholder|>"
    assert other.embed_multimodal(other(), **{video.mod.VIDEO_PIXEL_VALUES: torch.zeros(1)}) == "image-path"
    other()._mark_tower_model(None, "image")
    assert video.K3Model.marked[-1] == "image"
    assert "_hpu_kimi_k3_video" not in vars(other)


def test_install_is_idempotent(video):
    video.mod.install_kimi_k3_video()
    video.mod.install_kimi_k3_video()
    assert len(video.registrations) == 1


def test_tower_is_marked_for_both_modalities(video):
    video.mod.install_kimi_k3_video()
    video.K3Model()._mark_tower_model(None, "image")
    assert video.K3Model.marked == [{"image", "video"}]


def test_image_requests_take_the_unchanged_path(video):
    video.mod.install_kimi_k3_video()
    assert video.K3Model().embed_multimodal(pixel_values=torch.zeros(1), grid_thws=torch.zeros(1, 3)) == "image-path"


# --------------------------------------------------------- processor and model


def _processor(video):
    video.mod.install_kimi_k3_video()
    return video.registrations[0][1]()


def test_dummy_video_defaults_to_the_ibm_shape(video):
    video.mod.install_kimi_k3_video()
    builder = video.registrations[0][3]()
    assert builder.get_dummy_text({"video": 2}) == "<|kimi_video_placeholder|>" * 2
    data = builder.get_dummy_mm_data(1024, {"video": 2, "image": 0})
    frames, metadata = data["video"][0]
    assert frames.shape == (20, 480, 864, 3) and len(data["video"]) == 2
    assert metadata["fps"] == 2.0 and metadata["total_num_frames"] == 20
    override = types.SimpleNamespace(width=64, height=32, num_frames=8)
    frames, _ = builder.get_dummy_mm_data(1024, {"video": 1}, {"video": override})["video"][0]
    assert frames.shape == (8, 32, 64, 3)


def test_data_parser_asks_for_video_metadata(video):
    video.mod.install_kimi_k3_video()
    assert video.registrations[0][2]().get_data_parser().kwargs["video_needs_metadata"] is True
    assert video.registrations[0][2]().get_supported_mm_limits() == {"image": None, "video": None}


def test_processor_builds_one_grid_per_chunk(video):
    proc = _processor(video)
    out = proc._call_hf_processor({"videos": [_ibm_video(video.utils)]}, {})

    assert out[video.mod.VIDEO_GRID_THWS].tolist() == [[4, 36, 62]] * 5
    assert out[video.mod.VIDEO_NUM_CHUNKS].tolist() == [5]
    assert "pixel_values" not in out  # images are not touched
    (medias, ) = proc.info.preprocess_calls
    assert [len(m["video_chunk"]) for m in medias] == [4] * 5
    assert all(m["type"] == "video_chunk" for m in medias)


def test_images_and_videos_in_one_request(video):
    proc = _processor(video)
    out = proc._call_hf_processor({"images": ["img"], "videos": [_ibm_video(video.utils)]}, {})
    assert "pixel_values" in out and video.mod.VIDEO_PIXEL_VALUES in out


def test_fields_slice_the_flat_tensors_per_video(video):
    proc = _processor(video)
    hf_inputs = proc._call_hf_processor({"videos": [_ibm_video(video.utils)] * 2}, {})
    fields = proc._get_mm_fields_config(hf_inputs, {})

    chunk_patches = 4 * 36 * 62
    assert fields[video.mod.VIDEO_PIXEL_VALUES].sizes.tolist() == [5 * chunk_patches] * 2
    assert fields[video.mod.VIDEO_GRID_THWS].sizes.tolist() == [5, 5]
    assert fields[video.mod.VIDEO_GRID_THWS].keep_on_cpu and fields[video.mod.VIDEO_NUM_CHUNKS].keep_on_cpu
    assert all(f.modality == "video" for k, f in fields.items() if k != "pixel_values")


def test_fields_without_video_are_the_image_fields(video):
    proc = _processor(video)
    assert list(proc._get_mm_fields_config({"pixel_values": 1}, {})) == ["pixel_values"]


def test_prompt_replacement_has_2790_media_pads(video):
    proc = _processor(video)
    item = _ibm_video(video.utils)
    items = types.SimpleNamespace(get=lambda i: item)

    class Items(dict):

        def get_items(self, modality, typ):
            return items

    updates = proc._get_prompt_updates(Items(video=items), {}, None)
    assert updates[0] == "image-update"
    replacement = updates[1]
    assert replacement.modality == "video"
    assert replacement.target == ("enc", "<|kimi_video_placeholder|>")
    details = replacement.replacement(0)
    assert details.token_id == 7
    _, text = details.full
    assert text.count("<|media_pad|>") == IBM_VISION_TOKENS
    assert text.startswith("00:00:00.000<|media_begin|>video<|media_content|>")
    assert text.count("<|media_end|>") == 5 and "00:00:08.000<|media_begin|>" in text


def _prompt_text(proc, item):
    items = types.SimpleNamespace(get=lambda i: item)

    class Items(dict):

        def get_items(self, modality, typ):
            return items

    return proc._get_prompt_updates(Items(video=items), {}, None)[1].replacement(0).full[1]


def test_last_incomplete_chunk_through_processor_prompt_and_embed(video):
    """18 frames at 2 fps: four chunks of 4 frames and a tail chunk of 2 (a frame is never dropped or padded)."""
    import numpy as np
    frames = np.zeros((18, IBM_HEIGHT, IBM_WIDTH, 3), dtype=np.uint8)
    item = (frames, {"total_num_frames": 18, "fps": IBM_FPS, "frames_indices": list(range(18))})
    proc = _processor(video)

    out = proc._call_hf_processor({"videos": [item]}, {})
    assert out[video.mod.VIDEO_GRID_THWS].tolist() == [[4, 36, 62]] * 4 + [[2, 36, 62]]
    assert out[video.mod.VIDEO_NUM_CHUNKS].tolist() == [5]
    assert [len(m["video_chunk"]) for m in proc.info.preprocess_calls[0]] == [4, 4, 4, 4, 2]

    text = _prompt_text(proc, item)  # the tail chunk gets its own timestamped block with the usual token count
    assert text.count("<|media_begin|>video<|media_content|>") == 5
    assert text.count("<|media_pad|>") == 5 * 558
    assert text.startswith("00:00:00.000<") and "00:00:08.000<|media_begin|>" in text

    model = video.K3Model()
    model.vision_tower, model.mm_projector, model.use_data_parallel = _Tower(), object(), False
    (embeds, ) = model.embed_multimodal(
        **{
            video.mod.VIDEO_PIXEL_VALUES: out[video.mod.VIDEO_PIXEL_VALUES],
            video.mod.VIDEO_GRID_THWS: out[video.mod.VIDEO_GRID_THWS],
            video.mod.VIDEO_NUM_CHUNKS: out[video.mod.VIDEO_NUM_CHUNKS],
        })
    assert embeds.shape[0] == 5 * 558


def test_prompt_replacement_accepts_a_list_of_frames(video):
    frames = _ibm_video(video.utils)[0]
    item = (list(frames), {"total_num_frames": IBM_FRAMES, "fps": IBM_FPS})
    assert _prompt_text(_processor(video), item).count("<|media_pad|>") == IBM_VISION_TOKENS


def test_missing_fps_is_logged_as_a_warning(video, caplog):
    frames = _ibm_video(video.utils)[0]
    with caplog.at_level(logging.WARNING, logger="vllm"):
        chunks, _ = video.mod.video_chunk_plan(frames, {}, K3_CFG)
    assert len(chunks) == 5
    assert any("no fps" in r.message and r.name.startswith("vllm.") for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="vllm"):
        video.mod.video_chunk_plan(frames, {"fps": IBM_FPS}, K3_CFG)
    assert not caplog.records


def test_embed_returns_one_tensor_per_video(video):
    video.mod.install_kimi_k3_video()
    grid = torch.tensor([[4, 36, 62]] * 5 + [[4, 36, 62]] * 3)
    model = video.K3Model()
    model.vision_tower, model.mm_projector, model.use_data_parallel = _Tower(), object(), False
    out = model.embed_multimodal(
        **{
            video.mod.VIDEO_PIXEL_VALUES: torch.zeros(1),
            video.mod.VIDEO_GRID_THWS: grid,
            video.mod.VIDEO_NUM_CHUNKS: torch.tensor([5, 3]),
        })
    assert [t.shape[0] for t in out] == [5 * 558, 3 * 558]


def _video_model(video, use_data_parallel):
    model = video.K3Model()
    model.vision_tower, model.mm_projector, model.use_data_parallel = _Tower(), object(), use_data_parallel
    return model


@pytest.mark.parametrize("use_data_parallel", [False, True])
def test_embed_rows_equal_the_placeholder_count_for_a_20_frame_864x480_video(video, use_data_parallel):
    # Through the processor: the grids come from the processor, the placeholder count from the prompt replacement.
    # The model may have use_data_parallel=True (K3 with TP16); the video path must not use the DP-sharded tower
    # helper, whose outputs ignore the temporal pooling.
    proc = _processor(video)
    item = _ibm_video(video.utils)
    out = proc._call_hf_processor({"videos": [item]}, {})
    num_pads = _prompt_text(proc, item).count("<|media_pad|>")
    assert num_pads == IBM_VISION_TOKENS == 2790

    video.mod.install_kimi_k3_video()
    (embeds, ) = _video_model(video, use_data_parallel).embed_multimodal(
        **{
            video.mod.VIDEO_PIXEL_VALUES: out[video.mod.VIDEO_PIXEL_VALUES],
            video.mod.VIDEO_GRID_THWS: out[video.mod.VIDEO_GRID_THWS],
            video.mod.VIDEO_NUM_CHUNKS: out[video.mod.VIDEO_NUM_CHUNKS],
        })
    assert embeds.shape[0] == num_pads
    # Chunk order is kept: the rows of chunk i carry the marker value i of the stub tower.
    assert embeds[:, 0].unique_consecutive().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_embed_requests_the_plain_tower_even_with_data_parallel(video):
    seen = []
    orig = video.amd_model.vision_tower_forward

    def tower(tower_, pixels, grids, **kw):
        seen.append(kw["use_data_parallel"])
        return orig(tower_, pixels, grids, **kw)

    video.amd_model.vision_tower_forward = tower
    video.mod.install_kimi_k3_video()
    _video_model(video, True).embed_multimodal(
        **{
            video.mod.VIDEO_PIXEL_VALUES: torch.zeros(1),
            video.mod.VIDEO_GRID_THWS: torch.tensor([[4, 36, 62]]),
            video.mod.VIDEO_NUM_CHUNKS: torch.tensor([1]),
        })
    assert seen == [False]


def test_embed_rejects_a_chunk_with_the_wrong_row_count(video):
    video.mod.install_kimi_k3_video()
    video.amd_model.vision_tower_forward = lambda tower, pixels, grids, **kw: [torch.zeros(4 * 558, 3)]
    with pytest.raises(ValueError, match="chunk 0 has 2232 embedding rows, expected 558"):
        _video_model(video, False).embed_multimodal(
            **{
                video.mod.VIDEO_PIXEL_VALUES: torch.zeros(1),
                video.mod.VIDEO_GRID_THWS: torch.tensor([[4, 36, 62]]),
                video.mod.VIDEO_NUM_CHUNKS: torch.tensor([1]),
            })


def test_embed_rejects_a_wrong_chunk_count(video):
    video.mod.install_kimi_k3_video()
    video.amd_model.vision_tower_forward = lambda tower, pixels, grids, **kw: [torch.zeros(558, 3)]
    with pytest.raises(ValueError, match="1 chunk embeddings for 2 chunks"):
        _video_model(video, False).embed_multimodal(
            **{
                video.mod.VIDEO_PIXEL_VALUES: torch.zeros(1),
                video.mod.VIDEO_GRID_THWS: torch.tensor([[4, 36, 62]] * 2),
                video.mod.VIDEO_NUM_CHUNKS: torch.tensor([2]),
            })


def test_mixed_group_is_rejected(video):
    video.mod.install_kimi_k3_video()
    with pytest.raises(ValueError):
        video.K3Model().embed_multimodal(pixel_values=torch.zeros(1), **{video.mod.VIDEO_PIXEL_VALUES: torch.zeros(1)})


# ------------------------------------------------- multimodal warmup of the runner


@pytest.fixture
def warmup(video, monkeypatch):
    """A processor wired like the vLLM one (builder + parse_mm_data)."""

    class ProcessorInputs:

        def __init__(self, prompt, mm_data_items):
            self.prompt, self.mm_data_items = prompt, mm_data_items

    video.mod.install_kimi_k3_video()
    _, processor_cls, info_cls, builder_cls = video.registrations[0]

    class Tokenizer:

        def encode(self, text, add_special_tokens=False):
            return text.split("|>")[:-1]

    class Processor(processor_cls):  # type: ignore[valid-type, misc]
        # What BaseMultiModalProcessor.get_dummy_inputs does at the pinned vLLM.

        def __init__(self):
            super().__init__()
            self.dummy_inputs = builder_cls()
            self.info.parse_mm_data = lambda data, validate=False: data
            self.info.ctx.tokenizer = Tokenizer()
            self.info.default_tok_params = types.SimpleNamespace(get_encode_kwargs=lambda: {})
            self.info.get_hf_config = lambda: types.SimpleNamespace(image_placeholder="<|kimi_image_placeholder|>")
            self.calls = []

        def get_dummy_inputs(self, seq_len, mm_counts, mm_options):
            self.calls.append((seq_len, dict(mm_counts), mm_options))
            builder = self.dummy_inputs
            data = builder.get_dummy_mm_data(seq_len, mm_counts, mm_options)
            prompt = self.info.ctx.tokenizer.encode(builder.get_dummy_text(mm_counts))
            return ProcessorInputs(prompt, self.info.parse_mm_data(data))

    return types.SimpleNamespace(Processor=Processor, ProcessorInputs=ProcessorInputs)


def test_builder_has_no_get_dummy_processor_inputs(video):
    # The vLLM builders no longer have it, so the runner must not call it on the builder.
    video.mod.install_kimi_k3_video()
    assert not hasattr(video.registrations[0][3](), "get_dummy_processor_inputs")


def test_video_warmup_goes_through_the_processor(warmup):
    proc = warmup.Processor()
    options = {"video": types.SimpleNamespace(count=2, width=864, height=480, num_frames=20)}
    inputs = proc.get_dummy_inputs(seq_len=4096, mm_counts={"video": 2}, mm_options=options)
    assert proc.calls == [(4096, {"video": 2}, options)]
    assert inputs.prompt == ["<|kimi_video_placeholder"] * 2
    frames, metadata = inputs.mm_data_items["video"][0]
    assert len(inputs.mm_data_items["video"]) == 2
    assert frames.shape == (20, 480, 864, 3) and metadata["fps"] == 2.0


def test_video_warmup_follows_the_requested_size(warmup):
    proc = warmup.Processor()
    options = {"video": types.SimpleNamespace(count=1, width=64, height=32, num_frames=8)}
    inputs = proc.get_dummy_inputs(4096, {"video": 1}, options)
    assert inputs.mm_data_items["video"][0][0].shape == (8, 32, 64, 3)


def test_video_dummy_goes_through_processor_apply(warmup, video, utils):
    """get_dummy_inputs output feeds the K3 video processor: IBM shape -> 5 grids of 558 tokens."""
    proc = warmup.Processor()
    options = {"video": types.SimpleNamespace(count=1, width=IBM_WIDTH, height=IBM_HEIGHT, num_frames=IBM_FRAMES)}
    inputs = proc.get_dummy_inputs(4096, {"video": 1}, options)
    frames, metadata = inputs.mm_data_items["video"][0]
    out = proc._call_hf_processor({"videos": [(frames, metadata)]}, {})
    assert out[video.mod.VIDEO_GRID_THWS].tolist() == [[4, 36, 62]] * 5


# ------------------------------------------------------- dtype of the video pixels


def test_video_pixels_are_cast_to_the_model_dtype(video):
    # With device normalization off the processor returns normalized float32; the tower runs in the model dtype (bf16).
    proc = _processor(video)
    proc.info.device_normalize = False
    out = proc._call_hf_processor({"videos": [_ibm_video(video.utils)]}, {})
    assert out[video.mod.VIDEO_PIXEL_VALUES].dtype == torch.bfloat16
    assert out[video.mod.VIDEO_GRID_THWS].dtype == torch.long  # integer tensors are left alone
    assert out[video.mod.VIDEO_NUM_CHUNKS].dtype == torch.long


def test_video_pixels_reach_the_tower_in_the_model_dtype(video):
    seen = []
    orig = video.amd_model.vision_tower_forward

    def tower(tower_, pixels, grids, **kw):
        seen.append(pixels.dtype)
        return orig(tower_, pixels, grids, **kw)

    video.amd_model.vision_tower_forward = tower
    proc = _processor(video)
    proc.info.device_normalize = False
    out = proc._call_hf_processor({"videos": [_ibm_video(video.utils)]}, {})
    model = video.K3Model()
    model.vision_tower, model.mm_projector, model.use_data_parallel = _Tower(), object(), False
    embeds = model.embed_multimodal(
        **{
            video.mod.VIDEO_PIXEL_VALUES: out[video.mod.VIDEO_PIXEL_VALUES],
            video.mod.VIDEO_GRID_THWS: out[video.mod.VIDEO_GRID_THWS],
            video.mod.VIDEO_NUM_CHUNKS: out[video.mod.VIDEO_NUM_CHUNKS],
        })
    assert seen == [torch.bfloat16] and len(embeds) == 1


# ------------------------------------------- host rescale / normalize follow mm_device_do_normalize


def _video_pixels(video, device_normalize, hf_kwargs=None):
    proc = _processor(video)
    proc.info.device_normalize = device_normalize
    out = proc._call_hf_processor({"videos": [_ibm_video(video.utils)]}, hf_kwargs or {})
    return proc, out[video.mod.VIDEO_PIXEL_VALUES]


def test_device_normalize_on_sends_raw_pixels_to_the_tower(video):
    # The vision tower's input norm rescales and normalizes; the host must not do it first (double normalization).
    proc, pixels = _video_pixels(video, device_normalize=True)
    assert proc.info.preprocess_flags == (False, False)
    assert pixels.dtype == torch.uint8  # integer patches skip the float cast
    assert pixels.unique().tolist() == [RAW_PIXEL]


def test_device_normalize_off_normalizes_on_the_host(video):
    proc, pixels = _video_pixels(video, device_normalize=False)
    assert proc.info.preprocess_flags == (True, True)
    assert pixels.dtype == torch.bfloat16  # normalized float32 patches are cast to the model dtype
    assert pixels.float().unique().tolist() == [1.0]  # 255 -> 1.0 -> (1.0 - 0.5) / 0.5


def test_device_normalize_ignores_per_request_flags(video):
    proc, _ = _video_pixels(video, device_normalize=True, hf_kwargs={"do_normalize": True, "do_rescale": True})
    assert proc.info.preprocess_flags == (False, False)


def test_device_normalize_off_passes_per_request_flags_to_the_processor(video):
    # hf_kwargs must reach get_merged_mm_kwargs: dropping them would silently restore the defaults (True, True)
    proc, _ = _video_pixels(video, device_normalize=False, hf_kwargs={"do_normalize": False, "do_rescale": True})
    assert proc.info.preprocess_flags == (True, False)


# ------------------------------------------------------- data-parallel vision tower (HPU_KIMI_VIT_DP)


@pytest.mark.parametrize("num_chunks, world_size, expected", [
    (16, 16, [(i, i + 1) for i in range(16)]),
    (5, 16, [(i, i + 1) for i in range(5)] + [(5, 5)] * 11),
    (7, 3, [(0, 3), (3, 5), (5, 7)]),
    (64, 16, [(4 * i, 4 * i + 4) for i in range(16)]),
    (0, 2, [(0, 0), (0, 0)]),
    (3, 1, [(0, 3)]),
])
def test_split_chunks_across_ranks_is_contiguous_and_balanced(utils, num_chunks, world_size, expected):
    ranges = utils.split_chunks_across_ranks(num_chunks, world_size)
    assert ranges == expected
    assert [i for start, end in ranges for i in range(start, end)] == list(range(num_chunks))


def test_split_chunks_across_ranks_rejects_bad_sizes(utils):
    with pytest.raises(ValueError):
        utils.split_chunks_across_ranks(3, 0)
    with pytest.raises(ValueError):
        utils.split_chunks_across_ranks(-1, 2)


def test_rank_row_counts_and_patch_offsets(utils):
    assert utils.rank_row_counts([6, 2, 4], [(0, 2), (2, 3), (3, 3)]) == [8, 4, 0]
    assert utils.chunk_patch_offsets([[4, 4, 6], [4, 2, 4], [2, 4, 4]]) == [0, 96, 128, 160]


DP_DIM = 12  # 4 patches x 3 channels after the stub 2x2 merge


def _content_tower_forward(tower, pixels, grids, *, use_data_parallel=False, **kw):
    """Content-dependent stand-in for vision_tower_forward: temporal mean pool, then 2x2 merge of [patches, 3] rows."""
    assert not use_data_parallel
    out, start = [], 0
    for t, h, w in grids.tolist():
        seq = pixels[start:start + t * h * w]
        start += t * h * w
        out.append(seq.view(t, h * w, 3).mean(0).reshape(h * w // 4, DP_DIM))
    assert start == pixels.shape[0], "the tower got a pixel slice that does not match its grids"
    return out


class _Linear:
    output_size = DP_DIM


class _Projector:
    linear_2 = _Linear()

    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.float32)])


def _dp_model(video, use_data_parallel=True):
    model = video.K3Model()
    model.vision_tower, model.mm_projector, model.use_data_parallel = _Tower(), _Projector(), use_data_parallel
    return model


def _dp_distributed(monkeypatch, world_size):
    """Stub vllm.distributed for one simulated rank at a time; all_gather records or replays the local tensors."""
    dist = types.SimpleNamespace(rank=0, world_size=world_size, sent={}, replay=None)
    mod: Any = types.ModuleType("vllm.distributed")
    mod.get_tensor_model_parallel_rank = lambda: dist.rank
    mod.get_tensor_model_parallel_world_size = lambda: dist.world_size

    def all_gather(tensor, dim=-1):
        assert dim == 0
        dist.sent[dist.rank] = tensor.clone()
        if dist.replay is None:
            return torch.zeros(dist.world_size * tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype)
        return torch.cat([dist.replay[r] for r in range(dist.world_size)], dim=0)

    mod.tensor_model_parallel_all_gather = all_gather
    if "vllm" not in sys.modules:
        monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.distributed", mod)
    return dist


DP_GRIDS = [[4, 4, 6], [4, 2, 4], [2, 4, 4], [4, 4, 6], [1, 2, 2]]  # uneven t and sizes, rows 6, 2, 4, 6, 1
DP_NUM_CHUNKS = [3, 2]


def _dp_inputs(video):
    grids = torch.tensor(DP_GRIDS)
    num_patches = sum(t * h * w for t, h, w in DP_GRIDS)
    torch.manual_seed(0)
    return {
        video.mod.VIDEO_PIXEL_VALUES: torch.randn(num_patches, 3),
        video.mod.VIDEO_GRID_THWS: grids,
        video.mod.VIDEO_NUM_CHUNKS: torch.tensor(DP_NUM_CHUNKS),
    }


@pytest.mark.parametrize("world_size", [2, 3, 4, 16])
def test_dp_embed_matches_the_single_rank_path(video, monkeypatch, world_size):
    video.amd_model.vision_tower_forward = _content_tower_forward
    inputs = _dp_inputs(video)
    monkeypatch.setenv(video.mod.ENV_VIT_DP, "1")  # read at install time
    video.mod.install_kimi_k3_video()
    dist = _dp_distributed(monkeypatch, world_size)
    reference = _dp_model(video, use_data_parallel=False).embed_multimodal(**inputs)  # plain tower, all chunks
    assert dist.sent == {}
    for rank in range(world_size):  # pass 1: every rank encodes its share
        dist.rank = rank
        _dp_model(video).embed_multimodal(**inputs)
    dist.replay = dict(dist.sent)
    outputs = []
    for rank in range(world_size):  # pass 2: the all-gather returns the real per-rank tensors
        dist.rank = rank
        outputs.append(_dp_model(video).embed_multimodal(**inputs))

    for out in outputs:
        assert [t.shape for t in out] == [t.shape for t in reference]
        for got, want in zip(out, reference):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    # Padded to the largest share; idle ranks send zeros.
    max_rows = max(t.shape[0] for t in dist.sent.values())
    assert all(t.shape == (max_rows, DP_DIM) for t in dist.sent.values())
    if world_size > len(DP_GRIDS):
        assert all(not dist.sent[r].any() for r in range(len(DP_GRIDS), world_size))


def test_dp_idle_ranks_run_the_tower_on_the_first_chunk(video, monkeypatch):
    calls = []

    def tower(_tower, pixels, grids, **kw):
        calls.append(grids.tolist())
        return _content_tower_forward(_tower, pixels, grids, **kw)

    video.amd_model.vision_tower_forward = tower
    monkeypatch.setenv(video.mod.ENV_VIT_DP, "1")
    video.mod.install_kimi_k3_video()
    dist = _dp_distributed(monkeypatch, 16)
    for rank in range(16):
        calls.clear()
        dist.rank = rank
        _dp_model(video).embed_multimodal(**_dp_inputs(video))
        assert len(calls) == 1
        assert calls[0] == ([DP_GRIDS[rank]] if rank < len(DP_GRIDS) else [DP_GRIDS[0]])
    assert not dist.sent[len(DP_GRIDS)].any()


@pytest.mark.parametrize("num_chunks", [17, 32])
def test_dp_tower_gets_one_chunk_per_call_beyond_world_size(video, monkeypatch, num_chunks):
    world_size = 16
    grids = [DP_GRIDS[i % len(DP_GRIDS)] for i in range(num_chunks)]
    torch.manual_seed(0)
    inputs = {
        video.mod.VIDEO_PIXEL_VALUES: torch.randn(sum(t * h * w for t, h, w in grids), 3),
        video.mod.VIDEO_GRID_THWS: torch.tensor(grids),
        video.mod.VIDEO_NUM_CHUNKS: torch.tensor([num_chunks]),
    }
    seen = []

    def tower(_tower, pixels, g, **kw):
        seen.append(g.shape[0])
        return _content_tower_forward(_tower, pixels, g, **kw)

    video.amd_model.vision_tower_forward = _content_tower_forward
    monkeypatch.setenv(video.mod.ENV_VIT_DP, "1")
    video.mod.install_kimi_k3_video()
    dist = _dp_distributed(monkeypatch, world_size)
    reference = _dp_model(video, use_data_parallel=False).embed_multimodal(**inputs)
    video.amd_model.vision_tower_forward = tower
    for rank in range(world_size):
        dist.rank = rank
        _dp_model(video).embed_multimodal(**inputs)
    assert set(seen) == {1}
    dist.replay = dict(dist.sent)
    dist.rank = 0
    (out, ) = _dp_model(video).embed_multimodal(**inputs)
    torch.testing.assert_close(out, reference[0], rtol=0, atol=0)


def test_dp_embed_rejects_a_wrong_local_row_count(video, monkeypatch):
    monkeypatch.setenv(video.mod.ENV_VIT_DP, "1")
    video.mod.install_kimi_k3_video()
    video.amd_model.vision_tower_forward = lambda tower, pixels, grids, **kw: [torch.zeros(4 * 6, DP_DIM)]
    _dp_distributed(monkeypatch, 2)
    with pytest.raises(ValueError, match=r"rank 0 encoded chunks \[0, 3\) into \(72, 12\), expected \(12, 12\)"):
        _dp_model(video).embed_multimodal(**_dp_inputs(video))


@pytest.mark.parametrize("use_data_parallel, world_size", [(False, 4), (True, 1)])
def test_dp_env_falls_back_to_the_plain_tower(video, monkeypatch, caplog, use_data_parallel, world_size):
    monkeypatch.setenv(video.mod.ENV_VIT_DP, "1")
    video.mod.install_kimi_k3_video()
    video.amd_model.vision_tower_forward = _content_tower_forward
    dist = _dp_distributed(monkeypatch, world_size)
    with caplog.at_level(logging.WARNING, logger="vllm.gaudi.kimi_k3_video"):
        out = _dp_model(video, use_data_parallel).embed_multimodal(**_dp_inputs(video))
    assert [t.shape[0] for t in out] == [12, 7]
    assert dist.sent == {}
    assert "HPU_KIMI_VIT_DP=1 ignored" in caplog.text


def test_dp_env_unset_never_touches_the_tp_group(video, monkeypatch):
    monkeypatch.delenv(video.mod.ENV_VIT_DP, raising=False)
    video.mod.install_kimi_k3_video()
    video.amd_model.vision_tower_forward = _content_tower_forward
    dist = _dp_distributed(monkeypatch, 4)
    out = _dp_model(video).embed_multimodal(**_dp_inputs(video))
    assert [t.shape[0] for t in out] == [12, 7]
    assert dist.sent == {}
