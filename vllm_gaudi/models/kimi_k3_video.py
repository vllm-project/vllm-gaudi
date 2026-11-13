# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Video input for Kimi-K3 on HPU.

The vLLM Kimi-K3 model is image-only. This module adds the ``video`` modality to it, for Kimi-K3 only: nothing is
registered on Kimi-K2.5/K2.6 or any other class and the shared helpers of ``vllm.models.kimi_k3`` are not modified.

How a ``video_url`` request flows through the patched model:

1. The chat renderer decodes the video with the vLLM video loader (frames + metadata) and inserts
   ``<|kimi_video_placeholder|>`` into the prompt (``get_placeholder_str``). ``_K3_MEDIA_IO_DEFAULTS`` gets the loader
   defaults (fps / frame cap) of ``kimi_k3_video_utils``.
2. ``KimiK3VideoProcessor`` samples the frames at ``min(sample_fps, source fps)``, cuts them into chunks of
   ``temporal_merge_kernel_size`` (4) frames and runs the checkpoint image processor on ``video_chunk`` media, which
   gives one ``[t, h, w]`` grid per chunk. The placeholder is replaced by one timestamped media block per chunk.
3. ``embed_multimodal`` runs the shared vision tower on the chunks (``tpool_patch_merger`` mean-pools each chunk over
   time) and concatenates the chunk embeddings of a video into ONE tensor per video, the contract of the HPU runner.

Video tensors use their own keyword names (``video_pixel_values`` / ``video_grid_thws`` / ``video_num_chunks``) so the
model can tell a video group from an image group. The vision tower is not eager on HPU: in torch.compile mode regional
compilation compiles every ``nn.ModuleList`` of the model, including ``vision_tower.encoder.blocks``, and in lazy mode
the tower runs in an HPU graph. A video chunk (T=4, 36x62 patches for 864x480) is a new input shape, so it creates new
graphs. The multimodal warmup precompiles them from the video shape of ``--limit-mm-per-prompt``, e.g.
``{"video": {"count": 1, "width": 864, "height": 480, "num_frames": 20}}``; a video shape that was not warmed up
compiles on the first request.

The K3 classes cannot be imported while the plugin loads, so everything that subclasses them is built by
``install_kimi_k3_video`` when the K3 patches are applied.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from vllm_gaudi.models.kimi_k3_video_utils import (
    DEFAULT_MAX_SAMPLE_FPS,
    DEFAULT_TEMPORAL_CHUNK_SIZE,
    DUMMY_VIDEO_FPS,
    DUMMY_VIDEO_HEIGHT,
    DUMMY_VIDEO_NUM_FRAMES,
    DUMMY_VIDEO_WIDTH,
    K3_VIDEO_LOADER_FPS,
    K3_VIDEO_LOADER_MAX_FRAMES,
    K3_VIDEO_PLACEHOLDER,
    build_video_prompt,
    chunk_patch_offsets,
    plan_video_chunks,
    rank_row_counts,
    split_chunks_across_ranks,
)

# Names of the video keyword arguments of the model (see the module docstring).
VIDEO_PIXEL_VALUES = "video_pixel_values"
VIDEO_GRID_THWS = "video_grid_thws"
VIDEO_NUM_CHUNKS = "video_num_chunks"

# Opt-in data-parallel vision tower for video (see _encode_chunks_data_parallel). Read once at install time.
ENV_VIT_DP = "HPU_KIMI_VIT_DP"

# Marker set on the patched model class; makes the installation idempotent.
_INSTALLED_FLAG = "_hpu_kimi_k3_video"

# Must stay in the "vllm.*" tree: other loggers are silent in vllm-gaudi.
logger = logging.getLogger("vllm.gaudi.kimi_k3_video")


def _split_video_item(item: Any) -> tuple[Any, Mapping[str, Any]]:
    """``(frames, metadata)`` of a parsed video item; the K3 data parser always asks the loader for metadata."""
    if isinstance(item, tuple) and len(item) == 2:
        return item[0], item[1] or {}
    return item, {}


def video_chunk_plan(frames: Any, metadata: Mapping[str, Any], media_proc_cfg: Mapping[str, Any]):
    """Chunks and chunk start times of a decoded video (``frames``: ``[N, H, W, 3]``) from its loader metadata."""
    num_loaded = len(frames)
    source_fps = metadata.get("fps")
    if not source_fps:
        logger.warning("Video loader gave no fps; assuming %s fps for the frame sampling and chunk timestamps",
                       DUMMY_VIDEO_FPS)
        source_fps = DUMMY_VIDEO_FPS
    source_fps = float(source_fps)
    return plan_video_chunks(
        num_loaded,
        int(metadata.get("total_num_frames") or num_loaded),
        source_fps,
        metadata.get("frames_indices"),
        float(media_proc_cfg.get("sample_fps", DEFAULT_MAX_SAMPLE_FPS)),
        int(media_proc_cfg.get("temporal_merge_kernel_size", DEFAULT_TEMPORAL_CHUNK_SIZE)),
    )


def _make_classes():
    """Build the K3 video processing classes (needs the K3 classes of vLLM)."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import BatchFeature
    from vllm.models.kimi_k3.common.mm_preprocess import (
        KimiK3DummyInputsBuilder,
        KimiK3MultiModalProcessor,
        KimiK3ProcessingInfo,
    )
    from vllm.multimodal.inputs import MultiModalFieldConfig
    from vllm.multimodal.parse import MultiModalDataParser, VideoProcessorItems
    from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails, cached_encode

    class HpuKimiK3VideoProcessingInfo(KimiK3ProcessingInfo):

        def get_supported_mm_limits(self) -> Mapping[str, int | None]:
            return {"image": None, "video": None}

        def get_data_parser(self) -> MultiModalDataParser:
            # The chunk plan needs the source fps / frame positions of the loader metadata.
            return MultiModalDataParser(
                expected_hidden_size=self._get_expected_hidden_size(),
                allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
                video_needs_metadata=True,
            )

    class HpuKimiK3VideoDummyInputsBuilder(KimiK3DummyInputsBuilder):

        def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
            return (super().get_dummy_text(mm_counts) + K3_VIDEO_PLACEHOLDER * mm_counts.get("video", 0))

        def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
            data = dict(super().get_dummy_mm_data(seq_len, mm_counts, mm_options))
            num_videos = mm_counts.get("video", 0)
            if not num_videos:
                return data
            overrides = mm_options.get("video") if mm_options else None
            width = getattr(overrides, "width", None) or DUMMY_VIDEO_WIDTH
            height = getattr(overrides, "height", None) or DUMMY_VIDEO_HEIGHT
            num_frames = getattr(overrides, "num_frames", None) or DUMMY_VIDEO_NUM_FRAMES
            video = np.full((num_frames, height, width, 3), 255, dtype=np.uint8)
            metadata = {
                "total_num_frames": num_frames,
                "fps": DUMMY_VIDEO_FPS,
                "duration": num_frames / DUMMY_VIDEO_FPS,
                "video_backend": "dummy",
                "frames_indices": list(range(num_frames)),
                "do_sample_frames": False,
            }
            data["video"] = [(video, dict(metadata)) for _ in range(num_videos)]
            return data

    class KimiK3VideoProcessor(KimiK3MultiModalProcessor):

        def _media_proc_cfg(self) -> Mapping[str, Any]:
            return self.info.image_processor.media_proc_cfg

        def _call_hf_processor(self, hf_data, hf_kwargs):
            hf_data = dict(hf_data)
            videos = hf_data.pop("videos", None)
            outputs = {}
            if hf_data:  # images (and text): the unchanged image-only path
                outputs.update(super()._call_hf_processor(hf_data, hf_kwargs))
            if videos:
                outputs.update(self._process_videos(videos, hf_kwargs))
            return BatchFeature(data=outputs)

        def _process_videos(self, videos: Sequence[Any], hf_kwargs: Mapping[str, object]) -> dict[str, torch.Tensor]:
            cfg = self._media_proc_cfg()
            medias: list[dict[str, Any]] = []
            num_chunks: list[int] = []
            for item in videos:
                frames, metadata = _split_video_item(item)
                chunks, _ = video_chunk_plan(frames, metadata, cfg)
                num_chunks.append(len(chunks))
                for chunk in chunks:
                    # The checkpoint image processor takes PIL frames (it rejects ndarrays).
                    medias.append({
                        "type": "video_chunk",
                        "video_chunk": [Image.fromarray(np.asarray(frames[i])) for i in chunk],
                    })
            # Same flags the image path gets from InputProcessingContext.call_hf_processor. With
            # mm_device_do_normalize (default for Kimi-K3) the merge forces do_rescale=do_normalize=False because
            # the vision tower's input norm rescales and normalizes on the device; preprocessing on the host as
            # well would normalize twice. Without it the keys are absent and the host does both.
            merged = self.info.ctx.get_merged_mm_kwargs(hf_kwargs)
            processed = self.info.image_processor.preprocess(medias,
                                                             return_tensors="pt",
                                                             do_rescale=merged.get("do_rescale", True),
                                                             do_normalize=merged.get("do_normalize", True))
            # With device normalization on, the fused processor skips both host steps and returns uint8 0..255
            # patches, so the cast below does not apply. With it off the processor returns normalized float32
            # patches; cast those to the model dtype as call_hf_processor/_postprocess_output does for images (this
            # direct call bypasses it), which also halves the size of the tensors sent to the workers.
            pixel_values = processed["pixel_values"]
            if pixel_values.is_floating_point():
                pixel_values = pixel_values.to(dtype=self.info.ctx.model_config.dtype)
            return {
                VIDEO_PIXEL_VALUES: pixel_values,
                VIDEO_GRID_THWS: processed["grid_thws"],
                VIDEO_NUM_CHUNKS: torch.tensor(num_chunks, dtype=torch.long),
            }

        def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
            fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
            if VIDEO_GRID_THWS not in hf_inputs:
                return fields
            chunks_per_video = hf_inputs[VIDEO_NUM_CHUNKS]
            patches_per_chunk = hf_inputs[VIDEO_GRID_THWS].prod(-1)
            patches_per_video = torch.stack(
                [s.sum() for s in torch.split(patches_per_chunk, chunks_per_video.tolist())])
            fields[VIDEO_PIXEL_VALUES] = MultiModalFieldConfig.flat_from_sizes("video", patches_per_video)
            fields[VIDEO_GRID_THWS] = MultiModalFieldConfig.flat_from_sizes("video", chunks_per_video, keep_on_cpu=True)
            fields[VIDEO_NUM_CHUNKS] = MultiModalFieldConfig.batched("video", keep_on_cpu=True)
            return fields

        def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
            updates = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
            if "video" not in mm_items:
                return updates

            tokenizer = self.info.get_tokenizer()
            media_token = self.info.media_token
            media_token_id = self.info.media_token_id
            cfg = self._media_proc_cfg()
            videos = mm_items.get_items("video", VideoProcessorItems)

            def get_replacement(item_idx: int) -> PromptUpdateDetails:
                frames, metadata = _split_video_item(videos.get(item_idx))
                frames = np.asarray(frames)
                chunks, starts = video_chunk_plan(frames, metadata, cfg)
                # Tokens of a chunk do not depend on its pixels, only on the frame size: count them with blank frames
                # of the real size through the same calculator the image path uses.
                blank = Image.new("RGB", (frames.shape[2], frames.shape[1]))
                tokens = [
                    self.info.media_tokens_calculator({
                        "type": "video_chunk",
                        "video_chunk": [blank] * len(chunk)
                    }) for chunk in chunks
                ]
                return PromptUpdateDetails.select_token_id(
                    cached_encode(tokenizer, build_video_prompt(starts, tokens, media_token), add_special_tokens=False),
                    media_token_id,
                )

            updates.append(
                PromptReplacement(
                    modality="video",
                    target=cached_encode(tokenizer, K3_VIDEO_PLACEHOLDER, add_special_tokens=False),
                    replacement=get_replacement,
                ))
            return updates

    return (HpuKimiK3VideoProcessingInfo, HpuKimiK3VideoDummyInputsBuilder, KimiK3VideoProcessor)


def _vit_dp_requested() -> bool:
    import vllm_gaudi.envs as gaudi_envs
    return gaudi_envs.HPU_KIMI_VIT_DP


def _encode_chunks_data_parallel(amd_model, model, pixel_values, grid_thws, chunk_rows: list[int]):
    """Per-chunk embeddings of a video group, with the chunks split across the TP ranks (``HPU_KIMI_VIT_DP=1``).

    Each rank runs the plain tower on a contiguous chunk range (``split_chunks_across_ranks``), so the temporal pooling
    and the 2x2 merge are the same as on the single-rank path; every chunk is an independent varlen sequence of the
    tower, so the split does not change the math. The local outputs are padded to the largest per-rank row count,
    all-gathered over the TP group and cut back in rank order, which is the chunk order. Requires replicated tower
    weights (``use_data_parallel``) and that every TP rank calls ``embed_multimodal`` with the same inputs, which
    holds for the SPMD HPU runner (serving and multimodal warmup). The all-gather runs here, in eager code: only
    ``vision_tower.encoder.blocks`` is compiled, and the blocks see one chunk per call.
    """
    import torch
    from vllm.distributed import (get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
                                  tensor_model_parallel_all_gather)

    world_size = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    grid_list = grid_thws.tolist()
    ranges = split_chunks_across_ranks(len(grid_list), world_size)
    rows = rank_row_counts(chunk_rows, ranges)
    max_rows = max(rows)
    start, end = ranges[rank]
    out_dim = model.mm_projector.linear_2.output_size
    dtype = next(model.mm_projector.parameters()).dtype
    offsets = chunk_patch_offsets(grid_list)

    def encode(first: int, last: int):
        # One chunk per tower call: the compiled shape does not depend on the number of chunks of a rank.
        return [
            amd_model.vision_tower_forward(
                model.vision_tower,
                pixel_values[offsets[i]:offsets[i + 1]],
                grid_thws[i:i + 1],
                mm_projector=model.mm_projector,
                use_data_parallel=False,
            )[0] for i in range(first, last)
        ]

    if start == end and grid_list:
        # An idle rank encodes chunk 0 and drops the result, so every rank compiles the tower at warmup.
        encode(0, 1)
    if start < end:
        local = torch.cat(encode(start, end), dim=0)
        if tuple(local.shape) != (rows[rank], out_dim):
            raise ValueError(f"Kimi-K3 video DP: rank {rank} encoded chunks [{start}, {end}) into "
                             f"{tuple(local.shape)}, expected ({rows[rank]}, {out_dim})")
        if local.shape[0] < max_rows:
            pad = local.new_zeros(max_rows - local.shape[0], out_dim)
            local = torch.cat([local, pad], dim=0)
    else:
        local = torch.zeros(max_rows, out_dim, dtype=dtype, device=pixel_values.device)
    gathered = tensor_model_parallel_all_gather(local.contiguous(), dim=0).view(world_size, max_rows, out_dim)
    merged = torch.cat([gathered[r, :rows[r]] for r in range(world_size) if rows[r]], dim=0)
    if merged.shape[0] != sum(chunk_rows):
        raise ValueError(f"Kimi-K3 video DP: gathered {merged.shape[0]} rows, expected {sum(chunk_rows)}")
    return list(torch.split(merged, chunk_rows))


def _patch_model_class(model_cls, amd_model) -> None:
    """Video-aware ``get_placeholder_str``, ``_mark_tower_model`` and ``embed_multimodal`` on the K3 model class."""
    import torch

    vit_dp = _vit_dp_requested()
    dp_fallback_logged = []

    orig_placeholder = model_cls.get_placeholder_str.__func__
    orig_mark_tower = model_cls._mark_tower_model
    orig_embed = model_cls.embed_multimodal

    def get_placeholder_str(cls, modality: str, i: int):
        if modality == "video":
            return K3_VIDEO_PLACEHOLDER
        return orig_placeholder(cls, modality, i)

    def _mark_tower_model(self, vllm_config, modalities, *args, **kwargs):
        # The tower serves both modalities; marked as image-only it would be skipped (left as a stub layer) when
        # only video is enabled by --limit-mm-per-prompt.
        if modalities == "image" or modalities == {"image"}:
            modalities = {"image", "video"}
        return orig_mark_tower(self, vllm_config, modalities, *args, **kwargs)

    def embed_multimodal(self, **kwargs: object):
        pixel_values = kwargs.get(VIDEO_PIXEL_VALUES)
        if pixel_values is None:
            return orig_embed(self, **kwargs)
        if kwargs.get("pixel_values") is not None:
            raise ValueError("Kimi-K3: image and video items must be encoded in separate groups")
        grid_thws = kwargs[VIDEO_GRID_THWS]
        num_chunks = kwargs[VIDEO_NUM_CHUNKS]
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)
        if isinstance(grid_thws, list):
            grid_thws = torch.cat(list(grid_thws), dim=0)
        if isinstance(num_chunks, list):
            num_chunks = torch.stack(list(num_chunks))
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        # Plain tower call (use_data_parallel=False): the DP path (is_vit_use_data_parallel ->
        # run_dp_sharded_mrope_vision_model) skips the temporal pooling of tpool_patch_merger.
        merge_h, merge_w = self.vision_tower.merge_kernel_size
        chunk_rows = [(h // merge_h) * (w // merge_w) for _, h, w in grid_thws.tolist()]
        use_dp = False
        if vit_dp:
            from vllm.distributed import get_tensor_model_parallel_world_size
            use_dp = bool(getattr(self, "use_data_parallel", False)) and get_tensor_model_parallel_world_size() > 1
            if not use_dp and not dp_fallback_logged:
                dp_fallback_logged.append(True)
                logger.warning(
                    "%s=1 ignored: the vision tower is not replicated (use_data_parallel=%s) or TP world size is 1; "
                    "every rank encodes all video chunks", ENV_VIT_DP, getattr(self, "use_data_parallel", None))
        if use_dp:
            per_chunk = _encode_chunks_data_parallel(amd_model, self, pixel_values, grid_thws, chunk_rows)
        else:
            per_chunk = amd_model.vision_tower_forward(
                self.vision_tower,
                pixel_values,
                grid_thws,
                mm_projector=self.mm_projector,
                use_data_parallel=False,
            )
        chunks_per_video = num_chunks.reshape(-1).tolist()
        if len(per_chunk) != sum(chunks_per_video):
            raise ValueError(f"Kimi-K3 video: the vision tower returned {len(per_chunk)} chunk embeddings for "
                             f"{sum(chunks_per_video)} chunks")
        for i, (_, h, w) in enumerate(grid_thws.tolist()):
            expected = chunk_rows[i]
            if per_chunk[i].shape[0] != expected:
                raise ValueError(f"Kimi-K3 video: chunk {i} has {per_chunk[i].shape[0]} embedding rows, expected "
                                 f"{expected} ((h={h})//{merge_h} * (w={w})//{merge_w})")
        # One tensor per chunk row from the tower; the runner needs one per video.
        outputs = []
        start = 0
        for count in chunks_per_video:
            outputs.append(torch.cat(list(per_chunk[start:start + count]), dim=0))
            start += count
        return outputs

    model_cls.get_placeholder_str = classmethod(get_placeholder_str)
    model_cls._mark_tower_model = _mark_tower_model
    model_cls.embed_multimodal = embed_multimodal


def install_kimi_k3_video() -> None:
    """Add the ``video`` modality to ``vllm.models.kimi_k3.amd.model.KimiK3ForConditionalGeneration`` (K3 only).

    Raises on any problem; ``patches._patch_kimi_k3_arch`` then records the k3_arch patch as failed.
    """
    import vllm.models.kimi_k3.amd.model as amd_model
    import vllm.renderers.kimi_k3 as k3_renderer
    from vllm.multimodal import MULTIMODAL_REGISTRY

    model_cls = amd_model.KimiK3ForConditionalGeneration
    if getattr(model_cls, _INSTALLED_FLAG, False):
        return
    info_cls, dummy_cls, processor_cls = _make_classes()

    _patch_model_class(model_cls, amd_model)
    MULTIMODAL_REGISTRY.register_processor(processor_cls, info=info_cls, dummy_inputs=dummy_cls)(model_cls)
    # In place: the renderer reads the dict at request time. Server-level and request-level media_io_kwargs still win.
    k3_renderer._K3_MEDIA_IO_DEFAULTS.setdefault("video", {
        "num_frames": K3_VIDEO_LOADER_MAX_FRAMES,
        "fps": K3_VIDEO_LOADER_FPS,
    })
    setattr(model_cls, _INSTALLED_FLAG, True)


def log_video_settings() -> None:
    """Logs the video input settings when ``install_kimi_k3_video`` has installed the modality."""
    import vllm.models.kimi_k3.amd.model as amd_model
    if getattr(amd_model.KimiK3ForConditionalGeneration, _INSTALLED_FLAG, False):
        logger.info("Kimi-K3 video input enabled (loader fps=%s, max frames=%s, %s=%d)", K3_VIDEO_LOADER_FPS,
                    K3_VIDEO_LOADER_MAX_FRAMES, ENV_VIT_DP, int(_vit_dp_requested()))
