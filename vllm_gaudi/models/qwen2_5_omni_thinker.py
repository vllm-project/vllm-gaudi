import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2_5_omni_thinker import (Qwen2_5OmniThinkerForConditionalGeneration)
from vllm.model_executor.models.utils import maybe_prefix
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (Qwen2_5OmniAudioEncoderConfig)
from vllm_gaudi.models.qwen2_5_vl import Qwen2_5_VisionTransformerStaticShape
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.models.interfaces import MultiModalEmbeddings


class HpuQwen2_5OmniThinkerForConditionalGeneration(Qwen2_5OmniThinkerForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        thinker_config = vllm_config.model_config.hf_config.thinker_config
        quant_config = vllm_config.quant_config

        self.audio_tower = Qwen2_5OmniAudioEncoderStaticShape(thinker_config.audio_config)

        self.visual = Qwen2_5_VisionTransformerStaticShape(
            vision_config=thinker_config.vision_config,
            norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )

    def _process_image_input(self, image_input):
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"].type(self.visual.dtype)

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)

        image_embeds = self.visual.get_image_embeds(
            pixel_values,
            grid_thw=grid_thw,
            vision_buckets=self.vision_bucket_manager,
        )
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return image_embeds.split(sizes.tolist())

    def _process_video_input(self, video_input):
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"].type(self.visual.dtype)

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)

        video_embeds = self.visual.get_image_embeds(
            pixel_values_videos,
            grid_thw=grid_thw,
            vision_buckets=self.vision_bucket_manager,
        )
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return video_embeds.split(sizes.tolist())

    def _process_audio_input(self, audio_input):
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]
        if input_features.ndim == 3:
            assert input_features.shape[0] == 1
            input_features = input_features.squeeze(0)
        if audio_feature_lengths.ndim == 2:
            assert audio_feature_lengths.shape[0] == 1 or audio_feature_lengths.shape[1] == 1
            if audio_feature_lengths.shape[0] == 1:
                audio_feature_lengths = audio_feature_lengths.squeeze(0)
            else:
                audio_feature_lengths = audio_feature_lengths.squeeze(1)

        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(audio_feature_lengths))

        audio_outputs = self.audio_tower.get_audio_embeddings(input_features.to(self.audio_tower.dtype),
                                                              audio_feature_lengths=audio_feature_lengths,
                                                              audio_feat_lengths=audio_feat_lengths,
                                                              audio_buckets=self.vision_bucket_manager)

        return audio_outputs.split(audio_output_lengths.tolist())

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        is_multimodal_aligned = torch.zeros_like(input_ids, dtype=torch.bool)
        is_multimodal_aligned[is_multimodal.long()] = True

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal_aligned,
            handle_oov_mm_token=handle_oov_mm_token,
        )


class Qwen2_5OmniAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Qwen2_5OmniAudioEncoderConfig,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads "
                             f"(got `embed_dim`: {self.embed_dim}"
                             f" and `num_heads`: {self.num_heads}).")
        self.scaling = self.head_dim**-0.5
        self.is_decoder = False
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Detect attention implementation.
        self.attn_backend = get_vit_attn_backend(
            head_size=self.head_dim,
            dtype=torch.get_default_dtype(),
        )
        if self.attn_backend not in {
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.ROCM_AITER_FA,
        }:
            raise RuntimeError(f"Qwen2.5-VL does not support {self.attn_backend} backend now.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        seq_length, all_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states).\
            reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).\
            reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).\
            reshape(seq_length, self.num_heads, -1)

        if self.attn_backend == AttentionBackendEnum.FLASH_ATTN:
            from flash_attn import flash_attn_varlen_func

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            attn_output = flash_attn_varlen_func(query_states,
                                                 key_states,
                                                 value_states,
                                                 cu_seqlens,
                                                 cu_seqlens,
                                                 max_seqlen,
                                                 max_seqlen,
                                                 dropout_p=0.0)
            attn_output = attn_output.reshape(seq_length, all_dim)
        elif self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
            if attention_mask is None:
                attention_mask = torch.zeros([1, seq_length, key_states.shape[0]],
                                             device=query_states.device,
                                             dtype=torch.bool)
                for i in range(1, len(cu_seqlens)):
                    attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = True

            query_states = query_states.transpose(0, 1)
            key_states = key_states.transpose(0, 1)
            value_states = value_states.transpose(0, 1)

            from habana_frameworks.torch.hpex.kernels import FusedSDPA
            attn_output = FusedSDPA.apply(
                query_states,
                key_states,
                value_states,
                attention_mask,
                0.0,
            )

            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(seq_length, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output


class Qwen2_5OmniAudioEncoderLayer(nn.Module):

    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen2_5OmniAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = _ACTIVATION_REGISTRY[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape
                `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are
                indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in
                a given layer of size `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all
                attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 or \
           hidden_states.dtype == torch.bfloat16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states, )

        return outputs


class SinusoidsPositionEmbedding(nn.Module):

    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = \
          torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = \
          torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen2_5OmniAudioEncoder(nn.Module):

    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__()
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding \
                                                else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = \
            SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        self.layers = \
          nn.ModuleList([
              Qwen2_5OmniAudioEncoderLayer(config)
              for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.conv1.weight.device

    def postprocess_hpu(
        self,
        padded_token_audio,
        padded_aftercnn_lens,
        aftercnn_lens,
    ):
        # avgpool.stride is 2
        padded_aftercnn_lens = padded_aftercnn_lens // 2
        aftercnn_lens = aftercnn_lens // 2
        padded_token_audio_list = \
          padded_token_audio.split(padded_aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for i, each_audio_states in enumerate(padded_token_audio_list):
            token_audio_list.append(each_audio_states[:aftercnn_lens[i], :])
        token_audio = torch.cat(token_audio_list, dim=0)

        return token_audio

    def forward(
        self,
        input_features=None,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = \
          self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        cu_seqlens = torch.cat((
            torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0),
        )).to(torch.int32)

        padded_embed = \
          nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = \
          nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + \
            self.positional_embedding.positional_embedding[
                : padded_embed.shape[1], :
            ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat((
            torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0),
        )).to(torch.int32)

        for _, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]

        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = torch.cat(token_audio_list, dim=0)
        return token_audio

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        """
        Pads a sequence of tensors to their maximum length on indicated
        `padding_side`. Then prepares a mask so that pad tokens are not
        attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output
        length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class Qwen2_5OmniAudioEncoderStaticShape(Qwen2_5OmniAudioEncoder):

    def pre_attn(
        self,
        input_features,
        feature_lens,
        audio_buckets,
    ):
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = \
          self.padded_and_mask_function(
              chunk_list, chunk_lengths, padding_value=0,
              audio_buckets=audio_buckets
          )

        cu_seqlens = torch.cat((
            torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0),
        )).to(torch.int32)

        padded_mask_after_cnn = torch.fill(padded_mask_after_cnn, True)
        attention_mask = torch.zeros(
            [1, padded_mask_after_cnn.sum(), padded_mask_after_cnn.sum()],
            device=input_features.device,
            dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = True

        return padded_feature, padded_mask, padded_mask_after_cnn, \
                padded_mask_after_cnn.sum().unsqueeze(0), attention_mask

    def forward(
        self,
        input_features=None,
        aftercnn_lens=None,
        padded_mask=None,
        padded_mask_after_cnn=None,
        attention_mask=None,
    ):
        padded_embed = nn.functional.gelu(self.conv1(input_features)) * padded_mask
        padded_embed = nn.functional.gelu(\
            self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + \
          self.positional_embedding.positional_embedding[
                                    : padded_embed.shape[1],
                                    :
                                    ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]

        for _, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(hidden_states, None, attention_mask)

            hidden_states = layer_outputs[0]

        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.permute(1, 0)).permute(1, 0)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = torch.cat(token_audio_list, dim=0)
        return token_audio

    def post_attn(
        self,
        padded_token_audio,
        padded_aftercnn_lens,
        aftercnn_lens,
    ):
        # avgpool.stride is 2
        padded_aftercnn_lens = padded_aftercnn_lens // 2
        aftercnn_lens = aftercnn_lens // 2
        padded_token_audio_list = padded_token_audio.split(padded_aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for i, each_audio_states in enumerate(padded_token_audio_list):
            token_audio_list.append(each_audio_states[:aftercnn_lens[i], :])
        token_audio = torch.cat(token_audio_list, dim=0)

        return token_audio

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, audio_buckets=None):
        padded_len = audio_buckets.get_multimodal_bucket(tensor_len.sum())
        padded_len = padded_len if padded_len else tensor_len.sum()
        padded_chunk_num = math.ceil(padded_len / (self.n_window * 2))
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(padded_chunk_num, dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (padded_chunk_num, max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (padded_chunk_num, max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    def get_audio_embeddings(self, input_features, audio_feature_lengths, audio_feat_lengths, audio_buckets):
        offset = 0
        audio_outputs = []
        for i, feature_len in enumerate(audio_feature_lengths):
            input_feature = input_features[:, offset:offset + feature_len]
            offset = offset + feature_len
            padded_feature, padded_mask, padded_mask_after_cnn, \
              padded_aftercnn_lens, attention_mask = \
                  self.pre_attn(input_feature,
                                feature_len.unsqueeze(0),
                                audio_buckets)

            audio_features = self.forward(padded_feature, padded_aftercnn_lens, padded_mask, padded_mask_after_cnn,
                                          attention_mask)
            audio_features = self.post_attn(audio_features, padded_aftercnn_lens, audio_feat_lengths[i:i + 1])
            audio_outputs.append(audio_features)
        return torch.cat(audio_outputs, dim=0)
