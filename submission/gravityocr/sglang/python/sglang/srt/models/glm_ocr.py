# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modeling from:
# ./llama.py and
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/GlmOcr/modular_GlmOcr.py
"""Inference-only GLM-OCR model compatible with HuggingFace weights."""

import logging
from functools import lru_cache
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.glm_ocr.configuration_glm_ocr import (
    GlmOcrConfig,
    GlmOcrTextConfig,
    GlmOcrVisionConfig,
)

from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.attention import vision_utils
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.glm4 import Glm4Model
from sglang.srt.models.glm4v import (
    Glm4vForConditionalGeneration,
    Glm4vPatchMerger,
    Glm4vRMSNorm,
    Glm4vVisionMLP,
    Glm4vVisionModel,
    Glm4vVisionPatchEmbed,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


class GlmOcrRMSNorm(Glm4vRMSNorm):
    pass


class GlmOcrVisionMLP(Glm4vVisionMLP):
    pass


class GlmOcrVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        attn_qkv_bias: bool = True,
        num_dummy_heads: int = 0,
        rms_norm_eps: float = 1e-5,
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)
        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            qkv_bias=attn_qkv_bias,
            proj_bias=True,
            qk_normalization_by_head_size=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            num_dummy_heads=num_dummy_heads,
            use_data_parallel=use_data_parallel,
        )
        self.mlp = GlmOcrVisionMLP(
            dim,
            intermediate_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        S, B, H = x.shape
        # norm1: flatten to 2D -> [S*B, H], then reshape back
        x2d = x.reshape(-1, H)
        hidden_states = self.norm1(x2d).reshape(S, B, H)

        # Attention expects [B, S, H]
        hidden_states = rearrange(hidden_states, "s b h -> b s h")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
        )
        attn = rearrange(attn, "b s h -> s b h")

        # norm2 with fused residual-add: also 2D
        attn2d = attn.reshape(-1, H)
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)
        x_norm = x_norm_2d.reshape(S, B, H)
        x_after_add = x_after_add_2d.reshape(S, B, H)

        # MLP and final residual
        mlp_out = self.mlp(x_norm)
        x = x_after_add + mlp_out
        return x


class GlmOcrVisionPatchEmbed(Glm4vVisionPatchEmbed):
    pass


class GlmOcrVisionPatchMerger(Glm4vPatchMerger):
    pass


class GlmOcrVisionModel(Glm4vVisionModel):
    def __init__(
        self,
        vision_config: GlmOcrVisionConfig,
        text_config: GlmOcrTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__(vision_config, quant_config, prefix, use_data_parallel)

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size
        self.intermediate_size = vision_config.intermediate_size
        self.use_data_parallel = use_data_parallel

        self.patch_embed = GlmOcrVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )

        self.blocks = nn.ModuleList(
            [
                GlmOcrVisionBlock(
                    dim=self.hidden_size,
                    intermediate_dim=self.intermediate_size,
                    num_heads=self.num_heads,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                    rms_norm_eps=vision_config.rms_norm_eps,
                    attn_qkv_bias=vision_config.attention_bias,
                    use_data_parallel=use_data_parallel,
                )
                for layer_idx in range(depth)
            ]
        )
        self.merger = GlmOcrVisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=text_config.intermediate_size,
            quant_config=quant_config,
            bias=False,
            prefix=add_prefix("merger", prefix),
            use_data_parallel=use_data_parallel,
        )

        self.downsample = nn.Conv2d(
            in_channels=vision_config.hidden_size,
            out_channels=vision_config.out_hidden_size,
            kernel_size=vision_config.spatial_merge_size,
            stride=vision_config.spatial_merge_size,
        )
        self.post_layernorm = GlmOcrRMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )

    # SS_GLM_VIT_GRAPH_MAXM>0 sets self._ss_vit (multimodal/glm_vit_graph_runner.py) at the end of
    # GlmOcrForConditionalGeneration.load_weights; unset (SGLang's default) or a failed graph capture = the original
    # eager forward below, unchanged.
    _ss_vit = None

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        if self._ss_vit is not None:
            return self._ss_vit.run(x, grid_thw)
        return self._ss_forward_orig(x, grid_thw)

    def _ss_forward_orig(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # KEEP IN SYNC with GlmVitGraphRunner._body (multimodal/glm_vit_graph_runner.py), which replays this body from
        # `x = self.patch_embed(x)` on (the prologue here becomes its inputs); see the guards named there.
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb_cos, rotary_pos_emb_sin, image_type_ids = self.rot_pos_emb(
            grid_thw
        )
        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

        rotary_pos_emb_cos = torch.cat([rotary_pos_emb_cos, rotary_pos_emb_cos], dim=-1)
        rotary_pos_emb_sin = torch.cat([rotary_pos_emb_sin, rotary_pos_emb_sin], dim=-1)

        # x.shape: (s, b, d) where b=1 for vision processing
        # transformers
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )

        # adapter
        x = self.post_layernorm(x)
        x = x.view(-1, self.spatial_merge_size, self.spatial_merge_size, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.out_hidden_size)
        x = self.merger(x)

        return x



# [TRACK2-SHM-GPU-GET-IMAGE-FEATURE]
import mmap as _t2_mmap
import os as _t2_os
from torchvision.transforms.functional import InterpolationMode as _t2_Interp
from torchvision.transforms.functional import resize as _t2_tv_resize

_T2_SHM_RING_PATH = __import__("os").environ.get("TRACK2_SHM_RING_PATH", "/dev/shm/track2_raw_chw.bin")  # team taktak
_T2_SHM_MMAP = None
_T2_SHM_TRIED = False
_T2_CUDA_MOD = None
_T2_CUDA_TRIED = False

def _t2_get_shm_reader():
    global _T2_SHM_MMAP, _T2_SHM_TRIED
    if _T2_SHM_TRIED and _T2_SHM_MMAP is not None:
        return _T2_SHM_MMAP
    if not _t2_os.path.isfile(_T2_SHM_RING_PATH):
        return None
    _T2_SHM_TRIED = True
    try:
        fd = _t2_os.open(_T2_SHM_RING_PATH, _t2_os.O_RDWR)
        try:
            sz = _t2_os.fstat(fd).st_size
            _T2_SHM_MMAP = _t2_mmap.mmap(fd, sz, _t2_mmap.MAP_SHARED, _t2_mmap.PROT_READ | _t2_mmap.PROT_WRITE)
        finally:
            _t2_os.close(fd)
    except Exception:
        _T2_SHM_MMAP = None
    if _T2_SHM_MMAP is not None and _t2_os.environ.get("TRACK2_SHM_PIN", "1") == "1":
        # [TRACK2-SHM-PIN] page-lock the ring for CUDA: H2D copies from it become async DMA (a pageable copy blocks the
        # scheduler thread until the GPU has drained the stream, once per image)
        try:
            _buf = torch.frombuffer(_T2_SHM_MMAP, dtype=torch.uint8)
            _err = torch.cuda.cudart().cudaHostRegister(_buf.data_ptr(), _buf.numel(), 0)
            print(f"[track2-shm-pin] cudaHostRegister({_buf.numel() >> 20} MB) -> {_err}", flush=True)
        except Exception as _e:
            print(f"[track2-shm-pin] failed: {_e}", flush=True)
    return _T2_SHM_MMAP

def _t2_get_cuda_sm89():
    global _T2_CUDA_MOD, _T2_CUDA_TRIED
    if _T2_CUDA_TRIED:
        return _T2_CUDA_MOD
    _T2_CUDA_TRIED = True
    try:
        from gravity_ocr import cuda_sm89 as _m
        _T2_CUDA_MOD = _m if _m.available() else None
    except Exception:
        _T2_CUDA_MOD = None
    return _T2_CUDA_MOD


class GlmOcrForConditionalGeneration(Glm4vForConditionalGeneration):
    def __init__(
        self,
        config: GlmOcrConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)

        self.pp_group = get_pp_group()
        self.config = config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
        self.visual = GlmOcrVisionModel(
            vision_config=config.vision_config,
            text_config=config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
            use_data_parallel=self.use_data_parallel,
        )

        vision_utils.update_vit_attn_dummy_heads_config(self.config)

        self.model = Glm4Model(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False
        self._t2_shm_gpu_logged = False

    def _t2_is_desc_batch(self, items):
        return (
            items
            and not self.use_data_parallel
            and all(
                isinstance(it.feature, torch.Tensor)
                and it.feature.dtype == torch.int64
                and it.feature.shape == (1, 7)
                and int(it.feature[0, 0]) == -99
                for it in items
            )
        )

    def _t2_preprocess(self, items):
        """shm-ring descriptors -> (pixel_values [N, 588] bf16 on the GPU, grid_thw [n, 3] CPU); enqueued on the CURRENT
        stream: async H2D from the page-locked ring, bicubic antialias resize, fused patchify + normalize directly into
        one pre-allocated output tensor (no per-image alloc + torch.cat copy)."""
        mm = _t2_get_shm_reader()
        cuda_mod = _t2_get_cuda_sm89()
        if mm is None or cuda_mod is None:
            return None
        dev = self.visual.device
        meta = []
        total_patches = 0
        grid_rows = []
        for item in items:
            _, off, nbytes, h, w, gh, gw = (int(v) for v in item.feature[0].tolist())
            m = gh * gw
            meta.append((off, nbytes, h, w, gh, gw, m))
            total_patches += m
            grid_rows.append((1, gh, gw))
        pixel_values = torch.empty((total_patches, 588), dtype=torch.bfloat16, device=dev)
        p_off = 0
        for off, nbytes, h, w, gh, gw, m in meta:
            chw_cpu = torch.frombuffer(mm, dtype=torch.uint8, count=nbytes, offset=off).view(3, h, w)
            chw_gpu = chw_cpu.to(device=dev, non_blocking=True)
            if (gh * 14, gw * 14) != (h, w):
                chw_gpu = _t2_tv_resize(
                    chw_gpu,
                    [gh * 14, gw * 14],
                    interpolation=_t2_Interp.BICUBIC,
                    antialias=True,
                )
            cuda_mod.chw_u8_patchify_norm_bf16(chw_gpu, gh, gw, out=pixel_values[p_off : p_off + m])
            p_off += m
        image_grid_thw = torch.tensor(grid_rows, dtype=torch.int64)
        if not self._t2_shm_gpu_logged:
            self._t2_shm_gpu_logged = True
            print(
                f"[track2-k1-shm-gpu] FIRST HIT: count={len(items)} grid0={grid_rows[0]} "
                f"patches0=({meta[0][6]}, 588) {pixel_values.dtype}",
                flush=True,
            )
        return pixel_values, image_grid_thw

    def _t2_register_prefetch(self):
        # [TRACK2-T2LOG] multi-layer run log, scheduler layer (gravity_ocr/t2log.py; TRACK2_LOG=0 turns it off)
        try:
            import sys as _t2_sys
            if "sglang.srt.managers.scheduler" in _t2_sys.modules:
                from gravity_ocr import t2log as _t2l
                _t2l.install_scheduler()
        except Exception as _e:
            print(f"[t2log] scheduler layer failed: {_e}", flush=True)
        # [TRACK2-VIT-PREFETCH] see gravity_ocr/vit_prefetch.py
        self._t2_pf = None
        if _t2_os.environ.get("TRACK2_VIT_PREFETCH", "1") != "1" or self.use_data_parallel:
            return
        try:
            from gravity_ocr import vit_prefetch as _t2_vp
            if _t2_get_cuda_sm89() is None:
                return

            def _pre(items, _self=self):
                r = _self._t2_preprocess(items)
                if r is None:
                    raise RuntimeError("shm ring / sm89 kernels unavailable")
                return r

            self._t2_pf = _t2_vp.VitPrefetcher(self, _pre)
            _t2_vp.register(self._t2_pf)
            print("[track2-vit-prefetch] ON: packed ViT batches on a low-priority stream, decode on a high-priority one",
                  flush=True)
        except Exception as _e:
            print(f"[track2-vit-prefetch] OFF: {_e}", flush=True)
            self._t2_pf = None

    def _t2_vit_quantized(self):
        qm = getattr(self.visual.blocks[0].attn.qkv_proj, "quant_method", None)
        return qm is not None and type(qm).__name__ != "UnquantizedLinearMethod"

    def _t2_capture_if_deferred(self):
        if getattr(self, "_t2_capture_deferred", False):
            self._t2_capture_deferred = False
            self._ss_maybe_capture_vit_graphs()

    def get_image_feature(self, items):
        self._t2_capture_if_deferred()
        pf = getattr(self, "_t2_pf", None)
        if self._t2_is_desc_batch(items):
            if pf is not None:
                return pf.get_embeddings(items)
            r = self._t2_preprocess(items)
            if r is not None:
                return self.visual(r[0], grid_thw=r[1])
        if pf is not None:
            # any other ViT input still runs on the prefetch stream (one stream owns the graph runner's buffers)
            return pf.run_on_stream(lambda its: super(GlmOcrForConditionalGeneration, self).get_image_feature(its), items)
        return super().get_image_feature(items)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".up_proj", 1),
            (".gate_up_proj", ".gate_proj", 0),
        ]

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))

        # For the PP case, we add special handling for lm_head.weight,
        # - On non–last ranks: we continue, because this stage is supposed to
        #   be just an empty PPMissingLayer shell.
        # - On the last rank: params_dict is expected to contain lm_head.weight,
        #   so it will never hit the branch "if name not in params_dict".
        #
        # For all other parameters, such like
        # "model.visual.blocks.20.mlp.gate_proj.weight", the unified rule is:
        # If this name does not exist in the current rank’s params_dict,
        # it does not belong to this pipeline stage, thus we simply continue.

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if "model.visual." in name:
                name = name.replace("model.visual.", "visual.")

            if not is_nextn:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            continue
            else:
                if not name.startswith(nextn_layer_prefix):
                    continue

                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = name.replace(nextn_layer_prefix, "model")
                        is_decoder = False
                        break
                # For decoder layer weights
                if is_decoder:
                    name = name.replace(nextn_layer_prefix, "model.decoder")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")

                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if "visual" in name:
                    loaded_weight = vision_utils.pad_vit_attn_dummy_heads(
                        self.config, name, loaded_weight
                    )
                weight_loader(param, loaded_weight)

        if not is_nextn:
            if self._t2_vit_quantized():
                # quantized ViT linears get their FP8 weights only after load_weights returns
                # (process_weights_after_loading): capture at the first scheduler tick instead
                self._t2_capture_deferred = True
                print("[ss-vit-graph] deferred: quantized ViT, graphs are captured at the first scheduler tick", flush=True)
            else:
                self._ss_maybe_capture_vit_graphs()
            self._t2_register_prefetch()
            try:
                # v6 fused FP8 kernels (MLP silu*up+quant, fast M-RoPE); before any CUDA graph capture
                from gravity_ocr import fused_fp8 as _t2_ff
                _t2_ff.install(self)  # self: the GlmOcr model (layer / block stacks get chained)
            except Exception as _e:
                print(f"[track2-fused] OFF: {_e}", flush=True)
            try:
                # [TRACK2-JIT-STABLE] one compiled variant per SGLang index-copy Triton kernel (no alignment stalls)
                from gravity_ocr import jit_stable as _t2_js
                _t2_js.install()
            except Exception as _e:
                print(f"[track2-jit-stable] OFF: {_e}", flush=True)

    def _ss_maybe_capture_vit_graphs(self):
        """Exact-M ViT CUDA graphs (multimodal/glm_vit_graph_runner.py). Off unless SS_GLM_VIT_GRAPH_MAXM > 0 (the
        GravityOCR launchers set it under FAST=1). A malformed setting stops the startup with a ValueError naming the
        variable; a failed capture does not: GlmVitGraphRunner.setup() releases what it captured and returns False,
        and the ViT keeps its original eager forward."""
        from sglang.srt.multimodal.glm_vit_graph_runner import (
            GlmVitGraphRunner,
            report_settings,
            vit_graph_settings,
        )

        settings = vit_graph_settings()
        report_settings(settings)
        if settings.maxm <= 0 or self.visual._ss_vit is not None:
            return
        if self.use_data_parallel or self.visual.device.type != "cuda":
            print(f"[ss-vit-graph] OFF: dp_encoder={self.use_data_parallel} device={self.visual.device}", flush=True)
            return
        runner = GlmVitGraphRunner(self.visual, settings)
        if runner.setup():
            self.visual._ss_vit = runner


EntryClass = [GlmOcrForConditionalGeneration]
