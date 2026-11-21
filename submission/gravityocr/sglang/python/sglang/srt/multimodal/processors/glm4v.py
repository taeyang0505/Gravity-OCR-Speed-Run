"""Track 2 Zero-Pickle /dev/shm + GPU-Offloaded GLM-OCR Multimodal Processor."""
from __future__ import annotations

import asyncio
import base64
import io
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import unquote

import numpy as np
import torch
from PIL import Image

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration
from sglang.srt.models.glm4v_moe import Glm4vMoeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
    MultimodalSpecialTokens,
)

try:
    from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration
except ImportError:
    GlmOcrForConditionalGeneration = None

try:
    from torchvision.io import ImageReadMode, decode_image, read_image
    _HAS_TV_IO = True
except Exception:
    _HAS_TV_IO = False

try:
    from gravity_ocr import native as _g_native
    from gravity_ocr import preprocess as _g_preprocess

    _g_k = _g_native.load()
    if _g_k is not None:
        _g_k.set_threads(1)
    _HAS_RUST_PATCHIFY = _g_k is not None and _g_k.has("pixels")
except Exception:
    _g_preprocess = None
    _HAS_RUST_PATCHIFY = False

try:
    from gravity_ocr import textscale as _g_textscale
except Exception:
    _g_textscale = None

try:
    from gravity_ocr import margin_crop as _g_margin_crop
except Exception:
    _g_margin_crop = None

try:
    from gravity_ocr import cpu_prep as _g_cpu_prep
except Exception:
    _g_cpu_prep = None

try:  # multi-layer run log, tokenizer-manager layer (L2)
    from gravity_ocr import t2log as _T2LOG
    if _T2LOG.LEVEL < 1:
        _T2LOG = None
except Exception:
    _T2LOG = None

SHM_RING_PATH = os.environ.get("TRACK2_SHM_RING_PATH", "/dev/shm/track2_raw_chw.bin")  # team taktak: per lane
# 1 GB tmpfs ring (~4.6x one 402-crop dev job of raw uint8 CHW; every crop of a job is queued at once). The scheduler
# process registers the same mapping with CUDA (cudaHostRegister) so its H2D copies are async DMA.
SHM_RING_BYTES = int(os.environ.get("TRACK2_SHM_RING_MB", "1024")) * 1024 * 1024
_SHM_LOCK = threading.Lock()
_SHM_MMAP: Optional[mmap.mmap] = None
_SHM_NP: Optional[np.ndarray] = None
_SHM_OFFSET = 0


_T2_GC_DONE = False
_T2_CAP_TEXT = int(os.environ.get("TRACK2_TEXT_MAX_PATCHES", "0"))    # 0 = off
_T2_CAP_TABLE = int(os.environ.get("TRACK2_TABLE_MAX_PATCHES", "0"))  # 0 = off


def _t2_sm89_lib_present() -> bool:
    # the library must LOAD with its base kernels (missing or unloadable build -> CPU path, never broken descriptors)
    try:
        from gravity_ocr import cuda_sm89 as _c

        return _c.load() is not None
    except Exception:
        return False


def _get_shm_ring() -> Optional[mmap.mmap]:
    global _SHM_MMAP, _SHM_NP
    if _SHM_MMAP is not None:
        return _SHM_MMAP
    with _SHM_LOCK:
        if _SHM_MMAP is not None:
            return _SHM_MMAP
        try:
            fd = os.open(SHM_RING_PATH, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if os.fstat(fd).st_size < SHM_RING_BYTES:
                    # allocate the pages now: a /dev/shm smaller than the ring fails here (ENOSPC -> CPU fallback)
                    # instead of SIGBUS-ing the tokenizer on a later write into a sparse file
                    os.posix_fallocate(fd, 0, SHM_RING_BYTES)
                _SHM_MMAP = mmap.mmap(fd, SHM_RING_BYTES, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
                _SHM_NP = np.frombuffer(_SHM_MMAP, dtype=np.uint8)
            finally:
                os.close(fd)
        except Exception:
            _SHM_MMAP = None
            _SHM_NP = None
    return _SHM_MMAP


def _write_shm_ring(chw: torch.Tensor, y0: int = 0, y1: int = -1, x0: int = 0, x1: int = -1) -> Optional[int]:
    """Reserve [off, off + n) under the lock, copy outside it (decode workers write in parallel); 64-byte aligned."""
    global _SHM_OFFSET
    if _get_shm_ring() is None or _SHM_NP is None:
        return None
    c, h, w = chw.shape
    if y1 < 0:
        y1 = h
    if x1 < 0:
        x1 = w
    ch, cw = y1 - y0, x1 - x0
    n = int(c * ch * cw)
    if n <= 0 or n > SHM_RING_BYTES:
        return None
    with _SHM_LOCK:
        off = _SHM_OFFSET
        if off + n > SHM_RING_BYTES:
            off = 0
        _SHM_OFFSET = (off + n + 63) & ~63
    if _g_cpu_prep is not None and c == 3:
        if _g_cpu_prep.copy_crop_chw_c(chw.numpy(), y0, y1, x0, x1, int(_SHM_NP.ctypes.data) + off):
            return off
    sub = chw if (y0 == 0 and y1 == h and x0 == 0 and x1 == w) else chw[:, y0:y1, x0:x1]
    if sub.is_contiguous():
        _SHM_NP[off : off + n] = sub.numpy().reshape(-1)
    else:
        _SHM_NP[off : off + n].reshape(sub.shape)[:] = sub.numpy()
    return off


def _decode_raw_chw_u8(item: Union[str, bytes, Image.Image]) -> Optional[torch.Tensor]:
    """Decode image into `uint8 [3, H, W]` with C++ libpng/libjpeg (skips .contiguous() copy when C prep handles strides)."""
    _no_copy = _g_cpu_prep is not None and _g_cpu_prep.load() is not None
    if isinstance(item, str):
        path = None
        if item.startswith("file://"):
            path = unquote(item[7:])
        elif not item.startswith(("http://", "https://", "data:")) and os.path.isfile(item):
            path = item
        if path is not None:
            if _HAS_TV_IO and path.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    t = read_image(path, ImageReadMode.RGB)
                    if t.dtype == torch.uint8 and t.ndim == 3 and t.shape[0] == 3:
                        return t if _no_copy else t.contiguous()
                except Exception:
                    pass
            with Image.open(path) as im:
                rgb = np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
            t = torch.from_numpy(rgb).permute(2, 0, 1)
            return t if _no_copy else t.contiguous()

    if isinstance(item, bytes):
        if _HAS_TV_IO:
            try:
                buf = torch.frombuffer(bytearray(item), dtype=torch.uint8)
                t = decode_image(buf, ImageReadMode.RGB)
                if t.dtype == torch.uint8 and t.ndim == 3 and t.shape[0] == 3:
                    return t if _no_copy else t.contiguous()
            except Exception:
                pass
        with Image.open(io.BytesIO(item)) as im:
            rgb = np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
        t = torch.from_numpy(rgb).permute(2, 0, 1)
        return t if _no_copy else t.contiguous()

    if isinstance(item, str) and item.startswith("data:image/") and ";base64," in item:
        raw = base64.b64decode(item.split(";base64,", 1)[1])
        return _decode_raw_chw_u8(raw)

    if isinstance(item, Image.Image):
        rgb = np.ascontiguousarray(np.asarray(item.convert("RGB"), dtype=np.uint8))
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    return None


def _fast_prepare_shm_or_bf16(
    item: Union[str, bytes, Image.Image],
    use_bf16: bool,
    use_shm_gpu: bool,
    task: str = "",
) -> Optional[Tuple[torch.Tensor, Tuple[int, int]]]:
    if _g_preprocess is None:
        return None
    global _T2_GC_DONE
    if not _T2_GC_DONE:
        _T2_GC_DONE = True  # [TRACK2-GC] tokenizer manager: startup is over, keep cyclic-GC pauses out of the jobs
        try:
            from gravity_ocr import gcguard as _t2_gcguard
            _t2_gcguard.freeze("tokenizer")
        except Exception:
            pass
    _t2_t0 = time.perf_counter()
    chw = _decode_raw_chw_u8(item)
    if chw is None:
        return None
    _, h0, w0 = chw.shape
    h, w = h0, w0
    y0, y1, x0, x1 = 0, h0, 0, w0
    s = 1.0
    if _g_textscale is not None and (task in ("text", "table") or (task == "formula" and getattr(_g_textscale, "FORMULA_GLYPH", 0) > 0)):
        # [TRACK2-TEXT-DOWNSCALE] glyph/pitch normalization for text & table crops (downscale large, upscale tiny)
        s, _why = _g_textscale.text_downscale(chw, task)
    if _g_margin_crop is not None and _g_margin_crop.ENABLED:
        # [TRACK2-MARGIN-CROP] tight ink bounding box + 28 px grid snap (s decided on full crop first)
        (y0, y1, x0, x1), _mwhy = _g_margin_crop.crop_box(chw, task, s)
        h, w = y1 - y0, x1 - x0
    if s != 1.0 and _g_textscale is not None:
        sh, sw = _g_textscale.scaled_size(h, w, s)
        th, tw = _g_preprocess.smart_resize(sh, sw)
    else:
        th, tw = _g_preprocess.smart_resize(h, w)
    _cap = _T2_CAP_TEXT if task == "text" else (_T2_CAP_TABLE if task == "table" else 0)  # formulas: never capped
    if _cap > 0 and (th // 14) * (tw // 14) > _cap:
        # [TRACK2-PATCH-CAP] opt-in (TRACK2_TEXT_MAX_PATCHES / TRACK2_TABLE_MAX_PATCHES): an outlier crop above the cap is
        # resampled down to it (same single bicubic resample on the GPU, aspect kept)
        _f = ((_cap / ((th // 14) * (tw // 14))) ** 0.5) * 0.999
        for _ in range(12):
            nh, nw = _g_preprocess.smart_resize(max(28, int(th * _f)), max(28, int(tw * _f)))
            if (nh // 14) * (nw // 14) <= _cap:
                break
            _f *= 0.97
        if (nh // 14) * (nw // 14) <= _cap:
            s = s * ((nh * nw) / float(th * tw)) ** 0.5
            th, tw = nh, nw
    gh, gw = th // 14, tw // 14

    if use_shm_gpu:
        off = _write_shm_ring(chw, y0, y1, x0, x1)
        if off is not None:
            # 56-byte descriptor: [-99, shm_offset, nbytes, H, W, gh, gw]
            desc = torch.tensor([[-99, off, int(3 * h * w), h, w, gh, gw]], dtype=torch.int64)
            return desc, (gh, gw), {"decode_ms": round(1e3 * (time.perf_counter() - _t2_t0), 2), "scale": round(s, 3),
                                    "h": h, "w": w}

    # Fallback CPU patchify if /dev/shm unavailable
    if (y0, y1, x0, x1) != (0, h0, 0, w0):
        chw = chw[:, y0:y1, x0:x1]
    im = Image.fromarray(chw.permute(1, 2, 0).contiguous().numpy())
    if (th, tw) != (h, w):
        im = im.resize((tw, th), Image.BICUBIC)
    rgb = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
    p1, (gh, gw) = _g_preprocess.to_patches(rgb)
    t1 = torch.from_numpy(p1)
    if use_bf16:
        t1 = t1.to(torch.bfloat16)
    pv = t1.view(-1, 3, 1, 196).expand(-1, 3, 2, 196).reshape(-1, 1176).contiguous()
    return pv, (gh, gw)


def _fast_mrope_glm4v(head_len: int, rows: int, cols: int, tail_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized exact 3-axis M-RoPE matching MRotaryEmbedding.get_rope_index_glm4v bit-for-bit."""
    n_img = rows * cols
    total = head_len + n_img + tail_len
    pos = torch.empty((3, total), dtype=torch.long)
    if head_len > 0:
        h_idx = torch.arange(head_len, dtype=torch.long)
        pos[0, :head_len] = h_idx
        pos[1, :head_len] = h_idx
        pos[2, :head_len] = h_idx
    if n_img > 0:
        img_slice = slice(head_len, head_len + n_img)
        pos[0, img_slice] = head_len
        r_idx = torch.arange(rows, dtype=torch.long).repeat_interleave(cols) + head_len
        c_idx = torch.arange(cols, dtype=torch.long).repeat(rows) + head_len
        pos[1, img_slice] = r_idx
        pos[2, img_slice] = c_idx
    after = head_len + max(rows, cols)
    if tail_len > 0:
        t_idx = torch.arange(after, after + tail_len, dtype=torch.long)
        pos[0, head_len + n_img :] = t_idx
        pos[1, head_len + n_img :] = t_idx
        pos[2, head_len + n_img :] = t_idx
    delta = torch.tensor([[after + tail_len - total]], dtype=torch.long)
    return pos, delta


class Glm4vImageProcessor(SGLangBaseProcessor):
    models = [
        m
        for m in [
            Glm4vForConditionalGeneration,
            Glm4vMoeForConditionalGeneration,
            GlmOcrForConditionalGeneration,
        ]
        if m is not None
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self._ss_pv_bf16 = False
        if os.environ.get("SS_GLM_PV_BF16", "0") == "1":
            dt = str(getattr(hf_config, "torch_dtype", None) or getattr(hf_config, "dtype", None) or "")
            ok = "bfloat16" in dt and str(getattr(server_args, "dtype", "auto")) in ("auto", "bfloat16")
            self._ss_pv_bf16 = ok

        self.IMAGE_TOKEN = "<|image|>"
        self.VIDEO_TOKEN = "<|video|>"
        self.IMAGE_START_TOKEN = "<|begin_of_image|>"
        self.IMAGE_END_TOKEN = "<|end_of_image|>"
        self.VIDEO_START_TOKEN = "<|begin_of_video|>"
        self.VIDEO_END_TOKEN = "<|end_of_video|>"

        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id
        self.IMAGE_START_TOKEN_ID = hf_config.image_start_token_id
        self.IMAGE_END_TOKEN_ID = hf_config.image_end_token_id
        self.VIDEO_START_TOKEN_ID = hf_config.video_start_token_id
        self.VIDEO_END_TOKEN_ID = hf_config.video_end_token_id

        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 112 * 112
        self.MAX_PIXELS = 30000 * 28 * 28 * 2

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            video_token=self.VIDEO_TOKEN,
            video_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

        self._fast_mm_enabled = os.environ.get("TRACK2_FAST_MM", "1") == "1" and _g_preprocess is not None
        self._shm_gpu_enabled = (
            os.environ.get("TRACK2_SHM_GPU_MM", "1") == "1"
            and _t2_sm89_lib_present()  # the scheduler decodes shm descriptors with libtrack2_cuda_sm89.so
            and (_get_shm_ring() is not None)
        )
        workers = int(os.environ.get("TRACK2_MM_WORKERS", "2"))
        self._fast_pool = ThreadPoolExecutor(max_workers=max(1, workers)) if workers > 0 else None
        self._prompt_id_cache: Dict[str, Tuple[List[int], List[int]]] = {}
        self._fast_hits = 0
        print(
            f"[track2-fast-mm] enabled={self._fast_mm_enabled} shm_gpu={self._shm_gpu_enabled} "
            f"tv_io={_HAS_TV_IO} workers={workers}",
            flush=True,
        )

    def _get_prompt_parts(self, input_text: str) -> Optional[Tuple[List[int], List[int]]]:
        cached = self._prompt_id_cache.get(input_text)
        if cached is not None:
            return cached
        if self.IMAGE_TOKEN not in input_text:
            return None
        ids = self._tokenizer.encode(input_text, add_special_tokens=True)
        try:
            idx = ids.index(self.IM_TOKEN_ID)
        except ValueError:
            return None
        if ids.count(self.IM_TOKEN_ID) != 1:
            return None
        pair = (ids[:idx], ids[idx + 1 :])
        self._prompt_id_cache[input_text] = pair
        print(f"[track2-fast-mm] prompt template #{len(self._prompt_id_cache)}: {input_text!r}", flush=True)
        return pair

    def compute_mrope_positions(self, input_ids, mm_items):
        image_grid_thw = None
        video_grid_thw = None
        for item in mm_items:
            if "image_grid_thw" in item.model_specific_data:
                image_grid_thw = item.model_specific_data["image_grid_thw"]
            if "video_grid_thw" in item.model_specific_data:
                video_grid_thw = item.model_specific_data["video_grid_thw"]

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids_tensor)
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids_tensor,
            hf_config=self.hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        return mrope_positions.squeeze(1), mrope_position_delta

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        _t2_tin = time.perf_counter()
        if (
            self._fast_mm_enabled
            and image_data
            and len(image_data) == 1
            and not getattr(request_obj, "video_data", None)
            and isinstance(input_text, str)
        ):
            parts = self._get_prompt_parts(input_text)
            if parts is not None:
                raw_img = image_data[0]
                if hasattr(raw_img, "url"):
                    raw_img = raw_img.url
                elif isinstance(raw_img, dict) and "url" in raw_img:
                    raw_img = raw_img["url"]
                task = (
                    "text" if "Text Recognition" in input_text
                    else "table" if "Table Recognition" in input_text
                    else "formula" if "Formula Recognition" in input_text
                    else ""
                )
                if self._fast_pool is not None:
                    loop = asyncio.get_running_loop()
                    res = await loop.run_in_executor(
                        self._fast_pool,
                        _fast_prepare_shm_or_bf16,
                        raw_img,
                        self._ss_pv_bf16,
                        self._shm_gpu_enabled,
                        task,
                    )
                else:
                    res = _fast_prepare_shm_or_bf16(raw_img, self._ss_pv_bf16, self._shm_gpu_enabled, task)
                if res is not None:
                    pv, (gh, gw) = res[0], res[1]
                    if _T2LOG is not None:
                        _T2LOG.TM.add({"in": round(_t2_tin, 5), "out": round(time.perf_counter(), 5), "task": task,
                                       "patches": gh * gw, **(res[2] if len(res) > 2 else {})})
                    head_ids, tail_ids = parts
                    rows, cols = gh // 2, gw // 2
                    n_img = rows * cols
                    input_ids_list = head_ids + [self.IM_TOKEN_ID] * n_img + tail_ids
                    grid_thw = torch.tensor([[1, gh, gw]], dtype=torch.long)
                    mm_item = MultimodalDataItem(
                        modality=Modality.IMAGE,
                        feature=pv,
                        offsets=[(len(head_ids), len(head_ids) + n_img - 1)],
                        model_specific_data={"image_grid_thw": grid_thw},
                    )
                    mrope_positions, mrope_position_delta = _fast_mrope_glm4v(
                        len(head_ids), rows, cols, len(tail_ids)
                    )
                    if self._fast_hits == 0:
                        print(
                            f"[track2-fast-mm] FIRST HIT: grid=({gh},{gw}) n_img={n_img} desc={tuple(pv.shape)} {pv.dtype}",
                            flush=True,
                        )
                    self._fast_hits += 1
                    return MultimodalProcessorOutput(
                        input_ids=input_ids_list,
                        mm_items=[mm_item],
                        im_token_id=self.mm_tokens.image_token_id,
                        video_token_id=self.mm_tokens.video_token_id,
                        mrope_positions=mrope_positions,
                        mrope_position_delta=mrope_position_delta,
                    )

        # Fallback to stock SGLang multimodal processor
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        if base_output.videos:
            base_output.videos = request_obj.video_data
        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids.unsqueeze(0),
            hf_config=self.hf_config,
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            attention_mask=getattr(ret, "attention_mask", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )
