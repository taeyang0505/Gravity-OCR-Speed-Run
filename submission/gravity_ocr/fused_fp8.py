"""v6-v8 fused kernels wired into SGLang's GLM-OCR forward (installed from models/glm_ocr.py load_weights, i.e. before
any CUDA graph is captured). Every replacement reproduces the replaced arithmetic bit for bit (kernel tests:
tools/test_fused_kernels.py, tools/test_fp8_gemm.py; in-server check: TRACK2_FUSED_VERIFY=1, below).

  GEMM  fp8_utils.fp8_scaled_mm -> gravity_ocr.fp8_gemm dispatcher (bit-exact Triton on o/down at decode batch 8-15,
        43-44; CUTLASS elsewhere)
  MLP   (Glm4MLP: language model, Glm4vVisionMLP: ViT): silu_and_mul + per-token quant -> cuda_sm89.silu_mul_quant_fp8
  MROPE (MRotaryEmbedding.forward_triton) -> gravity_ocr.fast_mrope
  NORM  (Glm4DecoderLayer, GlmOcrVisionBlock): (residual add +) RMSNorm + per-token quant -> cuda_sm89.norm_quant_fp8,
        chained across layers (v8):
          LM : layer i returns its down_proj output un-normalized; layer i+1 applies layer i's post_mlp_layernorm, the
               residual add and its input_layernorm + FP8 quant in ONE kernel (mode 2). The last layer normalizes itself.
          ViT: block i returns x_after_add and leaves `+ mlp_out` pending; block i+1 does that bf16 add, norm1 and the FP8
               quant in ONE kernel (mode 3). The last block does the add itself.
        A layer that falls back to SGLang's forward first finishes a pending step exactly as SGLang would.
  K2    in-place ViT q/k norm + RoPE: lives in the vision.py patch (tools/patch_sglang_vit_l40s.py, TRACK2_K2_INPLACE).

Switches (default on): TRACK2_FUSED=0 turns all of them off; TRACK2_FUSED_MLP / _MROPE / _NORM / _NORM_VIT / _CHAIN.
TRACK2_FUSED_VERIFY=1 (diagnostic): the first TRACK2_FUSED_VERIFY_N (6) eager LM forwards and ViT forwards are re-run
with SGLang's original code (all patches swapped out) and compared bitwise; results go to stdout as [track2-fused-verify].
"""
from __future__ import annotations

import contextlib
import os

import torch

_ON = os.environ.get("TRACK2_FUSED", "1") == "1"
_MLP = _ON and os.environ.get("TRACK2_FUSED_MLP", "1") == "1"
_MROPE = _ON and os.environ.get("TRACK2_FUSED_MROPE", "1") == "1"
_NORM = _ON and os.environ.get("TRACK2_FUSED_NORM", "1") == "1"
_CHAIN = os.environ.get("TRACK2_FUSED_CHAIN", "1") == "1"
_F16ACC = os.environ.get("TRACK2_F16ACC", "off").strip().lower()
_F16_VIT = _F16ACC in ("vit", "all", "1")
_F16_LM = _F16ACC in ("lm", "all", "1")
_VERIFY = os.environ.get("TRACK2_FUSED_VERIFY", "0") == "1"
_VERIFY_N = int(os.environ.get("TRACK2_FUSED_VERIFY_N", "6"))
_INSTALLED = False
_PATCHES = []          # (owner, attribute, original, replacement): swapped back by _originals() (verify mode)


def _patch(owner, name, new) -> None:
    orig = getattr(owner, name)
    new._t2_fused = True
    new._t2_orig = orig
    _PATCHES.append((owner, name, orig, new))
    setattr(owner, name, new)


@contextlib.contextmanager
def _originals():
    """SGLang's original functions in place of every patch (and CUTLASS for fp8_scaled_mm) for the duration."""
    for owner, name, orig, _new in _PATCHES:
        setattr(owner, name, orig)
    try:
        yield
    finally:
        for owner, name, _orig, new in _PATCHES:
            setattr(owner, name, new)


class _Chain:
    """Per layer stack: the step layer i left for layer i + 1 (LM: the post_mlp_layernorm module; ViT: mlp_out)."""

    def __init__(self) -> None:
        self.pending = None


# team taktak (2026-11-21): SS_FUSED_SILU_VIT=1 -> ViT gate_up GEMM + SiLU*up in one kernel (default off)
_SS_FUSED_SILU_VIT = os.environ.get("SS_FUSED_SILU_VIT", "0") == "1"
if _SS_FUSED_SILU_VIT:
    from gravity_ocr import ss_fused_silu as _ss_fsilu


def _down_ok(layer) -> bool:
    """True when `layer` would run apply_fp8_linear's CUTLASS per-token path."""
    ok = getattr(layer, "_t2_fused_ok", None)
    if ok is not None:
        return ok
    w = getattr(layer, "weight", None)
    if w is None or w.dtype != torch.float8_e4m3fn:
        return False  # not quantized (yet): decide again on the next call, do not cache
    try:
        from sglang.srt.layers.quantization import fp8_utils as fu
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        qm = layer.quant_method
        ws = layer.weight_scale
        ok = bool(
            isinstance(qm, Fp8LinearMethod)
            and not getattr(qm, "use_marlin", False)
            and not getattr(qm, "use_mxfp8", False)
            and not getattr(qm, "block_quant", False)
            and getattr(layer, "input_scale", None) is None
            and getattr(qm, "cutlass_fp8_supported", False)
            and ws.numel() == w.shape[1]
            and w.shape[0] % 16 == 0
            and w.shape[1] % 16 == 0
            and not getattr(fu, "use_triton_w8a8_fp8_kernel", False)
            and getattr(fu, "_is_cuda", False)
            and int(getattr(layer, "tp_size", 1)) == 1
            and not getattr(layer, "skip_bias_add", False)
        )
    except Exception:
        ok = False
    layer._t2_fused_ok = ok
    return ok


def _install_mlp(C) -> int:
    from sglang.srt.layers.quantization import fp8_utils as fu

    scaled_mm = fu.fp8_scaled_mm
    n = 0

    def make(orig):
        def forward(self, x, *args, **kwargs):
            dp = self.down_proj
            if not _down_ok(dp):
                return orig(self, x, *args, **kwargs)
            gate_up, _ = self.gate_up_proj(x)
            if gate_up.shape[-1] != 2 * dp.weight.shape[0]:  # unexpected layout: SGLang's ops on this gate_up
                h, _ = dp(self.act_fn(gate_up))
                return h
            q, s = C.silu_mul_quant_fp8(gate_up)
            # == apply_fp8_linear's CUTLASS branch: fp8_scaled_mm(qinput, weight, x_scale, weight_scale, out_dtype, bias)
            out = scaled_mm(q, dp.weight, s, dp.weight_scale, out_dtype=gate_up.dtype, bias=dp.bias)
            return out.view(*gate_up.shape[:-1], out.shape[-1])

        return forward

    from sglang.srt.models import glm4, glm4v

    for cls in (glm4.Glm4MLP, glm4v.Glm4vVisionMLP):
        if not getattr(cls.forward, "_t2_fused", False):
            _patch(cls, "forward", make(cls.forward))
            n += 1
    return n


def _install_mrope() -> bool:
    from sglang.srt.layers.rotary_embedding import mrope as M

    from gravity_ocr import fast_mrope as F

    cls = M.MRotaryEmbedding
    if getattr(cls.forward_triton, "_t2_fused", False):
        return True
    orig = cls.forward_triton

    def forward_triton(self, positions, query, key):
        self._match_cos_sin_cache_dtype(query)
        if F.supported(query, key, self.cos_sin_cache, positions, self.mrope_section, self.head_size, self.rotary_dim,
                       self.mrope_interleaved, self.mrope_interleaved_glm, self.is_neox_style):
            # When query is a strided slice of qkv, write rotated query directly to a contiguous tensor so
            # FlashInferAttnBackend.forward_extend's `q = q.contiguous()` becomes a zero-cost no-op (-30ms/job).
            q_out = torch.empty(query.shape, dtype=query.dtype, device=query.device) if not query.is_contiguous() else None
            query = F.mrope_inplace(query, key, self.cos_sin_cache, positions, self.mrope_section, self.head_size,
                                    self.rotary_dim, q_out=q_out)
            return query, key
        return orig(self, positions, query, key)

    _patch(cls, "forward_triton", forward_triton)
    return True


def _install_norm(C) -> int:
    """(residual add +) RMSNorm -> FP8 quant fused into the layer forwards (kernel == flashinfer norm + sgl-kernel quant,
    bitwise), feeding the same fp8_scaled_mm calls apply_fp8_linear would make; chained across layers (module doc)."""
    from sglang.srt.layers.quantization import fp8_utils as fu
    from sglang.srt.models import glm4, glm_ocr
    from gravity_ocr import fp8_gemm as _G

    scaled_mm = fu.fp8_scaled_mm

    def lin(layer, q8, s, dtype):
        return scaled_mm(q8, layer.weight, s, layer.weight_scale, out_dtype=dtype, bias=layer.bias)

    def lin_f16(layer, q8, s, dtype):
        return _G.mm_f16(q8, layer.weight, s, layer.weight_scale, out_dtype=dtype, bias=layer.bias)

    n = 0
    L = glm4.Glm4DecoderLayer
    if not getattr(L.forward, "_t2_fused", False):
        orig_l = L.forward

        def layer_forward(self, positions, hidden_states, forward_batch, residual):
            st = getattr(self, "_t2_chain", None)
            pend = None
            if st is not None:
                pend, st.pending = st.pending, None
                if residual is None:  # first layer of a forward: nothing can be pending (drop a stale step)
                    pend = None
            at, mlp = self.self_attn, self.mlp
            if (hidden_states.shape[0] == 0 or hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2
                    or hidden_states.shape[-1] not in (1024, 1536) or not hidden_states.is_contiguous()
                    or (residual is not None and not residual.is_contiguous())
                    or not (_down_ok(at.qkv_proj) and _down_ok(mlp.gate_up_proj) and _down_ok(mlp.down_proj))):
                if pend is not None:
                    hidden_states = pend(hidden_states)  # the previous layer's post_mlp_layernorm, as SGLang runs it
                return orig_l(self, positions, hidden_states, forward_batch, residual)
            use_f16 = getattr(self, "_t2_f16_lm", None)
            if use_f16 is None:
                if _F16_LM and not torch.cuda.is_current_stream_capturing() and _down_ok(at.o_proj):
                    _G.rescale_f16_weight(at.qkv_proj)
                    _G.rescale_f16_weight(at.o_proj)
                    _G.rescale_f16_weight(mlp.gate_up_proj)
                    _G.rescale_f16_weight(mlp.down_proj)
                    use_f16 = True
                    if not getattr(L, "_t2_f16_logged", False):
                        L._t2_f16_logged = True
                        _be = "cutlass_f16+triton" if getattr(_G, "_CUTLASS_F16", None) and _G._CUTLASS_F16.available() else "triton"
                        print(f"[track2-f16acc] LM ON (QNT=3, shift=32, {_be})", flush=True)
                else:
                    use_f16 = False
                self._t2_f16_lm = use_f16
            nq_div = (C.NORM_DIV | 2) if use_f16 else None
            smq_qnt = 3 if use_f16 else None
            lin_fn = lin_f16 if use_f16 else lin
            dt = hidden_states.dtype
            ln = self.input_layernorm
            if residual is None:
                residual = hidden_states
                q8, s = C.norm_quant_fp8(hidden_states, ln.weight, ln.variance_epsilon, mode=C.NORM_MODE_PLAIN, div=nq_div)
            elif pend is not None:
                # == previous layer's post_mlp_layernorm(down), then fused_add_rmsnorm(., residual) of this layer
                q8, s = C.norm_quant_fp8(hidden_states, ln.weight, ln.variance_epsilon, residual=residual,
                                         mode=C.NORM_MODE_PRE_ADD, weight0=pend.weight, eps0=pend.variance_epsilon, div=nq_div)
            else:
                # == fused_add_rmsnorm(hidden_states, residual): residual += hidden_states (in place); the normalized rows
                # go straight to FP8 (SGLang also overwrote hidden_states with them; nothing reads that buffer again)
                q8, s = C.norm_quant_fp8(hidden_states, ln.weight, ln.variance_epsilon, residual=residual,
                                         mode=C.NORM_MODE_ADD, div=nq_div)
            qkv = lin_fn(at.qkv_proj, q8, s, dt)
            q, k, v = qkv.split([at.q_size, at.kv_size, at.kv_size], dim=-1)
            _saved_kv = False
            if (_MROPE and os.environ.get("TRACK2_FUSED_MROPE_KV", "1") == "1"
                    and forward_batch.forward_mode.is_target_verify()
                    and not getattr(at.attn, "is_cross_attention", False)
                    and not getattr(getattr(forward_batch.attn_backend, "forward_metadata", None), "use_ragged", True)
                    and getattr(at.attn, "k_scale", None) is None
                    and getattr(at.attn, "v_scale", None) is None):
                from gravity_ocr import fast_mrope as _F
                _re = at.rotary_emb
                _re._match_cos_sin_cache_dtype(q)
                _kvp = forward_batch.token_to_kv_pool
                _lid = at.attn.layer_id - _kvp.start_layer
                _kc = _kvp.k_buffer[_lid]
                _vc = _kvp.v_buffer[_lid]
                if (_kvp.dtype == dt and _kvp.store_dtype == dt and _kc.stride(-1) == 1 and _vc.stride(-1) == 1
                        and _kc.stride(-2) == _re.head_size and _vc.stride(-2) == _re.head_size
                        and v.stride(-1) == 1 and forward_batch.out_cache_loc is not None
                        and _F.supported(q, k, _re.cos_sin_cache, positions, _re.mrope_section, _re.head_size,
                                         _re.rotary_dim, _re.mrope_interleaved, _re.mrope_interleaved_glm, _re.is_neox_style)):
                    q = _F.mrope_kv_scatter(
                        q, k, v, _kc, _vc, forward_batch.out_cache_loc,
                        _re.cos_sin_cache, positions, _re.mrope_section, _re.head_size, _re.rotary_dim,
                    )
                    _saved_kv = True
            if not _saved_kv:
                q, k = at.rotary_emb(positions, q, k)
            ao = at.attn(q, k, v, forward_batch, save_kv_cache=not _saved_kv)
            if use_f16:
                q8, s = C.quant_fp8_rows(ao, qnt_mode=3)
                o = lin_f16(at.o_proj, q8, s, dt)
            else:
                o, _ = at.o_proj(ao)
            q8, s = C.norm_quant_fp8(o, self.post_attention_layernorm.weight, self.post_attention_layernorm.variance_epsilon,
                                     residual=residual, mode=C.NORM_MODE_PRE_ADD,
                                     weight0=self.post_self_attn_layernorm.weight,
                                     eps0=self.post_self_attn_layernorm.variance_epsilon, div=nq_div)
            gate_up = lin_fn(mlp.gate_up_proj, q8, s, dt)
            q8, s = C.silu_mul_quant_fp8(gate_up, qnt_mode=smq_qnt)
            down = lin_fn(mlp.down_proj, q8, s, dt)
            if st is not None and _CHAIN and not getattr(self, "_t2_last", True):
                st.pending = self.post_mlp_layernorm  # applied inside the next layer's input kernel
                return down, residual
            return self.post_mlp_layernorm(down), residual

        _patch(L, "forward", layer_forward)
        n += 1

    B = glm_ocr.GlmOcrVisionBlock
    if not getattr(B.forward, "_t2_fused", False) and os.environ.get("TRACK2_FUSED_NORM_VIT", "1") == "1":
        orig_b = B.forward

        def block_forward(self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin):
            st = getattr(self, "_t2_chain", None)
            pend = None
            if st is not None:
                pend, st.pending = st.pending, None
                if getattr(self, "_t2_first", False):  # first block of a forward: drop a stale step
                    pend = None
            at, mlp = self.attn, self.mlp
            # Rescale all 4 ViT block projections (qkv, proj, gate_up, down) once on the first non-capturing call
            # so CUDA graph capture and packed eager batches both run Ada sm_89 FP16-accum GEMMs!
            if (_F16_VIT and not getattr(self, "_t2_f16_vit", False)
                    and not torch.cuda.is_current_stream_capturing()
                    and _down_ok(at.qkv_proj) and _down_ok(mlp.gate_up_proj) and _down_ok(mlp.down_proj)):
                _G.rescale_f16_weight(at.qkv_proj)
                _G.rescale_f16_weight(mlp.gate_up_proj)
                _G.rescale_f16_weight(mlp.down_proj)
                _proj_ok = _down_ok(at.proj)
                if _proj_ok:
                    _G.rescale_f16_weight(at.proj)
                    at._t2_f16_proj = True
                    at._t2_cuda_mod = C
                    at._t2_f16_mm = _G.mm_f16
                self._t2_f16_vit = True
                if not getattr(B, "_t2_f16_logged", False):
                    B._t2_f16_logged = True
                    _be = "cutlass_f16+triton" if getattr(_G, "_CUTLASS_F16", None) and _G._CUTLASS_F16.available() else "triton"
                    print(f"[track2-f16acc] ViT ON (qkv+{'proj+' if _proj_ok else ''}gate_up+down: QNT=3, shift=32, {_be})", flush=True)
            if (x.dim() != 3 or x.shape[1] != 1 or x.shape[0] == 0 or x.dtype != torch.bfloat16
                    or x.shape[-1] not in (1024, 1536) or not x.is_contiguous()
                    or (pend is not None and (pend.shape != x.shape or not pend.is_contiguous()))
                    or not getattr(at, "_track2_k2_inplace", False)
                    or not (_down_ok(at.qkv_proj) and _down_ok(mlp.gate_up_proj) and _down_ok(mlp.down_proj))):
                if pend is not None:
                    x = x + pend  # the previous block's `x_after_add + mlp_out`, as SGLang runs it
                return orig_b(self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin)
            use_f16 = getattr(self, "_t2_f16_vit", False)
            nq_div = (C.NORM_DIV | 2) if use_f16 else None
            smq_qnt = 3 if use_f16 else None
            lin_fn = lin_f16 if use_f16 else lin
            S, _, H = x.shape
            x2d = x.view(S, H)
            if pend is not None:
                # == the previous block's `x = x_after_add + mlp_out` (bf16 torch add), then norm1(x) + FP8 quant;
                # the sum is written into pend's buffer, which is this block's input from here on
                p2d = pend.view(S, H)
                q8, s = C.norm_quant_fp8(x2d, self.norm1.weight, self.norm1.variance_epsilon, residual=p2d,
                                         mode=C.NORM_MODE_ROUND_ADD, div=nq_div)
                x2d = p2d
            else:
                q8, s = C.norm_quant_fp8(x2d, self.norm1.weight, self.norm1.variance_epsilon, mode=C.NORM_MODE_PLAIN, div=nq_div)
            at._t2_pre_qkv = lin_fn(at.qkv_proj, q8, s, x.dtype).view(1, S, -1)   # consumed by the patched attention forward
            attn = at(x2d.view(1, S, H), cu_seqlens=cu_seqlens, rotary_pos_emb_cos=rotary_pos_emb_cos,
                      rotary_pos_emb_sin=rotary_pos_emb_sin)
            at.__dict__.pop("_t2_pre_qkv", None)
            attn2d = attn.reshape(S, H)
            if not attn2d.is_contiguous():
                attn2d = attn2d.contiguous()
            # == norm2(x2d, residual=attn2d): attn2d <- x + attn (x_after_add), normalized rows -> FP8
            q8, s = C.norm_quant_fp8(x2d, self.norm2.weight, self.norm2.variance_epsilon, residual=attn2d,
                                     mode=C.NORM_MODE_ADD, div=nq_div)
            if use_f16 and _SS_FUSED_SILU_VIT and (_ss_fsilu.WARM or not torch.cuda.is_current_stream_capturing()):
                # team taktak: gate_up GEMM + SiLU*up fused (gravity_ocr/ss_fused_silu.py), then per-row FP8 quant.
                # The first launch (Triton compile) never happens inside a CUDA-graph capture.
                gu = mlp.gate_up_proj
                h = _ss_fsilu.gate_up_silu(q8, gu.weight, s, gu.weight_scale, gu.bias)
                _ss_fsilu.WARM = True
                q8, s = C.quant_fp8_rows(h, qnt_mode=smq_qnt)
            else:
                gate_up = lin_fn(mlp.gate_up_proj, q8, s, x.dtype)
                q8, s = C.silu_mul_quant_fp8(gate_up, qnt_mode=smq_qnt)
            mlp_out = lin_fn(mlp.down_proj, q8, s, x.dtype)
            if st is not None and _CHAIN and not getattr(self, "_t2_last", True):
                st.pending = mlp_out.view(S, 1, H)  # added inside the next block's norm1 kernel
                return attn2d.view(S, 1, H)
            return (attn2d + mlp_out).view(S, 1, H)

        _patch(B, "forward", block_forward)
        n += 1
    return n


def _mark_chains(model) -> str:
    """Give every LM decoder-layer stack / ViT block stack of `model` a shared _Chain and first / last flags."""
    if model is None or not _CHAIN:
        return "unchained"
    from sglang.srt.models import glm4, glm_ocr

    out = []
    for m in model.modules():
        if isinstance(m, glm4.Glm4Model) and not getattr(m, "layers_to_capture", None):
            layers = [x for x in m.layers if isinstance(x, glm4.Glm4DecoderLayer)]
            if len(layers) == len(m.layers) and layers:
                st = _Chain()
                for i, layer in enumerate(layers):
                    layer._t2_chain, layer._t2_last = st, i == len(layers) - 1
                out.append(f"LM {len(layers)} layers")
        if isinstance(m, glm_ocr.GlmOcrVisionModel):
            blocks = [b for b in m.blocks if isinstance(b, glm_ocr.GlmOcrVisionBlock)]
            if len(blocks) == len(m.blocks) and blocks:
                st = _Chain()
                for i, b in enumerate(blocks):
                    b._t2_chain, b._t2_first, b._t2_last = st, i == 0, i == len(blocks) - 1
                out.append(f"ViT {len(blocks)} blocks")
    return "chained " + " + ".join(out) if out else "unchained"


def _install_verify() -> str:
    """TRACK2_FUSED_VERIFY=1: compare eager forwards with SGLang's original code, bitwise (diagnostic)."""
    from sglang.srt.models import glm4, glm_ocr

    def report(tag, k, a, b, extra):
        a = a[0] if isinstance(a, (tuple, list)) else a
        b = b[0] if isinstance(b, (tuple, list)) else b
        same = bool(torch.equal(a, b))
        d = 0.0 if same else float((a.float() - b.float()).abs().max())
        print(f"[track2-fused-verify] {tag} eager forward #{k} {extra}: bitwise={same}"
              + ("" if same else f" max|diff|={d:.4g} differing={int((a != b).sum())}/{a.numel()}"), flush=True)

    cnt = {"lm": 0, "vit": 0}
    M = glm4.Glm4Model
    orig_m = M.forward

    def model_forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None, **kw):
        if cnt["lm"] >= _VERIFY_N or torch.cuda.is_current_stream_capturing():
            return orig_m(self, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors, **kw)
        cnt["lm"] += 1
        emb = input_embeds.clone() if input_embeds is not None else None  # layer 0 updates its residual in place
        out = orig_m(self, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors, **kw)
        with _originals():
            ref = orig_m(self, input_ids, positions, forward_batch, emb, pp_proxy_tensors, **kw)
        report("LM", cnt["lm"], out, ref, f"tokens={int(positions.shape[-1])} mode={forward_batch.forward_mode.name}")
        return out

    V = glm_ocr.GlmOcrVisionModel
    orig_v = V.forward

    def vision_forward(self, x, grid_thw):
        if cnt["vit"] >= _VERIFY_N or torch.cuda.is_current_stream_capturing():
            return orig_v(self, x, grid_thw)
        cnt["vit"] += 1
        x_ref = x.clone()
        out = orig_v(self, x, grid_thw)
        with _originals():
            ref = orig_v(self, x_ref, grid_thw)
        report("ViT", cnt["vit"], out, ref, f"images={int(grid_thw.shape[0])} patches={int(x.shape[0])}")
        return out

    M.forward = model_forward  # not in _PATCHES: the wrappers stay while _originals() runs the reference
    V.forward = vision_forward
    return f"VERIFY first {_VERIFY_N} eager LM / ViT forwards"


def install(model=None) -> None:
    global _INSTALLED
    if _INSTALLED or not _ON:
        if not _ON:
            print("[track2-fused] OFF (TRACK2_FUSED=0)", flush=True)
        return
    _INSTALLED = True
    from gravity_ocr import cuda_sm89 as C

    parts = []
    try:  # must precede _install_mlp / _install_norm: they capture fp8_utils.fp8_scaled_mm
        from sglang.srt.layers.quantization import fp8_utils as fu

        from gravity_ocr import fp8_gemm

        orig_mm = fu.fp8_scaled_mm
        parts.append(fp8_gemm.install())
        if fu.fp8_scaled_mm is not orig_mm:  # registered so verify mode can swap CUTLASS back in
            _PATCHES.append((fu, "fp8_scaled_mm", orig_mm, fu.fp8_scaled_mm))
    except Exception as e:
        parts.append(f"Triton FP8 GEMM unavailable ({e})")
    if _MLP:
        if C.fused_available():
            parts.append(f"silu*up+fp8-quant MLP ({_install_mlp(C)} classes)")
        else:
            parts.append("MLP fusion unavailable (libtrack2_cuda_sm89.so without fused kernels)")
    if _MROPE:
        try:
            _install_mrope()
            parts.append("fast M-RoPE")
        except Exception as e:  # triton / signature drift: keep SGLang's kernel
            parts.append(f"M-RoPE fusion unavailable ({e})")
    if _NORM and _MLP:
        if C.norm_available():
            try:
                parts.append(f"norm+fp8-quant layers ({_install_norm(C)} classes, {_mark_chains(model)})")
            except Exception as e:
                parts.append(f"norm fusion unavailable ({e})")
        else:
            parts.append("norm fusion unavailable (libtrack2_cuda_sm89.so without track2_norm_quant_fp8)")
    if _VERIFY:
        try:
            parts.append(_install_verify())
        except Exception as e:
            parts.append(f"verify unavailable ({e})")
    print(f"[track2-fused] ON: {', '.join(parts) or 'nothing'}", flush=True)
