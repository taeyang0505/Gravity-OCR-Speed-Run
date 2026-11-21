"""[TRACK2-JIT-STABLE] No Triton compile inside a timed job (TRACK2_JIT_STABLE=1, the default; 0 = stock behaviour).

Triton builds one kernel per specialization of the launch arguments: a tensor argument whose data pointer is a multiple of
16 selects another variant than an unaligned one (and an integer that is 1 or a multiple of 16 another one than any other
value). SGLang's index-copy kernels below take `out_cache_loc`, a slice of the KV allocator's free list whose start is
aligned or not depending on how many slots were handed out before, so each of them has variants that are compiled lazily
(45 - 80 ms on the scheduler thread) inside whatever prefill / decode round first meets them. The untimed warm-up job
meets most of them, never provably all: measured on the dev set, every cold timed job compiled at least one
(assign_req_to_token_pool, bs_upper 32, unaligned), and the batch-size tail (32 -> 16 -> 8 -> 4 -> 2 -> 1) can meet up to
five more at the very end of the job, where the GPU has nothing else to do.

install() marks the pointer arguments of those kernels "do not specialise on alignment" (Triton's decorator option of the
same name), in place, before their first launch. Each then has ONE variant: the code SGLang already runs for unaligned
pointers. They only copy int32 / int64 indices, so the vector-load hint they give up is worth nothing, and the values they
write are the same. (The batch-size bucket of assign_req_to_token_pool is fixed in speculative/ngram_info.py; the
FP8 GEMM's token-count variants in gravity_ocr/fp8_gemm.py.)
"""
from __future__ import annotations

import importlib
import os

ON = os.environ.get("TRACK2_JIT_STABLE", "1") == "1"

KERNELS = (
    ("sglang.srt.mem_cache.common", "write_req_to_token_pool_triton"),             # prefill batch preparation
    ("sglang.srt.speculative.spec_utils", "assign_req_to_token_pool"),             # every decode round (prepare_for_verify)
    ("sglang.srt.layers.attention.utils", "create_flashinfer_kv_indices_triton"),  # prefill attention metadata
    ("sglang.srt.model_executor.forward_batch_info", "compute_position_kernel"),   # prefill positions
)
_DONE = False


def install() -> str:
    """Call once in the scheduler process before the first request (models/glm_ocr.py load_weights)."""
    global _DONE
    if not ON:
        msg = "OFF (TRACK2_JIT_STABLE=0)"
        print(f"[track2-jit-stable] {msg}", flush=True)
        return msg
    if _DONE:
        return "already installed"
    _DONE = True
    parts = []
    for mod, name in KERNELS:
        try:
            fn = getattr(importlib.import_module(mod), name)
            n = 0
            for p in fn.params:
                if not p.is_constexpr and not p.do_not_specialize_on_alignment:
                    p.do_not_specialize_on_alignment = True
                    n += 1
            fn.device_caches.clear()  # the argument binder (and any kernel built with the old flags) is rebuilt on the next launch
            parts.append(f"{name} ({n} args)")
        except Exception as e:  # another SGLang / Triton layout: that kernel keeps its lazy variants
            parts.append(f"{name} skipped ({type(e).__name__}: {e})")
    msg = "ON: alignment variants off for " + ", ".join(parts)
    print(f"[track2-jit-stable] {msg}", flush=True)
    return msg
