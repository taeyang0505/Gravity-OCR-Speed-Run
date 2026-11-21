"""Keep Python's cyclic garbage collector out of the timed job.

The server processes hold millions of long-lived objects after model load and CUDA-graph capture; a generation-2
collection walks all of them and stops the process for hundreds of ms (seen as a +0.35-0.4 s stall inside a job).
freeze() moves everything alive now into the permanent generation and turns automatic collection off; reference
counting still frees ordinary objects immediately, only reference cycles created afterwards stay until process exit
(the harness boots a fresh process for every timed run, so that is a few jobs' worth at most).

Call freeze(tag) once per process after its startup work: the scheduler on its first request, the tokenizer manager on
its first image, the client before READY. TRACK2_GC_FREEZE=0 turns it off.
"""
from __future__ import annotations

import gc
import os

_DONE = False


def freeze(tag: str) -> bool:
    global _DONE
    if _DONE:
        return False
    _DONE = True
    if os.environ.get("TRACK2_GC_FREEZE", "1") != "1":
        return False
    try:
        gc.collect()
        gc.freeze()
        gc.disable()
        print(f"[track2-gc] {tag} (pid {os.getpid()}): {gc.get_freeze_count()} objects frozen, automatic GC off", flush=True)
        return True
    except Exception as e:  # pragma: no cover
        print(f"[track2-gc] {tag}: not applied ({e})", flush=True)
        return False
