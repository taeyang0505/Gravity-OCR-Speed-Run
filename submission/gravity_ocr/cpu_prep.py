"""Fast C CPU image-prep wrapper (`libtrack2_cpu_prep.so`): glyph_size, ink_box, single-pass crop->shm CHW copy."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_LIB: Optional[ctypes.CDLL] = None
_TRIED = False


def load() -> Optional[ctypes.CDLL]:
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    if os.environ.get("TRACK2_CPU_PREP_C", "1") != "1":
        return None
    p = Path(__file__).resolve().parent / "_native" / "libtrack2_cpu_prep.so"
    if not p.is_file():
        return None
    try:
        lib = ctypes.CDLL(str(p))
        u8p = ctypes.c_void_p
        i32, i64, f64p = ctypes.c_int, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)
        lib.t2_glyph_size_u8.argtypes = [u8p, i32, i32, i64, i64, f64p, f64p, f64p]
        lib.t2_glyph_size_u8.restype = i32

        lib.t2_ink_box_u8.argtypes = [
            u8p, i32, i32, i32, i32,
            i64, i64, i64,
            u8p, u8p, u8p, u8p,
            i32, i32, ctypes.POINTER(ctypes.c_int),
        ]
        lib.t2_ink_box_u8.restype = i32

        lib.t2_copy_crop_chw_u8.argtypes = [u8p, i64, i64, i64, i32, i32, i32, i32, u8p]
        lib.t2_copy_crop_chw_u8.restype = None
        _LIB = lib
    except Exception:
        _LIB = None
    return _LIB


def glyph_size_c(g: np.ndarray) -> Optional[Tuple[Optional[float], str]]:
    """Call C t2_glyph_size_u8 on a 2D uint8 array (any stride). Returns None if library unavailable."""
    lib = load()
    if lib is None or g.ndim != 2 or g.dtype != np.uint8:
        return None
    h, w = g.shape
    sh, sw = g.strides
    sz, a, b = ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0)
    ok = lib.t2_glyph_size_u8(
        ctypes.c_void_p(g.ctypes.data), int(h), int(w), int(sh), int(sw),
        ctypes.byref(sz), ctypes.byref(a), ctypes.byref(b),
    )
    if not ok:
        return None, "c_rej"
    return float(sz.value), f"a={a.value:.0f} b={b.value:.0f}"


def ink_box_c(
    a: np.ndarray,
    y_start: int, y_end: int, x_start: int, x_end: int,
    lo_f: bytes, hi_f: bytes, lo_s: bytes, hi_s: bytes,
    m: int, reach: int,
) -> Optional[Tuple[int, int, int, int, int, int, int, int]]:
    """Call C t2_ink_box_u8 on a 3-channel uint8 array [3, H, W] (any stride)."""
    lib = load()
    if lib is None or a.ndim != 3 or a.shape[0] != 3 or a.dtype != np.uint8:
        return None
    sc, sh, sw = a.strides
    out = (ctypes.c_int * 8)()
    ok = lib.t2_ink_box_u8(
        ctypes.c_void_p(a.ctypes.data),
        int(y_start), int(y_end), int(x_start), int(x_end),
        int(sc), int(sh), int(sw),
        lo_f, hi_f, lo_s, hi_s,
        int(m), int(reach), out,
    )
    if not ok:
        return (-1, -1, -1, -1, -1, -1, -1, -1)
    return tuple(int(out[i]) for i in range(8))


def copy_crop_chw_c(src: np.ndarray, y0: int, y1: int, x0: int, x1: int, dst_ptr: int) -> bool:
    """Single-pass crop + copy from strided [3, H, W] uint8 src into contiguous CHW dst_ptr."""
    lib = load()
    if lib is None or src.ndim != 3 or src.shape[0] != 3 or src.dtype != np.uint8:
        return False
    sc, sh, sw = src.strides
    lib.t2_copy_crop_chw_u8(
        ctypes.c_void_p(src.ctypes.data),
        int(sc), int(sh), int(sw),
        int(y0), int(y1), int(x0), int(x1),
        ctypes.c_void_p(dst_ptr),
    )
    return True
