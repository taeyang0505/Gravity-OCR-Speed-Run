"""Margin crop box for Track 2 crops (numpy only; NOT integrated anywhere yet).

Every LM image token is one 28x28 px cell of the resized crop, so blank paper around the ink costs ViT patches, ViT
attention (quadratic per image) and LM prefill tokens.  crop_box() returns a tighter crop rectangle: the bounding box of
everything that does not look like the margin of the crop, plus padding.  A tighter crop box is an ordinary input for
the model (the page-layout crops of the dev set have a median margin of 7-8 px); nothing inside the image is removed
and no patch position is skipped.

How the box is found
  paper    per-channel median of a BAND px ring just inside the border of the crop: the colour of the margin itself
           (a global percentile would call the card / zebra shading of a table on a tinted page "paper");
  noise    75th percentile of |ring - paper| per side, second smallest of the four sides (two sides may be full of
           ink): 0 on a clean render, 1-5 on JPEG / noisy paper;
  faint    pixels more than tol = TOL + NOISE_K * noise away from the paper in any channel.  On clean paper (tol 3)
           every faint rule, header / zebra shade and anti-aliasing halo is content; paper that would need more than
           TOL_MAX is not cropped at all.  On noisy paper a row / column needs NOISY_COUNT such pixels (or one
           strong pixel) to count, so single noise outliers do not block the crop;
  strong   pixels more than STRONG away from the paper: strokes and rules;
  box      union of (faint extent + PAD_FAINT) and (strong extent + PAD), found by scanning rows and columns from
           the outside in (only the margins are read).  The outer EDGE px of the crop are left out of that scan,
           because some renders draw a faint sheet-edge line there; when the result is a crop and that frame holds
           a strong pixel, the scan is redone with the frame as ordinary content.

Safety gates: each returns the full image, i.e. the request is served exactly as it is today
  off / task not enabled; crop too small; dark or inverted paper (luminance < MIN_PAPER); noisy or textured paper
  (photo background); strokes in the ring on all four sides (already tight); nothing but paper; a saving below
  MIN_GAIN of the patches and below MIN_SAVE patches.
The box is never smaller than MIN_SIDE px per side nor MIN_AREA px^2, so the resize never magnifies a tiny crop.
Grid snap (SNAP=1, crops served at native scale): each side length is moved onto a multiple of 28 px, where the
resample is the identity: grown when smart_resize would round it up, shrunk when the padding can give the pixels
(never below PAD_MIN around strokes), and grown to the next multiple when the resize would otherwise squeeze the
content by more than MAX_SQUEEZE (so a crop is never rendered smaller than MAX_SQUEEZE below native size unless the
uncropped image is squeezed at least as much).

Integration sketch (overlay multimodal/processors/glm4v.py, _fast_prepare_shm_or_bf16, after the decode):
    s, _why = _g_textscale.text_downscale(chw, task)            # decided on the FULL crop, exactly as today
    (y0, y1, x0, x1), _why = margin_crop.crop_box(chw, task, s)
    chw = chw[:, y0:y1, x0:x1]                                  # a view; _write_shm_ring copies it contiguously
    _, h, w = chw.shape                                         # then smart_resize / scaled_size as today

Env: TRACK2_MARGIN_CROP (1 = on), TRACK2_MARGIN_TASKS (text,table,formula), TRACK2_MARGIN_PAD (8),
     TRACK2_MARGIN_PAD_FAINT (2), TRACK2_MARGIN_PAD_MIN (4), TRACK2_MARGIN_TOL (3), TRACK2_MARGIN_TOL_MAX (20),
     TRACK2_MARGIN_NOISE_K (3), TRACK2_MARGIN_STRONG (24), TRACK2_MARGIN_NOISY_COUNT (4),
     TRACK2_MARGIN_MIN_GAIN (0.05), TRACK2_MARGIN_MIN_SAVE (64), TRACK2_MARGIN_MIN_PAPER (150),
     TRACK2_MARGIN_SNAP (1), TRACK2_MARGIN_MAX_SQUEEZE (0.04).
"""
from __future__ import annotations

import math
import os
import re
from typing import Tuple

import numpy as np

try:
    from gravity_ocr import cpu_prep as _cpu_prep
except Exception:
    _cpu_prep = None

ENABLED = os.environ.get("TRACK2_MARGIN_CROP", "1") == "1"
TASKS = tuple(t.strip() for t in re.split(r"[,:;]", os.environ.get("TRACK2_MARGIN_TASKS", "text,table,formula")) if t.strip())
PAD = int(os.environ.get("TRACK2_MARGIN_PAD", "8"))                # served px kept around strokes
PAD_FAINT = int(os.environ.get("TRACK2_MARGIN_PAD_FAINT", "2"))    # served px kept around faint content
PAD_MIN = int(os.environ.get("TRACK2_MARGIN_PAD_MIN", "4"))        # grid snap may shrink PAD down to this
TOL = int(os.environ.get("TRACK2_MARGIN_TOL", "3"))                # gray levels: faint threshold on clean paper
TOL_MAX = int(os.environ.get("TRACK2_MARGIN_TOL_MAX", "20"))       # paper that needs more is not cropped
NOISE_K = float(os.environ.get("TRACK2_MARGIN_NOISE_K", "3"))      # tol = TOL + NOISE_K * noise
STRONG = int(os.environ.get("TRACK2_MARGIN_STRONG", "24"))         # gray levels: a stroke / rule for sure
NOISY_COUNT = int(os.environ.get("TRACK2_MARGIN_NOISY_COUNT", "4"))  # faint pixels per line on noisy paper
MIN_GAIN = float(os.environ.get("TRACK2_MARGIN_MIN_GAIN", "0.05"))   # crop when >= this share of the patches goes
MIN_SAVE = int(os.environ.get("TRACK2_MARGIN_MIN_SAVE", "64"))       # ... or at least this many patches
MIN_PAPER = int(os.environ.get("TRACK2_MARGIN_MIN_PAPER", "150"))    # paper luminance below this: no crop
SNAP = os.environ.get("TRACK2_MARGIN_SNAP", "1") == "1"
MAX_SQUEEZE = float(os.environ.get("TRACK2_MARGIN_MAX_SQUEEZE", "0.04"))
EDGE = 2            # outer px that may hold a faint sheet-edge line
BAND = 4            # px of the ring (inside EDGE) that defines paper colour and noise
RING_SAMPLES = int(os.environ.get("TRACK2_MARGIN_RING_SAMPLES", "128"))  # ring pixels read per side and ring line
BLOCK = int(os.environ.get("TRACK2_MARGIN_BLOCK", "32"))                 # lines of the first scan block
MIN_SIDE = 56       # served px (2 cells): the box is never smaller, per side
MIN_AREA = 12544    # served px^2 (16 cells): keeps the MIN_PIXELS upscale of smart_resize out of play
FACTOR = 28
_MIN_PIXELS = 12_544
_MAX_PIXELS = 9_633_792
_TEMPORAL = 2

Box = Tuple[int, int, int, int]  # y0, y1, x0, x1 (half-open), in pixels of the input image


def smart_resize(height: int, width: int) -> Tuple[int, int]:
    """Same rule as gravity_ocr/preprocess.py::smart_resize (kept here so this file stands alone)."""
    if height < FACTOR or width < FACTOR:
        scale = max(FACTOR / height, FACTOR / width)
        height, width = int(height * scale), int(width * scale)
    h_bar = round(height / FACTOR) * FACTOR
    w_bar = round(width / FACTOR) * FACTOR
    if _TEMPORAL * h_bar * w_bar > _MAX_PIXELS:
        beta = math.sqrt(_TEMPORAL * height * width / _MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif _TEMPORAL * h_bar * w_bar < _MIN_PIXELS:
        beta = math.sqrt(_MIN_PIXELS / (_TEMPORAL * height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def served_patches(h: int, w: int, scale: float = 1.0) -> int:
    """ViT patches (14 px) the serving path spends on an h x w crop that is served at `scale` (text downscale)."""
    if scale < 1.0:
        h = max(1, int(math.floor(h * scale + 0.5)))
        w = max(1, int(math.floor(w * scale + 0.5)))
    th, tw = smart_resize(h, w)
    return (th // 14) * (tw // 14)


def _as_chw(img) -> np.ndarray:
    """uint8 view [C, H, W] (C = 1 or 3) of a CHW / HWC / HW image (numpy array or CPU torch tensor); no copy."""
    a = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if a.ndim == 2:
        a = a[None]
    elif a.ndim == 3 and a.shape[2] <= 4 and not (a.shape[0] <= 4 < a.shape[2]):
        a = a.transpose(2, 0, 1)  # HWC -> CHW view
    if a.ndim != 3 or a.dtype != np.uint8:
        raise ValueError("expected a uint8 image, CHW or HWC")
    return a[:3] if a.shape[0] >= 3 else a[:1]


def _ring(a: np.ndarray):
    """(paper colour per channel, noise, smallest per-side maximum deviation) from the BAND px ring inside the EDGE
    px frame.  At most 2 * BAND * RING_SAMPLES samples per side and channel."""
    c, h, w = a.shape
    e, b = EDGE, BAND
    sx = max(1, (w - 2 * e) // RING_SAMPLES)
    sy = max(1, (h - 2 * (e + b)) // RING_SAMPLES)
    strips = (
        a[:, e : e + b, e : w - e : sx].reshape(c, -1),
        a[:, h - e - b : h - e, e : w - e : sx].reshape(c, -1),
        a[:, e + b : h - e - b : sy, e : e + b].reshape(c, -1),
        a[:, e + b : h - e - b : sy, w - e - b : w - e].reshape(c, -1),
    )
    v = np.concatenate(strips, axis=1)
    n = v.shape[1]
    paper = np.partition(v, n // 2, axis=1)[:, n // 2 : n // 2 + 1]  # [C, 1] uint8
    d = np.maximum(v, paper) - np.minimum(v, paper)  # |v - paper| without leaving uint8
    d = d.max(axis=0) if c > 1 else d[0]
    p75, top = [], []
    off = 0
    for s in strips:
        k = s.shape[1]
        q = (3 * k) // 4
        t = np.partition(d[off : off + k], (q, k - 1))
        p75.append(int(t[q]))
        top.append(int(t[k - 1]))
        off += k
    return paper[:, 0].tolist(), sorted(p75)[1], min(top)


def _strong_at(v: np.ndarray, LO, HI, s: int, e: int, copy: bool) -> np.ndarray:
    """lines s .. e-1 of v [C, N, L] -> bool per line: it has a strong pixel"""
    blk = np.ascontiguousarray(v[:, s:e]) if copy else v[:, s:e]
    return ((blk.min(axis=2) < LO[1]) | (blk.max(axis=2) > HI[1])).any(axis=0)


def _scan(v: np.ndarray, LO, HI, m: int, reach: int, reverse: bool, copy: bool) -> Tuple[int, int]:
    """v [C, N, L] -> (first ink line, first strong line among it and the next reach - 1 lines); -1 = none.

    reverse: the last ink line, and the last strong line among it and the reach - 1 lines before it.
    A line is ink when it has a strong pixel or at least m faint pixels.  LO / HI [2, C, 1] hold the faint and the
    strong bounds.  Lines are read from the outside in, in growing blocks (most margins fit the first one).
    copy: the pixels are channel-interleaved in memory (an HWC image); blocks are then gathered per channel first."""
    n = v.shape[1]
    pos, step = 0, BLOCK
    while pos < n:
        end = min(n, pos + step)
        nb = end - pos
        s0 = n - end if reverse else pos  # the block is lines s0 .. s0 + nb - 1
        blk = np.ascontiguousarray(v[:, s0 : s0 + nb]) if copy else v[:, s0 : s0 + nb]
        mn = blk.min(axis=2)
        mx = blk.max(axis=2)
        r = ((mn < LO) | (mx > HI)).any(axis=1)  # [2, nb]: faint or stronger, strong
        ink = r[0]
        if m > 1:
            cand = np.flatnonzero(ink & ~r[1])
            if cand.size:
                sub = blk[:, cand]
                ink[cand] = ((sub < LO[0][:, :, None]) | (sub > HI[0][:, :, None])).any(axis=0).sum(axis=1) >= m
        j = nb - 1 - int(ink[::-1].argmax()) if reverse else int(ink.argmax())
        if ink[j]:
            idx = s0 + j
            sidx = -1
            if reach > 0:
                if reverse:
                    a0 = max(0, idx - reach + 1)
                    st = r[1, a0 - s0 : j + 1] if a0 >= s0 else _strong_at(v, LO, HI, a0, idx + 1, copy)
                    k = int(st[::-1].argmax())
                    if st[st.size - 1 - k]:
                        sidx = idx - k
                else:
                    a1 = min(n, idx + reach)
                    st = r[1, j : a1 - s0] if a1 <= s0 + nb else _strong_at(v, LO, HI, idx, a1, copy)
                    k = int(st.argmax())
                    if st[k]:
                        sidx = idx + k
            return idx, sidx
        pos, step = end, step * 2
    return -1, -1


def _ink_box(v: np.ndarray, LO, HI, m: int, pad: int, pad_f: int, pad_min: int, copy: bool):
    """Padded ink box of v [C, H, W] in the coordinates of v, not clipped: (y0, y1, x0, x1) and the innermost
    position (ly0, ly1, lx0, lx1) each edge may be moved to; None when v holds nothing but paper."""
    reach = pad - pad_f  # strokes this close to the edge of the faint extent push the box further out than it does
    c_res = None
    if _cpu_prep is not None and v.shape[0] == 3:
        c_res = _cpu_prep.ink_box_c(
            v, 0, int(v.shape[1]), 0, int(v.shape[2]),
            bytes(LO[0, :, 0].tolist()), bytes(HI[0, :, 0].tolist()),
            bytes(LO[1, :, 0].tolist()), bytes(HI[1, :, 0].tolist()),
            int(m), int(reach),
        )
    if c_res is not None:
        fy0, sy0, fy1, sy1, fx0, sx0, fx1, sx1 = c_res
        if fy0 < 0 or fx0 < 0:
            return None
    else:
        fy0, sy0 = _scan(v, LO, HI, m, reach, False, copy)
        if fy0 < 0:
            return None
        fy1, sy1 = _scan(v, LO, HI, m, reach, True, copy)
        vt = v[:, fy0 : fy1 + 1].transpose(0, 2, 1)  # [C, columns, rows of the ink extent]
        fx0, sx0 = _scan(vt, LO, HI, m, reach, False, copy)
        if fx0 < 0:
            return None
        fx1, sx1 = _scan(vt, LO, HI, m, reach, True, copy)
    y0 = ly0 = fy0 - pad_f
    y1 = ly1 = fy1 + 1 + pad_f
    x0 = lx0 = fx0 - pad_f
    x1 = lx1 = fx1 + 1 + pad_f
    if sy0 >= 0:
        y0, ly0 = sy0 - pad, min(ly0, sy0 - pad_min)
    if sy1 >= 0:
        y1, ly1 = sy1 + 1 + pad, max(ly1, sy1 + 1 + pad_min)
    if sx0 >= 0:
        x0, lx0 = sx0 - pad, min(lx0, sx0 - pad_min)
    if sx1 >= 0:
        x1, lx1 = sx1 + 1 + pad, max(lx1, sx1 + 1 + pad_min)
    return y0, y1, x0, x1, ly0, ly1, lx0, lx1


def _frame_is_ink(a: np.ndarray, paper: np.ndarray, strong: int) -> bool:
    """True when the outer EDGE px frame of a [C, H, W] holds a strong pixel (paper: [C, 1] uint8)."""
    c, h, w = a.shape
    e = EDGE
    f = np.concatenate(
        (a[:, :e].reshape(c, -1), a[:, h - e :].reshape(c, -1), a[:, :, :e].reshape(c, -1), a[:, :, w - e :].reshape(c, -1)),
        axis=1,
    )
    return int((np.maximum(f, paper) - np.minimum(f, paper)).max()) > strong


def _grow(lo: int, hi: int, limit: int, size: int) -> Tuple[int, int]:
    """Widen [lo, hi) to `size` px inside [0, limit), as evenly as the image allows."""
    need = min(size, limit) - (hi - lo)
    if need <= 0:
        return lo, hi
    before = min(lo, need // 2)
    after = min(limit - hi, need - before)
    before = min(lo, need - after)
    return lo - before, hi + after


def _snap(lo: int, hi: int, limit: int, flex_lo: int, flex_hi: int) -> Tuple[int, int]:
    """Move the length of [lo, hi) onto a multiple of 28 px (see the module docstring)."""
    size = hi - lo
    target = round(size / FACTOR) * FACTOR
    if target >= size:
        # smart_resize would round up: take the extra pixels from the margin instead of magnifying the content
        return _grow(lo, hi, limit, target) if target <= limit else (0, limit)
    cut = size - target
    if cut <= flex_lo + flex_hi and target >= MIN_SIDE:
        a = min(flex_lo, cut // 2)
        b = min(flex_hi, cut - a)
        a = cut - b
        return lo + a, hi - b
    if cut > MAX_SQUEEZE * size:
        if target + FACTOR <= limit:
            return _grow(lo, hi, limit, target + FACTOR)
        if target * limit < round(limit / FACTOR) * FACTOR * size:
            return 0, limit  # the uncropped side is squeezed less than the cropped one would be
    return lo, hi


def _finish(h: int, w: int, y0: int, y1: int, x0: int, x1: int, ly0: int, ly1: int, lx0: int, lx1: int,
            scale: float) -> Tuple[Box, str]:
    """Minimum size, grid snap and the gain gate.  l* = innermost allowed position of each edge."""
    full: Box = (0, h, 0, w)
    inv = 1.0 / scale if 0.0 < scale < 1.0 else 1.0
    min_side = int(math.ceil(MIN_SIDE * inv))
    y0, y1 = _grow(y0, y1, h, min_side)
    x0, x1 = _grow(x0, x1, w, min_side)
    area = (y1 - y0) * (x1 - x0)
    min_area = MIN_AREA * inv * inv
    if area < min_area:
        f = math.sqrt(min_area / area)
        y0, y1 = _grow(y0, y1, h, int(math.ceil((y1 - y0) * f)))
        x0, x1 = _grow(x0, x1, w, int(math.ceil((x1 - x0) * f)))
    if SNAP and scale >= 1.0:
        y0, y1 = _snap(y0, y1, h, max(0, ly0 - y0), max(0, y1 - ly1))
        x0, x1 = _snap(x0, x1, w, max(0, lx0 - x0), max(0, x1 - lx1))
    if (y0, y1, x0, x1) == full:
        return full, "tight"
    if max(y1 - y0, x1 - x0) > 190 * min(y1 - y0, x1 - x0):
        return full, "aspect"  # the serving smart_resize refuses crops beyond 200:1
    before = served_patches(h, w, scale)
    after = served_patches(y1 - y0, x1 - x0, scale)
    saved = before - after
    if saved <= 0 or (saved < MIN_GAIN * before and saved < MIN_SAVE):
        return full, "nogain"
    return (y0, y1, x0, x1), f"crop {before}->{after}"


def crop_box(img, task: str = "", scale: float = 1.0) -> Tuple[Box, str]:
    """-> ((y0, y1, x0, x1), reason).  The full image box means "do not crop".

    img    uint8 image, CHW (CPU torch tensor or numpy) or HWC / HW numpy; only read, never copied as a whole
    task   "text" / "table" / "formula" (TRACK2_MARGIN_TASKS selects which tasks are cropped)
    scale  the text-downscale factor the crop will be served at (1.0 = native).  Decide it on the FULL crop, as
           today, and pass it here: padding, minimum size and the gain test are then in served pixels.
    """
    a = _as_chw(img)
    c, h, w = a.shape
    full: Box = (0, h, 0, w)
    if not ENABLED or (task and task not in TASKS):
        return full, "off"
    inv = 1.0 / scale if 0.0 < scale < 1.0 else 1.0
    min_side = int(math.ceil(MIN_SIDE * inv))
    if h < 24 or w < 24 or (h <= min_side and w <= min_side):
        return full, "small"

    paper, noise, ring_top = _ring(a)
    lum = (0.299 * paper[0] + 0.587 * paper[1] + 0.114 * paper[2]) if c == 3 else float(paper[0])
    if lum < MIN_PAPER:
        return full, "dark"
    tol = int(math.ceil(TOL + NOISE_K * noise))
    if tol > TOL_MAX:
        return full, "noisy"
    strong = max(STRONG, tol)
    pad = int(math.ceil(PAD * inv))
    pad_f = min(pad, int(math.ceil(PAD_FAINT * inv)))
    pad_min = min(pad, max(pad_f, int(math.ceil(PAD_MIN * inv))))
    e = EDGE
    if ring_top > strong and pad >= e + BAND - 1:
        return full, "tight"  # strokes within EDGE + BAND px of all four sides: the padded box is the whole crop
    m = 1 if noise == 0 else NOISY_COUNT
    LO = np.array([[max(0, p - tol) for p in paper], [max(0, p - strong) for p in paper]], np.uint8)[:, :, None]
    HI = np.array([[min(255, p + tol) for p in paper], [min(255, p + strong) for p in paper]], np.uint8)[:, :, None]

    # HWC memory: every scan block is gathered per channel first (strided min / max reductions are ~10x slower)
    copy = c > 1 and a.strides[2] != 1
    # first without the outer EDGE px: a faint line there (a drawn sheet edge) is margin
    box, why = full, "blank"
    r = _ink_box(a[:, e : h - e, e : w - e], LO, HI, m, pad, pad_f, pad_min, copy)
    if r is not None:
        y0, y1, x0, x1, ly0, ly1, lx0, lx1 = r
        box, why = _finish(h, w, max(0, y0 + e), min(h, y1 + e), max(0, x0 + e), min(w, x1 + e),
                           max(0, ly0 + e), min(h, ly1 + e), max(0, lx0 + e), min(w, lx1 + e), scale)
    if box != full and _frame_is_ink(a, np.array(paper, np.uint8)[:, None], strong):
        # ... unless something strong is drawn in it: then the frame is content like everything else
        y0, y1, x0, x1, ly0, ly1, lx0, lx1 = _ink_box(a, LO, HI, m, pad, pad_f, pad_min, copy)
        box, why = _finish(h, w, max(0, y0), min(h, y1), max(0, x0), min(w, x1),
                           max(0, ly0), min(h, ly1), max(0, lx0), min(w, lx1), scale)
    return box, (f"{why} tol={tol}" if why.startswith("crop") else why)


__all__ = ["crop_box", "served_patches", "smart_resize"]
