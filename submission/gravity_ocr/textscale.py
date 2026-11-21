"""Font-size-aware downscale/upscale for text and table crops (Track 2 speed + accuracy path; formulas untouched).

GLM-OCR's processor keeps every crop at native resolution up to 4.8 MP, so a magazine or exam-paper paragraph scanned
at high DPI (line pitch 90-190 px) costs up to 24k ViT patches, and the ViT's full attention is quadratic in that. Such
crops carry the same text at a fraction of the resolution: the model reads documents whose glyph bodies are ~11-14 px.
Conversely, very small dense Korean text crops (glyph body < 12 px) gain accuracy when moderately upscaled to ~12 px.

Two rules:
1. Line pitch (text crops; legacy fallback). text_downscale() measures the text-line pitch from the horizontal ink
   profile and brings it down to TARGET_PITCH when unambiguous.
2. Glyph size (text and table crops). glyph_size() measures vertical stroke-edge band heights inside narrow (24 px,
   p25) and wide (64 px, p50 / 1.31) vertical strips, taking min(narrow, wide / 1.31) so mixed font sizes track the
   smaller body text and an over-estimate needs both views to fail. Crops above TRACK2_TEXT_GLYPH / TRACK2_TABLE_GLYPH
   are scaled down; tiny low-patch crops below TRACK2_TEXT_GLYPH_UP / TRACK2_TABLE_GLYPH_UP are enlarged up to
   TRACK2_GLYPH_UP_MAX_SCALE.

Env: TRACK2_TEXT_DOWNSCALE (1 = on), TRACK2_TEXT_PITCH (64), TRACK2_TEXT_MIN_PATCHES (4096), TRACK2_TEXT_MIN_SCALE (0.34),
     TRACK2_TEXT_GLYPH (12), TRACK2_TABLE_GLYPH (12), TRACK2_GLYPH_MIN_PATCHES (512), TRACK2_GLYPH_MIN_SCALE (0.25),
     TRACK2_TEXT_GLYPH_UP (12), TRACK2_TABLE_GLYPH_UP (0), TRACK2_GLYPH_UP_MAX_SCALE (1.5),
     TRACK2_GLYPH_UP_MIN_PATCHES (96), TRACK2_GLYPH_UP_MAX_PATCHES (2048).
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np

try:
    from gravity_ocr import cpu_prep as _cpu_prep
except Exception:
    _cpu_prep = None

ENABLED = os.environ.get("TRACK2_TEXT_DOWNSCALE", "1") == "1"
TARGET_PITCH = float(os.environ.get("TRACK2_TEXT_PITCH", "64"))
MIN_PATCHES = int(os.environ.get("TRACK2_TEXT_MIN_PATCHES", "4096"))
MIN_SCALE = float(os.environ.get("TRACK2_TEXT_MIN_SCALE", "0.34"))
TEXT_GLYPH = float(os.environ.get("TRACK2_TEXT_GLYPH", "12"))
TABLE_GLYPH = float(os.environ.get("TRACK2_TABLE_GLYPH", "12"))
FORMULA_GLYPH = float(os.environ.get("TRACK2_FORMULA_GLYPH", "0"))          # team taktak (default off)
FORMULA_GLYPH_MIN_PATCHES = int(os.environ.get("TRACK2_FORMULA_GLYPH_MIN_PATCHES", "256"))
FORMULA_GLYPH_MIN_SCALE = float(os.environ.get("TRACK2_FORMULA_GLYPH_MIN_SCALE", "0.5"))
GLYPH_MIN_PATCHES = int(os.environ.get("TRACK2_GLYPH_MIN_PATCHES", "512"))
GLYPH_MIN_SCALE = float(os.environ.get("TRACK2_GLYPH_MIN_SCALE", "0.25"))
TEXT_GLYPH_UP = float(os.environ.get("TRACK2_TEXT_GLYPH_UP", "12"))
TABLE_GLYPH_UP = float(os.environ.get("TRACK2_TABLE_GLYPH_UP", "0"))
GLYPH_UP_MAX_SCALE = float(os.environ.get("TRACK2_GLYPH_UP_MAX_SCALE", "1.5"))
GLYPH_UP_MIN_PATCHES = int(os.environ.get("TRACK2_GLYPH_UP_MIN_PATCHES", "96"))
GLYPH_UP_MAX_PATCHES = int(os.environ.get("TRACK2_GLYPH_UP_MAX_PATCHES", "2048"))
FACTOR = 28


def _patches(h: int, w: int) -> int:
    hb = max(FACTOR, round(h / FACTOR) * FACTOR)
    wb = max(FACTOR, round(w / FACTOR) * FACTOR)
    return (hb // 14) * (wb // 14)


def _gray(img) -> Tuple[np.ndarray, int, int]:
    """uint8 gray from HWC numpy or CHW torch uint8; columns subsampled to <= ~1024 for speed."""
    if hasattr(img, "numpy"):  # torch CHW
        a = img.numpy()
        h, w = a.shape[1], a.shape[2]
        step = max(1, w // 1024)
        g = a[1, :, ::step]
    else:
        a = np.asarray(img)
        h, w = a.shape[0], a.shape[1]
        step = max(1, w // 1024)
        g = a[:, ::step, 1] if a.ndim == 3 else a[:, ::step]
    return np.ascontiguousarray(g), h, w


def _plane(img) -> np.ndarray:
    """full-resolution green plane (a view) from HWC numpy or CHW torch uint8."""
    if hasattr(img, "numpy"):
        return img.numpy()[1]
    a = np.asarray(img)
    return a[:, :, 1] if a.ndim == 3 else a


def _hw(img) -> Tuple[int, int]:
    if hasattr(img, "numpy"):
        return int(img.shape[1]), int(img.shape[2])
    a = np.asarray(img)
    return int(a.shape[0]), int(a.shape[1])


# ------------------------------------------------------------------------------------------------ rule 2: glyph size
_T_EDGE = 32     # horizontal gradient (grey levels over 2 px) of a stroke edge
_SUB = 8         # sub-strip width; narrow strip = 3 sub-strips (24 px), wide strip = 8 sub-strips (64 px)
_R_WIDE = 1.31   # wide-strip median band / narrow-strip 25th-percentile band on ordinary crops


def _band_heights(on: np.ndarray, hmax: int) -> np.ndarray:
    """heights of the vertical runs of True in on [h, n] (runs never cross strips), limited to [4, hmax]."""
    n = on.shape[1]
    z = np.zeros((1, n), dtype=np.int8)
    d = np.diff(np.concatenate([z, on.view(np.int8), z], axis=0), axis=0).T.ravel()
    ht = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    return ht[(ht >= 4) & (ht <= hmax)]


def _robust_pct(ht: np.ndarray, q: float) -> float:
    med = float(np.median(ht))
    v = ht[(ht >= 0.35 * med) & (ht <= 3.0 * med)]
    return float(np.percentile(v, q))


def _half(g: np.ndarray) -> np.ndarray:
    h2, w2 = g.shape[0] // 2, g.shape[1] // 2
    a = g[: h2 * 2, : w2 * 2].reshape(h2, 2, w2, 2)
    return ((a[:, 0, :, 0].astype(np.uint16) + a[:, 0, :, 1] + a[:, 1, :, 0] + a[:, 1, :, 1] + 2) >> 2).astype(np.uint8)


def glyph_size(g: np.ndarray) -> Tuple[Optional[float], str]:
    """g: uint8 [H, W] -> (glyph body height in px of g, or None when it cannot be measured reliably; diagnostic)."""
    if _cpu_prep is not None:
        r = _cpu_prep.glyph_size_c(g)
        if r is not None:
            return r
    f = 1
    while g.shape[0] * g.shape[1] > 2_000_000 and f < 4:
        g = _half(g)
        f *= 2
    h, w = g.shape
    n8 = (w - 2) // _SUB
    if h < 12 or n8 < 8:
        return None, "tiny"
    e = np.subtract(g[:, 2:], g[:, :-2], dtype=np.int16)
    np.abs(e, out=e)
    e = e > _T_EDGE
    dens = float(e.mean())
    if not (0.004 <= dens <= 0.30):
        return None, f"dens{dens:.3f}"
    p8 = e[:, : n8 * _SUB].reshape(h, n8, _SUB).sum(axis=2, dtype=np.int16)
    hmax = max(8, int(0.8 * h))
    hts = []
    for k, dmax in ((3, 0.80), (8, 0.90)):
        nk = n8 // k
        p = p8[:, : nk * k].reshape(h, nk, k).sum(axis=2)
        on = p > (np.percentile(p, 10, axis=0)[None, :] + 1)
        duty = float(on.mean())
        if duty > dmax:
            return None, f"duty{duty:.2f}"
        hts.append(_band_heights(on, hmax))
    if len(hts[0]) < 8 or len(hts[1]) < 3:
        return None, f"few{len(hts[0])}/{len(hts[1])}"
    a = _robust_pct(hts[0], 25)
    b = _robust_pct(hts[1], 50)
    return f * min(a, b / _R_WIDE), f"a={f * a:.0f} b={f * b:.0f}"


def _glyph_scale(img, target: float, up: float, h: int, w: int) -> Tuple[float, str]:
    """target: glyph bodies above it are brought down to it; up: glyph bodies below it are enlarged to it (0 = off)."""
    p = _patches(h, w)
    down_ok = target > 0 and p >= GLYPH_MIN_PATCHES
    up_ok = up > 0 and GLYPH_UP_MIN_PATCHES <= p <= GLYPH_UP_MAX_PATCHES
    if not (down_ok or up_ok):
        return 1.0, ""
    size, why = glyph_size(_plane(img))
    if size is None:
        return 1.0, "g:" + why
    if down_ok and size > target:
        s = max(GLYPH_MIN_SCALE, target / size)
        if s <= 0.9:
            return s, f"g:{size:.1f} {why}"
    elif up_ok and size < up:
        s = min(GLYPH_UP_MAX_SCALE, up / size)
        if s >= 1.1:
            return s, f"g:up {size:.1f} {why}"
    return 1.0, f"g:fine {size:.1f} {why}"


# ------------------------------------------------------------------------------------------------ rule 1: line pitch
def _pitch_scale(img, task: str) -> Tuple[float, str]:
    if not ENABLED or task != "text":
        return 1.0, "off"
    h0, w0 = _hw(img)
    if _patches(h0, w0) < MIN_PATCHES:
        return 1.0, "small"
    g, h, w = _gray(img)
    paper = float(np.percentile(g, 90))
    if paper < 150:
        return 1.0, f"paper{paper:.0f}"
    ink = g < (paper - 70)
    cover = float(ink.mean())
    if not (0.01 <= cover <= 0.40):
        return 1.0, f"cover{cover:.3f}"
    row = ink.mean(axis=1)
    thr = max(0.004, 0.08 * float(np.percentile(row, 95)))
    on = row > thr
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    hts = ends - starts
    keep = hts >= 4
    starts, ends, hts = starts[keep], ends[keep], hts[keep]
    n = int(len(starts))
    if n == 0:
        return 1.0, "nolines"
    if n >= 3:
        pitches = np.diff(starts).astype(np.float64)
        pitch = float(np.median(pitches))
        cv = float(np.std(pitches) / max(pitch, 1.0))
        lh = float(np.median(hts))
        if cv > 0.35 or lh > 0.85 * pitch:
            return 1.0, f"irregular n={n} pitch={pitch:.0f} cv={cv:.2f} lh={lh:.0f}"
        eff = pitch
    else:
        lh = float(np.max(hts))
        if lh > 0.45 * h:
            return 1.0, f"merged n={n} lh={lh:.0f}"
        eff = 1.35 * lh
    scale = max(MIN_SCALE, min(1.0, TARGET_PITCH / eff))
    if scale > 0.9:
        return 1.0, f"fine n={n} pitch={eff:.0f}"
    return scale, f"n={n} pitch={eff:.0f}"


def text_downscale(img, task: str) -> Tuple[float, str]:
    """-> (scale, short diagnostic string). task: "text" or "table" (anything else: 1.0)."""
    if not ENABLED:
        return 1.0, "off"
    if task == "text":
        s, why = _pitch_scale(img, task)
        if TEXT_GLYPH > 0 or TEXT_GLYPH_UP > 0:
            s2, why2 = _glyph_scale(img, TEXT_GLYPH, TEXT_GLYPH_UP, *_hw(img))
            if why2:
                why = why + " | " + why2
            s = s2 if s2 > 1.0 and s >= 1.0 else min(s, s2)
        return s, why
    if task == "table" and (TABLE_GLYPH > 0 or TABLE_GLYPH_UP > 0):
        return _glyph_scale(img, TABLE_GLYPH, TABLE_GLYPH_UP, *_hw(img))
    if task == "formula" and FORMULA_GLYPH > 0:
        # team taktak: formula glyph target (default off)
        h, w = _hw(img)
        if _patches(h, w) < FORMULA_GLYPH_MIN_PATCHES:
            return 1.0, ""
        size, why = glyph_size(_plane(img))
        if size is None or size <= FORMULA_GLYPH:
            return 1.0, "f:" + why
        sc = max(FORMULA_GLYPH_MIN_SCALE, FORMULA_GLYPH / size)
        return (sc, f"f:{size:.1f} {why}") if sc <= 0.9 else (1.0, f"f:fine {size:.1f}")
    return 1.0, "off"


def scaled_size(h: int, w: int, scale: float) -> Tuple[int, int]:
    sh, sw = max(1, int(math.floor(h * scale + 0.5))), max(1, int(math.floor(w * scale + 0.5)))
    if scale < 1.0 and (TEXT_GLYPH > 0 or TABLE_GLYPH > 0 or FORMULA_GLYPH > 0):
        # Never let the 28 px grid round a side down by more than 8%: take the next multiple instead.
        gh = round(sh / FACTOR) * FACTOR
        if sh >= FACTOR and gh < 0.92 * sh:
            sh = gh + FACTOR
            sw = max(sw, min(w, int(math.floor(w * (sh / float(h)) + 0.5))))
        gw = round(sw / FACTOR) * FACTOR
        if sw >= FACTOR and gw < 0.92 * sw:
            sw = gw + FACTOR
    return sh, sw
