"""Ink map of a sheet or a crop: what is drawn, where the text lines are, how big they are.

Everything here is plain pixel arithmetic; boxes are ``(x0, y0, x1, y1)`` with
exclusive ends.  The pixel loops run in the native kernels when they are
loaded (``gravity_ocr.native``, checked against the ``*_np`` functions here,
which they reproduce exactly) and in numpy otherwise.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage

MAX_UPSCALE = 4.0
CROP_MAX_SIDE = 2048
INK_DROP = 70  # ink is this much darker than the paper
SKEW_RANGE = 3.0
SKEW_STEP = 0.2
MIN_BAND_PX = 4
MERGE_GAP_PX = 4
BAND_RATIO = 0.04
RULE_RATIO = 0.8  # a row / column inked over this share of the span is a rule
TEXT_UNIT = 0.012  # text unit of a page as a share of its height
RULE_MIN_UNITS = 3.5
GLYPH_MIN = 0.4
GLYPH_MAX = 2.8
CHROMA_MIN = 45  # max(R,G,B) - min(R,G,B) from which a pixel counts as coloured

Box = tuple[int, int, int, int]
EIGHT = np.ones((3, 3), bool)


def as_rgb(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    a = np.asarray(image, dtype=np.uint8)
    return np.repeat(a[:, :, None], 3, axis=2) if a.ndim == 2 else a[:, :, :3]


def _kernels():
    """The checked native pixel kernels, or ``None``."""
    from .. import native

    k = native.load()
    return k if k is not None and k.has("pixels") else None


def _is_rgb8(rgb) -> bool:
    return isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3


def gray_of_np(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.uint16) for i in range(3))
    return ((r * 77 + g * 150 + b * 29) >> 8).astype(np.uint8)


def chroma_of_np(rgb: np.ndarray) -> np.ndarray:
    return rgb.max(axis=-1) - rgb.min(axis=-1)


def gray_of(rgb: np.ndarray) -> np.ndarray:
    k = _kernels()
    return k.gray_chroma(rgb, chroma=False)[0] if k is not None and _is_rgb8(rgb) else gray_of_np(rgb)


def chroma_of(rgb: np.ndarray) -> np.ndarray:
    k = _kernels()
    return k.gray_chroma(rgb)[1] if k is not None and _is_rgb8(rgb) else chroma_of_np(rgb)


def paper_level(gray: np.ndarray) -> int:
    return int(np.percentile(gray, 90))


def level_from_hist(hist: np.ndarray, q: float = 90) -> int:
    """``int(np.percentile(values, q))`` from the histogram of uint8 values (linear method)."""
    cum = np.cumsum(np.asarray(hist, np.int64))
    n = int(cum[-1])
    vi = (n - 1) * (q / 100)
    lo = math.floor(vi)
    g = vi - lo
    a = int(np.searchsorted(cum, lo + 1))  # value of the lo-th smallest
    b = int(np.searchsorted(cum, min(lo + 1, n - 1) + 1))
    d = b - a
    return int(b - d * (1 - g) if g >= 0.5 else a + d * g)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square neighbourhood of ``radius`` pixels (zero border)."""
    if radius <= 0:
        return mask.copy()
    k = _kernels()
    if k is not None and mask.ndim == 2:
        return k.dilate(mask, radius)
    return ndimage.maximum_filter(mask, size=2 * radius + 1, mode="constant", cval=0)


def components(mask: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """8-connected pieces: labels, count and boxes ``[n, (y0, x0, y1, x1)]`` (``ndimage`` numbering)."""
    k = _kernels()
    if k is not None and mask.ndim == 2:
        return k.label(mask)
    lab, n = ndimage.label(mask, structure=EIGHT)
    boxes = np.array([(s[0].start, s[1].start, s[0].stop, s[1].stop) for s in ndimage.find_objects(lab)], np.int32)
    return lab, n, boxes.reshape(n, 4)


def long_runs(mask: np.ndarray, min_len: int, axis: int = 1) -> np.ndarray:
    """Pixels that belong to a straight run of at least ``min_len`` along ``axis``.

    ``axis=1`` looks for horizontal runs, ``axis=0`` for vertical ones.
    """
    k = _kernels()
    if k is not None and mask.ndim == 2:
        return k.long_runs(mask, min_len, axis)
    return long_runs_np(mask, min_len, axis)


def long_runs_np(mask: np.ndarray, min_len: int, axis: int = 1) -> np.ndarray:
    m = mask if axis == 1 else mask.T
    h, w = m.shape
    edge = np.zeros((h, w + 2), np.int8)
    edge[:, 1:-1] = m
    d = np.diff(edge, axis=1)
    ys, starts = np.nonzero(d == 1)
    _, ends = np.nonzero(d == -1)  # same order: row by row, left to right
    keep = (ends - starts) >= min_len
    marks = np.zeros((h, w + 1), np.int32)
    np.add.at(marks, (ys[keep], starts[keep]), 1)
    np.add.at(marks, (ys[keep], ends[keep]), -1)
    out = np.cumsum(marks, axis=1)[:, :w] > 0
    return out if axis == 1 else out.T


def shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``mask`` moved by ``(dy, dx)``; what leaves the frame is lost, what enters is empty."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = mask[ys, xs]
    return out


def bands(profile: np.ndarray, offset: int = 0) -> list[tuple[int, int]]:
    """Runs of a profile that rise above its floor: text lines of a row profile."""
    n = len(profile)
    if n < 8:
        return []
    ordered = np.sort(profile)
    floor, peak = float(ordered[n // 10]), float(ordered[n * 9 // 10])
    if peak - floor < 1:
        return []
    gate = floor + max((peak - floor) * BAND_RATIO, 2.0)
    on = profile > gate
    edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
    runs = [[int(a), int(b)] for a, b in zip(edges[0::2], edges[1::2])]
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= MERGE_GAP_PX:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a + offset, b + offset) for a, b in merged if b - a >= MIN_BAND_PX]


def upscale_factor(text_h: float | None, target: float, w: int, h: int) -> float:
    """Zoom that brings the text height to ``target`` px, inside the size budget."""
    if not text_h:
        return 1.0
    want = min(max(target / max(text_h, 1.0), 1.0), MAX_UPSCALE)
    return min(want, max(CROP_MAX_SIDE / max(w, h), 1.0))


class Ink:
    """Binary ink map with the measurements the planner needs."""

    def __init__(self, image, unit: float | None = None, mask: np.ndarray | None = None):
        self.rgb = as_rgb(image)
        k = _kernels()
        if k is not None and _is_rgb8(self.rgb):
            # one pass: gray, chroma and the histogram the paper level comes from
            self.gray, self.chroma, hist = k.gray_chroma(self.rgb)
            self.paper = level_from_hist(hist, 90)
        else:
            self.gray = gray_of_np(self.rgb)
            self.chroma = chroma_of_np(self.rgb)
            self.paper = paper_level(self.gray)
        self.h, self.w = self.gray.shape
        self.mask = self.gray < max(self.paper - INK_DROP, 32) if mask is None else mask
        self.color = self.mask & (self.chroma >= CHROMA_MIN)
        self.unit = float(unit) if unit else max(self.h * TEXT_UNIT, 4.0)

    @classmethod
    def of_crop(cls, image) -> "Ink":
        """A crop has no page height to derive the text unit from: use its own text lines."""
        ink = cls(image, unit=1.0)
        ink.unit = float(ink.text_height(ink.full) or max(ink.h * 0.5, 4.0))
        return ink

    def with_mask(self, mask: np.ndarray) -> "Ink":
        other = object.__new__(Ink)
        other.__dict__.update(self.__dict__)
        other.mask = mask
        other.color = mask & (self.chroma >= CHROMA_MIN)
        return other

    @property
    def full(self) -> Box:
        return (0, 0, self.w, self.h)

    # ------------------------------------------------------------------ counts
    def view(self, box: Box) -> np.ndarray:
        x0, y0, x1, y1 = box
        return self.mask[y0:y1, x0:x1]

    def count(self, box: Box) -> int:
        return int(self.view(box).sum())

    def row_counts(self, box: Box) -> np.ndarray:
        return self.view(box).sum(axis=1)

    def col_counts(self, box: Box) -> np.ndarray:
        return self.view(box).sum(axis=0)

    def frame_origin(self) -> tuple[int, int] | None:
        """Top-left corner of the sheet border: the first inked column / row near the edge."""
        col = self.mask.sum(axis=0)[: max(int(self.w * 0.06), 1)]
        row = self.mask.sum(axis=1)[: max(int(self.h * 0.06), 1)]
        xs = np.flatnonzero(col * 2 > self.h)
        ys = np.flatnonzero(row * 2 > self.w)
        return (int(xs[0]), int(ys[0])) if len(xs) and len(ys) else None

    # ------------------------------------------------------------------ boxes
    def tighten(self, box: Box) -> Box | None:
        """Shrink to the rows, then the columns, that hold more than a stray pixel."""
        x0, y0, x1, y1 = box
        v = self.mask[y0:y1, x0:x1]
        rows = np.flatnonzero(v.sum(axis=1) > 1)
        if not len(rows):
            return None
        v = v[rows[0] : rows[-1] + 1]
        cols = np.flatnonzero(v.sum(axis=0) > 1)
        if not len(cols):
            return None
        return (x0 + int(cols[0]), y0 + int(rows[0]), x0 + int(cols[-1]) + 1, y0 + int(rows[-1]) + 1)

    def rule_rows(self, box: Box) -> np.ndarray:
        """Rows of ``box`` that are a horizontal rule (boolean per row)."""
        return self.row_counts(box) >= RULE_RATIO * max(box[2] - box[0], 1)

    def rule_cols(self, box: Box) -> np.ndarray:
        return self.col_counts(box) >= RULE_RATIO * max(box[3] - box[1], 1)

    def text_box(self, box: Box) -> Box | None:
        """The text of a framed cell: the box without its rules."""
        t = self.tighten(box)
        if t is None:
            return None
        x0, y0, x1, y1 = t
        v = self.mask[y0:y1, x0:x1]
        keep_rows = v.sum(axis=1) < RULE_RATIO * (x1 - x0)
        if not keep_rows.any():
            return None
        cols = v[keep_rows].sum(axis=0)
        limit = RULE_RATIO * keep_rows.sum() if (y1 - y0) >= RULE_MIN_UNITS * self.unit else math.inf
        keep_cols = (cols > 1) & (cols < limit)
        xs = np.flatnonzero(keep_cols)
        if not len(xs):
            return None
        kept = v[:, keep_cols] & keep_rows[:, None]
        ys = np.flatnonzero(kept.sum(axis=1) > 1)
        if not len(ys):
            return None
        return (x0 + int(xs[0]), y0 + int(ys[0]), x0 + int(xs[-1]) + 1, y0 + int(ys[-1]) + 1)

    # ------------------------------------------------------------------ text lines
    def text_mask(self, box: Box) -> np.ndarray:
        """Ink of ``box`` without the rows and columns that are rules."""
        v = self.view(box).copy()
        h, w = v.shape
        if h and w:
            v[v.sum(axis=1) >= RULE_RATIO * w] = False
            if h >= RULE_MIN_UNITS * self.unit:
                v[:, v.sum(axis=0) >= RULE_RATIO * h] = False
        return v

    def row_bands(self, box: Box) -> list[tuple[int, int]]:
        return bands(self.text_mask(box).sum(axis=1).astype(np.float32), box[1])

    def text_height(self, box: Box) -> int | None:
        hs = sorted(b - a for a, b in self.row_bands(box))
        if not hs:
            return None
        return hs[len(hs) // 4] if len(hs) >= 4 else hs[0]

    def glyphs(self, box: Box, limit: int | None = None) -> int:
        """Connected pieces of ``box`` that have the size of a character."""
        _, _, boxes = components(self.view(box))
        lo, hi = GLYPH_MIN * self.unit, GLYPH_MAX * self.unit
        found = 0
        for y0, x0, y1, x1 in boxes.tolist():
            h, w = y1 - y0, x1 - x0
            long_, short = max(w, h), min(w, h)
            is_long = long_ >= 4 * max(short, 1) and long_ > 1.8 * self.unit and short <= max(0.25 * self.unit, 4)
            if not is_long and lo <= h <= hi and w <= hi:
                found += 1
                if limit is not None and found >= limit:
                    break
        return found


def estimate_skew(mask: np.ndarray) -> float:
    """Angle (degrees) whose row histogram is the sharpest; 0 when there is too little ink."""
    ys, xs = np.nonzero(mask[:, ::3])
    if len(ys) < 64:
        return 0.0
    xs = xs * 3.0 - mask.shape[1] / 2
    ys = ys - mask.shape[0] / 2
    rows = mask.shape[0] // 2 + 2
    best, best_score = 0.0, -1.0
    for deg in np.arange(-SKEW_RANGE, SKEW_RANGE + 1e-6, SKEW_STEP):
        r = ((ys - xs * math.tan(math.radians(deg))) / 2 + rows / 2).astype(np.int64)
        hist = np.bincount(r[(r >= 0) & (r < rows)], minlength=rows).astype(np.float64)
        score = float((hist * hist).sum())
        if score > best_score:
            best, best_score = float(deg), score
    return round(best, 2)


def deskew(image, deg: float, fill=(255, 255, 255)) -> np.ndarray:
    """Rotate about the centre so that text lines run level.

    ``estimate_skew`` returns the slope of the lines as ``y`` over ``x`` in image
    coordinates; turning the image by the same angle counter-clockwise on screen
    brings them level.
    """
    rgb = as_rgb(image)
    if abs(deg) < 1e-3:
        return rgb
    return np.asarray(Image.fromarray(rgb).rotate(deg, resample=Image.BICUBIC, fillcolor=tuple(fill)))
