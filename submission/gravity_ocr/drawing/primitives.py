"""Primitive shapes of a sheet (plan Part 1.1): what is drawn, by kind.

From one ink map the sheet is split into four ``node_type`` s:

``line``    straight runs (pipes, rules, frames, leaders) – horizontal and vertical
            runs longer than a few text heights;
``box``     line components closed on four sides (equipment boxes, tables, the
            sheet frame); small ones (a few text heights) are symbols;
``icon``    compact components that are neither text nor line (symbols, stamps);
``text``    everything else that has the size of a glyph, grouped into strips.

The text ink alone (lines, frames and symbols removed) is what the page layout
cuts into blocks: with the frame in place every gap of the sheet would be closed.

Every per-component decision is made on arrays of component statistics (boxes
from the labelling kernel, pixel counts from one ``bincount``) and turned back
into masks through a lookup table indexed by label, so the cost does not grow
with the number of glyphs on the sheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ink import Ink, _kernels, components, dilate, long_runs

LINE_UNITS = 2.6  # a straight run this many text heights long is a line
ICON_MIN_UNITS = 1.6  # a closed box this small (in text heights) is a symbol
ICON_MAX_UNITS = 5.0
ICON_FILL = (0.04, 0.62)
GLYPH_MAX_UNITS = 2.6


@dataclass
class Primitive:
    kind: str  # "line" | "box" | "icon" | "text"
    box: tuple  # (x0, y0, x1, y1)
    pixels: int


@dataclass
class Primitives:
    items: list[Primitive] = field(default_factory=list)
    line: np.ndarray | None = None  # line pixels
    text: np.ndarray | None = None  # text pixels (what the layout cuts)
    icon: np.ndarray | None = None

    def of(self, kind: str) -> list[Primitive]:
        return [p for p in self.items if p.kind == kind]


def _stats(mask: np.ndarray):
    """Labels, boxes ``[n, (y0, x0, y1, x1)]``, sizes ``[n]`` and pixel counts ``[n]`` of the components."""
    k = _kernels()
    if k is not None:
        lab, n, boxes, counts = k.label(mask, counts=True)
    else:
        lab, n, boxes = components(mask)
        counts = np.bincount(lab.reshape(-1), minlength=n + 1)[1:] if n else np.zeros(0, np.int64)
    h = (boxes[:, 2] - boxes[:, 0]).astype(np.int64) if n else np.zeros(0, np.int64)
    w = (boxes[:, 3] - boxes[:, 1]).astype(np.int64) if n else np.zeros(0, np.int64)
    return lab, n, boxes, w, h, counts


def _lookup(lab: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Mask of the pixels whose component is selected by ``keep`` (one flag per component)."""
    flags = np.concatenate([[False], keep.astype(bool)])
    k = _kernels()
    return k.select(lab, flags) if k is not None else flags[lab]


def extract(ink: Ink) -> Primitives:
    u = ink.unit
    mask = ink.mask
    n_run = max(int(round(LINE_UNITS * u)), 12)
    line = long_runs(mask, n_run, 1) | long_runs(mask, n_run, 0)
    # the line's own thickness around the runs, unless it is a glyph touching it
    rest0 = mask & ~line
    lab0, _, _, w0, h0, _ = _stats(rest0)
    glyph0 = _lookup(lab0, np.maximum(w0, h0) <= GLYPH_MAX_UNITS * u)
    line |= dilate(line, 1) & mask & ~glyph0
    out = Primitives(line=line)

    lab, n, boxes, w, h, counts = _stats(line)
    big = (w >= ICON_MIN_UNITS * u) & (h >= ICON_MIN_UNITS * u)
    for k in np.flatnonzero(big).tolist() if n else []:
        y0, x0, y1, x1 = (int(v) for v in boxes[k])
        piece = lab[y0:y1, x0:x1] == k + 1
        if _closed(piece):
            kind = "icon" if max(x1 - x0, y1 - y0) <= ICON_MAX_UNITS * u else "box"
            out.items.append(Primitive(kind, (x0, y0, x1, y1), int(counts[k])))
    for k in np.flatnonzero(~big).tolist() if n else []:
        y0, x0, y1, x1 = (int(v) for v in boxes[k])
        out.items.append(Primitive("line", (x0, y0, x1, y1), int(counts[k])))

    rest = mask & ~line
    lab2, n2, boxes2, w2, h2, counts2 = _stats(rest)
    longest = np.maximum(w2, h2)
    fill = counts2 / np.maximum(w2 * h2, 1)
    glyph = longest <= GLYPH_MAX_UNITS * u
    icon = ~glyph & (fill >= ICON_FILL[0]) & (fill <= ICON_FILL[1]) & (longest <= ICON_MAX_UNITS * u * 2)
    text_flags = ~icon  # long handwriting and stamps with text are left to the reader
    out.text = _lookup(lab2, text_flags) if n2 else np.zeros_like(mask)
    out.icon = _lookup(lab2, icon) if n2 else np.zeros_like(mask)
    for k in np.flatnonzero(icon).tolist():
        y0, x0, y1, x1 = (int(v) for v in boxes2[k])
        out.items.append(Primitive("icon", (x0, y0, x1, y1), int(counts2[k])))
    for b, c in _strips(out.text, u):
        out.items.append(Primitive("text", b, c))
    return out


def _closed(piece: np.ndarray) -> bool:
    """Ink on all four sides of the piece's bounding box (a frame, not an L or a T)."""
    if piece.shape[0] < 3 or piece.shape[1] < 3:
        return False
    rows = piece.sum(axis=1)
    cols = piece.sum(axis=0)
    w, h = piece.shape[1], piece.shape[0]
    return rows[:2].max() >= 0.7 * w and rows[-2:].max() >= 0.7 * w and cols[:2].max() >= 0.7 * h and cols[-2:].max() >= 0.7 * h


def _strips(text: np.ndarray, u: float) -> list[tuple]:
    """Glyphs joined into text strips: close the gaps between letters, then take the components holding text."""
    joined = dilate(text, max(int(round(0.45 * u)), 1))
    lab, n, boxes = components(joined)
    if not n:
        return []
    counts = np.bincount(lab[text], minlength=n + 1)[1:]
    return [((int(b[1]), int(b[0]), int(b[3]), int(b[2])), int(c)) for b, c in zip(boxes, counts) if c > 0]
