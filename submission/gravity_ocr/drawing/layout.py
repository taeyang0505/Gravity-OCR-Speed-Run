"""Recursive whitespace cuts: sheet → blocks → lines and paragraphs → tiles.

The cut works on an :class:`~gravity_ocr.drawing.ink.Ink` map.  On a drawing
that map is the one with pipes and frames already taken out (see
``primitives``), otherwise the sheet border would close every gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ink import Box, Ink

ROW_GAP_UNITS = 0.65
COL_GAP_RATIO = 0.035
COL_SIDE_RATIO = 0.2
EDGE_STRIP_RATIO = 0.06
LINE_COL_GAP_RATIO = 0.25
LINE_UNITS = 2.6
STACK_GAP_UNITS = 1.6
MIN_AREA_UNITS = 0.5
PARAGRAPH_MIN_LINES = 2
PARAGRAPH_HEIGHT_TOLERANCE = 0.4
PARAGRAPH_FILL = 0.6
PARAGRAPH_MAX_UNITS = 1.3
PATCH = 14
EMPTY_RATIO = 0.002
FORCED_OVERLAP = 0.2
MAX_DEPTH = 32


def runs_of(flags: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end)`` of every run of true values."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])]


def widest_gap(counts: np.ndarray, limit: float) -> tuple[int, int] | None:
    """Longest interior run of (nearly) empty rows or columns as ``(start, length)``."""
    best = None
    for a, b in runs_of(counts <= limit):
        if a == 0 or b == len(counts):
            continue
        if best is None or b - a > best[1]:
            best = (a, b - a)
    return best


def cut(ink: Ink, box: Box, out: list[Box], depth: int = 0) -> None:
    t = ink.tighten(box)
    if t is None:
        return
    x0, y0, x1, y1 = t
    w, h, u, pw = x1 - x0, y1 - y0, ink.unit, ink.w
    if depth < MAX_DEPTH:
        row = widest_gap(ink.row_counts(t), w * EMPTY_RATIO)
        col = widest_gap(ink.col_counts(t), h * EMPTY_RATIO)
        if row and row[1] < ROW_GAP_UNITS * u:
            row = None
        if col:
            gs, ge = x0 + col[0], x0 + col[0] + col[1]
            if gs <= EDGE_STRIP_RATIO * pw or ge >= (1 - EDGE_STRIP_RATIO) * pw:
                ok = col[1] >= COL_GAP_RATIO * pw * 0.5  # a margin strip comes off easily
            elif h <= LINE_UNITS * u:
                ok = col[1] >= LINE_COL_GAP_RATIO * pw  # one line: only a wide gap is a column break
            else:
                ok = col[1] >= COL_GAP_RATIO * pw and gs - x0 >= COL_SIDE_RATIO * pw and x1 - ge >= COL_SIDE_RATIO * pw
            if not ok:
                col = None
        if row and (not col or row[1] >= col[1]):
            cut(ink, (x0, y0, x1, y0 + row[0]), out, depth + 1)
            cut(ink, (x0, y0 + row[0] + row[1], x1, y1), out, depth + 1)
            return
        if col:
            cut(ink, (x0, y0, x0 + col[0], y1), out, depth + 1)
            cut(ink, (x0 + col[0] + col[1], y0, x1, y1), out, depth + 1)
            return
    in_strip = x1 <= EDGE_STRIP_RATIO * pw or x0 >= (1 - EDGE_STRIP_RATIO) * pw
    if w * h >= MIN_AREA_UNITS * u * u and not in_strip:
        out.append(t)


def stack_lines(ink: Ink, pieces: list[Box]) -> list[Box]:
    """Join single lines that sit right under each other into one block."""
    u = ink.unit
    out: list[Box] = []
    stacked: list[bool] = []  # the piece is a line or a stack of lines
    for b in sorted(pieces, key=lambda b: (b[1], b[0])):
        is_line = b[3] - b[1] <= LINE_UNITS * u
        if is_line:
            for i in range(len(out) - 1, -1, -1):
                a = out[i]
                share = min(a[2], b[2]) - max(a[0], b[0])
                if stacked[i] and share * 2 >= min(a[2] - a[0], b[2] - b[0]) and 0 <= b[1] - a[3] <= STACK_GAP_UNITS * u:
                    out[i] = (min(a[0], b[0]), a[1], max(a[2], b[2]), b[3])
                    break
            else:
                out.append(b)
                stacked.append(True)
        else:
            out.append(b)
            stacked.append(False)
    return out


def blocks(ink: Ink, box: Box | None = None) -> list[Box]:
    out: list[Box] = []
    cut(ink, box or ink.full, out)
    return stack_lines(ink, out)


def has_inner_rule(ink: Ink, box: Box) -> bool:
    """A rule inside the box (not its own frame): the mark of a table or a form."""
    x0, y0, x1, y1 = box
    inset = int(round(ink.unit))
    if y1 - y0 <= 2 * inset or x1 - x0 <= 2 * inset:
        return False
    return bool(ink.rule_rows((x0, y0 + inset, x1, y1 - inset)).any())


@dataclass
class Run:
    box: Box
    paragraph: bool


def runs(ink: Ink, box: Box) -> list[Run]:
    """Split a block into paragraphs (even, filled lines) and everything else."""
    x0, y0, x1, y1 = box
    bands = ink.row_bands(box)
    if has_inner_rule(ink, box) or len(bands) < PARAGRAPH_MIN_LINES:
        return [Run(box, False)]
    heights = sorted(b - a for a, b in bands)
    median = heights[len(heights) // 2]
    if median > PARAGRAPH_MAX_UNITS * ink.unit:
        return [Run(box, False)]
    w = x1 - x0
    even, filled = [], []
    for a, b in bands:
        t = ink.tighten((x0, a, x1, b))
        even.append(abs((b - a) - median) <= PARAGRAPH_HEIGHT_TOLERANCE * median)
        filled.append(t is not None and t[2] - t[0] >= PARAGRAPH_FILL * w)
    edges = [y0] + [(bands[i][1] + bands[i + 1][0]) // 2 for i in range(len(bands) - 1)] + [y1]
    out: list[Run] = []
    i = 0
    while i < len(bands):
        j = i
        while j < len(bands) and even[j] and filled[j]:
            j += 1
        if j - i >= PARAGRAPH_MIN_LINES - 1 and j > i:
            if j < len(bands) and even[j]:  # the short last line of a paragraph
                j += 1
            if j - i >= PARAGRAPH_MIN_LINES:
                out.append(Run((x0, edges[i], x1, edges[j]), True))
                i = j
                continue
        k = i + 1
        while k < len(bands) and not (even[k] and filled[k] and k + 1 < len(bands) and even[k + 1]):
            k += 1
        out.append(Run((x0, edges[i], x1, edges[k]), False))
        i = k
    return out


def tiles(ink: Ink, box: Box, scale: float, max_patches: int, overlap: float = FORCED_OVERLAP) -> tuple[list[Box], bool]:
    """Cut a block that is too tall for one vision pass between its text lines.

    Returns the pieces and whether they overlap (cut blindly because no gap
    between lines was available).
    """
    x0, y0, x1, y1 = box
    w = max(x1 - x0, 1)
    max_h = max(max_patches * PATCH * PATCH / (w * scale * scale), 4 * PATCH)
    if y1 - y0 <= max_h:
        return [box], False
    bands = ink.row_bands(box)
    pieces: list[Box] = []
    top = y0
    for (_, e), (s, e2) in zip(bands, bands[1:]):
        mid = (e + s) // 2
        if e2 - top > max_h and mid > top:  # the next line would not fit: cut in the gap before it
            pieces.append((x0, top, x1, mid))
            top = mid
    pieces.append((x0, top, x1, y1))
    if len(bands) < 2 or any(p[3] - p[1] > 1.5 * max_h for p in pieces):
        step = max(int(max_h * (1 - overlap)), 1)
        pieces = [(x0, y, x1, min(y + int(max_h), y1)) for y in range(y0, max(y1 - int(max_h * overlap), y0 + 1), step)]
        return pieces, True
    return pieces, False
