"""What a crop is and how it is shown to the model.

A crop is read once and decides, from its pixels alone, which of the branches
it takes:

``single_text``    one or two lines, no grid               → "Text Recognition:"
``text_block``     more lines without columns or rules     → "Text Recognition:"
``native_table``   ruled grid                              → "Table Recognition:", rules untouched
``reverse_table``  aligned columns without rules           → "Table Recognition:" after a virtual
                                                             grid is drawn into the whitespace

Conditioning only touches pixels that are not text: a line that runs through
text is damped to a light grey, except where a text stroke bridges it.  The
decode loop never sees any of this; it happens before the image is patchified.

Measured on the calibration crops (f32, text zoomed to 30 px): damping hurts –
the horizontal strokes of glyphs that a crossing line runs along join the line
and fade with it (crossed CER 0.024 → 0.062, overrule 0.000 → 0.018) – so it
is off by default.  The virtual grid of rule-less tables helps a little
(cell exact 0.933 → 0.938) and stays on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from ..model import TABLE_PROMPT, TEXT_PROMPT
from .ink import (
    CHROMA_MIN, Ink, _kernels, as_rgb, bands, chroma_of, components, deskew, dilate, estimate_skew, long_runs, shift,
    upscale_factor,
)  # fmt: skip
from .layout import runs_of

SINGLE_TEXT = "single_text"
TEXT_BLOCK = "text_block"
NATIVE_TABLE = "native_table"
REVERSE_TABLE = "reverse_table"

TEXT_PX = 30.0  # text height the crop is zoomed to
LINE_UNITS = 2.2  # a straight run this many text heights long is a line, not a stroke
DAMP_LUMA = 215
GRID_LUMA = 180
MIN_COL_CLUSTERS = 4
COL_GAP_UNITS = 1.0  # whitespace between two columns, in text heights
PROTECT_RADIUS = 2
BRIDGE_REACH = 8  # thickest line a stroke is followed through
BOOT_RUN = 12  # shortest straight run that can be a line
RESIDUE_PX = 3
ROW_ASPECT = 0.8  # a text line is at least this wide for its height
RULE_SPAN = 0.8
RULE_THICK_UNITS = 0.6
CROSS_MARGIN_UNITS = 0.12
COLOUR_LINE_THIN = 0.15
COLOUR_LINE_LONG = 2.5
MIN_SKEW = 0.3


@dataclass
class CropInfo:
    branch: str
    prompt: str
    unit: float
    rows: list = field(default_factory=list)  # text lines (y0, y1)
    cols: list = field(default_factory=list)  # column clusters (x0, x1)
    h_rules: list = field(default_factory=list)  # (y0, y1) of horizontal rules
    v_rules: list = field(default_factory=list)
    crossing: bool = False  # a rule runs through the inside of a text line
    lines: int = 0  # line pixels found
    scale: float = 1.0
    skew: float = 0.0
    damped: int = 0  # pixels softened
    grid: tuple = ((), ())  # virtual grid lines drawn (ys, xs) in output pixels

    def summary(self) -> dict:
        return {
            "branch": self.branch, "unit": round(self.unit, 1), "rows": len(self.rows), "cols": len(self.cols),
            "h_rules": len(self.h_rules), "v_rules": len(self.v_rules), "crossing": self.crossing, "lines": self.lines,
            "scale": round(self.scale, 3), "skew": self.skew, "damped": self.damped,
            "grid": [len(self.grid[0]), len(self.grid[1])],
        }  # fmt: skip


def line_masks(mask: np.ndarray, unit: float) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical straight runs that are too long to be a stroke of a glyph."""
    n = max(int(round(LINE_UNITS * unit)), BOOT_RUN)
    return long_runs(mask, n, 1), long_runs(mask, n, 0)


def bridges(line: np.ndarray, text: np.ndarray, axis: int) -> np.ndarray:
    """Line pixels a text stroke passes through: text right before and right after the line.

    ``axis=0`` follows strokes vertically (through a horizontal line), ``axis=1``
    horizontally (through a vertical line).
    """
    k = _kernels()
    if k is not None:
        return k.bridges(line, text, axis, BRIDGE_REACH)
    return bridges_np(line, text, axis)


def bridges_np(line: np.ndarray, text: np.ndarray, axis: int) -> np.ndarray:
    step = (1, 0) if axis == 0 else (0, 1)
    a = line & shift(text, *step)
    b = line & shift(text, -step[0], -step[1])
    for _ in range(BRIDGE_REACH):
        a |= line & shift(a, *step)
        b |= line & shift(b, -step[0], -step[1])
    return a & b


def mop_up(text: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Thin leftovers that hang on a line (the ends of a slightly turned rule) belong to the line."""
    lab, n, boxes = components(text)
    out = np.zeros_like(text)
    if not n:
        return out
    touching = np.unique(lab[dilate(line, 1) & text])
    for k in touching.tolist():
        y0, x0, y1, x1 = boxes[k - 1].tolist()
        if min(y1 - y0, x1 - x0) <= RESIDUE_PX:
            out[y0:y1, x0:x1] |= lab[y0:y1, x0:x1] == k
    return out


def text_rows(text: np.ndarray, box) -> tuple[list[tuple[int, int]], float | None]:
    """Text lines of ``box`` and their height (lower quartile: the smallest line decides the zoom)."""
    x0, y0, x1, y1 = box
    rows = []
    for a, b in bands(text[y0:y1, x0:x1].sum(axis=1).astype(np.float32), y0):
        xs = np.flatnonzero(text[a:b, x0:x1].sum(axis=0))
        if len(xs) and xs[-1] - xs[0] + 1 >= ROW_ASPECT * (b - a):  # a stray dash of a line is not a row
            rows.append((a, b))
    hs = sorted(b - a for a, b in rows)
    if not hs:
        return rows, None
    return rows, float(hs[len(hs) // 4] if len(hs) >= 4 else hs[0])


def column_clusters(text: np.ndarray, box, rows, gap: float) -> list[tuple[int, int]]:
    """Runs of inked columns separated by whitespace that is empty in every text line."""
    x0, _, x1, _ = box
    filled = np.zeros(x1 - x0, bool)
    for a, b in rows:
        filled |= text[a:b, x0:x1].any(axis=0)
    out: list[list[int]] = []
    for a, b in runs_of(filled):
        if out and a - out[-1][1] < gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a + x0, b + x0) for a, b in out]


def rules_of(line: np.ndarray, box, axis: int, unit: float) -> list[tuple[int, int]]:
    """Rules that span the box: ``(start, end)`` across the rule, sorted.  ``axis=1`` horizontal."""
    x0, y0, x1, y1 = box
    _, _, boxes = components(line[y0:y1, x0:x1])
    span = (x1 - x0) if axis == 1 else (y1 - y0)
    thick = max(RULE_THICK_UNITS * unit, 8)
    off = y0 if axis == 1 else x0
    out = []
    for by0, bx0, by1, bx1 in boxes.tolist():
        (a0, a1), (c0, c1) = ((bx0, bx1), (by0, by1)) if axis == 1 else ((by0, by1), (bx0, bx1))
        if a1 - a0 >= RULE_SPAN * span and c1 - c0 <= thick:
            out.append((c0 + off, c1 + off))
    return sorted(out)


def colour_line(ink: Ink, unit: float) -> np.ndarray | None:
    """A coloured straight line of any angle or dash pattern over text that is not coloured.

    The coloured pixels of a pipe or a pen line lie on one thin straight band;
    coloured handwriting does not.
    """
    col = ink.color
    n = int(col.sum())
    if n < 2 * unit or n * 2 >= int(ink.mask.sum()):  # nothing coloured, or the pen is the text
        return None
    ys, xs = np.nonzero(col)
    pts = np.stack([xs - xs.mean(), ys - ys.mean()]).astype(np.float64)
    vals, vecs = np.linalg.eigh(pts @ pts.T / n)
    along = pts.T @ vecs[:, 1]
    thin = np.sqrt(max(vals[0], 0.0)) <= max(COLOUR_LINE_THIN * unit, 2.0)
    return col if thin and along.max() - along.min() >= COLOUR_LINE_LONG * unit else None


def analyse(image, lines: np.ndarray | None = None) -> tuple[CropInfo, dict]:
    """Measure a crop.  ``lines`` marks pixels already known to be pipes or rules (page context)."""
    rgb = as_rgb(image)
    ink = Ink(rgb, unit=1.0)
    mask = ink.mask
    known = None if lines is None else lines & mask
    # first guess of the text height: whatever is left when every longish straight run is gone
    boot = max(BOOT_RUN, min(ink.h, ink.w) // 3)
    rough = mask & ~long_runs(mask, boot, 1) & ~long_runs(mask, boot, 0)
    _, unit = text_rows(rough, ink.full)
    unit = unit or max(ink.h * 0.5, 4.0)
    for _ in range(3):
        hl, vl = line_masks(mask, unit)
        if known is not None:
            hl = hl | (known & ~vl)
        text = mask & ~(hl | vl)
        keep = bridges(hl, text, 0) | bridges(vl, text, 1)
        text |= keep
        box = ink.with_mask(text).tighten(ink.full) or ink.full
        _, seen = text_rows(text, box)
        if seen is None or abs(seen - unit) <= 0.15 * unit:
            break
        unit = seen
    coloured = colour_line(ink, unit)
    extra = np.zeros_like(mask) if coloured is None else coloured & ~(hl | vl)
    text &= ~extra
    keep &= ~extra
    residue = mop_up(text & ~keep, hl | vl | extra)
    text &= ~residue
    line = hl | vl | extra | residue
    box = ink.with_mask(text).tighten(ink.full) or ink.full
    rows, seen = text_rows(text, box)
    unit = seen or unit
    ink.unit = unit
    frame = ink.tighten(ink.full) or ink.full  # rules are measured against everything drawn
    h_rules = rules_of(hl, frame, 1, unit)
    v_rules = rules_of(vl, frame, 0, unit)
    cols = column_clusters(text, box, rows, COL_GAP_UNITS * unit)
    # ruled paper under handwriting is not a table: a rule runs through the inside of a text line
    m = max(CROSS_MARGIN_UNITS * unit, 2.0)
    crossing = any(a + m <= r0 and r1 <= b - m for r0, r1 in h_rules for a, b in rows)
    inner = max(len(h_rules) - 2, 0) + max(len(v_rules) - 2, 0)
    grid = len(h_rules) >= 2 and len(v_rules) >= 2 and inner >= 1
    if grid and not crossing:
        branch = NATIVE_TABLE
    elif len(rows) >= 2 and len(cols) >= MIN_COL_CLUSTERS and not (h_rules or v_rules):
        branch = REVERSE_TABLE
    elif len(rows) <= 2:
        branch = SINGLE_TEXT
    else:
        branch = TEXT_BLOCK
    prompt = TABLE_PROMPT if branch in (NATIVE_TABLE, REVERSE_TABLE) else TEXT_PROMPT
    info = CropInfo(branch, prompt, unit, rows, cols, h_rules, v_rules, crossing, int(line.sum()))
    return info, {"ink": ink, "line": line, "text": text, "keep": keep, "box": box}


def damp_lines(
    rgb: np.ndarray, line: np.ndarray, keep: np.ndarray, text: np.ndarray, protect: np.ndarray | None = None
) -> tuple[np.ndarray, int]:
    """Lift line pixels and their blurred rim to ``DAMP_LUMA`` unless a stroke bridges them or they are protected."""
    soft = dilate(line & ~keep, 1) & ~text
    if protect is not None:
        soft &= ~protect
    out = np.array(rgb, dtype=np.uint8, order="C", copy=True)
    k = _kernels()
    if k is not None and out.ndim == 3 and out.shape[-1] == 3:
        k.damp(out, soft, DAMP_LUMA)
    else:
        out[soft] = np.maximum(out[soft], DAMP_LUMA)
    return out, int((soft & line).sum())


def protected_zone(rgb: np.ndarray, new_strokes: np.ndarray | None = None, radius: int = PROTECT_RADIUS) -> np.ndarray:
    """Coloured pen strokes and strokes that are new against the master, grown by ``radius``."""
    zone = chroma_of(rgb) >= CHROMA_MIN
    if new_strokes is not None:
        zone = zone | new_strokes
    return dilate(zone, radius)


def coloured_text(work: dict) -> bool:
    """Most of the text ink (lines aside) is coloured: a pen wrote it."""
    ink: Ink = work["ink"]
    text = work["text"]
    n = int(text.sum())
    return n > 0 and int((text & (ink.chroma >= CHROMA_MIN)).sum()) >= 0.5 * n


def virtual_grid(rgb: np.ndarray, info: CropInfo, scale: float, frame: bool = False, width: int = 1) -> tuple[np.ndarray, tuple]:
    """Draw light grey lines in the whitespace valleys between rows and between columns."""
    out = rgb.copy()
    h, w = out.shape[:2]
    ys = [int(round((a[1] + b[0]) / 2 * scale)) for a, b in zip(info.rows, info.rows[1:])]
    xs = [int(round((a[1] + b[0]) / 2 * scale)) for a, b in zip(info.cols, info.cols[1:])]
    if frame and info.rows and info.cols:
        pad = info.unit * 0.4
        ys = [int((info.rows[0][0] - pad) * scale)] + ys + [int((info.rows[-1][1] + pad) * scale)]
        xs = [int((info.cols[0][0] - pad) * scale)] + xs + [int((info.cols[-1][1] + pad) * scale)]
    ys = [min(max(y, 0), h - width) for y in ys]
    xs = [min(max(x, 0), w - width) for x in xs]
    y0, y1 = (min(ys), max(ys) + width) if frame and ys else (0, h)
    x0, x1 = (min(xs), max(xs) + width) if frame and xs else (0, w)
    for y in ys:
        out[y : y + width, x0:x1] = GRID_LUMA
    for x in xs:
        out[y0:y1, x : x + width] = GRID_LUMA
    return out, (tuple(ys), tuple(xs))


def resize(rgb: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-3:
        return rgb
    h, w = rgb.shape[:2]
    size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.BICUBIC))


@dataclass
class Prepared:
    image: np.ndarray  # what the vision tower gets
    prompt: str
    info: CropInfo


def trim_outer_whitespace(rgb: np.ndarray, pad: int = 6, bg_thresh: int = 245) -> np.ndarray:
    """Losslessly trim pure outer paper whitespace while preserving original glyph resolution and a ``pad`` px halo."""
    rgb = as_rgb(rgb)
    h, w = rgb.shape[:2]
    if h <= 32 or w <= 32:
        return rgb
    k = _kernels()
    if k is not None and k.has("pixels"):
        gray, _, _ = k.gray_chroma(rgb, chroma=False)
        ink_mask = k.below(gray, bg_thresh)
    else:
        ink_mask = rgb.min(axis=2) < bg_thresh
    if not ink_mask.any():
        return rgb
    rows = np.flatnonzero(ink_mask.any(axis=1))
    cols = np.flatnonzero(ink_mask.any(axis=0))
    r0 = max(0, int(rows[0]) - pad)
    r1 = min(h, int(rows[-1]) + 1 + pad)
    c0 = max(0, int(cols[0]) - pad)
    c1 = min(w, int(cols[-1]) + 1 + pad)
    if (r1 - r0) >= h - 4 and (c1 - c0) >= w - 4:
        return rgb
    return np.ascontiguousarray(rgb[r0:r1, c0:c1])


def prepare_crop(
    image,
    prompt: str | None = None,
    text_px: float | None = TEXT_PX,
    damp: bool = False,
    grid: bool = True,
    grid_frame: bool = False,
    level: bool = False,
    lines: np.ndarray | None = None,
    new_strokes: np.ndarray | None = None,
) -> Prepared:
    """Classify, condition and scale one crop.

    ``prompt`` forces the prompt (``"text"`` / ``"table"`` or a full prompt
    string); by default the branch decides.  ``text_px=None`` keeps the size.
    ``level`` turns a tilted crop upright first (page crops are already level).
    """
    rgb = as_rgb(image)
    skew = 0.0
    if level and lines is None and new_strokes is None:
        skew = estimate_skew(Ink(rgb, unit=1.0).mask)
        if abs(skew) >= MIN_SKEW:
            paper = tuple(int(v) for v in np.percentile(rgb.reshape(-1, 3), 90, axis=0))
            rgb = deskew(rgb, skew, fill=paper)
        else:
            skew = 0.0
    info, work = analyse(rgb, lines)
    info.skew = skew
    out = rgb
    if damp and info.branch in (SINGLE_TEXT, TEXT_BLOCK) and work["line"].any():
        # with a pen on the sheet the pen is the text: printed lines never win against it
        zone = protected_zone(rgb, new_strokes) if (new_strokes is not None or coloured_text(work)) else None
        out, info.damped = damp_lines(out, work["line"], work["keep"], work["text"], zone)
    info.scale = upscale_factor(info.unit, text_px, rgb.shape[1], rgb.shape[0]) if text_px else 1.0
    out = resize(out, info.scale)
    if grid and info.branch == REVERSE_TABLE:
        out, info.grid = virtual_grid(out, info, info.scale, frame=grid_frame)
    if prompt is not None:
        info.prompt = {"text": TEXT_PROMPT, "table": TABLE_PROMPT}.get(prompt, prompt)
    return Prepared(out, info.prompt, info)
