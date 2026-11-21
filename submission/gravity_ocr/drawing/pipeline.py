"""Reading a whole sheet: ink → primitives → text blocks → roles → crops → text.

The layout cut runs on the text ink only (``primitives``): pipes, frames and
symbols removed, so that the gaps between blocks are open.  Every block becomes a
crop (padded, classified and zoomed by ``crops.prepare_crop``); the crops of a
sheet are read together (``decode.read_many``: one batched vision pass on a GPU).

**Title-block protection.**  The title block and the signature box are never
dropped by a filter.  Two sheet layouts exist and both are covered:

``corner``  title block bottom right: the block starts at ``x0 ≥ 0.60 W`` and
            ``y0 ≥ 0.75 H`` (the plan's ``ProtectedRole::TitleBlock`` rule);
``top``     title bar along the top edge: the block ends above ``0.12 H``; its
            signature box sits bottom right and falls under the first rule.

A block inside a closed box primitive that itself lies in one of those regions
is protected too (a title cell that starts a little left of 0.60 W), and so is a
block with ``PROTECT_SHARE`` of its area inside a region: the whitespace cut can
join a title line with text beside the title block, and such a block must not be
dropped either.

**Rules as walls at the title block.**  The cut runs on the text ink, so the
frame of the title block is gone and a line of notes can run into the title
block across it.  A block is therefore split where a rule crosses it from edge to
edge and runs on beyond it (``WALL_REACH`` text heights on at least one side) –
but only where the two sides differ in protection: the edge of the title block
separates protected from unprotected text, the rules of a table (inside or
outside the title block) do not, so tables are never cut into cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import layout, primitives
from .crops import Prepared, prepare_crop
from .ink import Ink, as_rgb

CORNER = (0.60, 0.75)
TOP_BAND = 0.12
PROTECT_SHARE = 0.2  # a block with this share of its area in a protected region is protected
WALL_SPAN = 0.9  # a rule crossing this share of a block's height (width) …
WALL_REACH = 1.5  # … and reaching this many text heights past it may split the block
PAD = 6


@dataclass
class Block:
    box: tuple  # (x0, y0, x1, y1) in sheet pixels
    protected: bool = False
    region: str = ""  # "corner" | "top" | ""
    prepared: Prepared | None = None
    text: str | None = None
    dropped: bool = False
    score: float | None = None  # boilerplate score of the bank
    vector: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def _share(box, region) -> float:
    """Share of ``box``'s area inside ``region``."""
    x0, y0, x1, y1 = box
    rx0, ry0, rx1, ry1 = region
    ix = max(0, min(x1, rx1) - max(x0, rx0))
    iy = max(0, min(y1, ry1) - max(y0, ry0))
    return ix * iy / max((x1 - x0) * (y1 - y0), 1)


def protected_region(box, w: int, h: int, frames=()) -> str:
    x0, y0, x1, y1 = box
    if x0 >= CORNER[0] * w and y0 >= CORNER[1] * h:
        return "corner"
    if y1 <= TOP_BAND * h:
        return "top"
    for fx0, fy0, fx1, fy1 in frames:
        inside = x0 >= fx0 - 2 and y0 >= fy0 - 2 and x1 <= fx1 + 2 and y1 <= fy1 + 2
        if inside and fx1 - fx0 < 0.95 * w:  # not the sheet frame itself
            if fx0 >= 0.5 * w and fy0 >= CORNER[1] * h * 0.95:
                return "corner"
            if fy1 <= TOP_BAND * h * 1.2:
                return "top"
    if _share(box, (CORNER[0] * w, CORNER[1] * h, w, h)) >= PROTECT_SHARE:
        return "corner"
    if _share(box, (0, 0, w, TOP_BAND * h)) >= PROTECT_SHARE:
        return "top"
    return ""


def _walls(rule: np.ndarray, box, reach: int, along: int) -> list[tuple[int, int]]:
    """Runs of positions (columns for ``along=1``, rows for ``along=0``) where a rule crosses the
    whole box and continues ``reach`` pixels past it on at least one side."""
    x0, y0, x1, y1 = box
    hh, ww = rule.shape
    if along == 1:  # vertical rules: columns
        inner = rule[y0:y1, x0:x1].mean(axis=0) >= WALL_SPAN
        above = rule[max(y0 - reach, 0) : y0, x0:x1].all(axis=0) if y0 >= reach else np.zeros(x1 - x0, bool)
        below = rule[y1 : min(y1 + reach, hh), x0:x1].all(axis=0) if y1 + reach <= hh else np.zeros(x1 - x0, bool)
    else:  # horizontal rules: rows
        inner = rule[y0:y1, x0:x1].mean(axis=1) >= WALL_SPAN
        above = rule[y0:y1, max(x0 - reach, 0) : x0].all(axis=1) if x0 >= reach else np.zeros(y1 - y0, bool)
        below = rule[y0:y1, x1 : min(x1 + reach, ww)].all(axis=1) if x1 + reach <= ww else np.zeros(y1 - y0, bool)
    return layout.runs_of(inner & (above | below))


def split_at_walls(text_ink: Ink, rule: np.ndarray, box, frames=(), depth: int = 0) -> list[tuple]:
    """``box`` cut where a rule crosses it and the protection differs on the two sides (see the
    module notes); pieces tightened to their ink."""
    x0, y0, x1, y1 = box
    w, h = text_ink.w, text_ink.h
    reach = max(int(round(WALL_REACH * text_ink.unit)), 4)
    for along in (1, 0):
        lo, hi = (x0, x1) if along == 1 else (y0, y1)
        cuts = [(lo + a, lo + b) for a, b in _walls(rule, box, reach, along) if a > 0 and lo + b < hi]
        if not cuts or depth > 4:
            continue
        pieces, start = [], lo
        for a, b in cuts + [(hi, hi)]:
            t = text_ink.tighten((start, y0, a, y1) if along == 1 else (x0, start, x1, a))
            if t is not None:
                pieces.append(t)
            start = b
        # rejoin neighbours on the same side of the protection boundary
        joined: list[tuple] = []
        for p in pieces:
            if joined and bool(protected_region(joined[-1], w, h, frames)) == bool(protected_region(p, w, h, frames)):
                q = joined[-1]
                joined[-1] = (min(q[0], p[0]), min(q[1], p[1]), max(q[2], p[2]), max(q[3], p[3]))
            else:
                joined.append(p)
        if len(joined) < 2:
            continue
        return [r for p in joined for r in split_at_walls(text_ink, rule, p, frames, depth + 1)]
    return [box]


def page_blocks(image) -> tuple[list[Block], dict]:
    """Text blocks of a sheet with their protection, plus the measurements behind them."""
    rgb = as_rgb(image)
    ink = Ink(rgb)
    prim = primitives.extract(ink)
    text_ink = ink.with_mask(prim.text)
    boxes = layout.blocks(text_ink)
    frames = [p.box for p in prim.of("box")]
    if prim.line is not None:
        boxes = [p for b in boxes for p in split_at_walls(text_ink, prim.line, b, frames)]
    h, w = ink.h, ink.w
    out = []
    for b in boxes:
        region = protected_region(b, w, h, frames)
        out.append(Block(b, bool(region), region))
    return out, {"ink": ink, "primitives": prim}


def crop_of(rgb: np.ndarray, box, pad: int = PAD) -> np.ndarray:
    x0, y0, x1, y1 = box
    hh, ww = rgb.shape[:2]
    return rgb[max(y0 - pad, 0) : min(y1 + pad, hh), max(x0 - pad, 0) : min(x1 + pad, ww)]


def prepare(rgb: np.ndarray, blocks: list[Block], text_px: float = 30.0) -> None:
    for b in blocks:
        b.prepared = prepare_crop(crop_of(rgb, b.box), text_px=text_px)


def read_page(model, image, bank=None, text_px: float = 30.0, max_tokens: int = 768, batch_patches: int = 4_096) -> list[Block]:
    """Blocks of a sheet with their text; ``bank`` (a :class:`bank.Bank`) may drop boilerplate first."""
    from ..decode import read_many

    rgb = as_rgb(image)
    blocks, _ = page_blocks(rgb)
    prepare(rgb, blocks, text_px)
    if bank is not None:
        bank.apply(model, rgb, blocks)
    todo = [b for b in blocks if not b.dropped]
    outs = read_many(model, [(b.prepared.image, b.prepared.prompt) for b in todo], max_tokens=max_tokens, batch_patches=batch_patches)
    for b, o in zip(todo, outs):
        b.text = o.text
        b.extra["stats"] = o.stats
    return blocks
