"""Blank-patch pruning (plan Part 1.2 / 4.3): skip what is only paper.

A merge group (2×2 patches = 28×28 px, one image token) whose pixels all sit at
the paper level carries nothing to read.  Two levels of skipping:

``vision``  the group's four patches are left out of the 24 vision blocks (a
            gather of the kept rows with their own rotary positions); its image
            token is filled with the *blank token* – the token the tower gives a
            blank region of a blank page, computed once per model;
``prefill`` the blank image tokens are also left out of the language prefill;
            the kept tokens keep their original ``(row, col)`` positions, so the
            3-axis rotary embedding still places every one of them;
``vq``      as ``vision``, and line-only groups that hit an enabled word of the VQ
            codebook (``vq.py``, ``model.vq``) take the word's token instead of
            going through the tower.

Both change what the model sees (attention no longer spans the blank groups),
so they are measured, not assumed: ``tools/eval.py --prune vision|prefill|vq``.
"""

from __future__ import annotations

import numpy as np

from .preprocess import FACTOR, MERGE, PATCH

BLANK_SPREAD = 16  # a blank group has no pixel more than this darker than the paper


def blank_groups(rgb: np.ndarray, spread: int = BLANK_SPREAD, halo: int = 1) -> np.ndarray:
    """Boolean per merge group (merge-group order, like the image tokens) of an on-grid RGB image.

    When ``halo > 0`` (default ``1``), a 1-hop safety halo around every non-blank (ink) merge group
    is protected via binary dilation so glyph edges, thin table rules and formula dots are never cut.
    """
    from .drawing.ink import dilate, gray_of

    h, w = rgb.shape[:2]
    gray = gray_of(np.ascontiguousarray(rgb)).astype(np.int16)
    paper = int(np.percentile(gray, 90))
    gh, gw = h // FACTOR, w // FACTOR
    g = gray.reshape(gh, FACTOR, gw, FACTOR)
    dark = g.min(axis=(1, 3))
    bright = g.max(axis=(1, 3))
    blank_2d = ((paper - dark) <= spread) & ((bright - dark) <= spread)
    if halo > 0 and gh > 1 and gw > 1:
        ink_2d = ~blank_2d
        if ink_2d.any():
            keep_2d = dilate(ink_2d, halo)
            blank_2d = ~keep_2d
    # image tokens come group row by group row: the row-major order of this grid
    flat = blank_2d.reshape(-1)
    if flat.size >= 2:
        flat[0] = False
        flat[-1] = False
    return flat


def kept_rows(blank: np.ndarray) -> np.ndarray:
    """Patch rows (4 per group, merge-group order) of the groups that are not blank."""
    keep = np.flatnonzero(~blank)
    per = MERGE * MERGE
    return (keep[:, None] * per + np.arange(per)[None, :]).reshape(-1)


def blank_token(model) -> np.ndarray:
    """The image token of a blank region of a blank page (cached on the model)."""
    tok = getattr(model, "_blank_token", None)
    if tok is None:
        from . import preprocess

        rgb = np.full((4 * FACTOR, 4 * FACTOR, 3), 255, np.uint8)
        patches, grid = preprocess.to_patches(rgb)
        tok = model.vision.forward(patches, grid).mean(axis=0).astype(np.float32)
        model._blank_token = tok
    return tok


def encode(model, rgb: np.ndarray, level: str = "vision") -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """Image tokens of an on-grid RGB image with blank (and VQ-hit) groups skipped in the tower.

    Returns the tokens (``[groups, 1536]``), the merged grid and the boolean mask of
    the blank groups (the ones a ``prefill`` level may leave out of the prompt).
    """
    from . import preprocess

    patches, (gh, gw) = preprocess.to_patches(rgb)
    grid = (gh // MERGE, gw // MERGE)
    blank = blank_groups(rgb) if level != "off" else np.zeros(grid[0] * grid[1], bool)
    skip, fill = blank.copy(), {}
    if level == "vq" and getattr(model, "vq", None) is not None:
        from . import vq

        simple = vq.simple_groups(rgb) & ~blank
        idx = np.flatnonzero(simple)
        if idx.size:
            # the patch rows of the simple groups (kept_rows names the rows of the groups a mask leaves)
            word, hit = model.vq.lookup(vq.group_keys(model, patches[kept_rows(~simple)]))
            for g, w in zip(idx[hit].tolist(), word[hit].tolist()):
                fill[g] = w
                skip[g] = True
    if not skip.any():
        return model.vision.forward(patches, (gh, gw)), grid, blank
    out = np.empty((skip.size, 1536), np.float32)
    out[blank] = blank_token(model)
    for g, w in fill.items():
        out[g] = model.vq.values[w]
    if (~skip).any():
        rows = kept_rows(skip)
        out[~skip] = model.vision.forward(patches[rows], (gh, gw), rows=rows)
    model._last_skip = {"blank": int(blank.sum()), "vq": len(fill), "groups": int(skip.size)}
    return out, grid, blank


__all__ = ["BLANK_SPREAD", "blank_groups", "blank_token", "encode", "kept_rows", "PATCH"]
