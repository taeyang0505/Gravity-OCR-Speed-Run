"""Image → patch rows for the vision tower.

Two ways to bring a crop onto the 28 px grid (patch 14 × merge 2):

``fit="pad"``     no resampling; the crop is extended with paper colour to the
                  next multiple of 28 (the plan's "resolution bucket padding").
                  Strokes keep their exact pixels.
``fit="resize"``  the reference behaviour: ``smart_resize`` + bicubic.

A crop outside the pixel budget (below ``min_pixels`` or above ``max_pixels``)
is always resized first, as the reference does.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

PATCH = 14
MERGE = 2
FACTOR = PATCH * MERGE
TEMPORAL = 2
MIN_PIXELS = 12_544
MAX_PIXELS = 9_633_792
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """Target size on the 28 px grid inside the pixel budget (reference rule)."""
    if height < FACTOR or width < FACTOR:
        scale = max(FACTOR / height, FACTOR / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio above 200")
    h_bar = round(height / FACTOR) * FACTOR
    w_bar = round(width / FACTOR) * FACTOR
    if TEMPORAL * h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt(TEMPORAL * height * width / MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif TEMPORAL * h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (TEMPORAL * height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def paper_colour(rgb: np.ndarray) -> tuple[int, int, int]:
    """Median of the border pixels: what the sheet looks like where nothing is drawn."""
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    return tuple(int(v) for v in np.median(edge, axis=0))


def pad_to_grid(rgb: np.ndarray, fill=None) -> np.ndarray:
    """Extend to multiples of 28 (content centred) without resampling."""
    h, w = rgb.shape[:2]
    th = max(FACTOR, math.ceil(h / FACTOR) * FACTOR)
    tw = max(FACTOR, math.ceil(w / FACTOR) * FACTOR)
    while TEMPORAL * th * tw < MIN_PIXELS:
        if tw <= th * 4:
            tw += FACTOR
        else:
            th += FACTOR
    if (th, tw) == (h, w):
        return rgb
    fill = paper_colour(rgb) if fill is None else fill
    out = np.empty((th, tw, 3), np.uint8)
    out[:] = np.asarray(fill, np.uint8)
    y0, x0 = (th - h) // 2, (tw - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = rgb
    return out


def fit_image(image, fit: str = "pad") -> np.ndarray:
    """uint8 RGB array whose sides are multiples of 28 and inside the budget."""
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
    else:
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    h, w = rgb.shape[:2]
    if fit == "pad" and TEMPORAL * h * w <= MAX_PIXELS and max(h, w) / min(h, w) <= 200:
        return pad_to_grid(rgb)
    th, tw = smart_resize(h, w)
    if (th, tw) != (h, w):
        rgb = np.asarray(Image.fromarray(rgb).resize((tw, th), Image.BICUBIC))
    return rgb


def patch_positions(grid_h: int, grid_w: int) -> np.ndarray:
    """(row, col) of every patch in 2×2 merge-group order → ``[N, 2]``."""
    rows = np.arange(grid_h)[:, None].repeat(grid_w, axis=1)
    cols = np.arange(grid_w)[None, :].repeat(grid_h, axis=0)

    def grouped(a):
        return a.reshape(grid_h // MERGE, MERGE, grid_w // MERGE, MERGE).transpose(0, 2, 1, 3).reshape(-1)

    return np.stack([grouped(rows), grouped(cols)], axis=-1)


def to_patches(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Normalise and cut an on-grid RGB image into ``[N, 3·14·14]`` rows.

    Rows come in 2×2 merge-group order (the order the merger consumes them);
    inside a row the layout is channel, y, x.  The reference feeds every still
    image as two identical frames; the runtime folds that into the patch
    embedding weight instead (see :class:`gravity_ocr.vision.VisionTower`).
    """
    h, w = rgb.shape[:2]
    if h % FACTOR or w % FACTOR:
        raise ValueError("image is not on the 28 px grid; call fit_image first")
    gh, gw = h // PATCH, w // PATCH
    from . import native

    k = native.load()
    if k is not None and k.has("pixels") and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3:
        return k.patchify(rgb, (gh, gw), MEAN, STD, np.float32(1 / 255)), (gh, gw)
    return to_patches_np(rgb)


def to_patches_np(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = rgb.shape[:2]
    gh, gw = h // PATCH, w // PATCH
    x = (rgb.astype(np.float32) * np.float32(1 / 255) - MEAN) / STD
    x = x.reshape(gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH, 3)
    x = x.transpose(0, 3, 1, 4, 6, 2, 5)  # group row, group col, i, j, C, y, x
    return np.ascontiguousarray(x.reshape(gh * gw, 3 * PATCH * PATCH)), (gh, gw)


def fold_reference_patches(pixel_values: np.ndarray) -> np.ndarray:
    """Reference ``pixel_values`` ``[N, 3·2·14·14]`` → single-frame rows."""
    n = pixel_values.shape[0]
    return np.ascontiguousarray(pixel_values.reshape(n, 3, TEMPORAL, PATCH * PATCH)[:, :, 0].reshape(n, -1))
