"""What a scanner does to a sheet: a slight turn, blur, paper tint, sensor noise."""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageFilter


def rotate_box(box, deg: float, w: int, h: int) -> list[int]:
    """Bounding box of ``box`` after the sheet is rotated by ``deg`` about its centre."""
    a = math.radians(deg)
    cx, cy = w / 2, h / 2
    xs, ys = [], []
    for x, y in ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])):
        dx, dy = x - cx, y - cy
        xs.append(cx + dx * math.cos(a) + dy * math.sin(a))
        ys.append(cy - dx * math.sin(a) + dy * math.cos(a))
    return [int(max(min(xs), 0)), int(max(min(ys), 0)), int(min(max(xs), w)), int(min(max(ys), h))]


def effects(img: Image.Image, rng: random.Random, blur=(0.2, 0.5), noise: float = 2.0, edge: bool = True) -> Image.Image:
    """Blur, tint and noise without moving anything."""
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(*blur)))
    a = np.asarray(img, dtype=np.float32)
    tint = np.array([rng.uniform(246, 255), rng.uniform(245, 254), rng.uniform(240, 252)], np.float32) / 255.0
    a = a * tint
    a += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0.0, noise, a.shape).astype(np.float32)
    if edge:
        h, w = a.shape[:2]
        if rng.random() < 0.4:  # the lid lets light in along one edge
            n = min(rng.randint(6, 26), w // 4)
            ramp = np.linspace(rng.uniform(0.6, 0.85), 1.0, n, dtype=np.float32)
            a[:, :n] *= ramp[None, :, None]
        if rng.random() < 0.25 and h > 12:  # a streak of the scan head
            y = rng.randint(h // 3, 2 * h // 3)
            a[y : y + 2] *= 0.88
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def scanned(img: Image.Image, gt: list[dict], rng: random.Random, max_deg: float = 0.8) -> tuple[Image.Image, list[dict], float]:
    """A scan of a page; ground-truth boxes follow the rotation."""
    deg = rng.uniform(-max_deg, max_deg)
    out = img.rotate(deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    out = effects(out, rng)
    moved = [{**g, "box": rotate_box(g["box"], deg, img.width, img.height)} for g in gt]
    return out, moved, deg
