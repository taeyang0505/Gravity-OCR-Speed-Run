"""A drawing surface that remembers what was written where.

Every text that is drawn with a ``kind`` leaves a ground-truth record
``{"kind", "box", "text"}``; the scenario generators and the evaluation read
those records instead of guessing from pixels.
"""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

from .fonts import font

INK = (20, 20, 25)
BLUE = (25, 35, 120)
RED = (205, 25, 25)
GRAY = (130, 130, 135)
SHADE = (236, 238, 241)
WHITE = (255, 255, 255)
DASHES = {"solid": None, "dash": [18, 10], "dot": [4, 8], "dashdot": [22, 7, 4, 7]}
MEDIUM_STYLE = {
    "power": ("dash", 2), "draft": ("dashdot", 3), "oil": ("dot", 3), "shaft": ("solid", 2),
    "steam": ("solid", 4), "water": ("solid", 4),
}  # fmt: skip


class Canvas:
    def __init__(self, w: int, h: int, rng: random.Random, background=WHITE):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w, h), background)
        self.d = ImageDraw.Draw(self.img)
        self.rng = rng
        self.gt: list[dict] = []

    # ------------------------------------------------------------------ ground truth
    def note(self, kind: str, box, text: str, **extra) -> dict:
        rec = {"kind": kind, "box": [int(round(v)) for v in box], "text": text, **extra}
        self.gt.append(rec)
        return rec

    # ------------------------------------------------------------------ printed text
    def width(self, s: str, size: int, role: str = "gothic") -> float:
        return self.d.textlength(s, font=font(role, size))

    def text(self, x, y, s: str, size: int = 20, role: str = "gothic", fill=INK, kind: str | None = None):
        """Draw ``s`` with its top-left at ``(x, y)``; returns the right and bottom edge."""
        f = font(role, size)
        self.d.text((x, y), s, font=f, fill=fill)
        x0, y0, x1, y1 = self.d.textbbox((x, y), s, font=f)
        if kind:
            self.note(kind, (x0, y0, x1, y1), s)
        return x1, y1

    def fit(self, s: str, size: int, room: float, role: str = "gothic", floor: int = 10) -> int:
        """Largest size from ``size`` down to ``floor`` at which ``s`` fits in ``room``."""
        while size > floor and self.width(s, size, role) > room:
            size -= 1
        return size

    def tag(self, cx, cy, s: str, size: int = 15, fill=GRAY, kind: str | None = "label"):
        """Text centred on a point with the paper knocked out underneath."""
        f = font("gothic", size)
        x0, y0, x1, y1 = self.d.textbbox((0, 0), s, font=f)
        w, h = x1 - x0, y1 - y0
        x, y = cx - w / 2 - x0, cy - h / 2 - y0
        self.d.rectangle([cx - w / 2 - 3, cy - h / 2 - 2, cx + w / 2 + 3, cy + h / 2 + 2], fill=WHITE)
        self.d.text((x, y), s, font=f, fill=fill)
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if kind:
            self.note(kind, box, s)
        return box

    def vtext(self, cx, cy, s: str, size: int = 13, fill=GRAY, kind: str | None = "label"):
        """Text rotated a quarter turn (reads bottom to top), centred on a point."""
        f = font("gothic", size)
        x0, y0, x1, y1 = self.d.textbbox((0, 0), s, font=f)
        tile = Image.new("RGB", (x1 - x0 + 6, y1 - y0 + 4), WHITE)
        ImageDraw.Draw(tile).text((3 - x0, 2 - y0), s, font=f, fill=fill)
        tile = tile.rotate(90, expand=True)
        x, y = int(cx - tile.width / 2), int(cy - tile.height / 2)
        self.img.paste(tile, (x, y))
        box = (x, y, x + tile.width, y + tile.height)
        if kind:
            self.note(kind, box, s, rotated=True)
        return box

    # ------------------------------------------------------------------ handwriting
    def hand(self, x, y, s: str, size: int = 30, color=BLUE, kind: str | None = "value"):
        """Pen writing: every character on its own slightly rotated, shifted tile."""
        rng = self.rng
        tilt = rng.uniform(-2.5, 2.5)
        cx = float(x)
        drop = 0.0
        x0 = y0 = 1e9
        x1 = y1 = -1e9
        for ch in s:
            if ch == " ":
                cx += size * 0.45
                continue
            sz = int(size * rng.uniform(0.92, 1.1))
            f = font("hand", sz)
            pad = sz
            tile = Image.new("RGBA", (sz * 2, sz * 2), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((pad // 2, pad // 2), ch, font=f, fill=(*color, 235))
            tile = tile.rotate(rng.uniform(-7, 7) + tilt, resample=Image.BICUBIC, expand=False)
            bbox = tile.getbbox()
            if bbox is None:
                cx += size * 0.3
                continue
            # keep the vertical placement of the glyph inside its line box
            tile = tile.crop((bbox[0], 0, bbox[2], tile.height))
            top = int(y - pad // 2 + drop + rng.uniform(-3, 3))
            self.img.paste(tile, (int(cx), top), tile)
            x0, y0 = min(x0, cx), min(y0, top + bbox[1])
            x1, y1 = max(x1, cx + tile.width), max(y1, top + bbox[3])
            cx += tile.width + rng.uniform(1, 6)
            drop += rng.uniform(-0.6, 0.6)
        self.d = ImageDraw.Draw(self.img)
        if x1 < x0:
            return x, y
        if kind:
            self.note(kind, (x0, y0, x1, y1), s, hand=True)
        return x1, y1

    def scribble(self, x, y, w, h, color=BLUE, kind: str | None = "sign", text: str = ""):
        """A signature: a few joined quadratic curves."""
        rng = self.rng
        pts = []
        px, py = x + rng.uniform(0, w * 0.15), y + h * rng.uniform(0.3, 0.7)
        for _ in range(rng.randint(3, 5)):
            cx, cy = px + rng.uniform(0.05, 0.25) * w, y + rng.uniform(0, h)
            ex, ey = min(px + rng.uniform(0.15, 0.3) * w, x + w), y + rng.uniform(0.15, 0.85) * h
            for k in range(40):
                t = k / 39
                pts.append(
                    ((1 - t) ** 2 * px + 2 * (1 - t) * t * cx + t * t * ex, (1 - t) ** 2 * py + 2 * (1 - t) * t * cy + t * t * ey)
                )
            px, py = ex, ey
        self.d.line(pts, fill=color, width=3, joint="curve")
        if kind:
            self.note(kind, (x, y, x + w, y + h), text)

    def stamp(self, cx, cy, name: str, r: int = 40, color=RED):
        self.d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=4)
        f = font("bold", 24)
        s = name[:3]
        x0, y0, x1, y1 = self.d.textbbox((0, 0), s, font=f)
        self.d.text((cx - (x1 - x0) / 2 - x0, cy - (y1 - y0) / 2 - y0), s, font=f, fill=color)
        self.note("stamp", (cx - r, cy - r, cx + r, cy + r), s)

    # ------------------------------------------------------------------ strokes
    def line(self, pts, style: str = "solid", width: int = 3, fill=INK):
        pattern = DASHES.get(style)
        pts = [(float(x), float(y)) for x, y in pts]
        if not pattern:
            self.d.line(pts, fill=fill, width=width, joint="curve")
            return
        idx, left = 0, float(pattern[0])
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg == 0:
                continue
            ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
            t = 0.0
            while t < seg - 1e-9:
                step = min(left, seg - t)
                if idx % 2 == 0:
                    self.d.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * (t + step), y0 + uy * (t + step))], fill=fill, width=width)
                t += step
                left -= step
                if left <= 1e-9:
                    idx = (idx + 1) % len(pattern)
                    left = float(pattern[idx])

    def arrow(self, x, y, direction: str, size: int = 14, fill=INK):
        """A filled arrowhead whose tip is at ``(x, y)``."""
        s, h = size, size * 0.55
        tri = {
            "right": [(x, y), (x - s, y - h), (x - s, y + h)],
            "left": [(x, y), (x + s, y - h), (x + s, y + h)],
            "down": [(x, y), (x - h, y - s), (x + h, y - s)],
            "up": [(x, y), (x - h, y + s), (x + h, y + s)],
        }[direction]
        self.d.polygon(tri, fill=fill)


def draw_symbol(d: ImageDraw.ImageDraw, cls: str, cx: float, cy: float, s: float, w: int = 3, fill=INK, variant: int = 0):
    """One of the equipment symbols, centred on ``(cx, cy)`` in a square of side ``s``."""
    h = s / 2
    if cls == "pump":
        r = 0.32 * s
        d.polygon([(cx - 0.3 * s, cy + h), (cx + 0.3 * s, cy + h), (cx, cy + 0.1 * s)], outline=fill, width=w)
        d.ellipse([cx - r, cy - r - 0.08 * s, cx + r, cy + r - 0.08 * s], outline=fill, width=w, fill=WHITE)
        d.line([(cx, cy - r - 0.08 * s), (cx + h, cy - r - 0.08 * s)], fill=fill, width=w)
        d.line([(cx + h, cy - r - 0.2 * s), (cx + h, cy - r + 0.04 * s)], fill=fill, width=w)
    elif cls == "valve":
        a = 0.26 * s
        d.polygon([(cx - h, cy - a), (cx - h, cy + a), (cx, cy)], outline=fill, width=w)
        d.polygon([(cx + h, cy - a), (cx + h, cy + a), (cx, cy)], outline=fill, width=w)
        if variant % 2 == 0:
            d.line([(cx, cy), (cx, cy - 0.34 * s)], fill=fill, width=w)
            d.pieslice([cx - 0.2 * s, cy - 0.54 * s, cx + 0.2 * s, cy - 0.14 * s], 180, 360, outline=fill, width=w)
    elif cls == "turbine":
        d.polygon(
            [(cx - h, cy - 0.2 * s), (cx + h, cy - 0.46 * s), (cx + h, cy + 0.46 * s), (cx - h, cy + 0.2 * s)],
            outline=fill, width=w,
        )  # fmt: skip
    elif cls == "fan":
        r = 0.42 * s
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fill, width=w)
        blades = 3 + variant % 2
        for k in range(blades):
            a = 2 * math.pi * k / blades + 0.4
            d.line([(cx, cy), (cx + r * 0.9 * math.cos(a), cy + r * 0.9 * math.sin(a))], fill=fill, width=w)
    elif cls == "exchanger":
        r = 0.4 * s
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fill, width=w)
        zig = [(cx - h, cy), (cx - 0.2 * s, cy), (cx - 0.1 * s, cy - 0.2 * s), (cx + 0.1 * s, cy + 0.2 * s), (cx + 0.2 * s, cy), (cx + h, cy)]
        d.line(zig, fill=fill, width=w, joint="curve")
    elif cls == "vessel":
        d.rounded_rectangle([cx - 0.3 * s, cy - h, cx + 0.3 * s, cy + h], radius=0.22 * s, outline=fill, width=w)
    elif cls == "breaker":
        a = 0.36 * s
        d.rectangle([cx - a, cy - a, cx + a, cy + a], outline=fill, width=w)
        d.line([(cx - a, cy - a), (cx + a, cy + a)], fill=fill, width=w)
        d.line([(cx - a, cy + a), (cx + a, cy - a)], fill=fill, width=w)
    elif cls == "transformer":
        r, o = 0.27 * s, 0.15 * s
        d.ellipse([cx - r, cy - o - r, cx + r, cy - o + r], outline=fill, width=w)
        d.ellipse([cx - r, cy + o - r, cx + r, cy + o + r], outline=fill, width=w)
    elif cls == "generator":
        r = 0.42 * s
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fill, width=w)
        wave = [(cx - 0.26 * s + 0.52 * s * k / 24, cy - 0.14 * s * math.sin(2 * math.pi * k / 24)) for k in range(25)]
        d.line(wave, fill=fill, width=w, joint="curve")
    elif cls == "panel":
        d.rectangle([cx - 0.4 * s, cy - 0.44 * s, cx + 0.4 * s, cy + 0.44 * s], outline=fill, width=w)
        d.line([(cx - 0.4 * s, cy - 0.14 * s), (cx + 0.4 * s, cy - 0.14 * s)], fill=fill, width=w)
        d.line([(cx - 0.4 * s, cy + 0.14 * s), (cx + 0.4 * s, cy + 0.14 * s)], fill=fill, width=w)
    # data-centre symbols
    elif cls == "rack":
        d.rectangle([cx - 0.28 * s, cy - h, cx + 0.28 * s, cy + h], outline=fill, width=w)
        for k in range(1, 4 + variant % 2):
            y = cy - h + k * s / (4 + variant % 2)
            d.line([(cx - 0.2 * s, y), (cx + 0.2 * s, y)], fill=fill, width=w)
    elif cls == "manifold":
        d.line([(cx - h, cy - 0.12 * s), (cx + h, cy - 0.12 * s)], fill=fill, width=w + 2)
        for k in (-0.3, 0.0, 0.3):
            d.line([(cx + k * s, cy - 0.12 * s), (cx + k * s, cy + 0.3 * s)], fill=fill, width=w)
    elif cls == "ups":
        a = 0.4 * s
        d.rectangle([cx - a, cy - a, cx + a, cy + a], outline=fill, width=w)
        d.line([(cx - a, cy + a), (cx + a, cy - a)], fill=fill, width=w)
        wave = [(cx - 0.32 * s + 0.24 * s * k / 12, cy - 0.2 * s - 0.06 * s * math.sin(2 * math.pi * k / 12)) for k in range(13)]
        d.line(wave, fill=fill, width=w, joint="curve")
        d.line([(cx + 0.08 * s, cy + 0.18 * s), (cx + 0.32 * s, cy + 0.18 * s)], fill=fill, width=w)
        d.line([(cx + 0.08 * s, cy + 0.26 * s), (cx + 0.32 * s, cy + 0.26 * s)], fill=fill, width=w)
    elif cls == "battery":
        for k in range(2 + variant % 2):
            x = cx - 0.3 * s + k * 0.3 * s
            d.line([(x, cy - 0.36 * s), (x, cy + 0.36 * s)], fill=fill, width=w)
            d.line([(x + 0.12 * s, cy - 0.18 * s), (x + 0.12 * s, cy + 0.18 * s)], fill=fill, width=w + 2)
    elif cls == "switch":
        a = 0.4 * s
        d.rectangle([cx - a, cy - 0.3 * s, cx + a, cy + 0.3 * s], outline=fill, width=w)
        d.line([(cx - 0.26 * s, cy - 0.1 * s), (cx + 0.26 * s, cy - 0.1 * s)], fill=fill, width=w)
        d.polygon([(cx + 0.26 * s, cy - 0.18 * s), (cx + 0.34 * s, cy - 0.1 * s), (cx + 0.26 * s, cy - 0.02 * s)], fill=fill)
        d.line([(cx - 0.26 * s, cy + 0.1 * s), (cx + 0.26 * s, cy + 0.1 * s)], fill=fill, width=w)
        d.polygon([(cx - 0.26 * s, cy + 0.02 * s), (cx - 0.34 * s, cy + 0.1 * s), (cx - 0.26 * s, cy + 0.18 * s)], fill=fill)
    else:
        raise ValueError(cls)


SYMBOL_CLASSES = ["pump", "valve", "turbine", "fan", "exchanger", "vessel", "breaker", "transformer", "generator", "panel", "rack", "manifold", "ups", "battery", "switch"]
