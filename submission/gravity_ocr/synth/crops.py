"""Crop-level scenarios: one image, one prompt, one ground truth.

These are the pieces a page is cut into, drawn directly so that every difficulty
can be dosed on its own:

``line``       one printed line (codes, names, readings), font 12 … 26 px
``box``        an equipment box: frame, symbol, three to five lines
``table``      a ruled table
``ruleless``   the same content with the rules left out (aligned columns only)
``crossed``    a printed line with a pipe or rule running through it
``title``      a title bar or a corner title block
``hand``       a hand-written value
``form``       form rows: printed label, hand-written value
``note``       a hand-written multi-line note in red pen
``overrule``   pen writing across printed rules
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from PIL import Image

from . import scan
from .canvas import BLUE, INK, MEDIUM_STYLE, RED, WHITE, Canvas, draw_symbol
from .drawings import SYMBOL_PAD, SYMBOL_SIZE, TABLE_HEAD, draw_table, draw_title_bar, draw_title_corner, sensor_line, table_cells, title_cells
from .ledger import Ledger, fmt_value, short_tag

SCENARIOS = ["line", "box", "table", "ruleless", "crossed", "title", "hand", "form", "note", "overrule"]
DEFAULT_COUNTS = {"line": 16, "box": 8, "table": 8, "ruleless": 8, "crossed": 12, "title": 6, "hand": 10, "form": 6, "note": 6, "overrule": 6}


@dataclass
class Sample:
    scenario: str
    image: Image.Image
    prompt: str  # "text" or "table"
    gt_text: str
    gt_cells: list | None = None
    meta: dict = field(default_factory=dict)
    id: str = ""


def finish(c: Canvas, rng: random.Random, tilt: float = 0.5) -> Image.Image:
    img = c.img
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-tilt, tilt), resample=Image.BICUBIC, fillcolor=WHITE)
    return scan.effects(img, rng, edge=False)


def text_line(led: Ledger, rng: random.Random) -> tuple[str, str, str]:
    """A line as it appears on a sheet: ``(what, text, font role)``."""
    e = rng.choice(led.equipment)
    sensors = led.sensors or [None]
    s = rng.choice(led.sensors_of(e.kks) or sensors)
    d = rng.choice(led.drawings)
    what = rng.choice(["kks", "name", "loc", "sensor", "docno", "tag", "limit", "kksname"])
    if what == "kks":
        return what, f"{led.code_label}: {e.kks}", "bold"
    if what == "name":
        return what, e.name, "gothic"
    if what == "loc":
        return what, f"위치: {led.location(e)}", "gothic"
    if what == "sensor":
        return what, sensor_line(s), "gothic"
    if what == "docno":
        return what, f"도면번호: {d.id} ({d.rev})", "bold"
    if what == "tag":
        return what, s.tag, rng.choice(["gothic", "bold"])
    if what == "limit":
        return what, f"경보한계 {fmt_value(s.alarm, s.decimals)} {s.unit}", "gothic"
    return what, f"{e.kks} {e.name}", "gothic"


def line(led: Ledger, rng: random.Random) -> Sample:
    what, s, role = text_line(led, rng)
    size = rng.choice([12, 13, 14, 15, 16, 16, 18, 20, 21, 24, 26])
    probe = Canvas(8, 8, rng)
    w = int(probe.width(s, size, role)) + rng.randint(14, 30)
    h = int(size * 1.5) + rng.randint(8, 18)
    c = Canvas(w, h, rng)
    c.text(rng.randint(5, 12), rng.randint(3, 8), s, size, role)
    return Sample("line", finish(c, rng), "text", s, meta={"what": what, "font_px": size, "role": role})


def box(led: Ledger, rng: random.Random) -> Sample:
    e = rng.choice([x for x in led.equipment if led.sensors_of(x.kks)] or led.equipment)
    bw, bh, m = 330, 175, rng.randint(8, 18)
    c = Canvas(bw + 2 * m, bh + 2 * m, rng)
    x0, y0, x1, y1 = m, m, m + bw, m + bh
    style, width = MEDIUM_STYLE[rng.choice(sorted(MEDIUM_STYLE))]
    c.line([(0, (y0 + y1) / 2), (x0, (y0 + y1) / 2)], style, width)
    c.d.rectangle([x0, y0, x1, y1], fill=WHITE, outline=INK, width=3)
    sx, sy = x1 - SYMBOL_PAD - SYMBOL_SIZE / 2, y0 + SYMBOL_PAD + SYMBOL_SIZE / 2
    draw_symbol(c.d, e.kind, sx, sy, SYMBOL_SIZE, 3, INK, variant=rng.randint(0, 1))
    lines = [(f"{led.code_label}: {e.kks}", 21, "bold"), (e.name, 21, "gothic"), (f"위치: {led.location(e)}", 16, "gothic")]
    lines += [(sensor_line(s), 16, "gothic") for s in led.sensors_of(e.kks)[:2]]
    ty = y0 + 10
    for s, size, role in lines:
        beside = ty < y0 + SYMBOL_PAD * 2 + SYMBOL_SIZE
        size = c.fit(s, size, bw - 22 - (SYMBOL_SIZE + SYMBOL_PAD * 2 + 4 if beside else 0), role, floor=12)
        c.text(x0 + 11, ty, s, size, role)
        ty += size * 1.45
    text = "\n".join(s for s, _, _ in lines)
    return Sample("box", finish(c, rng), "text", text, meta={"kks": e.kks, "kind": e.kind, "lines": len(lines)})


def _table(led: Ledger, rng: random.Random, rules: bool) -> Sample:
    sensors = rng.sample(led.sensors, min(len(led.sensors), rng.randint(3, 6)))
    cols = sorted(rng.sample(range(6), rng.randint(4, 6)))
    if 1 not in cols:  # always keep the tag column: it is what the ledger is matched on
        cols = sorted(cols[1:] + [1])
    head = [TABLE_HEAD[i] for i in cols]
    rows = [[table_cells(s)[i] for i in cols] for s in sensors]
    size = rng.choice([14, 15, 16])
    probe = Canvas(8, 8, rng)
    widths = [max(probe.width(r[i], size + 1, "bold") for r in [head] + rows) + rng.randint(22, 46) for i in range(len(cols))]
    total = sum(widths)
    row_h = rng.choice([26, 28, 30, 34])
    m = rng.randint(8, 16)
    c = Canvas(int(total) + 2 * m, row_h * (len(rows) + 1) + 2 * m, rng)
    t = draw_table(c, (m, m, m + total, m + row_h * (len(rows) + 1)), head, rows, [w / total for w in widths], rules=rules, size=size)
    cells = t["gt"]["cells"]
    name = "table" if rules else "ruleless"
    meta = {"rows": len(cells), "cols": len(cols), "font_px": size, "row_h": row_h}
    return Sample(name, finish(c, rng, 0.3), "table", t["gt"]["text"], cells, meta)


def table(led: Ledger, rng: random.Random) -> Sample:
    return _table(led, rng, True)


def ruleless(led: Ledger, rng: random.Random) -> Sample:
    return _table(led, rng, False)


def crossed(led: Ledger, rng: random.Random) -> Sample:
    what, s, role = text_line(led, rng)
    size = rng.choice([16, 18, 20, 21, 24])
    probe = Canvas(8, 8, rng)
    tw = int(probe.width(s, size, role))
    mx, my = rng.randint(16, 30), rng.randint(14, 24)
    w, h = tw + 2 * mx, int(size * 1.4) + 2 * my
    c = Canvas(w, h, rng)
    mode = rng.choice(["h", "h", "v", "v", "d"])
    colour_name = rng.choice(["ink", "ink", "blue", "red"])
    colour = {"ink": INK, "blue": BLUE, "red": RED}[colour_name]
    style = rng.choice(["solid", "solid", "dash", "dashdot"])
    width = rng.choice([2, 3, 4])
    # the line goes down first for half of them (text printed over a pipe) …
    under = rng.random() < 0.5
    top = my

    def stroke():
        if mode == "h":
            y = top + size * rng.uniform(0.3, 0.95)
            c.line([(0, y), (w, y)], style, width, colour)
        elif mode == "v":
            for _ in range(rng.randint(1, 2)):
                x = mx + tw * rng.uniform(0.1, 0.9)
                c.line([(x, 0), (x, h)], style, width, colour)
        else:
            y0, y1 = rng.choice([(0, h), (h, 0)])
            x0 = mx + tw * rng.uniform(0.0, 0.5)
            c.line([(x0, y0), (x0 + tw * rng.uniform(0.3, 0.5), y1)], style, width, colour)

    if under:
        stroke()
    c.text(mx, top, s, size, role)
    if not under:
        stroke()
    meta = {"what": what, "font_px": size, "mode": mode, "colour": colour_name, "style": style, "width": width, "under": under}
    return Sample("crossed", finish(c, rng, 0.3), "text", s, meta=meta)


def title(led: Ledger, rng: random.Random) -> Sample:
    d = rng.choice(led.drawings)
    cells = title_cells(led, d.id)
    m = rng.randint(8, 14)
    if rng.random() < 0.5:
        w, h = 1880, 105
        c = Canvas(w + 2 * m, h + 2 * m, rng)
        draw_title_bar(c, (m, m, m + w, m + h), cells)
        style = "bar"
    else:
        w, h = 660, 273
        c = Canvas(w + 2 * m, h + 2 * m, rng)
        p = led.person(d.signer)
        draw_title_corner(c, (m, m, m + w, m + h), cells, f"{p.name} ({p.id})", d.issued)
        style = "corner"
    rec = next(g for g in c.gt if g["kind"] == "title")
    return Sample("title", finish(c, rng, 0.3), "text", "\n".join(rec["cells"]), meta={"style": style, "drawing": d.id})


def hand_value(led: Ledger, rng: random.Random) -> tuple[str, str]:
    s = rng.choice(led.sensors)
    p = rng.choice(led.people)
    what = rng.choice(["reading", "reading", "tag", "kks", "name", "id", "when"])
    if what == "reading":
        return what, f"{fmt_value(round(s.normal * rng.uniform(0.9, 1.1), s.decimals), s.decimals)} {s.unit}"
    if what == "tag":
        return what, s.tag
    if what == "kks":
        return what, s.kks
    if what == "name":
        return what, p.name
    if what == "id":
        return what, p.id
    return what, f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(6, 22):02d}:{rng.choice(['00', '15', '30', '45'])}"


def hand(led: Ledger, rng: random.Random) -> Sample:
    what, s = hand_value(led, rng)
    size = rng.choice([28, 30, 32, 34, 36])
    c = Canvas(int(len(s) * size * 0.95) + 80, size * 2 + 30, rng)
    colour_name = rng.choice(["blue", "blue", "red"])
    right, bottom = c.hand(18, 16, s, size, BLUE if colour_name == "blue" else RED)
    c.img = c.img.crop((0, 0, min(int(right) + rng.randint(12, 24), c.w), min(int(bottom) + rng.randint(10, 18), c.h)))
    return Sample("hand", finish(c, rng), "text", s, meta={"what": what, "font_px": size, "colour": colour_name})


def form(led: Ledger, rng: random.Random) -> Sample:
    s = rng.choice(led.sensors)
    p = rng.choice(led.people)
    reading = fmt_value(round(s.normal * rng.uniform(0.9, 1.1), s.decimals), s.decimals)
    pool = [("점검자", p.name), ("사번", p.id), (f"설비 {led.code_label}", s.kks), ("센서 태그", s.tag), ("현장 실측치", reading), ("단위", s.unit)]
    rows = sorted(rng.sample(range(len(pool)), rng.randint(2, 4)))
    fields = [pool[i] for i in rows]
    label_w, value_w, row_h, m = 260, 560, 78, 10
    c = Canvas(label_w + value_w + 2 * m, row_h * len(fields) + 2 * m, rng)
    for i, (label, value) in enumerate(fields):
        y = m + row_h * i
        c.d.rectangle([m, y, m + label_w, y + row_h], fill=(236, 238, 241), outline=INK, width=2)
        c.d.rectangle([m + label_w, y, m + label_w + value_w, y + row_h], outline=INK, width=2)
        c.text(m + 18, y + 22, label, 24, "bold")
        c.hand(m + label_w + 24, y + 18, value, 34, BLUE)
    cells = [list(f) for f in fields]
    return Sample("form", finish(c, rng, 0.3), "table", "\n".join(" | ".join(r) for r in cells), cells, {"rows": len(cells), "cols": 2})


def note(led: Ledger, rng: random.Random) -> Sample:
    moc = rng.choice(led.mocs)
    s = next(x for x in led.sensors if x.tag == moc.tag)
    p = led.person(moc.approved_by)
    lines = [
        moc.moc_no,
        f"{short_tag(moc.tag)} {moc.param}",
        f"{fmt_value(moc.old, s.decimals)} → {fmt_value(moc.new, s.decimals)} {moc.unit}",
        f"{moc.approved_on} {p.name}",
    ]
    lines = lines[: rng.randint(2, 4)]
    size = rng.choice([24, 26, 30])
    c = Canvas(int(max(len(x) for x in lines) * size * 0.95) + 80, (size + 22) * len(lines) + 60, rng)
    y, right = 14, 0
    for x in lines:
        rx, ry = c.hand(16, y, x, size, RED, kind="note")
        right, y = max(right, rx), ry + 8
    c.img = c.img.crop((0, 0, min(int(right) + 18, c.w), min(int(y) + 8, c.h)))
    return Sample("note", finish(c, rng), "text", "\n".join(lines), meta={"lines": len(lines), "font_px": size})


def overrule(led: Ledger, rng: random.Random) -> Sample:
    what, s = hand_value(led, rng)
    size = rng.choice([28, 32, 34])
    w, h = int(len(s) * size * 0.95) + 90, size * 2 + 40
    c = Canvas(w, h, rng)
    grid_h = rng.choice([24, 28, 34])
    y = rng.randint(4, grid_h)
    rules = 0
    while y < h:
        c.d.line([(0, y), (w, y)], fill=INK, width=rng.choice([1, 2]))
        y += grid_h
        rules += 1
    for _ in range(rng.randint(1, 3)):
        x = rng.randint(20, w - 20)
        c.d.line([(x, 0), (x, h)], fill=INK, width=1)
    colour_name = rng.choice(["blue", "red"])
    c.hand(20, 14, s, size, BLUE if colour_name == "blue" else RED)
    return Sample("overrule", finish(c, rng, 0.3), "text", s, meta={"what": what, "font_px": size, "colour": colour_name, "rules": rules})


GENERATORS = {
    "line": line, "box": box, "table": table, "ruleless": ruleless, "crossed": crossed,
    "title": title, "hand": hand, "form": form, "note": note, "overrule": overrule,
}  # fmt: skip


def make_set(led: Ledger, seed: int, counts: dict | None = None, prefix: str = "") -> list[Sample]:
    """Scenario samples for one ledger; each scenario has its own random stream."""
    counts = counts or DEFAULT_COUNTS
    out = []
    for name in SCENARIOS:
        rng = random.Random(f"{seed}:{name}")
        seen = set()
        for i in range(counts.get(name, 0)):
            for _ in range(20):  # do not repeat a ground truth inside a set
                s = GENERATORS[name](led, rng)
                if s.gt_text not in seen:
                    break
            seen.add(s.gt_text)
            s.id = f"{prefix}{name}-{i:02d}"
            out.append(s)
    return out
