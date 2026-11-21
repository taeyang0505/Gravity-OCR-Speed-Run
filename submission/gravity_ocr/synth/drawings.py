"""Whole pages: a P&ID master sheet, its red-lined copy and a field gauge card.

Two sheet layouts are produced on purpose.  ``top`` carries the title in a bar
along the top edge, ``corner`` in a block at the bottom right – the place a
coordinate rule for title blocks expects it.  A rule that only works for one of
them is a rule that fails on real drawings.
"""

from __future__ import annotations

import math
import random

from .canvas import BLUE, GRAY, INK, MEDIUM_STYLE, RED, SHADE, WHITE, Canvas, draw_symbol
from .ledger import Ledger, fmt_value, short_tag

DW, DH = 2000, 1400  # drawing sheet
PW, PH = 1240, 1754  # portrait form
COL_X = [0.140, 0.325, 0.510, 0.695]
COL_W = 0.165
ROW_H = 0.125
OPC_W = 0.083
SYMBOL_SIZE, SYMBOL_PAD = 60, 12
TABLE_HEAD = ["설비코드", "센서태그", "항목", "정상치", "경보한계", "트립한계"]
TABLE_SHARES = [0.18, 0.18, 0.22, 0.13, 0.15, 0.14]
TABLE_MAX_ROWS = 8
LAYOUTS = {
    "top": {
        "title": (0.030, 0.020, 0.970, 0.095),
        "rows": [0.130, 0.310, 0.490],
        "table": (0.035, 0.665, 0.660, 0.870),
        "guide": (0.035, 0.895, 0.660, 0.975),
        "sign": (0.720, 0.860, 0.970, 0.975),
    },
    "corner": {
        "title": (0.640, 0.780, 0.970, 0.975),
        "rows": [0.050, 0.240, 0.430],
        "table": (0.035, 0.630, 0.600, 0.850),
        "guide": (0.035, 0.885, 0.600, 0.975),
        "sign": None,
    },
}
# The notices printed on every sheet.  They never change, which is what makes
# them boilerplate; ``NOTICES_ALT`` is a different wording for testing whether a
# filter learnt "boilerplate" or just these three sentences.
NOTICES = [
    "1. 모든 설비 식별 코드는 KKS 규격을 따르며, 센서 태그의 C 다음 글자는 측정 종류를 뜻한다.",
    "2. 실선은 급수 · 증기, 파선은 전력, 일점쇄선은 통풍, 점선은 윤활유 계통을 나타낸다.",
    "3. 이 도면은 합성 시험용이며 실제 발전소 설비와 관계가 없다. 무단 복제를 금한다.",
]
NOTICES_AIDC = [
    "1. 모든 자산 식별 코드는 DCIM 자산코드 규격을 따르며, 센서 태그의 C 다음 글자는 측정 종류를 뜻한다.",
    "2. 실선은 냉각수 · 냉수, 파선은 전력, 일점쇄선은 공조, 점선은 데이터 회선 계통을 나타낸다.",
    "3. 이 도면은 합성 시험용이며 실제 데이터센터 설비와 관계가 없다. 무단 복제를 금한다.",
]
NOTICES_ALT = [
    "1. 치수 단위는 별도 표기가 없는 한 밀리미터이며, 축척은 도면 안의 표기를 따른다.",
    "2. 현장 시공 전에 최신 개정 도면인지 확인하고, 구판 도면은 즉시 폐기한다.",
    "3. 이 도면의 권리는 발주처에 있으며 사전 승인 없이 외부로 반출할 수 없다.",
]


def default_notices(led) -> list[str]:
    return NOTICES_AIDC if getattr(led, "domain", "plant") == "aidc" else NOTICES


def frac(box, w: int, h: int) -> tuple:
    return box[0] * w, box[1] * h, box[2] * w, box[3] * h


def limit_text(s) -> tuple[str, str]:
    return f"{fmt_value(s.alarm, s.decimals)} {s.unit}", f"{fmt_value(s.trip, s.decimals)} {s.unit}"


def sensor_line(s) -> str:
    return f"{short_tag(s.tag)} {s.label} {fmt_value(s.normal, s.decimals)} {s.unit}"


def table_rows(led: Ledger, drawing_id: str, first_tag: str | None = None) -> list:
    """Sensors listed in the limits table, the one named by ``first_tag`` guaranteed to be in."""
    sensors = [s for e in led.equipment_of(drawing_id) for s in led.sensors_of(e.kks)]
    rows = sensors[:TABLE_MAX_ROWS]
    if first_tag and all(s.tag != first_tag for s in rows):
        rows[-1] = next(s for s in sensors if s.tag == first_tag)
    return rows


def table_cells(s) -> list[str]:
    alarm, trip = limit_text(s)
    return [s.kks, s.tag, s.label, f"{fmt_value(s.normal, s.decimals)} {s.unit}", alarm, trip]


def draw_table(c: Canvas, box, head: list[str], rows: list[list[str]], shares: list[float], title: str | None = None, rules: bool = True, size: int = 15, kind: str = "table") -> dict:
    """A table inside ``box``; returns its geometry.  ``rules=False`` leaves only aligned text."""
    tx0, ty0, tx1, ty1 = box
    d = c.d
    head_y = ty0
    if title:
        c.text(tx0, ty0 - 2, title, 20, "bold", kind="table_title")
        head_y = ty0 + 32
    n = len(rows) + 1
    row_h = min(34.0, (ty1 - head_y) / n)
    xs = [tx0]
    for s in shares:
        xs.append(xs[-1] + (tx1 - tx0) * s)
    bottom = head_y + row_h * n
    if rules:
        d.rectangle([tx0, head_y, tx1, head_y + row_h], fill=SHADE)
        d.rectangle([tx0, head_y, tx1, bottom], outline=INK, width=2)
        for x in xs[1:-1]:
            d.line([(x, head_y), (x, bottom)], fill=INK, width=1)
        for r in range(1, n):
            d.line([(tx0, head_y + row_h * r), (tx1, head_y + row_h * r)], fill=INK, width=1)
    pad_y = max((row_h - size * 1.3) / 2, 1)
    for i, h in enumerate(head):
        sz = c.fit(h, size + 1, xs[i + 1] - xs[i] - 12, "bold")
        c.text(xs[i] + 8, head_y + pad_y, h, sz, "bold")
    for r, row in enumerate(rows, 1):
        for i, cell in enumerate(row):
            sz = c.fit(cell, size, xs[i + 1] - xs[i] - 10)
            c.text(xs[i] + 6, head_y + row_h * r + pad_y, cell, sz)
    cells = [head] + rows
    rec = c.note(
        kind, (tx0, head_y, tx1, bottom), "\n".join(" | ".join(r) for r in cells), cells=cells, ruled=rules
    )
    return {"xs": xs, "head_y": head_y, "row_h": row_h, "bottom": bottom, "gt": rec}


def title_cells(led: Ledger, drawing_id: str) -> list[str]:
    dr = led.drawing(drawing_id)
    sec = led.sector(dr.sector)
    return [f"도면번호: {dr.id} ({dr.rev})", dr.title, f"{sec.building} {sec.label} ({sec.code})"]


def draw_title_bar(c: Canvas, box, cells: list[str], size: int = 28) -> None:
    """Three cells side by side in one ruled bar."""
    x0, y0, x1, y1 = box
    c.d.rectangle([x0, y0, x1, y1], outline=INK, width=3)
    shares = [0.30, 0.42, 0.28]
    x = x0
    for i, (s, share) in enumerate(zip(cells, shares)):
        w = (x1 - x0) * share
        sz = c.fit(s, size, w - 30, "bold", floor=16)
        tw = c.width(s, sz, "bold")
        c.text(x + (w - tw) / 2, y0 + (y1 - y0 - sz * 1.3) / 2, s, sz, "bold")
        x += w
        if i < 2:
            c.d.line([(x, y0), (x, y1)], fill=INK, width=2)
    c.note("title", box, " | ".join(cells), cells=cells)


def draw_title_corner(c: Canvas, box, cells: list[str], signer: str, issued: str) -> None:
    """The bottom-right title block: three stacked cells and a signature row."""
    x0, y0, x1, y1 = box
    d = c.d
    d.rectangle([x0, y0, x1, y1], outline=INK, width=3)
    row = (y1 - y0) / 4
    for i, s in enumerate(cells):
        sz = c.fit(s, 26 if i < 2 else 22, x1 - x0 - 30, "bold", floor=14)
        c.text(x0 + 15, y0 + row * i + (row - sz * 1.3) / 2, s, sz, "bold")
        d.line([(x0, y0 + row * (i + 1)), (x1, y0 + row * (i + 1))], fill=INK, width=2)
    left = f"기술책임자: {signer}"
    sz = c.fit(left, 18, (x1 - x0) * 0.62 - 20, "bold", floor=12)
    c.text(x0 + 15, y0 + row * 3 + 8, left, sz, "bold")
    c.text(x0 + 15, y0 + row * 3 + 8 + sz * 1.5, f"검토일 {issued}", 15)
    split = x0 + (x1 - x0) * 0.62
    d.line([(split, y0 + row * 3), (split, y1)], fill=INK, width=2)
    c.scribble(split + 14, y0 + row * 3 + 8, x1 - split - 28, row - 16, BLUE, kind="sign", text=signer)
    c.note("title", box, " | ".join(cells + [left, f"검토일 {issued}"]), cells=cells + [left, f"검토일 {issued}"])


def draw_master(led: Ledger, drawing_id: str, rng: random.Random, notices: list[str] | None = None, values: dict | None = None) -> tuple[Canvas, dict]:
    """The master sheet of one drawing.  ``values`` overrides sensor readings by tag."""
    dr = led.drawing(drawing_id)
    lay = LAYOUTS[dr.layout]
    c = Canvas(DW, DH, rng)
    d = c.d
    d.rectangle([14, 14, DW - 14, DH - 14], outline=INK, width=3)
    d.rectangle([22, 22, DW - 22, DH - 22], outline=INK, width=1)
    signer = led.person(dr.signer)
    signer_text = f"{signer.name} ({signer.id})"
    cells = title_cells(led, drawing_id)
    if dr.layout == "top":
        draw_title_bar(c, frac(lay["title"], DW, DH), cells)
    else:
        draw_title_corner(c, frac(lay["title"], DW, DH), cells, signer_text, dr.issued)

    info: dict = {"layout": dr.layout, "boxes": {}, "pipes": [], "free_cells": []}
    eqs = led.equipment_of(drawing_id)
    values = values or {}
    bw, bh = COL_W * DW, ROW_H * DH

    def cell_box(cell):
        col, row = cell
        x, y = COL_X[col] * DW, lay["rows"][row] * DH
        return x, y, x + bw, y + bh

    taken = {e.cell for e in eqs}
    info["free_cells"] = [cell_box((col, row)) for row in range(3) for col in range(4) if (col, row) not in taken]

    # ---- pipes first, boxes on top
    def pipe(pts, medium, arrow_dir):
        style, width = MEDIUM_STYLE.get(medium, ("solid", 4))
        c.line(pts, style, width)
        c.arrow(pts[-1][0], pts[-1][1], arrow_dir)
        info["pipes"].append({"points": [[round(x), round(y)] for x, y in pts], "medium": medium})

    by_code = {e.kks: e for e in eqs}
    for edge in led.edges_of(drawing_id):
        a, b = by_code[edge.src], by_code[edge.dst]
        ax0, ay0, ax1, ay1 = cell_box(a.cell)
        bx0, by0, bx1, by1 = cell_box(b.cell)
        if a.cell[1] == b.cell[1]:
            y = (ay0 + ay1) / 2
            if b.cell[0] > a.cell[0]:
                pts, direction = [(ax1, y), (bx0, y)], "right"
            else:
                pts, direction = [(ax0, y), (bx1, y)], "left"
            pipe(pts, edge.medium, direction)
            run = abs(pts[1][0] - pts[0][0])
            mid = (pts[0][0] + pts[1][0]) / 2
            if run > 160:
                c.tag(mid, y - 16, edge.label)
            else:
                c.vtext(mid, y - 14 - c.width(edge.label, 13) / 2 - 6, edge.label)
        elif a.cell[0] == b.cell[0]:
            x = (ax0 + ax1) / 2
            pipe([(x, ay1), (x, by0)], edge.medium, "down")
            c.text(x + 10, (ay1 + by0) / 2 - 10, edge.label, 15, fill=GRAY, kind="label")
        else:
            xa, xb = (ax0 + ax1) / 2, (bx0 + bx1) / 2
            gy = ay1 + (lay["rows"][a.cell[1] + 1] * DH - ay1) / 2
            pipe([(xa, ay1), (xa, gy), (xb, gy), (xb, by0)], edge.medium, "down")
            c.tag((xa + xb) / 2, gy, edge.label)

    # ---- off-page connectors
    def connector(eq, incoming: bool, other: str):
        x0, y0, x1, y1 = cell_box(eq.cell)
        y = (y0 + y1) / 2
        left = incoming or eq.cell[1] % 2 == 1
        w, h, notch = OPC_W * DW, 70, 22
        if left:
            px0 = 0.035 * DW
            px1 = px0 + w
            line = [(px1, y), (x0, y)] if incoming else [(x0, y), (px1, y)]
            direction = "right" if incoming else "left"
        else:
            px0 = (1 - 0.035) * DW - w
            px1 = px0 + w
            line, direction = [(x1, y), (px0, y)], "right"
        tip_right = (left and incoming) or (not left)
        if tip_right:
            poly = [(px0, y - h / 2), (px1 - notch, y - h / 2), (px1, y), (px1 - notch, y + h / 2), (px0, y + h / 2)]
            tx = px0 + 8
        else:
            poly = [(px1, y - h / 2), (px0 + notch, y - h / 2), (px0, y), (px0 + notch, y + h / 2), (px1, y + h / 2)]
            tx = px0 + notch + 4
        style, width = MEDIUM_STYLE.get(led.edges_of(drawing_id)[0].medium if led.edges_of(drawing_id) else "", ("solid", 4))
        c.line(line, style, width)
        c.arrow(line[-1][0], line[-1][1], direction)
        d.polygon(poly, fill=SHADE, outline=INK, width=2)
        head = f"{other} → 인입" if incoming else f"점프 → {other}"
        sec = led.sector(led.drawing(other).sector)
        lines = [(head, 15, "bold"), (f"{sec.label} 계통", 14, "gothic")]
        ty = y - h / 2 + 12
        for s, size, role in lines:
            size = c.fit(s, size, w - notch - 14, role)
            c.text(tx, ty, s, size, role)
            ty += size * 1.5
        c.note("connector", (px0, y - h / 2, px1, y + h / 2), "\n".join(s for s, _, _ in lines))

    if eqs and dr.jump_in:
        connector(eqs[0], True, dr.jump_in)
    if eqs and dr.jump_out:
        connector(eqs[-1], False, dr.jump_out)

    # ---- equipment boxes
    for e in eqs:
        x0, y0, x1, y1 = cell_box(e.cell)
        d.rectangle([x0, y0, x1, y1], fill=WHITE, outline=INK, width=3)
        sx, sy = x1 - SYMBOL_PAD - SYMBOL_SIZE / 2, y0 + SYMBOL_PAD + SYMBOL_SIZE / 2
        draw_symbol(d, e.kind, sx, sy, SYMBOL_SIZE, 3, INK, variant=rng.randint(0, 1))
        c.note("symbol", (sx - SYMBOL_SIZE / 2, sy - SYMBOL_SIZE / 2, sx + SYMBOL_SIZE / 2, sy + SYMBOL_SIZE / 2), e.kind)
        lines = [(f"{led.code_label}: {e.kks}", 21, "bold"), (e.name, 21, "gothic"), (f"위치: {led.location(e)}", 16, "gothic")]
        for s in led.sensors_of(e.kks)[:2]:
            v = values.get(s.tag, s.normal)
            lines.append((f"{short_tag(s.tag)} {s.label} {fmt_value(v, s.decimals)} {s.unit}", 16, "gothic"))
        ty = y0 + 10
        for s, size, role in lines:
            beside = ty < y0 + SYMBOL_PAD * 2 + SYMBOL_SIZE
            room = bw - 22 - (SYMBOL_SIZE + SYMBOL_PAD * 2 + 4 if beside else 0)
            size = c.fit(s, size, room, role, floor=12)
            c.text(x0 + 11, ty, s, size, role)
            ty += size * 1.45
        box = (x0, y0, x1, y1)
        c.note("value", box, "\n".join(s for s, _, _ in lines), kks=e.kks)
        info["boxes"][e.kks] = [round(v) for v in box]

    # ---- limits table
    moc = next((m for m in led.mocs if m.kks in by_code), None)
    rows = table_rows(led, drawing_id, moc.tag if moc else None)
    table = draw_table(c, frac(lay["table"], DW, DH), TABLE_HEAD, [table_cells(s) for s in rows], TABLE_SHARES, "운전 기준치 일람표")
    info["table"] = {"xs": table["xs"], "head_y": table["head_y"], "row_h": table["row_h"], "tags": [s.tag for s in rows]}

    # ---- fixed notices
    gx0, gy0, gx1, gy1 = frac(lay["guide"], DW, DH)
    c.line([(gx0, gy0), (gx1, gy0), (gx1, gy1), (gx0, gy1), (gx0, gy0)], "dash", 1, GRAY)
    notices = notices or default_notices(led)
    pitch = (gy1 - gy0 - 16) / len(notices)
    for i, s in enumerate(notices):
        sz = c.fit(s, 15, gx1 - gx0 - 20)
        c.text(gx0 + 10, gy0 + 8 + pitch * i, s, sz, fill=GRAY)
    c.note("guide", (gx0, gy0, gx1, gy1), "\n".join(notices))

    # ---- signature box
    if lay["sign"]:
        sx0, sy0, sx1, sy1 = frac(lay["sign"], DW, DH)
        d.rectangle([sx0, sy0, sx1, sy1], outline=INK, width=2)
        c.text(sx0 + 12, sy0 + 10, f"기술책임자: {signer_text}", 18, "bold")
        c.text(sx0 + 12, sy1 - 34, f"검토일 {dr.issued}", 16)
        c.scribble(sx0 + 250, sy0 + 40, 170, 60, BLUE, kind="sign", text=signer.name)
        c.note("sign_text", (sx0, sy0, sx1, sy1), f"기술책임자: {signer_text}\n검토일 {dr.issued}")
    return c, info


def redline(c: Canvas, info: dict, led: Ledger, drawing_id: str, rng: random.Random) -> dict | None:
    """Pen marks of a management-of-change on a master sheet (drawn in place).

    The old alarm limit is struck out with the new one written above it, the
    equipment is circled, and a four-line note with a stamp sits in a free cell.
    """
    codes = {e.kks for e in led.equipment_of(drawing_id)}
    moc = next((m for m in led.mocs if m.kks in codes), None)
    if moc is None or moc.tag not in info["table"]["tags"]:
        return None
    d = c.d
    t = info["table"]
    r = t["tags"].index(moc.tag) + 1
    y0 = t["head_y"] + t["row_h"] * r
    x0, x1 = t["xs"][4], t["xs"][5]
    sensor = next(s for s in led.sensors if s.tag == moc.tag)
    d.line([(x0 + 4, y0 + t["row_h"] / 2 + rng.uniform(-2, 2)), (x1 - 6, y0 + t["row_h"] / 2 + rng.uniform(-2, 2))], fill=RED, width=4)
    new_text = fmt_value(moc.new, sensor.decimals)
    c.hand(x0 + 36, y0 - 21, new_text, 24, RED, kind="new_value")

    bx0, by0, bx1, by1 = info["boxes"][moc.kks]
    for out in (12, 15):
        d.rounded_rectangle([bx0 - out, by0 - out, bx1 + out, by1 + out], radius=26, outline=RED, width=3)

    person = led.person(moc.approved_by)
    lines = [
        moc.moc_no,
        f"{short_tag(moc.tag)} {moc.param}",
        f"{fmt_value(moc.old, sensor.decimals)} → {new_text} {moc.unit}",
        f"{moc.approved_on} {person.name}",
    ]
    note_box = None
    if info["free_cells"]:
        cx = (bx0 + bx1) / 2
        fx0, fy0, fx1, fy1 = min(info["free_cells"], key=lambda b: abs((b[0] + b[2]) / 2 - cx) + abs(b[1] - by0))
        y = fy0 + 14
        right = fx0
        for s in lines:
            rx, ry = c.hand(fx0 + 8, y, s, 24, RED, kind="note")
            right = max(right, rx)
            y = ry + 8
        note_box = (fx0, fy0, right, y)
        c.stamp(min(right + 50, fx1 + 30), fy1 - 46, person.name, 38)
        # leader from the note to the circled equipment
        ax, ay = (fx0 + fx1) / 2, (fy0 if by0 < fy0 else fy1)
        tx, ty = (bx0 + bx1) / 2, (by1 + 15 if by0 < fy0 else by0 - 15)
        if abs(ax - tx) + abs(ay - ty) < 0.45 * DW:
            d.line([(ax, ay), (tx, ty)], fill=RED, width=3)
    return {"moc": moc.moc_no, "tag": moc.tag, "kks": moc.kks, "old": moc.old, "new": moc.new, "lines": lines, "note_box": note_box}


def gauge_card(led: Ledger, rng: random.Random, sensor_tag: str | None = None, notices: list[str] | None = None) -> tuple[Canvas, dict]:
    """A filled-in field inspection card: printed labels, hand-written values."""
    c = Canvas(PW, PH, rng)
    d = c.d
    s = next(x for x in led.sensors if x.tag == sensor_tag) if sensor_tag else rng.choice(led.sensors)
    eq = next(e for e in led.equipment if e.kks == s.kks)
    person = rng.choice(led.people)
    reading = round(s.normal * rng.uniform(0.9, 1.12), s.decimals)
    when = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(6, 22):02d}:{rng.choice(['00', '10', '20', '30', '40', '50'])}"
    title = "현장 계기 점검표"
    c.text((PW - c.width(title, 44, "bold")) / 2, 90, title, 44, "bold", kind="form_title")
    c.text(90, 165, "모든 값은 현장에서 직접 읽어 손으로 적는다", 20, fill=GRAY, kind="guide")
    sec = led.sector(eq.sector)
    fields = [
        ("점검 일시", when), ("점검자", person.name), ("사번", person.id), ("관할 섹터", f"{sec.building} {sec.label}"),
        (f"설비 {led.code_label}", eq.kks), ("센서 태그", s.tag), ("현장 실측치", fmt_value(reading, s.decimals)), ("단위", s.unit),
    ]  # fmt: skip
    x0, y0, label_w, value_w, row_h = 90, 230, 260, 800, 78
    cells = []
    for i, (label, value) in enumerate(fields):
        y = y0 + row_h * i
        d.rectangle([x0, y, x0 + label_w, y + row_h], fill=SHADE, outline=INK, width=2)
        d.rectangle([x0 + label_w, y, x0 + label_w + value_w, y + row_h], outline=INK, width=2)
        c.text(x0 + 18, y + 22, label, 24, "bold")
        c.hand(x0 + label_w + 24, y + 18, value, 34, BLUE, kind="value")
        cells.append([label, value])
    c.note("form", (x0, y0, x0 + label_w + value_w, y0 + row_h * len(fields)), "\n".join(" | ".join(r) for r in cells), cells=cells)

    # analogue gauge
    gx, gy, gr = 380, 1080, 190
    d.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], outline=INK, width=4)
    vmax = max(s.normal, s.alarm, s.trip) * 1.25
    for k in range(11):
        a = math.radians(225 - 270 * k / 10)
        d.line([(gx + (gr - 22) * math.cos(a), gy - (gr - 22) * math.sin(a)), (gx + (gr - 4) * math.cos(a), gy - (gr - 4) * math.sin(a))], fill=INK, width=3)
    a = math.radians(225 - 270 * min(reading / vmax, 1.0))
    d.line([(gx, gy), (gx + (gr - 40) * math.cos(a), gy - (gr - 40) * math.sin(a))], fill=RED, width=6)
    d.ellipse([gx - 10, gy - 10, gx + 10, gy + 10], fill=INK)
    c.text(gx - c.width(s.unit, 26, "bold") / 2, gy + 70, s.unit, 26, "bold")

    c.text(660, 900, "현장 판독 메모", 24, "bold", kind="memo_title")
    memo = [f"{s.label} {fmt_value(reading, s.decimals)} {s.unit}", rng.choice(["이상 없음", "소음 약간 있음", "누설 흔적 없음", "재점검 필요"])]
    for i, line in enumerate(memo):
        c.hand(660, 960 + 70 * i, line, 32, BLUE, kind="memo")
    for i, line in enumerate(notices or default_notices(led)):
        sz = c.fit(line, 18, PW - 180)
        c.text(90, 1330 + 30 * i, line, sz, fill=GRAY)
    c.note("guide", (90, 1326, PW - 90, 1330 + 30 * 3), "\n".join(notices or default_notices(led)))
    d.rectangle([760, 1480, 1150, 1620], outline=INK, width=2)
    c.text(775, 1490, "점검자 서명", 20, "bold")
    c.scribble(800, 1530, 300, 70, BLUE, kind="sign", text=person.name)
    return c, {"tag": s.tag, "kks": eq.kks, "reading": reading, "unit": s.unit, "person": person.id, "when": when, "fields": cells}
