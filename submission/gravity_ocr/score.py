"""Scoring a reading against ground truth: character error rate and table cells."""

from __future__ import annotations

import unicodedata

import numpy as np

from .tables import has_table, rows_of, strip_tags


def normalise(s: str) -> str:
    """NFKC (℃ → °C, full-width forms → ASCII) and single spaces."""
    return " ".join(unicodedata.normalize("NFKC", s).split())


def squeeze(s: str) -> str:
    return "".join(unicodedata.normalize("NFKC", s).split())


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    bb = np.array([hash(x) for x in b], dtype=np.int64)
    prev = np.arange(len(b) + 1, dtype=np.int32)
    steps = np.arange(len(b) + 1, dtype=np.int32)
    for i, ca in enumerate(a, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        np.minimum(prev[1:] + 1, prev[:-1] + (bb != hash(ca)), out=cur[1:])
        # insertions run left to right: cur[j] = min(cur[j], cur[j-1] + 1)
        cur = np.minimum.accumulate(cur - steps) + steps
        prev = cur
    return int(prev[-1])


def substring_distance(text, part) -> int:
    """Fewest edits that turn ``part`` into some substring of ``text`` (Sellers): how well a
    ground-truth item was read inside a block that also holds other items."""
    if not part:
        return 0
    if not text:
        return len(part)
    tt = np.array([hash(x) for x in text], dtype=np.int64)
    steps = np.arange(len(text) + 1, dtype=np.int32)
    prev = np.zeros(len(text) + 1, dtype=np.int32)  # the match may start anywhere in the text
    for i, cp in enumerate(part, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        np.minimum(prev[1:] + 1, prev[:-1] + (tt != hash(cp)), out=cur[1:])
        cur = np.minimum.accumulate(cur - steps) + steps
        prev = cur
    return int(prev.min())


def plain(text: str) -> str:
    """Reading or ground truth as plain text: table markup and cell bars become spaces."""
    if has_table(text):
        text = strip_tags(text)
    return normalise(text.replace(" | ", " "))


def cer(pred: str, truth: str) -> float:
    p, t = plain(pred), plain(truth)
    return edit_distance(p, t) / max(len(t), 1)


def cer_tight(pred: str, truth: str) -> float:
    """CER with every space removed (spacing mistakes are not counted)."""
    p, t = squeeze(plain(pred)), squeeze(plain(truth))
    return edit_distance(p, t) / max(len(t), 1)


def table_score(pred: str, cells: list[list[str]]) -> dict:
    """Cell-by-cell comparison of a table reading with the true cells.

    ``shifted_rows`` counts rows whose cells do not sit in their own column:
    a different number of cells, or a true cell found in another column.
    """
    rows = rows_of(pred)
    want = [[squeeze(c) for c in r] for r in cells]
    got = [[squeeze(c) for c in r] for r in rows]
    total = sum(len(r) for r in want)
    exact = 0
    shifted = 0
    for i, r in enumerate(want):
        g = got[i] if i < len(got) else []
        exact += sum(1 for j, c in enumerate(r) if j < len(g) and g[j] == c)
        moved = any(c and c in g and (j >= len(g) or g[j] != c) for j, c in enumerate(r))
        shifted += int(len(g) != len(r) or moved)
    return {
        "cells": total,
        "cells_exact": exact,
        "rows": len(want),
        "rows_read": len(got),
        "shifted_rows": shifted,
        "is_table": has_table(pred),
    }


def summarise(records: list[dict]) -> dict:
    """Totals over scored records (each with ``edits``, ``chars``, optional table fields)."""
    chars = sum(r["chars"] for r in records)
    out = {
        "n": len(records),
        "cer": sum(r["edits"] for r in records) / max(chars, 1),
        "cer_tight": sum(r["edits_tight"] for r in records) / max(sum(r["chars_tight"] for r in records), 1),
        "exact": sum(1 for r in records if r["edits_tight"] == 0) / max(len(records), 1),
    }
    tables = [r["table"] for r in records if r.get("table")]
    if tables:
        out["cell_exact"] = sum(t["cells_exact"] for t in tables) / max(sum(t["cells"] for t in tables), 1)
        out["shifted_rows"] = sum(t["shifted_rows"] for t in tables)
        out["table_rows"] = sum(t["rows"] for t in tables)
    return out


def score(pred: str, truth: str, cells: list[list[str]] | None = None) -> dict:
    p, t = plain(pred), plain(truth)
    ps, ts = squeeze(p), squeeze(t)
    rec = {
        "edits": edit_distance(p, t),
        "chars": len(t),
        "edits_tight": edit_distance(ps, ts),
        "chars_tight": len(ts),
    }
    if cells:
        rec["table"] = table_score(pred, cells)
    return rec
