"""Post-OCR verification against the ledger (plan Part 6, ``Shadow Top-3``).

Two kinds of correction, kept apart so that nothing is invented:

* **mask repair** – an equipment code (KKS ``99AAA99AA999`` or DCIM
  ``AA9AAA99AA99``) whose characters only need the digit/letter confusions of
  its positions undone (``1``↔``I``, ``0``↔``O``, ``5``↔``S`` …) and that then
  matches exactly one ledger code is corrected at once;
* **evidence repair** – anything else (two characters misread in a row, a
  name off by a syllable, a code one edit away from a ledger entry) is only
  corrected when the candidate can be *reassembled from the decoder's own
  alternatives*: walking the decoding steps that produced the span, each step
  contributes either its chosen byte piece or one of the runner-up pieces it
  logged with probability ≥ ``MIN_ALT_PROB`` (0.08), and the bytes must spell
  the candidate exactly (a prefix DP over byte pieces, so a syllable split
  across tokens is handled).  Without that evidence the reading is kept and
  flagged; a ledger look-up alone never overrides what the model saw.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .decode import MIN_ALT_PROB, Pick
from .drawing import codes

CODE_SPAN = re.compile(r"(?<![0-9A-Za-z])[0-9A-Za-z|!][0-9A-Za-z|!\-. ]{8,16}[0-9A-Za-z|!](?![0-9A-Za-z])")
HANGUL_WORD = re.compile(r"[가-힣]{2,4}")
FIELD_BREAK = re.compile(r"<[^>]*>|\s\|\s|\n|:\s")
MAX_NAME_DISTANCE = 2


def _code_spans(text: str):
    """Code-like spans inside one field each (table cells, lines, ``label: value`` parts)."""
    pos = 0
    bounds = [(m.start(), m.end()) for m in FIELD_BREAK.finditer(text)] + [(len(text), len(text))]
    for a, b in bounds:
        field_text = text[pos:a]
        for m in CODE_SPAN.finditer(field_text):
            raw = m.group(0).strip()
            start = pos + m.start() + m.group(0).index(raw)
            yield raw, start, start + len(raw)
        pos = b


# characters a scan or a crossing line turns into one another (same position class)
CONFUSABLE = {frozenset(p) for p in (
    "17", "1T", "7T", "1I", "IT", "IL", "LT", "1L", "08", "38", "56", "68", "69", "35", "2Z", "0O", "0D", "OD", "OQ", "CO",
    "CG", "EF", "BR", "MN", "UV", "PR", "HN", "KX",
)}  # fmt: skip


def _confusable_neighbours(candidates, code_set) -> list[str]:
    """Ledger codes that differ from a candidate in exactly one position, by a confusable pair."""
    found = set()
    for cand in candidates:
        if len(cand) != 12:
            continue
        for c in code_set:
            if len(c) != 12:
                continue
            diff = [(a, b) for a, b in zip(cand, c) if a != b]
            if len(diff) == 1 and frozenset(diff[0]) in CONFUSABLE:
                found.add(c)
    return sorted(found)


@dataclass
class Fix:
    start: int  # character offsets in the text
    end: int
    raw: str
    new: str
    kind: str  # "mask" | "confusion" | "evidence"
    target: str  # "code" | "name"


@dataclass
class Report:
    text: str
    fixes: list[Fix] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # candidates without evidence
    flagged: list[dict] = field(default_factory=list)  # ambiguous ledger matches


# --------------------------------------------------------------------------- evidence
class Steps:
    """Byte pieces of a decoded text with the alternatives each step logged."""

    def __init__(self, picks: list[Pick], piece):
        self.pieces: list[bytes] = []
        self.alts: list[list[bytes]] = []
        self.offsets: list[int] = []  # byte offset of every step
        pos = 0
        for p in picks:
            b = piece(p.token)
            self.offsets.append(pos)
            self.pieces.append(b)
            self.alts.append([piece(t) for t, prob in p.alts if prob >= MIN_ALT_PROB])
            pos += len(b)
        self.total = pos

    def span(self, b0: int, b1: int) -> tuple[int, int, int, int]:
        """Steps covering bytes ``[b0, b1)`` and the bytes they cover: (first step, end step, byte start, byte end)."""
        first = max(i for i, o in enumerate(self.offsets) if o <= b0) if self.offsets else 0
        end = first
        while end < len(self.pieces) and self.offsets[end] < b1:
            end += 1
        start_b = self.offsets[first] if self.offsets else 0
        end_b = self.offsets[end] if end < len(self.offsets) else self.total
        return first, end, start_b, end_b

    def reassembles(self, first: int, end: int, want: bytes) -> bool:
        """Can steps ``first..end`` spell ``want`` from their chosen or alternative pieces?"""
        reach = {0}
        for i in range(first, end):
            options = [self.pieces[i]] + self.alts[i]
            nxt = set()
            for off in reach:
                rem = want[off:]
                for piece in options:
                    if rem.startswith(piece):
                        nxt.add(off + len(piece))
            if not nxt:
                return False
            reach = nxt
        return len(want) in reach


def supported(text: str, steps: Steps, start: int, end: int, candidate: str) -> bool:
    """Is replacing ``text[start:end]`` by ``candidate`` backed by the logged alternatives?"""
    b0 = len(text[:start].encode("utf-8"))
    b1 = len(text[:end].encode("utf-8"))
    raw = text.encode("utf-8")
    first, last, sb, eb = steps.span(b0, b1)
    want = raw[sb:b0] + candidate.encode("utf-8") + raw[b1:eb]
    return steps.reassembles(first, last, want)


# --------------------------------------------------------------------------- verification
def verify(text: str, picks: list[Pick] | None, piece, ledger_codes=(), names=(), name_fields: bool = False) -> Report:
    """Check codes (and, with ``name_fields``, person names) of ``text`` against the ledger.

    ``picks`` are the decoding steps (with their Shadow Top-3); ``piece``
    turns a token id into its bytes (``Vocab.piece``).  Without picks only mask
    repairs can be made.
    """
    rep = Report(text)
    steps = Steps(picks, piece) if picks else None
    code_set = set(ledger_codes)
    edits: list[Fix] = []
    for m in _code_spans(text):
        raw, start, end = m
        if raw in code_set:
            continue
        squeezed = re.sub(r"[\s\-._:·]", "", raw)
        if len(squeezed) < 10 or len(squeezed) > 14:
            continue
        # 1. mask repair: the confusions of the positions alone give one ledger code
        repaired = {codes.repair_code(raw, s) for s in (codes.KKS, codes.DCIM)}
        exact = sorted(c for c in repaired if c in code_set)
        if len(exact) == 1 and len(squeezed) == 12:
            edits.append(Fix(start, end, raw, exact[0], "mask", "code"))
            continue
        # 1b. one character away from exactly one ledger code, by a pair that looks alike
        one = _confusable_neighbours(repaired, code_set)
        if len(one) == 1:
            edits.append(Fix(start, end, raw, one[0], "confusion", "code"))
            continue
        # 2. nearest ledger code, only with evidence
        match = codes.match_code(raw, code_set)
        if match.kind == "ambiguous":
            rep.flagged.append({"raw": raw, "candidates": list(match.candidates)})
            continue
        if match.kind != "corrected":
            continue
        if steps is not None and supported(text, steps, start, end, match.code):
            edits.append(Fix(start, end, raw, match.code, "evidence", "code"))
        else:
            rep.rejected.append({"raw": raw, "candidate": match.code, "target": "code", "distance": match.distance})
    if name_fields and names:
        name_set = set(names)
        for m in HANGUL_WORD.finditer(text):
            word = m.group(0)
            if word in name_set:
                continue
            near = sorted((codes.distance(word, n), n) for n in name_set if len(n) == len(word))
            near = [(d, n) for d, n in near if 0 < d <= MAX_NAME_DISTANCE]
            if not near:
                continue
            best = [n for d, n in near if d == near[0][0]]
            if len(best) > 1:
                rep.flagged.append({"raw": word, "candidates": best})
                continue
            if steps is not None and supported(text, steps, m.start(), m.end(), best[0]):
                edits.append(Fix(m.start(), m.end(), word, best[0], "evidence", "name"))
            else:
                rep.rejected.append({"raw": word, "candidate": best[0], "target": "name", "distance": near[0][0]})
    out = text
    for f in sorted(edits, key=lambda f: -f.start):
        out = out[: f.start] + f.new + out[f.end :]
    rep.text = out
    rep.fixes = sorted(edits, key=lambda f: f.start)
    return rep
