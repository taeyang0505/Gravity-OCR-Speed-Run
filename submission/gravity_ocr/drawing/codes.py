"""Equipment code schemes and their OCR repair.

Two 12-character schemes, one per domain, each a fixed pattern of digit (``9``)
and letter (``A``) positions:

``KKS``   power plant   ``99AAA99AA999``  ``40LAC20AP001`` unit · system · system no. · kind · serial
``DCIM``  AI data centre ``AA9AAA99AA99``  ``DC1CDU20AA02`` site · centre · system · system no. · kind · serial

A sensor tag has the shape of its equipment code with the kind letters starting
with ``C`` (``40LAC20CP003``, ``DC1CDU20CG02``); the part from the kind letters
on is the short tag written in drawing boxes (``CP003``, ``CG02``).

Because every position is either a digit or a letter, the common OCR confusions
(``1``/``I``/``l``, ``0``/``O``/``D``, ``5``/``S``, ``8``/``B`` …) can be undone
position by position (:func:`repair_code`); :func:`match_code` then checks the
result against the ledger and never guesses between two equally close codes.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

MAX_DISTANCE = 2


@dataclass(frozen=True)
class Scheme:
    name: str
    mask: str
    kind_at: int  # first character of the kind letters
    label: str  # how drawings label the code

    def shape(self, code: str) -> bool:
        return len(code) == 12 and all(c.isdigit() if m == "9" else ("A" <= c <= "Z") for c, m in zip(code, self.mask))

    def short_tag(self, tag: str) -> str:
        return tag[self.kind_at :]

    def is_sensor(self, code: str) -> bool:
        return self.shape(code) and code[self.kind_at] == "C"

    @property
    def pattern(self) -> str:
        """Regular expression of the code shape."""
        return "".join("[0-9]" if m == "9" else "[A-Z]" for m in self.mask)


KKS = Scheme("kks", "99AAA99AA999", 7, "KKS")
DCIM = Scheme("dcim", "AA9AAA99AA99", 8, "자산코드")
SCHEMES = {"kks": KKS, "dcim": DCIM}


def scheme_of(code: str) -> Scheme | None:
    """The scheme whose shape ``code`` has (``None`` when neither)."""
    return next((s for s in (KKS, DCIM) if s.shape(code)), None)


_AS_DIGIT = {"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "i": "1", "|": "1", "L": "1", "J": "1", "!": "1",
             "Z": "2", "z": "2", "S": "5", "s": "5", "G": "6", "b": "6", "T": "7", "B": "8", "g": "9", "q": "9"}  # fmt: skip
_AS_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
_DROP = re.compile(r"[\s\-._:·]")


def repair_code(raw: str, scheme: Scheme) -> str:
    """Drop separators; when 12 characters remain, fix each position by the mask."""
    s = _DROP.sub("", raw)
    if len(s) != 12:
        return s.upper()
    out = []
    for c, m in zip(s, scheme.mask):
        if m == "9":
            out.append(c if c.isdigit() else _AS_DIGIT.get(c, c.upper()))
        else:
            out.append(c.upper() if c.isalpha() and c.isascii() else _AS_LETTER.get(c, c.upper()))
    return "".join(out)


def distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class Match:
    kind: str  # "exact" | "corrected" | "ambiguous" | "unknown"
    raw: str
    code: str | None = None
    distance: int = 0
    candidates: tuple = ()


def match_code(raw: str, codes, scheme: Scheme | None = None, max_distance: int = MAX_DISTANCE) -> Match:
    """Ledger match of an OCR'd code: exact, repaired + unique nearest, ambiguous or unknown."""
    codes = set(codes)
    if raw in codes:
        return Match("exact", raw, raw)
    schemes = [scheme] if scheme else [KKS, DCIM]
    best: dict[str, int] = {}
    for sc in schemes:
        fixed = repair_code(raw, sc)
        for c in codes:
            if not sc.shape(c):
                continue
            d = distance(fixed, c)
            if d <= max_distance and d < best.get(c, 99):
                best[c] = d
    if not best:
        return Match("unknown", raw)
    dmin = min(best.values())
    top = sorted(c for c, d in best.items() if d == dmin)
    if len(top) > 1:
        return Match("ambiguous", raw, None, dmin, tuple(top))
    return Match("corrected", raw, top[0], dmin, tuple(top))


# --------------------------------------------------------------------------- seeds
KKS_SYSTEMS = ["LAB", "LAC", "LAD", "LBA", "LBC", "HAH", "MAA", "MAC", "MAG", "PAB", "PAC", "PCB", "HLB", "HNC", "HFC",
               "MKA", "BAT", "BBA", "MAV", "MAX", "LCA", "LCB", "LCC"]  # fmt: skip
KKS_KINDS = ["AP", "AA", "AD", "AN", "AC", "BB", "GS", "GT", "GA", "GH"]
DCIM_SYSTEMS = ["GPU", "MNF", "RTN", "CDU", "CHW", "CHL", "CTW", "CWS", "SWG", "UPS", "PDU", "BES", "TRF", "FAB", "OPT", "CRA",
                "IRC", "AHU"]  # fmt: skip
DCIM_KINDS = ["RK", "MV", "AP", "AA", "AC", "BB", "AN", "GU", "BT", "GS", "GT", "SW", "GH"]
SENSOR_KINDS = ["CP", "CT", "CF", "CL", "CS", "CE", "CY", "CG", "CW"]


def sample_codes(n: int = 400, seed: int = 11) -> list[str]:
    """Codes and sensor tags of both schemes (for vocabulary coverage and n-gram seeds)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n // 2):
        unit = rng.choice([10, 20, 30, 40, 50, 60, 70, 80])
        sysc, no = rng.choice(KKS_SYSTEMS), rng.choice([10, 11, 12, 20, 21, 30, 40, 50])
        out.append(f"{unit}{sysc}{no:02d}{rng.choice(KKS_KINDS)}{rng.randint(1, 30):03d}")
        out.append(f"{unit}{sysc}{no:02d}{rng.choice(SENSOR_KINDS)}{rng.randint(1, 30):03d}")
        site = rng.choice(["DC1", "DC2", "DC0"])
        sysd, nd = rng.choice(DCIM_SYSTEMS), rng.choice([10, 20, 30])
        out.append(f"{site}{sysd}{nd:02d}{rng.choice(DCIM_KINDS)}{rng.randint(1, 9):02d}")
        out.append(f"{site}{sysd}{nd:02d}{rng.choice(SENSOR_KINDS)}{rng.randint(1, 9):02d}")
    return out
