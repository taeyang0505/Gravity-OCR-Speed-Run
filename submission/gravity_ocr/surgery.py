"""Output-vocabulary slicing of ``lm_head`` (plan Part 4.1).

The full head has 59,283 rows; every decoding step multiplies all of them.
A drawing reads Korean, ASCII codes and numbers, English engineering words,
table markup and a handful of symbols, so most rows (Cyrillic, Arabic, rare
Chinese, ...) are never chosen.  Cutting the rows by token id would be wrong:
the 232 tokens that hold whole Hangul syllables sit between ids 5,993 and
59,117 and only 21 of them come before id 16,384.  The slice is therefore
built by priority:

1. **must keep** – control / image tokens; every token with a Hangul syllable
   or jamo; every piece that is not valid UTF-8 on its own (the byte
   fragments a syllable is assembled from: lead bytes ``0xEA..0xED`` and
   continuation bytes ``0x80..0xBF`` included) and all 256 single bytes, so
   any text stays expressible; HTML table tags; symbols, CJK / full-width
   punctuation and Greek (units such as ``μm``, ``Ω``, ``Δ``); every run of
   capitals and digits (equipment codes, and English as drawings write it:
   ``PRESS`` + ``URE``); the punctuation of running text and numbers
   (``.\\n``, ``, ``, `` (``…);
2. **domain** – every token the tokenizer produces for the domain corpus
   (drawing texts, KKS / DCIM codes, units, numbers, table markup, and an
   English engineering lexicon in capitals, title case and lower case);
3. **fill** – the most frequent Chinese characters (Hanja), then English word
   pieces, then the other ASCII tokens, each in id order (the merge order of
   the BPE, i.e. by frequency) until the budget.

Measured (``tools/slice_eval.py``): every text token of the four synthetic sets
is kept; of the 660 word forms of lexicon words *not* in the corpus, 122 (18 %)
still need a cut row – none in capitals – against 181 (27 %, 20 in capitals) with
the earlier rule (capitals only up to 4 letters, fill in pure id order).  No
slice of 16,384 rows holds English, so the runtime keeps the cut rows apart and
checks every pass that none of them wins (``GravityOCR.set_rest_parts``).

The rows keep their order of token id; ``ids[row]`` maps a row back to the
full vocabulary.  Nothing is retrained: the kept rows are the original rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .tokenizer import Vocab

BUDGET = 16_384
CJK_FILL = 1_024  # the most frequent Hanja

CAPS_PIECE = re.compile(rb" ?[A-Z0-9]+")
TEXT_PUNCT = re.compile(rb"[0-9 \n\t.,:;\-/()%+=#&*'\"!?{}\\\^_$<>\[\]|~`]+")
WORD_PIECE = re.compile(rb" ?[A-Za-z]+")
HTML_RE = re.compile(r"</?(table|thead|tbody|tr|td|th|br|caption)\b|^\s*/?>\s*$")

# English on plant / data-centre drawings (instrument, equipment, title-block and note words)
ENGLISH = """pressure temperature flow level valve pump motor control gauge transmitter indicator switch alarm high low normal
open close closed inlet outlet drain vent supply return cooling water steam oil air gas fuel tank vessel heat exchanger
compressor fan filter strainer check gate globe ball butterfly safety relief isolation bypass sample analyzer conductivity
differential position limit speed vibration current voltage power frequency breaker transformer panel cabinet cable tray
local remote auto manual start stop run trip reset emergency standby duty main auxiliary boiler turbine generator condenser
deaerator feedwater chiller cooler heater humidity damper duct rack server battery rectifier inverter drawing revision
checked approved designed scale sheet title project note notes detail section elevation plan view typical reference
specification material quantity description remarks size rating class schedule insulation painting welding support hanger
flange elbow reducer nozzle orifice venturi thermowell controller recorder totalizer solenoid actuator positioner
interlock sequence logic signal input output analog digital setpoint override permissive shutdown startup maintenance
inspection test calibration operation instrument instrumentation electrical mechanical piping civil structural architectural
ventilation fire protection sprinkler hydrant detector smoke lighting grounding earthing lightning
substation switchgear busbar feeder load capacity efficiency demand consumption meter metering redundancy tower
containment aisle hot cold raised floor ceiling distribution unit precision humidifier dehumidifier makeup blowdown chemical
dosing injection treatment softener demineralizer condensate deaeration economizer superheater reheater attemperator drum"""


def english_forms(words: list[str] | None = None) -> list[str]:
    """CAPS, Title and lower case of every word, each also after a space (as it follows another word)."""
    out = []
    for w in words if words is not None else ENGLISH.split():
        forms = [w.upper(), w.title(), w]
        out += forms + [" " + f for f in forms]
    return out


def _script(c: str) -> str:
    o = ord(c)
    if o < 128:
        return "ascii"
    if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F or 0xA960 <= o <= 0xA97F or 0xD7B0 <= o <= 0xD7FF:
        return "hangul"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF:
        return "cjk"
    if 0x0370 <= o <= 0x03FF:
        return "greek"
    if 0x2000 <= o <= 0x2BFF or 0x3200 <= o <= 0x33FF or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF or 0x00A0 <= o <= 0x00BF:
        return "symbol"
    return "other"


def token_class(piece: bytes) -> str:
    """``partial`` (not valid UTF-8 alone), ``hangul``, ``html``, ``symbol``, ``greek``, ``ascii``, ``cjk`` or ``other``."""
    try:
        s = piece.decode("utf-8")
    except UnicodeDecodeError:
        return "partial"
    if not s:
        return "other"
    kinds = {_script(c) for c in s}
    if "hangul" in kinds:
        return "hangul"
    if kinds <= {"ascii"}:
        return "html" if HTML_RE.search(s) else "ascii"
    if kinds <= {"ascii", "symbol"}:
        return "symbol"
    if kinds <= {"ascii", "symbol", "greek"}:
        return "greek"
    if kinds <= {"ascii", "symbol", "cjk"}:
        return "cjk"
    return "other"


@dataclass
class Slice:
    ids: np.ndarray  # full-vocabulary id of every kept row, ascending
    tiers: dict = field(default_factory=dict)  # tier name → number of ids it added
    classes: dict = field(default_factory=dict)  # class → (kept, total)

    @property
    def size(self) -> int:
        return int(self.ids.size)

    def rows_of(self, tokens) -> np.ndarray:
        """Row of each full id (-1 where the token was cut)."""
        pos = np.searchsorted(self.ids, tokens)
        pos = np.clip(pos, 0, self.ids.size - 1)
        return np.where(self.ids[pos] == np.asarray(tokens), pos, -1)


def domain_corpus(extra: list[str] | None = None, english: list[str] | None = None) -> list[str]:
    """Strings a drawing produces besides its free text: codes, units, numbers, table markup, English
    words (``english``: the words to include, default the whole ``ENGLISH`` lexicon)."""
    from .drawing import codes

    out = list(extra or [])
    out += codes.sample_codes()
    out += english_forms(english)
    out += [
        "<table>", "</table>", "<tr>", "</tr>", "<td>", "</td>", "</td><td>", "</td></tr>", "<tr><td>", "</td></tr><tr><td>",
        "</td></tr></table>", "<th>", "</th>", "<br>", "<td colspan=\"2\">", "<td rowspan=\"2\">",
    ]  # fmt: skip
    units = ["mm", "mm/s", "m/s", "m³/h", "t/h", "kg/h", "bar", "kPa", "MPa", "kV", "kW", "MW", "A", "V", "Hz", "rpm",
             "℃", "°C", "%", "ppm", "μm", "Ω", "kΩ", "Nm³/h", "L/min", "dB", "lux", "kVA", "MVA", "pH", "μS/cm"]  # fmt: skip
    rng = np.random.default_rng(7)
    for u in units:
        for _ in range(4):
            v = rng.uniform(0, 2000)
            out.append(f"{v:,.1f} {u}")
            out.append(f"{v:.2f}{u}")
    for _ in range(200):
        y, m, d = rng.integers(2015, 2031), rng.integers(1, 13), rng.integers(1, 29)
        out.append(f"{y}-{m:02d}-{d:02d}")
        out.append(f"{y}.{m:02d}.{d:02d}")
        out.append(f"Rev.{rng.integers(0, 9)}")
    return out


def build(vocab: Vocab, corpus: list[str], budget: int = BUDGET, cjk_fill: int = CJK_FILL) -> Slice:
    n = vocab.size
    cls = {}
    for i in range(n):
        cls[i] = "special" if vocab.is_special(i) else token_class(vocab.piece(i))
    keep: set[int] = set()
    tiers: dict[str, int] = {}

    def add(name: str, ids) -> None:
        before = len(keep)
        for i in ids:
            if len(keep) >= budget:
                break
            keep.add(int(i))
        tiers[name] = len(keep) - before

    def piece(i: int) -> bytes:
        return vocab.piece(i)

    must = [i for i in range(n) if cls[i] in ("special", "hangul", "partial", "html", "symbol", "greek")]
    must += [i for i in range(n) if len(piece(i)) == 1]
    must += [i for i in range(n) if cls[i] == "ascii" and CAPS_PIECE.fullmatch(piece(i))]
    must += [i for i in range(n) if cls[i] == "ascii" and TEXT_PUNCT.fullmatch(piece(i)) and b"   " not in piece(i)]
    add("must", sorted(set(must)))
    seen = set()
    for text in corpus:
        seen.update(vocab.encode(text))
    add("domain", sorted(seen))
    add("cjk", [i for i in range(n) if cls[i] == "cjk" and i not in keep][:cjk_fill])
    add("words", [i for i in range(n) if cls[i] == "ascii" and i not in keep and WORD_PIECE.fullmatch(piece(i))])
    add("ascii", [i for i in range(n) if cls[i] == "ascii" and i not in keep])
    add("rest", [i for i in range(n) if i not in keep and cls[i] in ("cjk", "other")])
    if len(keep) < budget:
        raise ValueError("the vocabulary is smaller than the budget")
    ids = np.array(sorted(keep), dtype=np.int64)
    counts = {}
    for c in sorted(set(cls.values())):
        total = sum(1 for v in cls.values() if v == c)
        kept = sum(1 for i in ids.tolist() if cls[i] == c)
        counts[c] = (kept, total)
    return Slice(ids, tiers, counts)


def coverage(vocab: Vocab, sl: Slice, texts: list[str]) -> dict:
    """Share of the tokens of ``texts`` (as the tokenizer encodes them) that the slice keeps."""
    total = missing = 0
    lost = {}
    for t in texts:
        toks = vocab.encode(t)
        rows = sl.rows_of(np.asarray(toks, dtype=np.int64)) if toks else np.zeros(0)
        total += len(toks)
        for tok, r in zip(toks, rows.tolist()):
            if r < 0:
                missing += 1
                lost[tok] = lost.get(tok, 0) + 1
    return {"tokens": total, "missing": missing, "coverage": 1.0 - missing / max(total, 1),
            "lost": {vocab.piece(k).decode("utf-8", "replace"): c for k, c in sorted(lost.items(), key=lambda kv: -kv[1])[:20]}}  # fmt: skip


FORMULA_SEEDS = [
    r"\frac{a}{b}", r"\dfrac{x}{y}", r"\tfrac{1}{2}", r"\sqrt{x}", r"\sqrt[n]{x}",
    r"\sum_{i=1}^{n}", r"\prod_{k=1}^{m}", r"\int_{a}^{b}", r"\iint", r"\oint",
    r"\lim_{x \to \infty}", r"\log", r"\ln", r"\exp", r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc",
    r"\arcsin", r"\arccos", r"\arctan", r"\sinh", r"\cosh", r"\tanh", r"\max", r"\min", r"\arg\max", r"\arg\min",
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon", r"\zeta", r"\eta", r"\theta", r"\vartheta",
    r"\iota", r"\kappa", r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi", r"\rho", r"\sigma", r"\tau", r"\upsilon",
    r"\phi", r"\varphi", r"\chi", r"\psi", r"\omega",
    r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi", r"\Pi", r"\Sigma", r"\Upsilon", r"\Phi", r"\Psi", r"\Omega",
    r"\cdot", r"\cdots", r"\ldots", r"\vdots", r"\ddots", r"\times", r"\div", r"\pm", r"\mp", r"\ast", r"\star",
    r"\circ", r"\bullet", r"\oplus", r"\otimes", r"\odot", r"\nabla", r"\partial", r"\infty", r"\forall", r"\exists",
    r"\in", r"\notin", r"\subset", r"\subseteq", r"\supset", r"\supseteq", r"\cup", r"\cap", r"\emptyset", r"\varnothing",
    r"\leq", r"\geq", r"\le", r"\ge", r"\neq", r"\ne", r"\approx", r"\equiv", r"\sim", r"\simeq", r"\cong", r"\propto",
    r"\ll", r"\gg", r"\to", r"\rightarrow", r"\leftarrow", r"\leftrightarrow", r"\Rightarrow", r"\Leftarrow",
    r"\Leftrightarrow", r"\mapsto", r"\uparrow", r"\downarrow",
    r"\left(", r"\right)", r"\left[", r"\right]", r"\left\{", r"\right\}", r"\left|", r"\right|", r"\left\|", r"\right\|",
    r"\langle", r"\rangle", r"\lfloor", r"\rfloor", r"\lceil", r"\rceil",
    r"\begin{matrix}", r"\end{matrix}", r"\begin{pmatrix}", r"\end{pmatrix}", r"\begin{bmatrix}", r"\end{bmatrix}",
    r"\begin{cases}", r"\end{cases}", r"\begin{aligned}", r"\end{aligned}", r"\begin{array}", r"\end{array}",
    r"\mathbf{x}", r"\mathrm{d}", r"\mathcal{L}", r"\mathbb{R}", r"\mathbb{E}", r"\mathbb{N}", r"\mathbb{Z}",
    r"\text{if }", r"\operatorname{softmax}", r"\hat{y}", r"\bar{x}", r"\tilde{z}", r"\vec{v}", r"\dot{x}", r"\ddot{x}",
    r"\overline{AB}", r"\underline{x}", r"\overbrace{a+b}", r"\underbrace{x_1+\dots+x_n}",
    r"\quad", r"\qquad", r"\,", r"\;", r"\!", r"\\", r"&", r"^", r"_", r"{", r"}",
    r"(B \cdot S, V)", r"x_i^2 + y_j^2 = r^2", r"P(y \mid x)",
]

TABLE_SEEDS = [
    "<table>", "</table>", "<thead>", "</thead>", "<tbody>", "</tbody>",
    "<tr>", "</tr>", "<td>", "</td>", "<th>", "</th>", "<caption>", "</caption>",
    "</td><td>", "</td></tr>", "<tr><td>", "</td></tr><tr><td>", "</td></tr></table>",
    "</th><th>", "</th></tr>", "<tr><th>", "</td></tr></tbody></table>",
    '<td rowspan="2">', '<td rowspan="3">', '<td rowspan="4">',
    '<td colspan="2">', '<td colspan="3">', '<td colspan="4">',
    '<th rowspan="2">', '<th colspan="2">', '<td rowspan="2" colspan="2">',
    "O", "X", "△", "○", "×", "◎", "✓", "✗", "—", "-", "N/A",
]


def build_task_slices(
    vocab: Vocab,
    extra_table: list[str] | None = None,
    extra_text: list[str] | None = None,
    extra_formula: list[str] | None = None,
    table_budget: int = 16_384,
    text_budget: int = 16_384,
    formula_budget: int = 8_192,
) -> dict[str, Slice]:
    """Build Track 2 Byte-BPE Safe core vocabulary slices for ``table``, ``text``, and ``formula``."""
    base_corpus = domain_corpus()
    table_slice = build(vocab, base_corpus + TABLE_SEEDS + list(extra_table or []), budget=table_budget)
    text_slice = build(vocab, base_corpus + FORMULA_SEEDS + list(extra_text or []), budget=text_budget)

    # Formula slice: keep all special/byte-safe/partial/greek/symbol/ASCII math tokens + LaTeX commands
    n = vocab.size
    cls = {i: ("special" if vocab.is_special(i) else token_class(vocab.piece(i))) for i in range(n)}
    keep: set[int] = set()
    tiers: dict[str, int] = {}

    def add(name: str, ids) -> None:
        before = len(keep)
        for i in ids:
            if len(keep) >= formula_budget:
                break
            keep.add(int(i))
        tiers[name] = len(keep) - before

    must_formula = [i for i in range(n) if cls[i] in ("special", "partial", "greek", "symbol")]
    must_formula += [i for i in range(n) if len(vocab.piece(i)) == 1]
    add("must", sorted(set(must_formula)))
    seen_f = set()
    for text in FORMULA_SEEDS + list(extra_formula or []):
        seen_f.update(vocab.encode(text))
    add("formula_domain", sorted(seen_f))
    add("ascii_punct", [i for i in range(n) if cls[i] == "ascii" and i not in keep and TEXT_PUNCT.fullmatch(vocab.piece(i))])
    add("ascii_caps", [i for i in range(n) if cls[i] == "ascii" and i not in keep and CAPS_PIECE.fullmatch(vocab.piece(i))])
    add("ascii_words", [i for i in range(n) if cls[i] == "ascii" and i not in keep and WORD_PIECE.fullmatch(vocab.piece(i))])
    add("hangul", [i for i in range(n) if cls[i] == "hangul" and i not in keep])
    add("ascii_rest", [i for i in range(n) if cls[i] == "ascii" and i not in keep])
    add("rest", [i for i in range(n) if i not in keep])
    f_ids = np.array(sorted(keep), dtype=np.int64)
    formula_slice = Slice(f_ids, tiers, {})
    return {"table": table_slice, "text": text_slice, "formula": formula_slice}


def build_universal_slice(
    vocab: Vocab,
    extra: list[str] | None = None,
    budget: int = BUDGET,
    cjk_fill: int = CJK_FILL,
) -> Slice:
    """Build a single 16,384-row Byte-BPE Safe universal slice covering table, text, and formula."""
    corpus = domain_corpus() + TABLE_SEEDS + FORMULA_SEEDS + list(extra or [])
    return build(vocab, corpus, budget=budget, cjk_fill=cjk_fill)
