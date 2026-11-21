"""Output normalization toward the reference conventions the Track 2 scorer compares against (no model change).

formula : Hangul inside \\mathrm{...} -> \\text{...}. pdflatex (CDM renders both strings) sets \\mathrm content in the math
          roman font, which has no Hangul glyphs: the characters vanish or the render fails (a crop scored 0). The
          references write Korean words in formulas as \\text{...}.
table   : the model writes ±, Greek letters and the down arrow as Unicode; the references write them as inline LaTeX
          (``90.4 $\\pm$ 1.0``, ``$\\lambda$``, ``Loss $\\downarrow$``). No reference table on dev uses those Unicode
          characters, so the rewrite cannot cost a correct cell. (× and ↑ are left alone: references use them both ways.)

Env: TRACK2_POSTFIX (1 = on), TRACK2_POSTFIX_TABLE / TRACK2_POSTFIX_FORMULA (per-task switches, default on).
"""
from __future__ import annotations

import os
import re

ENABLED = os.environ.get("TRACK2_POSTFIX", "1") == "1"
TABLE_ON = os.environ.get("TRACK2_POSTFIX_TABLE", "1") == "1"
FORMULA_ON = os.environ.get("TRACK2_POSTFIX_FORMULA", "1") == "1"

_HANGUL = re.compile("[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")
_MATHRM = re.compile(r"\\mathrm\s*\{([^{}]*)\}")

_GREEK = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta",
    "θ": r"\theta", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi", "π": r"\pi",
    "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau", "φ": r"\phi", "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi",
    "Ω": r"\Omega",
}
_GREEK_SUB = re.compile("([" + "".join(_GREEK) + "])(\\d+)?")
_PM = re.compile(r"\s*±\s*")
_DOWN = re.compile(r"\s*↓")
# "5×10-5": the model drops the superscript of scientific notation; the references write $5 \times 10^{-5}$
_SCI = re.compile(r"(\d+(?:\.\d+)?)\s*×\s*10\s*[-−]\s*(\d+)")
# "Acc. (%)↑": a unit in parentheses followed by the arrow is written "(%) $\uparrow$" by the references
# (a bare "Overall↑" is kept: those references use the Unicode arrow)
_PAREN_UP = re.compile(r"\)\s*↑")
_CELL = re.compile(r"(<t[dh][^>]*>)(.*?)(</t[dh]>)", re.S)


def _fix_formula(s: str) -> str:
    def rm(m: re.Match) -> str:
        body = m.group(1)
        return "\\text{" + body + "}" if _HANGUL.search(body) else m.group(0)

    return _MATHRM.sub(rm, s)


def _fix_cell(text: str) -> str:
    if "$" in text:  # the model already wrote LaTeX in this cell: leave it
        return text
    t = _PM.sub(" $\\\\pm$ ", text) if "±" in text else text
    if "↓" in t:
        t = _DOWN.sub(" $\\\\downarrow$", t)
    if "×" in t:
        t = _SCI.sub(lambda m: f"${m.group(1)} \\times 10^{{-{m.group(2)}}}$", t)
    if "↑" in t:
        t = _PAREN_UP.sub(") $\\\\uparrow$", t)

    def greek(m: re.Match) -> str:
        g, sub = _GREEK[m.group(1)], m.group(2)
        return f"${g}_{sub}$" if sub else f"${g}$"

    t = _GREEK_SUB.sub(greek, t)
    return t.strip() if t != text else text


def _fix_table(s: str) -> str:
    if "<t" not in s:
        return s
    return _CELL.sub(lambda m: m.group(1) + _fix_cell(m.group(2)) + m.group(3), s)


def fix(text: str, task: str) -> str:
    if not ENABLED or not text:
        return text
    try:
        if task == "formula" and FORMULA_ON:
            return _fix_formula(text)
        if task == "table" and TABLE_ON:
            return _fix_table(text)
    except Exception:
        return text
    return text
