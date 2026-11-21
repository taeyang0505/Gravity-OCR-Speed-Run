"""patch_minion_formula_glyph.py <team dir> (team taktak, 2026-11-21): formula crops get the same glyph-height downscale as text
and tables (they are never resized in the winning team's v17). Default OFF.
  TRACK2_FORMULA_GLYPH=<px>               glyph bodies above it are brought down to it (0 = off, the default)
  TRACK2_FORMULA_GLYPH_MIN_PATCHES=<n>    only crops with at least n 14-px patches (default 256)
  TRACK2_FORMULA_GLYPH_MIN_SCALE=<s>      never below this scale (default 0.5)"""
import os
import sys
team = sys.argv[1]
p = os.path.join(team, "gravity_ocr/textscale.py"); s = open(p).read()
if "FORMULA_GLYPH" in s:
    print("already patched"); sys.exit(0)
old_tail = '''    if task == "table" and (TABLE_GLYPH > 0 or TABLE_GLYPH_UP > 0):
        return _glyph_scale(img, TABLE_GLYPH, TABLE_GLYPH_UP, *_hw(img))
    return 1.0, "off"
'''
assert s.count(old_tail) == 1
s = s.replace(old_tail, '''    if task == "table" and (TABLE_GLYPH > 0 or TABLE_GLYPH_UP > 0):
        return _glyph_scale(img, TABLE_GLYPH, TABLE_GLYPH_UP, *_hw(img))
    if task == "formula" and FORMULA_GLYPH > 0:
        # team taktak: formula glyph target (default off)
        h, w = _hw(img)
        if _patches(h, w) < FORMULA_GLYPH_MIN_PATCHES:
            return 1.0, ""
        size, why = glyph_size(_plane(img))
        if size is None or size <= FORMULA_GLYPH:
            return 1.0, "f:" + why
        sc = max(FORMULA_GLYPH_MIN_SCALE, FORMULA_GLYPH / size)
        return (sc, f"f:{size:.1f} {why}") if sc <= 0.9 else (1.0, f"f:fine {size:.1f}")
    return 1.0, "off"
''')
old_g = 'TABLE_GLYPH = float(os.environ.get("TRACK2_TABLE_GLYPH", "12"))\n'
assert s.count(old_g) == 1
s = s.replace(old_g, old_g + '''FORMULA_GLYPH = float(os.environ.get("TRACK2_FORMULA_GLYPH", "0"))          # team taktak (default off)
FORMULA_GLYPH_MIN_PATCHES = int(os.environ.get("TRACK2_FORMULA_GLYPH_MIN_PATCHES", "256"))
FORMULA_GLYPH_MIN_SCALE = float(os.environ.get("TRACK2_FORMULA_GLYPH_MIN_SCALE", "0.5"))
''')
old_sz = "    if scale < 1.0 and (TEXT_GLYPH > 0 or TABLE_GLYPH > 0):\n"
assert s.count(old_sz) == 1
s = s.replace(old_sz, "    if scale < 1.0 and (TEXT_GLYPH > 0 or TABLE_GLYPH > 0 or FORMULA_GLYPH > 0):\n")
open(p, "w").write(s); print("patched", p)
# the processor calls text_downscale only for text and table crops: let formulas in when the target is set
g = os.path.join(team, "gravityocr/sglang/python/sglang/srt/multimodal/processors/glm4v.py"); t = open(g).read()
old = '    if _g_textscale is not None and task in ("text", "table"):\n'
assert t.count(old) == 1
t = t.replace(old, '    if _g_textscale is not None and (task in ("text", "table") or (task == "formula" and getattr(_g_textscale, "FORMULA_GLYPH", 0) > 0)):\n')
open(g, "w").write(t); print("patched", g)
