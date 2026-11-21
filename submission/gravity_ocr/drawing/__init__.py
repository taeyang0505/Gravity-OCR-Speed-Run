"""Drawing analysis in pixel space and on the model's own vision features.

``ink``         ink map, rules, row bands, text height, skew (pixel kernels: native)
``layout``      recursive whitespace cuts into blocks, lines and tiles
``primitives``  the four node types (box, icon, line, text strip)
``crops``       three-way crop branch and prefill conditioning
``codes``       KKS / DCIM equipment codes: shapes, mask repair, ledger match
``bank``        block vectors (palette + box + vision shape), vision n-grams by
                message passing, the double-centred boilerplate bank, archive index
``pipeline``    sheet → blocks (title block / signature protected) → crops → text
"""
