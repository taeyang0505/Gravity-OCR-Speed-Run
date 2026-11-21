"""Reading the model's table output: HTML tags → rows of cells."""

from __future__ import annotations

import re

TAG = re.compile(r"<\s*(/?)\s*([A-Za-z][A-Za-z0-9]*)[^>]*>")
ROW_TAGS = {"table", "tr", "tbody", "thead"}
CELL_TAGS = {"td", "th"}


def has_table(text: str) -> bool:
    return any(m.group(2).lower() in ROW_TAGS | CELL_TAGS for m in TAG.finditer(text))


def strip_tags(text: str) -> str:
    """Tags out: row tags become a line break, cell tags a space."""

    def swap(m: re.Match) -> str:
        name = m.group(2).lower()
        return "\n" if name in ROW_TAGS else " " if name in CELL_TAGS else ""

    return TAG.sub(swap, text)


def rows_of(text: str) -> list[list[str]]:
    """Rows of cells.  Outside a table every line is a row with one cell."""
    rows: list[list[str]] = []
    row: list[str] = []
    buf: list[str] = []
    in_table = False

    def push_cell(force: bool = False) -> None:
        cell = " ".join("".join(buf).split())
        buf.clear()
        if cell or force:
            row.append(cell)

    def push_row() -> None:
        nonlocal row
        if row:
            rows.append(row)
        row = []

    at = 0
    for m in TAG.finditer(text):
        chunk = text[at : m.start()]
        at = m.end()
        closing, name = bool(m.group(1)), m.group(2).lower()
        if in_table:
            buf.append(chunk)
        else:
            for i, line in enumerate(chunk.split("\n")):
                if i:
                    push_cell()
                    push_row()
                buf.append(line)
        if name in CELL_TAGS:
            in_table = True
            if closing:
                push_cell(force=True)  # an empty cell still holds its column
            else:
                push_cell()
        elif name in ROW_TAGS:
            push_cell()
            push_row()
            in_table = not (closing and name == "table")
        # any other tag is dropped, its text stays
    for i, line in enumerate(text[at:].split("\n")):
        if i:
            push_cell()
            push_row()
        buf.append(line)
    push_cell()
    push_row()
    return rows
