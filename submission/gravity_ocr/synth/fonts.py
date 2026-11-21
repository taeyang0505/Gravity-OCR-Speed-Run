"""Font lookup for the generator.

Fonts are never shipped with this repository.  Each role lists candidate files
that are looked up in the system font folders (and in ``GRAVITY_OCR_FONTS``);
the first one found is used.  The rendered glyph shapes therefore depend on the
machine, which is why the generated sets record the files that were used.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

# role → candidates ``(file name, face index)``
CANDIDATES: dict[str, list[tuple[str, int]]] = {
    "gothic": [
        ("malgun.ttf", 0), ("NotoSansKR-Regular.otf", 0), ("NotoSansKR-Regular.ttf", 0),
        ("NotoSansCJK-Regular.ttc", 1), ("NanumGothic.ttf", 0), ("AppleSDGothicNeo.ttc", 0), ("gulim.ttc", 0),
    ],
    "bold": [
        ("malgunbd.ttf", 0), ("NotoSansKR-Bold.otf", 0), ("NotoSansKR-Bold.ttf", 0),
        ("NotoSansCJK-Bold.ttc", 1), ("NanumGothicBold.ttf", 0), ("AppleSDGothicNeo.ttc", 6), ("gulim.ttc", 0),
    ],
    "hand": [
        ("batang.ttc", 2), ("NanumPen.ttf", 0), ("NanumBrush.ttf", 0), ("NanumMyeongjo.ttf", 0),
        ("NotoSerifKR-Regular.otf", 0), ("NotoSerifCJK-Regular.ttc", 1), ("AppleMyungjo.ttf", 0),
    ],
}  # fmt: skip


class FontMissing(RuntimeError):
    pass


def font_dirs() -> list[Path]:
    dirs = [Path(p) for p in os.environ.get("GRAVITY_OCR_FONTS", "").split(os.pathsep) if p]
    dirs += [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
        Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
        Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
        Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library/Fonts",
    ]  # fmt: skip
    return [d for d in dirs if d.is_dir()]


@lru_cache(maxsize=None)
def resolve(role: str) -> tuple[str, int]:
    """Path and face index of the first available candidate for ``role``."""
    dirs = font_dirs()
    for name, index in CANDIDATES[role]:
        for d in dirs:
            direct = d / name
            if direct.is_file():
                return str(direct), index
        for d in dirs:
            if d.parts[:2] in (("/", "usr"), ("\\", "usr")) or "share" in d.parts:
                hit = next(iter(d.rglob(name)), None)
                if hit is not None:
                    return str(hit), index
    wanted = ", ".join(n for n, _ in CANDIDATES[role])
    raise FontMissing(
        f"no font for role '{role}' (looked for {wanted}); "
        "install a Korean font or point GRAVITY_OCR_FONTS at a folder that has one"
    )


@lru_cache(maxsize=512)
def font(role: str, size: int) -> ImageFont.FreeTypeFont:
    path, index = resolve(role)
    return ImageFont.truetype(path, int(size), index=index)


def available() -> bool:
    try:
        for role in CANDIDATES:
            resolve(role)
    except FontMissing:
        return False
    return True


def used() -> dict[str, str]:
    """Role → file name, for the manifest of a generated set."""
    return {role: f"{Path(resolve(role)[0]).name}#{resolve(role)[1]}" for role in CANDIDATES}
