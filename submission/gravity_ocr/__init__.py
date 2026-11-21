"""GravityOCR-KO runtime: drawing OCR on a 0.9B vision-language model.

Python keeps the orchestration; the heavy work runs on the best backend that is
there and passes its self-check: a GPU engine (``gpu``: ROCm / Vulkan / CUDA on
candle-rocm-vulkan), else the Rust CPU kernels (``native``), else numpy.

    from gravity_ocr import GravityOCR, read
    model = GravityOCR.from_package("gravity-ocr-ko.gocr")      # device="auto"
    out = read(model, image, kind="text")                        # Decoded: .text, .picks, .stats

``read`` prepares the crop (``drawing.crops``), decodes (speculative, exact),
and with ``ledger=`` checks equipment codes and names against it (``verify``).
"""

from __future__ import annotations

__version__ = "0.2.0"


def __getattr__(name):  # lazy: importing the package does not load numpy-heavy modules
    if name == "GravityOCR":
        from .model import GravityOCR

        return GravityOCR
    raise AttributeError(name)


def read(model, image, kind: str | None = None, ledger: dict | None = None, text_px: float = 30.0, prune: str = "off", max_tokens: int = 1024):
    """Read one crop: prepare it, decode it, verify it against a ledger.

    ``kind``    ``"text"`` / ``"table"`` forces the prompt; ``None`` lets the crop decide
    ``ledger``  ``{"codes": [...], "names": [...]}`` – equipment codes / sensor tags and
                person names to check the reading against (see ``verify``)
    """
    from . import decode, verify
    from .drawing import crops

    prep = crops.prepare_crop(image, prompt=kind, text_px=text_px)
    out = decode.read(model, prep.image, prep.prompt, max_tokens=max_tokens, prune=prune)
    out.stats["crop"] = prep.info.summary()
    if ledger:
        rep = verify.verify(out.text, out.picks, model.vocab.piece, ledger.get("codes", ()), ledger.get("names", ()),
                            name_fields=bool(ledger.get("names")))  # fmt: skip
        out.stats["verify"] = {"fixes": [f.__dict__ for f in rep.fixes], "rejected": rep.rejected, "flagged": rep.flagged}
        out.text = rep.text
    return out
