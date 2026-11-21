"""Decoding strategies on top of :class:`gravity_ocr.model.GravityOCR`.

``greedy``       one token per forward pass (the reference behaviour).
``speculative``  a draft proposes several tokens, one causal pass verifies them
                 and the longest prefix that the model would have produced
                 token by token is kept.  The emitted tokens are the greedy
                 tokens; only the number of passes changes.

A draft comes from the model's block-diffusion head (a block of mask tokens
filled in one pass) and/or from a :class:`Lookahead` table of deterministic
continuations.  The single exception to "identical to greedy" is the early
exit: a diffusion block that reaches the end of the text with every token at or
above ``early_exit`` confidence is taken without verification.

Every emitted token carries its probability and, when it is below
``SHADOW_CEIL``, the runner-up tokens ("Shadow Top-3") – the evidence the
post-OCR verifier needs.  The log never changes what is decoded.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .model import TEXT_PROMPT, GravityOCR, Session
from .ops import top_k

EARLY_EXIT = 0.99
SHADOW_CEIL = 0.95  # log alternatives below this confidence
MIN_ALT_PROB = 0.08  # an alternative counts as evidence from here
MAX_TOKENS = 4096
# L40S 48GB VRAM default: 24,576 patch rows per ragged batched vision pass (overridable via env)
BATCH_PATCHES = int(os.environ.get("GRAVITY_OCR_BATCH_PATCHES", "24576"))


@dataclass
class Pick:
    token: int  # full-vocabulary id
    conf: float
    alts: tuple = ()  # ((token, prob), ...) runner-ups, only below SHADOW_CEIL
    rescued: bool = False  # a row the output slice cut, taken because it beat every kept row


@dataclass
class Decoded:
    picks: list[Pick]
    text: str
    stats: dict = field(default_factory=dict)

    @property
    def tokens(self) -> list[int]:
        return [p.token for p in self.picks]


def pick_of(model: GravityOCR, logits: np.ndarray, rest: tuple | None = None) -> Pick:
    """The token of one row of logits.  ``rest`` = ``(logit, token)`` of the best output row the
    slice cut (``GravityOCR.rest_at``): it is taken when it beats every kept row."""
    if rest is not None and rest[0] > float(logits.max()):
        return _rescued(model, logits, rest)
    rows, probs = top_k(logits, 3)
    conf = float(probs[0])
    alts = ()
    if conf < SHADOW_CEIL:
        alts = tuple((model.token_of(r), float(p)) for r, p in zip(rows[1:], probs[1:]) if p >= MIN_ALT_PROB)
    return Pick(model.token_of(rows[0]), conf, alts)


def _rescued(model: GravityOCR, logits: np.ndarray, rest: tuple) -> Pick:
    """A cut row wins.  Probabilities over the kept rows plus that row (the other cut rows are not known)."""
    rows, _ = top_k(logits, 2)
    e = np.exp((logits[rows] - rest[0]).astype(np.float64))
    z = float(np.sum(np.exp((logits - rest[0]).astype(np.float64)))) + 1.0
    conf = 1.0 / z
    alts = ()
    if conf < SHADOW_CEIL:
        alts = tuple((model.token_of(r), float(p / z)) for r, p in zip(rows, e) if p / z >= MIN_ALT_PROB)
    return Pick(int(rest[1]), conf, alts, rescued=True)


def _finish(model: GravityOCR, picks: list[Pick], stats: dict, t0: float) -> Decoded:
    stats["seconds"] = time.perf_counter() - t0
    stats["tokens"] = len(picks)
    stats["rescued"] = sum(p.rescued for p in picks)
    return Decoded(picks, model.vocab.decode([p.token for p in picks]), stats)


def greedy(model: GravityOCR, session: Session, max_tokens: int = MAX_TOKENS) -> Decoded:
    t0 = time.perf_counter()
    picks: list[Pick] = []
    nxt = pick_of(model, session.logits, session.rest)
    passes = 0
    while len(picks) < max_tokens and nxt.token not in model.eos:
        picks.append(nxt)
        if len(picks) == max_tokens:
            break
        logits = model.step(session, [nxt.token])[0]
        nxt = pick_of(model, logits, model.rest_at(0))
        passes += 1
    return _finish(model, picks, {"strategy": "greedy", "passes": passes, "rows": passes}, t0)


class Lookahead:
    """Deterministic continuations: the last ``order`` tokens → the tokens that follow.

    Built from token sequences (seed phrases, earlier outputs).  A context that
    was seen with two different continuations keeps their common prefix, so the
    table never guesses between alternatives.
    """

    def __init__(self, order: int = 2, max_len: int = 6):
        self.order = order
        self.max_len = max_len
        self.table: dict[tuple, tuple] = {}

    def add(self, tokens) -> None:
        tokens = list(tokens)
        for i in range(self.order, len(tokens)):
            key = tuple(tokens[i - self.order : i])
            cont = tuple(tokens[i : i + self.max_len])
            old = self.table.get(key)
            if old is None:
                self.table[key] = cont
                continue
            n = 0
            while n < min(len(old), len(cont)) and old[n] == cont[n]:
                n += 1
            self.table[key] = old[:n]

    def propose(self, history) -> tuple:
        if len(history) < self.order:
            return ()
        return self.table.get(tuple(history[-self.order :]), ())

    def __len__(self) -> int:
        return sum(1 for v in self.table.values() if v)


def table_lookahead(vocab, order: int = 2, max_len: int = 6) -> Lookahead:
    """Seed a :class:`Lookahead` table with deterministic HTML table tag continuations (``</td><td>``, ``</tr><tr><td>``)."""
    la = Lookahead(order=order, max_len=max_len)
    for seq in (
        "<table><tr><td>",
        "</td><td>",
        "</td></tr><tr><td>",
        "</td></tr></table>",
        "</th><th>",
        "</th></tr><tr><td>",
        "</td></tr></tbody></table>",
    ):
        la.add(vocab.encode(seq))
    return la


def speculative(
    model: GravityOCR,
    session: Session,
    max_tokens: int = MAX_TOKENS,
    early_exit: float | None = EARLY_EXIT,
    block: int | None = None,
    lookahead: Lookahead | None = None,
    diffusion: bool = True,
) -> Decoded:
    """Draft-and-verify decoding; see the module docstring."""
    t0 = time.perf_counter()
    block = block or model.bd_size
    picks: list[Pick] = []
    stats = {
        "strategy": "speculative", "passes": 0, "rows": 0, "draft_passes": 0, "verify_passes": 0,
        "drafted": 0, "accepted": 0, "lookahead_hits": 0, "lookahead_accepted": 0, "early_exit": False,
    }  # fmt: skip
    nxt = pick_of(model, session.logits, session.rest)
    while len(picks) < max_tokens and nxt.token not in model.eos:
        room = max_tokens - len(picks)
        if room == 1:
            picks.append(nxt)
            break
        history = [p.token for p in picks] + [nxt.token]
        cand: list[Pick] = [nxt]
        from_table = False
        guess = lookahead.propose(history) if lookahead is not None else ()
        if guess and room > 1:
            cand += [Pick(t, 0.0) for t in guess[: room - 1]]
            from_table = True
            stats["lookahead_hits"] += 1
        elif diffusion and room > 1:
            n = min(block, room)
            rows = model.step(session, [nxt.token] + [model.mask_id] * (n - 1), mode="block", commit=False)
            stats["passes"] += 1
            stats["draft_passes"] += 1
            stats["rows"] += n
            draft = [pick_of(model, r, model.rest_at(i)) for i, r in enumerate(rows)]  # draft[i] fills offset i + 1
            end = next((i for i, p in enumerate(draft) if p.token in model.eos), None)
            cand += draft[: n - 1] if end is None else draft[:end]
            if (
                end is not None
                and early_exit is not None
                and draft[end].conf >= early_exit
                and all(p.conf >= early_exit for p in cand[1:])
            ):
                picks += cand
                stats["drafted"] += len(cand) - 1
                stats["accepted"] += len(cand) - 1
                stats["early_exit"] = True
                return _finish(model, picks, stats, t0)
        stats["drafted"] += len(cand) - 1
        # verification: what would the model have produced token by token?
        rows = model.step(session, [p.token for p in cand], mode="causal", commit=False)
        stats["passes"] += 1
        stats["verify_passes"] += 1
        stats["rows"] += len(cand)
        auto = [pick_of(model, r, model.rest_at(i)) for i, r in enumerate(rows)]
        m = 0
        while m + 1 < len(cand) and auto[m].token == cand[m + 1].token:
            m += 1
        picks.append(cand[0])
        picks += auto[:m]  # verified tokens carry the autoregressive confidence
        stats["accepted"] += m
        if from_table:
            stats["lookahead_accepted"] += m
        session.accept([p.token for p in cand[: m + 1]])
        nxt = auto[m]
    del picks[max_tokens:]
    return _finish(model, picks, stats, t0)


def read_many(
    model: GravityOCR,
    items: list,
    fit: str = "pad",
    strategy: str = "speculative",
    max_tokens: int = MAX_TOKENS,
    lookahead: Lookahead | None = None,
    prune: str = "off",
    batch_patches: int = BATCH_PATCHES,
) -> list[Decoded]:
    """Read several ``(image, prompt)`` crops.

    On a GPU the vision passes of consecutive crops are batched (one pass for up to
    ``batch_patches`` patch rows: the row-wise work runs once over all of them, attention
    stays inside each crop); prefill and decoding follow crop by crop.  On the CPU this is
    the same as calling :func:`read` for each crop.
    """
    from . import preprocess
    from .prune import blank_groups, blank_token, kept_rows

    work = []
    for image, prompt in items:
        rgb = preprocess.fit_image(image, fit)
        patches, (gh, gw) = preprocess.to_patches(rgb)
        grid = (gh // preprocess.MERGE, gw // preprocess.MERGE)
        blank = blank_groups(rgb) if prune != "off" else np.zeros(grid[0] * grid[1], bool)
        if blank.all():
            blank = np.zeros_like(blank)  # nothing to read: let the model see the page as it is
        rows = kept_rows(blank) if blank.any() else None
        work.append({"patches": patches if rows is None else patches[rows], "gh_gw": (gh, gw), "grid": grid, "rows": rows,
                     "blank": blank, "prompt": prompt})  # fmt: skip
    batched = hasattr(model.vision, "forward_batch")
    i = 0
    while i < len(work):
        j = i + 1
        if batched:
            total = work[i]["patches"].shape[0]
            while j < len(work) and total + work[j]["patches"].shape[0] <= batch_patches:
                total += work[j]["patches"].shape[0]
                j += 1
        t0 = time.perf_counter()
        if batched and j - i > 1:
            outs = model.vision.forward_batch([(w["patches"], w["gh_gw"], w["rows"]) for w in work[i:j]])
        else:
            outs = [model.vision.forward(w["patches"], w["gh_gw"], rows=w["rows"]) for w in work[i:j]]
        dt = time.perf_counter() - t0
        n_all = sum(w["patches"].shape[0] for w in work[i:j])
        for w, out in zip(work[i:j], outs):
            if w["blank"].any():
                full = np.empty((w["blank"].size, out.shape[1]), np.float32)
                full[w["blank"]] = blank_token(model)
                full[~w["blank"]] = out
                out = full
            w["tokens"] = out
            w["vision"] = dt * w["patches"].shape[0] / n_all
        i = j
    results = []
    for w in work:
        t1 = time.perf_counter()
        keep = ~w["blank"] if prune == "prefill" and w["blank"].any() else None
        tokens = w["tokens"]
        session = model.prefill(tokens, w["grid"], w["prompt"], capacity=tokens.shape[0] + 16 + min(max_tokens, 512), keep=keep)
        t2 = time.perf_counter()
        if strategy == "greedy":
            out = greedy(model, session, max_tokens)
        else:
            out = speculative(model, session, max_tokens, lookahead=lookahead)
        out.stats.update(
            vision=w["vision"], prefill=t2 - t1, decode=out.stats["seconds"], total=w["vision"] + time.perf_counter() - t1,
            grid=list(w["grid"]), image_tokens=int(tokens.shape[0]), prompt_tokens=session.prompt_tokens,
            blank_share=round(float(w["blank"].mean()) if w["blank"].size else 0.0, 4), batched=batched,
        )  # fmt: skip
        results.append(out)
    return results


def read(
    model: GravityOCR,
    image,
    prompt: str = TEXT_PROMPT,
    fit: str = "pad",
    strategy: str = "speculative",
    max_tokens: int = MAX_TOKENS,
    lookahead: Lookahead | None = None,
    prune: str = "off",
) -> Decoded:
    """Image → text in one call; ``stats`` carries the time of every stage.

    ``prune``: ``"off"``, ``"vision"`` (blank merge groups skip the vision tower) or
    ``"prefill"`` (they are also left out of the language prefill), see ``prune.py``.
    """
    t0 = time.perf_counter()
    keep = None
    blank_share = 0.0
    if prune == "off":
        tokens, grid = model.encode_image(image, fit)
    else:
        from . import preprocess
        from .prune import encode

        tokens, grid, blank = encode(model, preprocess.fit_image(image, fit), level=prune)
        blank_share = float(blank.mean()) if blank.size else 0.0
        if prune == "prefill" and blank.any() and not blank.all():
            keep = ~blank
    t1 = time.perf_counter()
    session = model.prefill(tokens, grid, prompt, capacity=tokens.shape[0] + 16 + min(max_tokens, 512), keep=keep)
    t2 = time.perf_counter()
    if strategy == "greedy":
        out = greedy(model, session, max_tokens)
    else:
        out = speculative(model, session, max_tokens, lookahead=lookahead)
    out.stats.update(
        vision=t1 - t0, prefill=t2 - t1, decode=out.stats["seconds"], total=time.perf_counter() - t0,
        grid=list(grid), image_tokens=int(tokens.shape[0]), prompt_tokens=session.prompt_tokens,
        blank_share=round(blank_share, 4),
    )  # fmt: skip
    return out
