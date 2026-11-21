"""Block vectors, the vision n-grams and the boilerplate bank (plan Parts 1-3).

**Unigram** (Part 1).  A block of a sheet becomes one vector

    v⁰ = L2Norm(h_shape + α · c Φ_color + β · b Φ_bbox)

``h_shape``  the mean of the block's image tokens (the vision tower's 2×2-merged
             outputs, 1536 wide) – computed by the same vision pass that reads it;
``c``        the 16-colour palette histogram of its ink (4 grey levels for pixels
             with chroma below 45, 12 hue sectors of 30° for coloured ones);
``b``        its box ``(x0, y0, x1, y1, w, h, cx, cy)`` relative to the sheet;
``Φ``        fixed random orthogonal bases (seeded), so the three parts occupy
             near-orthogonal directions without any training.

**N-grams** (Part 2).  Blocks are linked by spatial relations – left / right /
above / below neighbours within a few text heights and containment (one box
inside another by more than 80 % of its area).  One round of message passing

    m_i = Σ_r 1/|N_r(i)| Σ_{j∈N_r(i)} R_r(v_j),   v' = L2Norm(β₀ v + β₁ m)

with ``R_r`` a relation-specific cyclic shift (an orthogonal map) and β₀ = 0.65,
β₁ = 0.35 turns unigrams into 2-grams, a second round into 3-grams.

**Bank** (Part 3.2).  Exemplar vectors of boilerplate blocks (the notes printed on
every sheet: *prejudice*) and of content blocks (codes, values, tables, titles:
*bias*) form a key matrix that is double-centred – its row means, then its column
means removed – so that what all blocks share cancels.  A block scores the mean
of its top similarities to the prejudice keys minus that to the bias keys; it is
dropped (not read) above a threshold set on the calibration sheets so that no
content block of those sheets would have been dropped.  Protected blocks (title
block, signature) are never dropped.

**Archive** (Part 6).  A sheet vector is the normalised mean of its block 2-grams;
a scan is matched to its master by inner product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ink import CHROMA_MIN, Ink, as_rgb

WIDTH = 1536
ALPHA = 0.5  # weight of the colour part
BETA = 0.5  # weight of the box part
BETA0, BETA1 = 0.65, 0.35
RELATIONS = ("left", "right", "above", "below", "contains", "inside")
NEAR_UNITS = 3.0  # neighbours within this many text heights
CONTAINED = 0.8
TOP_K = 3


def _orthonormal_rows(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((WIDTH, n)))
    return q.T.astype(np.float32)  # [n, WIDTH], orthonormal rows


PHI_COLOR = _orthonormal_rows(16, 1601)
PHI_BOX = _orthonormal_rows(8, 2802)
SHIFTS = {r: 97 * (i + 1) for i, r in enumerate(RELATIONS)}


def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def palette(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """16-bin colour histogram of the ink: 4 grey levels, 12 hue sectors."""
    ink = Ink(rgb) if mask is None else None
    mask = ink.mask if mask is None else mask
    px = rgb[mask].astype(np.int16)
    out = np.zeros(16, np.float32)
    if not len(px):
        return out
    mx, mn = px.max(axis=1), px.min(axis=1)
    chroma = mx - mn
    grey = chroma < CHROMA_MIN
    luma = (px[:, 0] * 77 + px[:, 1] * 150 + px[:, 2] * 29) >> 8
    np.add.at(out, np.minimum(luma[grey] // 64, 3), 1)
    c = px[~grey].astype(np.float32)
    if len(c):
        r, g, b = c[:, 0], c[:, 1], c[:, 2]
        hue = (np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) + 360) % 360
        np.add.at(out, 4 + (hue // 30).astype(int) % 12, 1)
    return out / out.sum()


def box_vector(box, w: int, h: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.array([x0 / w, y0 / h, x1 / w, y1 / h, (x1 - x0) / w, (y1 - y0) / h, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h], np.float32)


def unigram(shape: np.ndarray, colour: np.ndarray, box: np.ndarray) -> np.ndarray:
    return l2(l2(shape.astype(np.float32)) + ALPHA * (colour @ PHI_COLOR) + BETA * (box @ PHI_BOX))


def relations(boxes: list, unit: float) -> dict[str, list[list[int]]]:
    """Neighbours of every block per relation."""
    n = len(boxes)
    out = {r: [[] for _ in range(n)] for r in RELATIONS}
    near = NEAR_UNITS * unit
    for i, (ax0, ay0, ax1, ay1) in enumerate(boxes):
        a_area = max((ax1 - ax0) * (ay1 - ay0), 1)
        for j, (bx0, by0, bx1, by1) in enumerate(boxes):
            if i == j:
                continue
            ix = min(ax1, bx1) - max(ax0, bx0)
            iy = min(ay1, by1) - max(ay0, by0)
            if ix > 0 and iy > 0:
                inter = ix * iy
                b_area = max((bx1 - bx0) * (by1 - by0), 1)
                if inter >= CONTAINED * b_area:
                    out["contains"][i].append(j)
                elif inter >= CONTAINED * a_area:
                    out["inside"][i].append(j)
                continue
            if iy > 0:  # side by side
                if 0 <= bx0 - ax1 <= near:
                    out["right"][i].append(j)
                elif 0 <= ax0 - bx1 <= near:
                    out["left"][i].append(j)
            elif ix > 0:
                if 0 <= by0 - ay1 <= near:
                    out["below"][i].append(j)
                elif 0 <= ay0 - by1 <= near:
                    out["above"][i].append(j)
    return out


def message_pass(v: np.ndarray, rel: dict[str, list[list[int]]]) -> np.ndarray:
    """One round: v' = L2(β₀ v + β₁ Σ_r mean_{N_r} R_r(v_j))."""
    m = np.zeros_like(v)
    for r, nbrs in rel.items():
        shifted = np.roll(v, SHIFTS[r], axis=1)
        for i, js in enumerate(nbrs):
            if js:
                m[i] += shifted[js].mean(axis=0)
    return l2(BETA0 * v + BETA1 * m)


def ngrams(v0: np.ndarray, boxes: list, unit: float) -> tuple[np.ndarray, np.ndarray]:
    rel = relations(boxes, unit)
    v1 = message_pass(v0, rel)
    return v1, message_pass(v1, rel)


def neutral_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Double centring: row means, then column means removed.  Returns the keys and what a query loses."""
    k = keys - keys.mean(axis=1, keepdims=True)
    col = k.mean(axis=0)
    return k - col, col, 0.0


@dataclass
class Bank:
    keys: np.ndarray  # [n, WIDTH] double-centred
    col: np.ndarray  # column means removed from the keys
    prejudice: np.ndarray  # bool per key: boilerplate
    threshold: float
    # which vectors are compared: v0, v1 or v2.  Measured (tools/page_eval.py, the package, held-out
    # sheets, no content block dropped at any level): of 53 printed-note blocks the unigrams drop 39,
    # the 2-grams 40, the 3-grams 35 (with an earlier block cut 44 / 30 / 7 of 51: the n-grams depend
    # on how the neighbours are cut).  The unigrams are also what the archive and the pattern search
    # do best with, so they are the default.
    level: str = "v0"

    @classmethod
    def build(cls, vectors: np.ndarray, prejudice: np.ndarray, level: str = "v0") -> "Bank":
        keys, col, _ = neutral_keys(vectors)
        return cls(l2(keys), col, prejudice.astype(bool), np.inf, level)

    def score(self, v: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
        """Boilerplate score of each row of ``v``; ``exclude`` masks keys (leave-one-sheet-out)."""
        q = v - v.mean(axis=1, keepdims=True) - self.col
        sims = l2(q) @ self.keys.T
        if exclude is not None:
            sims = np.where(exclude, -np.inf, sims)
        pre = np.sort(np.where(self.prejudice[None, :], sims, -np.inf), axis=1)[:, -TOP_K:]
        bia = np.sort(np.where(~self.prejudice[None, :], sims, -np.inf), axis=1)[:, -TOP_K:]
        pre = np.where(np.isfinite(pre), pre, np.nan)
        bia = np.where(np.isfinite(bia), bia, np.nan)
        return np.nan_to_num(np.nanmean(pre, axis=1), nan=-1.0) - np.nan_to_num(np.nanmean(bia, axis=1), nan=-1.0)

    def apply(self, model, rgb: np.ndarray, blocks: list) -> None:
        """Score the blocks of a sheet and drop the boilerplate ones (never a protected block)."""
        vecs = sheet_vectors(model, rgb, blocks)
        s = self.score(vecs[self.level])
        for b, sc in zip(blocks, s):
            b.score = float(sc)
            b.dropped = (not b.protected) and sc > self.threshold


def block_tokens(model, blocks: list) -> list[np.ndarray]:
    """Image tokens of every prepared block (batched vision passes on a GPU)."""
    from .. import preprocess
    from ..decode import BATCH_PATCHES

    items = []
    for b in blocks:
        patches, grid = preprocess.to_patches(preprocess.fit_image(b.prepared.image))
        items.append((patches, grid, None))
    if hasattr(model.vision, "forward_batch"):
        out, i = [], 0
        while i < len(items):
            j, total = i, 0
            while j < len(items) and (j == i or total + items[j][0].shape[0] <= BATCH_PATCHES):
                total += items[j][0].shape[0]
                j += 1
            out += model.vision.forward_batch(items[i:j]) if j - i > 1 else [model.vision.forward(items[i][0], items[i][1])]
            i = j
        return out
    return [model.vision.forward(p, g) for p, g, _ in items]


def sheet_vectors(model, rgb: np.ndarray, blocks: list, tokens: list | None = None) -> dict[str, np.ndarray]:
    rgb = as_rgb(rgb)
    h, w = rgb.shape[:2]
    ink = Ink(rgb)
    toks = tokens if tokens is not None else block_tokens(model, blocks)
    v0 = []
    for b, t in zip(blocks, toks):
        x0, y0, x1, y1 = b.box
        b.extra["tokens"] = t
        colour = palette(rgb[y0:y1, x0:x1], ink.mask[y0:y1, x0:x1])
        v0.append(unigram(t.mean(axis=0), colour, box_vector(b.box, w, h)))
    v0 = np.stack(v0) if v0 else np.zeros((0, WIDTH), np.float32)
    v1, v2 = ngrams(v0, [b.box for b in blocks], ink.unit) if len(v0) else (v0, v0)
    for b, a, c, d in zip(blocks, v0, v1, v2):
        b.vector = d
    return {"v0": v0, "v1": v1, "v2": v2}


def sheet_vector(v2: np.ndarray) -> np.ndarray:
    """The plan's archive key: the normalised mean of a sheet's block 2-grams (kept for comparison;
    ``archive_scores`` is what the archive uses)."""
    return l2(v2.mean(axis=0)) if len(v2) else np.zeros(WIDTH, np.float32)


def archive_scores(query: np.ndarray, masters: list[np.ndarray]) -> np.ndarray:
    """Similarity of a sheet to every master, block to block: each block's best match among the
    other sheet's blocks, averaged over both sheets, on unigram vectors centred on the mean block
    of the masters.  Measured (``tools/archive_eval.py``, 48 scans / red-lined / re-noticed scans
    against 16 masters): 42/48 at the first rank, against 9/48 for the plan's mean of 2-grams – a
    mean of blocks drowns what tells two sheets of one kind apart, and message passing blurs it
    further."""
    c = np.concatenate([m for m in masters if len(m)]).mean(axis=0) if masters else 0.0
    q = l2(query - c)
    out = np.zeros(len(masters), np.float32)
    for i, m in enumerate(masters):
        if not len(m) or not len(q):
            continue
        s = q @ l2(m - c).T
        out[i] = 0.5 * (s.max(axis=1).mean() + s.max(axis=0).mean())
    return out
