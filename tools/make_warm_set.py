#!/usr/bin/env python3
"""A warm-up set for local runs that behaves like dev-vs-test on the eval server (2026-11-19).

On the eval server the warm-up reads dev crops and the timed job is the private test set: other images. Locally the dev set
stands in for the test set, so a warm-up on dev crops fills the server's prefix cache with the very images that are timed
(measured: 27% of the prompt tokens of the timed job came from the cache with the reference's 96-crop warm-up, and a second
job in the same process ran from the cache entirely, 5.4 s instead of 11.3 s). This writes a copy of the dev set whose images
are resized by a random factor in [0.85, 1.15] (fixed seed): different pixels and sizes, same kind of content.

  python3 make_warm_set.py <dev-set dir> <out dir>
"""
import json, os, random, sys
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, "images"), exist_ok=True)
rng = random.Random(20261119)
rows = [json.loads(l) for l in open(os.path.join(src, "manifest.jsonl")) if l.strip()]
for r in rows:
    im = Image.open(os.path.join(src, r["image"])).convert("RGB")
    f = rng.uniform(0.85, 1.15)
    im = im.resize((max(8, round(im.width * f)), max(8, round(im.height * f))), Image.BICUBIC)
    im.save(os.path.join(out, r["image"]))
with open(os.path.join(out, "manifest.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(rows), "images to", out)
