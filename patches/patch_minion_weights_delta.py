"""Ship only the changed tensors of our draft-tuned weights (team taktak, 2026-11-21).

The submission storage left 2.1 GB; a full GravityOCR-KO model folder is 2.2 GB. Our draft-tuned weights (submission 278)
differ from the stock ones in 118 tensors only (language model 117 + lm_head, 1.35 GB); the vision tower is identical.

  make  : python patch_minion_weights_delta.py make <stock weights dir> <tuned weights dir> <out dir>
          -> <out dir>/model_delta.safetensors (only the tensors that differ)
  patch : python patch_minion_weights_delta.py patch <team serve.py>
          -> the launcher assembles the full model folder before it starts the server (untimed load): the stock weights of
             the image ($GRAVITYOCR_WEIGHTS, /opt/gravityocr/weights) with the delta's tensors put in place, written under
             $TRACK2_DIR (fallback: the system temp dir); then GRAVITYOCR_WEIGHTS points at it. A team folder that has a
             full weights/ folder, or no weights_delta/, is left as it was.
"""
import os
import sys


def make(stock, tuned, out):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    delta = {}
    with safe_open(os.path.join(stock, "model.safetensors"), "pt") as a, safe_open(os.path.join(tuned, "model.safetensors"), "pt") as b:
        assert set(a.keys()) == set(b.keys())
        for k in b.keys():
            tb = b.get_tensor(k)
            ta = a.get_tensor(k)
            if ta.shape != tb.shape or not torch.equal(ta, tb):
                delta[k] = tb.contiguous()
    save_file(delta, os.path.join(out, "model_delta.safetensors"), metadata={"format": "pt"})
    print(f"delta: {len(delta)} tensors, {sum(t.numel() * t.element_size() for t in delta.values()) / 1e9:.2f} GB -> {out}")


MERGE = '''

def _merge_weights_delta() -> None:
    """team taktak: weights_delta/model_delta.safetensors holds only the tensors that differ from the stock GravityOCR-KO
    weights (our draft-tuned language model, 1.35 GB), to fit the submission storage. The full model folder is assembled here,
    before the server starts (untimed load), from the stock weights of the image."""
    import shutil
    import tempfile
    delta = HERE / "weights_delta" / "model_delta.safetensors"
    if not delta.is_file() or (HERE / "weights").exists():
        return
    t0 = time.time()
    stock = Path(os.environ.get("GRAVITYOCR_WEIGHTS", "/opt/gravityocr/weights"))
    from safetensors import safe_open
    from safetensors.torch import save_file
    tensors = {}
    with safe_open(str(stock / "model.safetensors"), "pt") as s:
        meta = s.metadata() or {"format": "pt"}
        for k in s.keys():
            tensors[k] = s.get_tensor(k)
    n = 0
    with safe_open(str(delta), "pt") as d:
        for k in d.keys():
            t = d.get_tensor(k)
            if k not in tensors or tensors[k].shape != t.shape or tensors[k].dtype != t.dtype:
                raise RuntimeError(f"weights_delta: {k} does not match the stock weights")
            tensors[k] = t
            n += 1
    for base in (WORK, tempfile.gettempdir()):
        out = Path(base) / "weights_merged"
        try:
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True)
            for f in stock.iterdir():
                if f.is_file() and f.name != "model.safetensors":
                    shutil.copy2(f, out / f.name)
            save_file(tensors, str(out / "model.safetensors"), metadata=meta)
            break
        except OSError as e:
            log(f"weights: could not write {out} ({e!r}); trying the next place")
    else:
        raise RuntimeError("weights_delta: nowhere to write the merged weights")
    os.environ["GRAVITYOCR_WEIGHTS"] = str(out)
    log(f"weights: stock + {n} tensors of weights_delta -> {out} ({time.time() - t0:.1f} s)")


def main() -> None:
    _merge_weights_delta()
'''


def patch(serve_py):
    s = open(serve_py).read()
    if "_merge_weights_delta" in s:
        print("already patched", serve_py)
        return
    old = "\n\ndef main() -> None:\n"
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, MERGE, 1)
    open(serve_py, "w").write(s)
    print("patched", serve_py)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(*sys.argv[2:5])
    else:
        patch(sys.argv[2])
