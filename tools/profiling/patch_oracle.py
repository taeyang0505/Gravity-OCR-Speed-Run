"""patch_oracle.py <serve.py>: MEASUREMENT ONLY (never for a submission). TRACK2_ORDER_ORACLE=<json {id: {"rounds": n}}> replaces
the predicted decode work of each crop by its measured decode rounds from an earlier dev run, to measure how much a perfect
work predictor could gain. TRACK2_ORDER_ORACLE_KEY=rounds|ms picks the field."""
import sys
p = sys.argv[1]; s = open(p).read()
assert s.count('"pred2": predict_rounds_v2(task_key, w, h, nbytes),') == 1
s = s.replace('"pred2": predict_rounds_v2(task_key, w, h, nbytes),',
              '"pred2": _t2_oracle(cid, predict_rounds_v2(task_key, w, h, nbytes)),')
s = s.replace("\n\ndef order_items(", '''

_T2_ORACLE = None


def _t2_oracle(cid, default):
    """MEASUREMENT ONLY: measured decode rounds of a dev crop (TRACK2_ORDER_ORACLE), else the prediction"""
    global _T2_ORACLE
    path = os.environ.get("TRACK2_ORDER_ORACLE", "")
    if not path:
        return default
    if _T2_ORACLE is None:
        import json as _j
        _T2_ORACLE = _j.load(open(path))
    r = _T2_ORACLE.get(cid)
    return float(r[os.environ.get("TRACK2_ORDER_ORACLE_KEY", "rounds")]) * 40.0 if r else default


def order_items(''', 1)
open(p, "w").write(s); print("patched", p)
