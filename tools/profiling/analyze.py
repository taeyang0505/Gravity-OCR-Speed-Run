"""analyze.py <rep.sqlite>: kernel time of the timed job (the last busy cluster) by stream and kernel family."""
import sqlite3, sys, re, collections
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
rows = db.execute("select start, end, streamId, demangledName, shortName, gridX*gridY*gridZ from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
print("kernels", len(rows))
# busy clusters: split where nothing runs for > GAP
GAP = int(float(sys.argv[2]) * 1e9) if len(sys.argv) > 2 else int(0.12e9)
cl = []; s0 = rows[0][0]; e0 = rows[0][1]; n = 0; i0 = 0
for i, r in enumerate(rows):
    if r[0] > e0 + GAP:
        cl.append((s0, e0, i0, i)); s0 = r[0]; i0 = i
    e0 = max(e0, r[1])
cl.append((s0, e0, i0, len(rows)))
t00 = rows[0][0]
for c in cl[-8:]:
    print(f"cluster +{(c[0]-t00)/1e9:8.3f}s span {(c[1]-c[0])/1e9:6.3f}s kernels {c[3]-c[2]}")
pick = int(sys.argv[3]) if len(sys.argv) > 3 else -1
s, e, a, b = cl[pick]
print(f"\n== picked cluster span {(e-s)/1e9:.3f} s")
def fam(nm):
    l = nm.lower()
    if "flash" in l and ("fwd" in l or "flash_fwd" in l): return "attn_fa2"
    if "batchprefill" in l or "batchdecode" in l or "flashinfer" in l or "mergestate" in l or "persistent" in l: return "attn_flashinfer"
    if "cutlass" in l or "gemm" in l or "sm89_xmma" in l or "cublas" in l or "_mm" in l or "matmul" in l or "fp8_" in l and "quant" not in l: return "gemm"
    if "track2" in l or "t2_" in l: return "track2_fused"
    if "triton" in l: return "triton"
    if "elementwise" in l or "vectorized" in l or "reduce" in l or "copy" in l or "cat" in l or "index" in l: return "torch_misc"
    return "other"
by_stream = collections.defaultdict(float); by = collections.defaultdict(float); top = collections.defaultdict(lambda: [0.0, 0])
for st, en, sid, dn, sn, g in rows[a:b]:
    d = (en - st) / 1e6
    nm = names.get(dn, str(dn))
    by_stream[sid] += d; by[(sid, fam(nm))] += d
    k = (sid, nm[:110]); top[k][0] += d; top[k][1] += 1
# stream busy time (union per stream)
busy = {}
for sid in by_stream:
    iv = sorted((r[0], r[1]) for r in rows[a:b] if r[2] == sid)
    tot = 0; cs, ce = iv[0]
    for x, y in iv[1:]:
        if x > ce: tot += ce - cs; cs, ce = x, y
        else: ce = max(ce, y)
    tot += ce - cs; busy[sid] = tot / 1e6
iv = sorted((r[0], r[1]) for r in rows[a:b]); tot = 0; cs, ce = iv[0]
for x, y in iv[1:]:
    if x > ce: tot += ce - cs; cs, ce = x, y
    else: ce = max(ce, y)
tot += ce - cs
print(f"GPU busy (any stream) {tot/1e6:.1f} ms of {(e-s)/1e6:.1f} ms")
for sid in sorted(by_stream, key=lambda k: -by_stream[k]):
    fr = [r for r in rows[a:b] if r[2] == sid]
    print(f"\nstream {sid}: kernel sum {by_stream[sid]:.1f} ms, busy {busy[sid]:.1f} ms, first +{(fr[0][0]-s)/1e9:.3f}s last +{(fr[-1][1]-s)/1e9:.3f}s")
    for (sd, f), v in sorted(by.items(), key=lambda kv: -kv[1]):
        if sd == sid: print(f"   {f:16s} {v:8.1f} ms")
    for (sd, nm), (v, c) in sorted(top.items(), key=lambda kv: -kv[1][0])[:60]:
        if sd == sid and v > 4: print(f"      {v:7.1f} ms {c:6d}x  {nm}")
