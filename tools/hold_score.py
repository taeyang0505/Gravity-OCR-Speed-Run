"""Per-task dev quality on the 100 held-out crops only (split.json "hold"), from a scorer details.json.
Usage: python3 hold_score.py <run dir or details.json> [more ...]"""
import glob, json, os, sys
H = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.json")))["hold"])
for arg in sys.argv[1:]:
    paths = [arg] if arg.endswith(".json") else sorted(glob.glob(os.path.join(arg, "**", "details.json"), recursive=True))
    for p in paths:
        crops = json.load(open(p))["dev"]["crops"]
        r = {}
        for c in crops:
            if c["id"] in H:
                r.setdefault(c["task"], []).append(c["score"])
        print(p, "| hold:", " ".join(f"{t} {sum(v)/len(v):.2f} (n={len(v)})" for t, v in sorted(r.items())))
