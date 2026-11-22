"""Compare temp-0 panel outputs of two runs: common prefix length (chars and words) per prompt.
Usage: python -I scripts/greedy_cmp.py RUN_A RUN_B"""
import json, os, sys
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "outputs")
a = json.load(open(os.path.join(R, sys.argv[1] + ".json")))
b = json.load(open(os.path.join(R, sys.argv[2] + ".json")))
for x, y in zip(a, b):
    s, t = x["text"], y["text"]
    n = next((i for i, (c, d) in enumerate(zip(s, t)) if c != d), min(len(s), len(t)))
    full = s == t
    print(f"prompt {x['prompt']}: identical={full}  common prefix {n} chars / {len(s[:n].split())} words of {len(s)}")
    if not full:
        print(f"    A: ...{s[max(0,n-40):n+40]!r}\n    B: ...{t[max(0,n-40):n+40]!r}")
