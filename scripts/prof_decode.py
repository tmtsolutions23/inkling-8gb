"""Per-token decode breakdown from an nsys sqlite export (CUDA trace).
Usage: .venv\\Scripts\\python -I scripts/prof_decode.py results/prof/X.sqlite [n_decode_tokens]
Uses the last 75% of the GPU timeline (decode, after prompt) and reports: wall/token, GPU kernel busy (union), memcpy busy,
and the remaining time (= CPU expert matmuls + sync/launch gaps)."""
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 128


def intervals(table):
    try:
        return db.execute(f"select start, end from {table} order by start").fetchall()
    except sqlite3.OperationalError:
        return []


kern = intervals("CUPTI_ACTIVITY_KIND_KERNEL")
mcpy = intervals("CUPTI_ACTIVITY_KIND_MEMCPY")
t0, t1 = kern[0][0], kern[-1][1]
w0 = t0 + 0.25 * (t1 - t0)
win = t1 - w0


def union(iv, lo, hi):
    tot, cur_s, cur_e = 0, None, None
    for s, e in iv:
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


gk = union(kern, w0, t1)
gm = union(mcpy, w0, t1)
both = union(sorted(kern + mcpy), w0, t1)
frac_tok = n_tok * 0.75  # approx tokens in window (decode dominates the timeline)
print(f"window {win/1e9:.2f} s  (~{frac_tok:.0f} tokens -> {win/1e6/frac_tok:.1f} ms/token)")
print(f"GPU kernels busy : {gk/win*100:5.1f}%  ({gk/1e6/frac_tok:.1f} ms/token)")
print(f"memcpy busy      : {gm/win*100:5.1f}%  ({gm/1e6/frac_tok:.2f} ms/token)")
print(f"GPU idle (CPU work + sync gaps): {(win-both)/win*100:5.1f}%  ({(win-both)/1e6/frac_tok:.1f} ms/token)")
n_k = sum(1 for s, e in kern if s >= w0)
n_m = sum(1 for s, e in mcpy if s >= w0)
print(f"kernels/token ~{n_k/frac_tok:.0f}, memcpys/token ~{n_m/frac_tok:.0f}")
