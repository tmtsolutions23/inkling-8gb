"""Decode breakdown from an nsys sqlite with --cuda-graph-trace=graph. Usage: python -I scripts/prof_graphs.py X.sqlite [n_tok]"""
import sqlite3, sys
db = sqlite3.connect(sys.argv[1]); ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 127
g = db.execute("select start,end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start").fetchall()
k = db.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
m = db.execute("select start,end from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
lo, hi = g[0][0], g[-1][1]
kw = [x for x in k if x[0] >= lo and x[1] <= hi]
mw = [x for x in m if x[0] >= lo and x[1] <= hi]
def union(iv):
    tot = 0; cs = ce = None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None: tot += ce - cs
            cs, ce = s, e
        else: ce = max(ce, e)
    return tot + (ce - cs if ce is not None else 0)
win = hi - lo
busy = union(g + kw + mw)
ev = sorted(g + kw + mw)
gaps = [ev[i+1][0] - ev[i][1] for i in range(len(ev)-1) if ev[i+1][0] > ev[i][1]]
print(f"{win/1e6/ntok:.1f} ms/tok | graphs {len(g)/ntok:.1f}/tok busy {union(g)/1e6/ntok:.2f} ms | non-graph kernels {len(kw)/ntok:.0f}/tok "
      f"busy {union(kw)/1e6/ntok:.2f} ms | memcpy {len(mw)/ntok:.0f}/tok | idle {(win-busy)/1e6/ntok:.1f} ms "
      f"(short gaps <100us: {sum(1 for x in gaps if x<1e5)/ntok:.0f}/tok {sum(x for x in gaps if x<1e5)/1e6/ntok:.2f} ms; long: {sum(x for x in gaps if x>=3e5)/1e6/ntok:.1f} ms)")
