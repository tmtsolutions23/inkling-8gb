"""RAM smoke test (NOT a replacement for MemTest86/TestMem5): N processes each fill a large buffer with seeded pseudo-random
uint64 patterns and verify them, for a fixed time. Any mismatch = memory error at the current XMP settings.
Usage: .venv\\Scripts\\python -I scripts/ramtest.py [total_gib=20] [minutes=10] [procs=8]"""
import multiprocessing as mp
import sys
import time

import numpy as np


def worker(wid, gib, deadline, q):
    n = int(gib * 2**30 // 8)
    buf = np.empty(n, dtype=np.uint64)
    rounds = errors = 0
    while time.time() < deadline:
        seed = (wid << 32) | rounds
        rng = np.random.default_rng(seed)
        chunk = 1 << 24
        for off in range(0, n, chunk):  # write
            buf[off:off + chunk] = rng.integers(0, 2**63, size=min(chunk, n - off), dtype=np.uint64)
        rng = np.random.default_rng(seed)
        for off in range(0, n, chunk):  # verify
            exp = rng.integers(0, 2**63, size=min(chunk, n - off), dtype=np.uint64)
            errors += int(np.count_nonzero(buf[off:off + chunk] != exp))
        buf ^= np.uint64(0xFFFFFFFFFFFFFFFF)  # inverted pattern pass
        rng = np.random.default_rng(seed)
        for off in range(0, n, chunk):
            exp = rng.integers(0, 2**63, size=min(chunk, n - off), dtype=np.uint64) ^ np.uint64(0xFFFFFFFFFFFFFFFF)
            errors += int(np.count_nonzero(buf[off:off + chunk] != exp))
        rounds += 1
    q.put((wid, rounds, errors))


if __name__ == "__main__":
    total = float(sys.argv[1]) if len(sys.argv) > 1 else 20
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 10
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    deadline = time.time() + minutes * 60
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(i, total / procs, deadline, q)) for i in range(procs)]
    t0 = time.time()
    for p in ps:
        p.start()
    res = [q.get() for _ in ps]
    for p in ps:
        p.join()
    rounds = sum(r[1] for r in res)
    errors = sum(r[2] for r in res)
    print(f"{total:.1f} GiB over {procs} procs, {(time.time()-t0)/60:.1f} min, {rounds} write/verify rounds "
          f"({rounds * total / procs * 2:.0f} GiB verified), mismatches: {errors}")
    print("RESULT:", "PASS" if errors == 0 else "FAIL")
    sys.exit(0 if errors == 0 else 1)
