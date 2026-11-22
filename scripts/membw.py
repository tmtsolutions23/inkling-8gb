"""DDR4 read / copy bandwidth probe (1 thread vs N processes).

Each worker owns a 512 MiB buffer (>> 96 MB L3 V-Cache) and streams it repeatedly;
all workers start together behind a barrier, aggregate GB/s = total bytes / slowest wall time.
Usage: python -I scripts/membw.py [max_workers]
"""
import multiprocessing as mp
import sys
import time

import numpy as np

BUF_BYTES = 512 * 1024 * 1024
REPS = 8


def worker(mode, barrier, q):
    a = np.ones(BUF_BYTES // 8, dtype=np.float64)
    b = np.empty_like(a) if mode == "copy" else None
    a.sum()  # fault pages in / warm TLB
    if b is not None:
        np.copyto(b, a)
    barrier.wait()
    t0 = time.perf_counter()
    for _ in range(REPS):
        if mode == "read":
            a.sum()
        else:
            np.copyto(b, a)
    dt = time.perf_counter() - t0
    moved = BUF_BYTES * REPS * (2 if mode == "copy" else 1)  # copy = read + write
    q.put((moved, dt))


def run(mode, n):
    barrier = mp.Barrier(n)
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(mode, barrier, q)) for _ in range(n)]
    for p in ps:
        p.start()
    res = [q.get() for _ in ps]
    for p in ps:
        p.join()
    total = sum(r[0] for r in res)
    wall = max(r[1] for r in res)
    return total / wall / 1e9


if __name__ == "__main__":
    max_w = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for mode in ("read", "copy"):
        for n in sorted({1, 2, 4, 6, max_w}):
            samples = [run(mode, n) for _ in range(3)]
            print(f"{mode:4s} workers={n:2d}  GB/s median={sorted(samples)[1]:6.1f}  "
                  f"(samples {', '.join(f'{s:.1f}' for s in samples)})", flush=True)
