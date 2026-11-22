"""GPU probe: device-to-device copy bandwidth, H2D/D2H PCIe bandwidth, FP16 GEMM TFLOPS (cupy).
Usage: .venv\\Scripts\\python -I scripts/gpubw.py
"""
import cupy as cp
import numpy as np


def timeit(fn, reps=20):
    s, e = cp.cuda.Event(), cp.cuda.Event()
    fn()
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(cp.cuda.get_elapsed_time(s, e) / 1e3)
    return min(ts), float(np.median(ts))


n = 1024 * 1024 * 1024  # 1 GiB
a = cp.ones(n // 4, dtype=cp.float32)
b = cp.empty_like(a)
tmin, tmed = timeit(lambda: cp.copyto(b, a))
print(f"D2D copy 1 GiB: {2 * n / tmed / 1e9:7.1f} GB/s (median)  {2 * n / tmin / 1e9:7.1f} GB/s (best)   [read+write]")
tmin, tmed = timeit(lambda: a.sum())
print(f"D2D read (sum) : {n / tmed / 1e9:7.1f} GB/s (median)  {n / tmin / 1e9:7.1f} GB/s (best)")
del b

h = cp.cuda.alloc_pinned_memory(256 * 2**20)
hn = np.frombuffer(h, dtype=np.uint8, count=256 * 2**20)
d = cp.empty(256 * 2**20, dtype=cp.uint8)
tmin, tmed = timeit(lambda: d.set(hn))
print(f"H2D pinned 256MiB: {256 * 2**20 / tmed / 1e9:6.1f} GB/s (median)")
tmin, tmed = timeit(lambda: d.get(out=hn))
print(f"D2H pinned 256MiB: {256 * 2**20 / tmed / 1e9:6.1f} GB/s (median)")
del a, d

for N in (4096, 8192):
    for dt in (cp.float16,):
        x = cp.random.rand(N, N).astype(dt)
        y = cp.random.rand(N, N).astype(dt)
        tmin, tmed = timeit(lambda: x @ y, reps=10)
        print(f"GEMM {N}^3 {np.dtype(dt).name}: {2 * N**3 / tmed / 1e12:6.1f} TFLOPS (median)  {2 * N**3 / tmin / 1e12:6.1f} (best)")
        del x, y
cp.get_default_memory_pool().free_all_blocks()
