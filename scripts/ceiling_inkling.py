"""Hybrid (GPU + DDR4) decode / prefill ceilings for Inkling-2-Mini AD-Q4_K.
All byte counts come from results/tensors.json (scripts/gguf_dump.py). Bandwidths are measured (THEORY.md).
Usage: .venv\\Scripts\\python -I scripts/ceiling_inkling.py [ctx_tokens]
"""
import json
import sys

MiB = 2**20
CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 512
BW_GPU = 422e9          # measured D2D read (gpubw.py)
BW_CPU_LO = 41e9        # measured DDR4 copy plateau (membw.py)
BW_CPU_HI = 48e9        # plausible true read ceiling for DDR4-3600 dual channel
PCIE = 14.1e9           # measured H2D pinned
GPU_TFLOPS = 52e12      # measured FP16 GEMM 8192^3
LOOP = set(range(6, 18))
TOPK, NEXP = 8, 128

t = json.load(open("results/tensors.json"))["tensors"]
exp = {L: 0 for L in range(24)}
oth = {L: 0 for L in range(24)}
glob = 0
for x in t:
    if x["layer"] < 0:
        if x["name"].startswith("output"):
            glob += x["bytes"]          # LM head read every token; token_embd is a row lookup (negligible)
    elif "_exps" in x["name"]:
        exp[x["layer"]] += x["bytes"]
    else:
        oth[x["layer"]] += x["bytes"]

passes = {L: (3 if L in LOOP else 1) for L in range(24)}
act = {L: exp[L] * TOPK / NEXP for L in range(24)}          # bytes of the 8 routed experts per pass
kv_full = 8 * CTX * 8 * 128 * 2 * 2                          # 8 non-SWA instances, f16 K+V
kv_swa = 40 * min(CTX, 1024) * 8 * 128 * 2 * 2
gpu_fixed = glob + sum(oth[L] * passes[L] for L in range(24)) + kv_full + kv_swa

print(f"ctx={CTX}  active experts/pass={act[0]/MiB:.1f} MiB  GPU fixed bytes/token={gpu_fixed/MiB:.0f} MiB "
      f"(LM head {glob/MiB:.0f}, attn+ctrl {sum(oth[L]*passes[L] for L in range(24))/MiB:.0f}, KV {(kv_full+kv_swa)/MiB:.0f})")
print(f"all-CPU expert bytes/token = {sum(act[L]*passes[L] for L in range(24))/MiB:.0f} MiB\n")


def decode(gpu_layers, bw_cpu):
    cpu = sum(act[L] * passes[L] for L in range(24) if L not in gpu_layers)
    gpu = gpu_fixed + sum(act[L] * passes[L] for L in gpu_layers)
    return 1 / (cpu / bw_cpu + gpu / BW_GPU), cpu, gpu


order_loop = [11, 12, 10, 13, 9, 14, 8, 15, 7, 16, 6, 17]   # middle-out inside the looped block
order_outer = [0, 1, 2, 3, 4, 5]
print("experts-on-GPU | VRAM for experts | DDR MiB/tok | tok/s @41GB/s | tok/s @48GB/s")
for k in range(0, 9):
    for label, order in (("looped", order_loop), ("outer", order_outer)):
        if k > len(order) or (label == "outer" and k == 0):
            continue
        g = set(order[:k])
        lo, cpu, _ = decode(g, BW_CPU_LO)
        hi, _, _ = decode(g, BW_CPU_HI)
        print(f"{k:2d} {label:6s}       | {k*exp[0]/MiB:7.0f} MiB      | {cpu/MiB:7.0f}     | {lo:6.1f}        | {hi:6.1f}")

# prefill
flops_tok = 0
for x in t:
    if x["layer"] < 0:
        if x["name"] == "output.weight":
            flops_tok += 2 * x["shape"][0] * x["shape"][1]
        continue
    n = 1
    for s in x["shape"][:2]:
        n *= s
    if "_exps" in x["name"]:
        flops_tok += 2 * n * TOPK * passes[x["layer"]]
    elif len(x["shape"]) >= 2 and x["shape"][0] > 16 and x["shape"][1] > 16:
        flops_tok += 2 * n * passes[x["layer"]]
all_exp = sum(exp.values())
print(f"\nPrefill FLOPs/token (weights only) = {flops_tok/1e9:.2f} GFLOP")
print(f"  all-GPU compute ceiling @50% of {GPU_TFLOPS/1e12:.0f} TFLOPS = {0.5*GPU_TFLOPS/flops_tok:,.0f} tok/s")
for ub in (128, 256, 512, 1024, 2048):
    t_pcie = all_exp / PCIE          # op-offload streams every CPU expert tensor over PCIe once per ubatch
    print(f"  op-offload PCIe bound, ub={ub:4d}: {ub / t_pcie:6.0f} tok/s  (expert stream {all_exp/2**30:.1f} GiB / {PCIE/1e9:.1f} GB/s per ubatch)")
