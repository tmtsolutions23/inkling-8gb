# THEORY.md — Hardware, Toolchain & Ceilings

## Phase 0 audit (2026-11-21)

| Item | Found | Note |
| :--- | :--- | :--- |
| GPU | RTX 5060 Ti, 8151 MiB, compute cap 12.0 (sm_120), driver 621.14 | PCIe **Gen4 x8** (card is x8-native) |
| Idle VRAM used | **~1.7 GB** (Chrome, Edge, Antigravity, OrcaSlicer, NZXT CAM, Claude, DWM) | Budget for the model ≈ 8.15 − 1.7 − 0.5 safety ≈ **~6.0 GB** unless GPU apps are closed |
| CPU | Ryzen 7 5800X3D, 8C/16T | |
| RAM | 4 × 8 GB TEAMGROUP-UD4-**3600**, running at **2400 MT/s** | **XMP/DOCP is off.** Spec BW 38.4 GB/s now vs 57.6 GB/s at 3600 (+50%) |
| Power plan | High performance | OK |
| Disk | C: 110 GB free | OK (~20 GB needed) |
| MSVC | VS Build Tools 2019 (16.11) + 2022 (17.14) | OK for source build |
| CUDA Toolkit (nvcc) | **missing** | Needed only for source build (Phase 3) |
| CMake / Ninja | **missing** | Needed only for source build |
| git / Python | 2.29.1 / 3.12.0 | OK |
| hf CLI | missing | Can download with `Invoke-WebRequest` / `pip install huggingface_hub` |

## Upstream state
- llama.cpp PR #32694 "Inkling 2 Mini" merged 2026-11-20; first release containing it: **b12600**. Latest: **b12615** (prebuilt `win-cuda-13.5-x64` available).
- Model repo: `Inkling-2-Mini-AD-Q4_K.gguf` = 15.39 GiB (16.53 GB). `mmproj` F16/BF16 = 0.60 GiB. Also a 0.04 GiB LoRA and IQ2_M / IQ3_S variants.

## Why XMP matters most
CPU-resident experts (~10 GB of the file) are streamed from DDR4 every token. Decode on the CPU side is roughly
`tok/s ≈ BW_ddr4 / bytes_read_from_RAM_per_token`, so RAM bandwidth scales decode almost linearly. Going 2400 → 3600 MT/s
is the single largest lever available and costs nothing but a BIOS toggle.

## XMP enabled (2026-11-21, after reboot)
- All 4 DIMMs report `Speed = ConfiguredClockSpeed = 3600`. (WMI reports `ConfiguredVoltage = 1200 mV`; this is commonly the SPD
  value, not the live XMP voltage — confirm "DRAM Voltage 1.35 V" in BIOS if any instability appears.)
- Idle VRAM after reboot: **1026 MiB** (was 1.7 GB with more GPU apps open).

## Measured DDR4-3600 bandwidth (`scripts/membw.py`, 512 MiB/worker, median of 3)

| Mode | 1 | 2 | 4 | 6 | 8 workers |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Read (`np.sum`, GB/s) | 8.7 | 14.4 | 27.3 | 35.9 | 37.7 |
| Copy (read+write, GB/s) | 39.0 | 38.3 | 41.2 | 40.9 | 40.0 |

- Spec peak 57.6 GB/s. Copy saturates at **~41 GB/s** (~71% of spec) already at 1 thread. Read via `np.sum` is
  partly compute-bound per core (~9 GB/s/core) and still climbing at 8, so **~38–41 GB/s is a lower bound** on the real
  read ceiling; realistic DDR4-3600 dual-channel read is ~45–50 GB/s. The decisive number will be the *effective* BW
  llama.cpp achieves on CPU expert matmuls (bytes/token × tok/s), measured in Phase 1.
- Taken while the GGUF download was running (minor disk/NIC load).
- Runtime: llama.cpp **b12615** (commit a818a46f3, Clang 22.1.8) prebuilt CUDA 13.5 in `llama.cpp\b12615\`.
  CPU backend is the dynamic-dispatch set — no `zen3` variant; expect `haswell` (AVX2/FMA/F16C) to be selected.

## Measured GPU (`scripts/gpubw.py`, cupy 14.2)
| Probe | Result |
| :--- | :--- |
| D2D read (sum, 1 GiB) | **422 GB/s** (94% of 448 spec) |
| D2D copy (r+w) | 375 GB/s |
| PCIe H2D / D2H pinned | **14.1 / 14.3 GB/s** (Gen4 x8) |
| FP16 GEMM 8192³ | **~52 TFLOPS** (48 median, clocks ramping) |
| Clocks under load | SM 2917 MHz, mem 13801 MHz |

## Model byte anatomy (`results/tensors.json`)
- Experts: 587.2 MiB/layer (gate_up Q4_K 364.5 + down Q5_0 222.75). **Active per pass (top-8/128) = 36.7 MiB.**
- Loop: layers **6–17 run 3×** → 48 layer-passes/token. Full-attn layers 5, 11, 17, 23 → **8 KV instances** (11 & 17 ×3).
- Per token (decode) bytes: LM head 469 MiB + attention/ctrl 1461 MiB (incl. loop repeats) = **1.93 GiB GPU-fixed**;
  experts **1762 MiB** if all on CPU; KV 96 MiB @512 ctx, **1184 MiB @32K** (f16).

## VRAM budget (from llama.cpp logs, see PLAN.md "VRAM anatomy")
- CUDA reports ~7013 MiB free at start. Target ≤ ~6.5 GB allocated (≥1 GB headroom for Windows).
- Fixed: non-expert weights 1185 + KV@32K f16 1264 + compute (no op-offload, ub128) 840 ≈ **3.3 GB** → **~5 expert layers** fit (~2.9 GB).
- q8_0 KV saves ~630 MiB (~1 more layer); ub64 saves ~420 MiB more.
- **Op-offload (default) reserves a 7–10 GB compute buffer** → oversubscribes VRAM (WDDM spills to shared RAM).

## Decode ceilings (bandwidth-only, serial CPU→GPU per layer, `scripts/ceiling_inkling.py`)
`t/token = DDR_bytes / BW_ddr + GPU_bytes / 422 GB/s`. DDR BW bracket: 41 GB/s (measured copy) – 48 GB/s (plausible read).

| Experts on GPU | Expert VRAM | DDR MiB/tok | ctx 512 tok/s | ctx 32K tok/s (@41) |
| :--- | ---: | ---: | :---: | ---: |
| none | 0 | 1762 | 20.0 – 23.0 | 18.9 |
| 3 looped | 1.7 GB | 1431 | 23.5 – 26.9 | 22.1 |
| 3 outer | 1.7 GB | 1652 | 21.0 – 24.2 | — |
| **5 looped** | 2.9 GB | 1211 | **26.8 – 30.4** | ~25 |
| 5 outer | 2.9 GB | 1578 | 21.8 – 25.0 | — |
| 6 looped | 3.5 GB | 1101 | 28.7 – 32.5 | ~27 |
| 8 looped | 4.7 GB | 881 | 33.6 – 37.8 | — |

- A looped layer on GPU is worth **3×** an outer one (110 vs 37 MiB of DDR traffic per token). Confirms the plan's priority.
- **Practical ceiling ≈ 85% of these** (split/sync overhead: 98 graph splits/token, CPU↔GPU hops on every MoE layer).
  Realistic target with ~5–6 looped layers on GPU: **~23–27 tok/s** short context.
- Caveat: assumes 8 *distinct* experts read per pass. If the three loop passes reuse experts, the 96 MB V-Cache could absorb some reads (upside).

## Prefill ceilings
- Weights-only FLOPs/token = **9.82 GFLOP** (48 passes). All-GPU compute ceiling @50% of 52 TFLOPS ≈ 2,650 tok/s (unreachable — weights don't fit).
- **Op-offload path** (stream CPU experts over PCIe once per ubatch): 13.8 GiB / 14.1 GB/s ≈ 1.05 s per ubatch →
  ub128 ≈ 122, ub512 ≈ 488, ub2048 ≈ 1950 tok/s. If looped layers are re-streamed on each pass: halve these.
- **CPU path** (no op-offload): experts ≈ 6.1 GFLOP/token on 8 Zen 3 cores (~0.5–1 TFLOPS effective) → ~100–150 tok/s.
- Trade-off: op-offload wins prefill by 3–10× but its staging buffer competes with resident experts for VRAM (or spills).

## KV capacity
- f16: 32 KiB/token across 8 full-attn instances (4 KiB each) + fixed SWA 240 MiB. 32K = 1.0 GiB, 64K = 2.0 GiB.
- Capacity is not the binding constraint at 32K; VRAM for experts is.

## Achieved vs ceiling (end of Phase 2, BEST2 = 6 looped layers on GPU, q8 KV)
| | Ceiling | Achieved | % |
| :--- | ---: | ---: | ---: |
| Decode short ctx | 28.7 (41 GB/s) – 32.5 (48 GB/s) | 21.1–21.3 | 65–74% |
| Decode @31K | ~27 | 19.64 | ~73% |
| Prefill (CPU experts) | ~100–150 (hypothesis) | ~92 | — |
- Gap attribution (hypothesis, not yet profiled): ~74 graph splits/token with CPU↔GPU syncs + CPU MoE kernel efficiency below DDR peak.
- Prefill ceilings for op-offload remain hypotheses: the 7 GB staging buffer could not be tested without spilling.

## Achieved, final (2026-11-22, BEST3 = patched build)
| | Ceiling | BEST3 | % |
| :--- | ---: | ---: | ---: |
| Decode short ctx | 28.7 – 32.5 | 21.1 | 65–74% |
| Decode @31K | ~27 | 19.71 | ~73% |
| Prefill ~2K tok | op-offload PCIe-bound ~0.95 x ubatch (hypothesis) = ~490 at ub512 | ~319 | ~65% of that hypothesis |
| Prefill @30.7K | — | 279 | — |
- Measured decode anatomy (nsys): CPU expert reads ~32 ms/token (~36 GB/s), GPU ~14.5 ms, hand-offs ~4.4 ms.
