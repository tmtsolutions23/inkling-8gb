# Inkling 2 Mini (AD-Q4_K) on RTX 5060 Ti 8 GB + 5800X3D + 32 GB DDR4-3600 — final report

*2026-11-22. Full experiment log: [LAB-NOTEBOOK.md](LAB-NOTEBOOK.md). Hardware numbers and ceilings: [THEORY.md](THEORY.md). Run it: `.\run-inkling.ps1`.*

## Result

| Metric | Stock llama.cpp (auto-fit) | Control C0 | BEST2 (config only, official binary) | **BEST3 (patched build, default)** | Ceiling | BEST3 % of ceiling |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode, short context (tok/s) | 8.16 (VRAM spill) | 15.68 | 21.1 | **21.1** | 28.7–32.5 (bandwidth) | 65–74% |
| Decode at ~31K context | — | — | 19.69 | **19.71** | ~27 | ~73% |
| Cold prefill, ~512 tok | 47.3 | 49.7 | ~90 | **~258** | (no validated ceiling) | — |
| Cold prefill, ~2K tok | 49.6 | 45.2 | ~93 | **~319** | | |
| Prefill of a 30.7K prompt | — | — | 89 tok/s (~5.8 min) | **279 tok/s (~1.8 min)** | | |
| Needles 16K / 31K (in final answer) | — | — | 3/3, 3/3 | **3/3, 3/3** | | |
| Perplexity (llama.cpp docs, c512) | NaN with op-offload | — | 4.537 | **4.535** | | |
| TTFT, 45 / 82 / 146 / 266 new tokens (ms) | — | — | 1459 / 1239 / 1886 / 3273 | **817 / 1037 / 1337 / 1541** | | |
| VRAM peak incl. desktop (MiB) | 7714 + 4.4 GB spilled | 4203 | ~6.5–7.0 K | ~300 MiB above BEST2 (~6.8–7.3 K) | 8151 | >= 1 GB free only while desktop < ~0.95 GB |

Decode numbers: median of 6 x 256-token greedy panel requests (coding / reasoning / chat), llama-server `timings`, A/B-sandwiched
between controls. Gates on BEST3 (and BEST2): chat, reasoning_effort none..high, tool-call round trip, strict needles — all PASS.
BEST3 = BEST2 placement + patched build + op-offload at ubatch 512 for batches >= 128 tokens (`GGML_OP_OFFLOAD_MIN_BATCH=128`,
measured: never slower than BEST2 at any prompt size) + `LLAMA_ATTN_Q_CHUNK=32`. BEST3 has ~300 MiB less VRAM margin than BEST2;
the launcher warns below 7200 MiB free — close GPU-heavy apps or use `-Safe` if it does.

## What made the difference
1. **Not spilling VRAM.** llama.cpp's op-offload reserved 7–10 GB of staging on this model; Windows silently spilled it to shared RAM.
2. **Loop-aware placement.** The 12 middle layers run 3x per token, so their experts are worth 3.2x an outer layer's on the GPU.
   Experts of layers 9–14 live on the GPU (`-ot`), everything else in RAM. +28% (5 layers), +5% more (6th).
3. **q8_0 KV + small ubatch** freed ~1 GB for that 6th looped layer. (The "q8_0 KV slows decode" gotcha does not apply here.)
4. **`-tb 16`** for prompt processing (+22%). CUDA graphs (default) are worth +52%.
5. **Two llama.cpp patches** (BEST3, `patches/0002-inkling-opt-full.patch`, private, not upstreamed):
   - *Scheduler:* weights shared by several splits (the looped layers) get a staging copy per split. Fixes op-offload producing
     **NaN** on this model (reproduced on the official binary) and its +587 MiB-per-looped-layer VRAM cost.
   - *Chunked biased attention:* Inkling folds its relative-position bias into a dense per-head FA mask (210 B x ubatch x context);
     the mask is now built per 32-query chunk (`LLAMA_ATTN_Q_CHUNK=32`), cutting the 32K/ub512 buffer 3.4 GB -> ~0.7 GB, which lets
     op-offload run at ubatch 512 next to the 6 GPU layers -> ~3x prefill, decode unchanged.

## Why decode stops at ~21 tok/s
Measured per token (nsys): ~32 ms CPU expert reads (30 RAM-resident layer-passes, ~36 GB/s), ~14.5 ms GPU work,
~4.4 ms CPU<->GPU hand-offs (consistent with DDR4-bandwidth-bound expert reads). More decode speed needs more experts in VRAM (bigger GPU) or faster RAM; the remaining software
levers (GPU kernel fusion for ~8.8K tiny kernels/token, hand-off latency) are open-ended with unquantified payoff.

## Rejected (measured)
`-t 6/7`, `--poll 100`, `--load-mode none` (no gain, 7x startup), outer-layer placement, built-in `--moe-cache-mib` (-9% to -22%),
n-gram speculation as default (kept as `-Coding`: no cost when it doesn't match), larger ubatch without chunking (VRAM).

## Not verified
- RAM stability: only a 10-minute pattern smoke test (20 GiB x 8 procs, 1640 GiB verified, 0 mismatches); run MemTest86/TestMem5 overnight for real assurance, and confirm
  DRAM voltage 1.35 V in BIOS (Windows reports 1.20 V).
- Concurrency (> 1 user) not measured; contexts above 32K not tested; tg128/tg512 and peak system RAM not measured separately.
- MoE cache tested only at cpu-share 1.0, hit rate not inspected.
- `-Coding` (n-gram) on BEST3: works, 39% draft acceptance on a repeated prompt, no measured speedup; output can diverge from
  non-speculative greedy (verify-batch numerics).
- BEST2's raw gate output files were deleted by mistake (results recorded in the lab notebook).
- Exact upstream mechanism of the op-offload NaN not pinned down (the patch fixes it; facts for an upstream report are in the lab notebook).

