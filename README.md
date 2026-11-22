# A 16.5 GB MoE model at 21 tok/s on an 8 GB GPU

Measured, end-to-end optimisation of **Inkling 2 Mini** (AD-Q4_K GGUF: 128-expert MoE, 3.4B active parameters, a looped
architecture) with llama.cpp, on a mid-range desktop:

**RTX 5060 Ti 8 GB (Blackwell, sm_120) · Ryzen 7 5800X3D · 32 GB DDR4-3600 · Windows 10**

| | Stock llama.cpp | Best config, official binary | **Best, with two llama.cpp patches** |
|---|---:|---:|---:|
| Decode, short context | 8.2 tok/s (VRAM spilled) | 21.1 tok/s | **21.1 tok/s** (65–74% of the bandwidth ceiling) |
| Decode at ~31K context | — | 19.7 tok/s | **19.7 tok/s** |
| Prefill, ~2K-token prompt | 50 tok/s | 93 tok/s | **319 tok/s** |
| Prefill, 30.7K-token prompt | — | 5.8 min | **1.8 min** |
| Needle retrieval at 16K / 31K | — | 3/3, 3/3 | **3/3, 3/3** |
| Perplexity (c512) | NaN with op-offload | 4.537 | **4.535** |

Every number is the median of six or more runs, A/B-sandwiched between control runs (control → candidate → control). The
model was only accepted if it also passed correctness gates: greedy equivalence, needles, tool calls and perplexity.
Full write-up: **[docs/REPORT.md](docs/REPORT.md)**.

## What made the difference

1. **Don't let Windows spill VRAM.** llama.cpp's op-offload reserved 7–10 GB of staging buffers on this model, and WDDM
   silently moved the overflow to shared system RAM. That cost 2–3x on every measurement. Watch the GPU's
   "Shared Usage" counter, not just "Dedicated".
2. **Loop-aware expert placement.** Inkling runs its 12 middle layers three times per token. An expert placed in VRAM
   saves three times the DDR4 traffic of an outer layer's expert, so the experts of the middle layers go to the GPU
   (`-ot "blk\.(9|10|11|12|13|14)\.ffn_.*_exps\.weight=CUDA0,..."`): +28%, then +5% for the sixth layer.
3. **q8_0 KV and a small ubatch** freed the ~1 GB that the sixth looped layer needed. The common warning that "q8_0 KV slows
   decode" didn't hold here, which the A/B test showed.
4. **Two patches to llama.cpp** ([patches/](patches/)):
   - *Scheduler:* weights shared by several graph splits (the looped layers) now get a staging copy per split. This fixes
     op-offload producing **NaN** on this model (reproduced on the official binary) and its +587 MiB of VRAM per looped
     layer.
   - *Chunked biased attention:* Inkling folds its relative-position bias into a dense per-head mask of
     ~210 B × ubatch × context. Building it per 32-query chunk cuts the 32K-context buffer from 3.4 GB to ~0.7 GB.
     Op-offload then fits at ubatch 512, giving **~3x prefill** with decode unchanged.
5. **What was tried and rejected:** `-t 6/7`, polling, `--load-mode none`, llama.cpp's built-in MoE cache (−9 to −22%),
   n-gram speculation as the default, and outer-layer placement. Each rejection is in the lab notebook with its numbers.

## Why decode stops at ~21 tok/s

An Nsight Systems profile per token shows ~32 ms of CPU expert reads (30 RAM-resident layer passes at ~36 GB/s), ~14.5 ms
of GPU work and ~4.4 ms of CPU↔GPU hand-offs. Decode is DDR4-bandwidth-bound: going faster needs more experts in VRAM or
faster RAM. [docs/THEORY.md](docs/THEORY.md) derives the ceilings from measured bandwidth (GDDR7 422 GB/s, DDR4 ~41 GB/s,
PCIe Gen4 x8 14 GB/s).

## Repository

```
run-inkling.ps1      launcher: Agent (64K ctx, default) / -Fast (32K, BEST3) / -Safe (official binary) / -Coding / -Lan
best-config.json     the verified champion; configs/ = every experiment (each one change from the champion)
patches/             0001 scheduler fix alone; 0002 full patch set (vs llama.cpp tag b12615, commit a818a46f3)
scripts/             bench harness (bench_inkling.py), gates (gates.py), perplexity (ppl.ps1), profiling (prof_*.py),
                     bandwidth probes (membw.py, gpubw.py), ceilings (ceiling_inkling.py), build (get_cuda.py, build.cmd)
data/ledger.jsonl    machine-readable log of every benchmark run
docs/                REPORT.md (results), THEORY.md (hardware + ceilings), LAB-NOTEBOOK.md (every experiment, accepted and rejected)
```

## Reproduce

1. Download an **AD-Q4_K GGUF of Inkling 2 Mini** (~16.5 GB) and save it as `models\Inkling-2-Mini-AD-Q4_K.gguf`.
2. Unpack the official llama.cpp **b12615** Windows CUDA 13.5 build (and its cudart zip) into `llama.cpp\b12615\`
   ([releases](https://github.com/ggml-org/llama.cpp/releases)). `.\run-inkling.ps1 -Safe` now runs the best
   configuration on the official binary.
3. For the patched build:
   - Clone llama.cpp at `b12615` into `llama.cpp\src` and apply `patches\0002-inkling-opt-full.patch`.
   - Run `python scripts\get_cuda.py` (portable CUDA toolchain), then `scripts\build.cmd inkling`. This needs MSVC Build
     Tools 2022, CMake and Ninja.
   - Run `.\run-inkling.ps1`. It sets `LLAMA_ATTN_Q_CHUNK` and `GGML_OP_OFFLOAD_MIN_BATCH` itself.
4. The server requires an API key from `.secrets\inkling-api-key.txt` (create it, one line), or pass `-NoKey` for
   local-only use.

The placement, thread counts and VRAM budgets are tuned for this exact hardware. On other hardware, re-run the method
rather than copying the flags:
- measure bandwidth (`scripts\membw.py`, `scripts\gpubw.py`);
- compute the ceilings;
- hill-climb one change at a time with `scripts\mkcfg.py` and `scripts\bench_inkling.py`.

## Status of the patches

These are local patches, not submitted upstream. The op-offload NaN is a real upstream bug: the facts needed to report it
are in the lab notebook. The upstream project doesn't accept AI-generated issues or PRs, so any report will be written by
hand.

## Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
- The Inkling 2 Mini model authors and its GGUF quantizers.
- The methodology follows [inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill).
- Built with an AI coding agent (Claude Code), directed, reviewed and tested end to end by the author. Every number
  comes from a logged run.
- License: [MIT](LICENSE).
