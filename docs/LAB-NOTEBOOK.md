# PLAN.md — Running log (every step, change, test, result)

> Context-survival file. Append-only, newest at the bottom. Every experiment gets: timestamp, hypothesis, exact command/config,
> numbers, verdict (ACCEPT / REJECT / INFO) and reason. Machine-readable mirror: `ledger.jsonl`.
> **Resume point** is always the last "NEXT:" line.

## Status board
| Phase | State |
| :--- | :--- |
| 0 Audit / ceilings | done (source build deferred) |
| 1 Harness + baseline | done — control C0 = 15.68 tok/s |
| 2 Config hill-climb | **done — plateau; BEST2 = 21.2 tok/s decode, ~92 tok/s prefill, needles 3/3 @16K+31K** |
| 3 Kernels | **done** — op-offload NaN/VRAM fix + chunked biased attention -> BEST3 (prefill ~3x, decode same) |
| 4 Packaging | **done** — run-inkling.ps1 (BEST3 default, -Safe, -Coding), REPORT.md, all gates pass |

## Files created so far
- `CLAUDE.md` — project instructions, decisions, discipline.
- `THEORY.md` — audit, measured HW numbers, ceilings.
- `skill\` — local copy of inference-optimisation-skill (SKILL.md, references\, scripts\).
- `scripts\membw.py` — DDR4 read/copy bandwidth probe.
- `llama.cpp\b12615\` — prebuilt llama.cpp b12615 (commit a818a46f3), CUDA 13.5 x64 + cudart DLLs. Zips in `llama.cpp\dl\`.
- `models\Inkling-2-Mini-AD-Q4_K.gguf` — verified (sha256 matches HF).
- `.venv\` — Python venv (gguf, numpy, requests, cupy-cuda12x). Use `.venv\Scripts\python.exe -I`.
- `scripts\gguf_dump.py` (tensor sizes), `scripts\gpubw.py` (GPU BW/PCIe/GEMM), `scripts\ceiling_inkling.py` (ceilings).
- `scripts\server.ps1` (start/stop), `scripts\probe_mem.ps1` (buffer sizes), `scripts\bench_inkling.py` (harness, writes ledger.jsonl),
  `scripts\mkcfg.py` (derive config from best-config.json with one change), `scripts\queue.ps1` (serialized run queue -> logs\queue.txt).
- `results\` — tensors.json, gguf_dump.txt, ceiling_*.txt, outputs\<run>.json (panel texts).  `logs\` — server + bench logs.
- `configs\*.json`, `best-config.json` (currently = C0).

---

## Log

### 2026-11-21 — Phase 0 audit
- Toolchain: driver 621.14, no nvcc/CMake/Ninja (not needed until Phase 3), MSVC BT 2019+2022, git 2.29, Python 3.12 + numpy 2.4.2.
- GPU: RTX 5060 Ti 8151 MiB, sm_120, PCIe Gen4 x8. CPU 5800X3D. RAM 4x8 GB DDR4-3600 kit found running at **2400** (XMP off).
- Upstream: PR #32694 (Inkling 2) merged 2026-11-20 -> first release b12600; latest b12615.
- Decisions confirmed by user: project dir, text-only, 32K target, stop after Phase 2, enable XMP first.

### 2026-11-21 — XMP enabled by user (BIOS, MSI MEG X570 UNIFY, A-XMP Profile 1)
- Verified all DIMMs `ConfiguredClockSpeed = 3600`. Idle VRAM after reboot 1026 MiB.
- INFO `scripts/membw.py`: copy ~41 GB/s (1–8 workers), read (np.sum) 37.7 GB/s @8 workers (lower bound, numpy compute-limited).

### 2026-11-21 — Runtime + download
- Downloaded + unpacked llama.cpp b12615 CUDA 13.5 prebuilt. `llama-cli --version` OK.
- GGUF download interrupted when previous session ended (~3.8 GiB); resumed with `curl -C -`.

### 2026-11-21 — GGUF verified + tensor dump
- Download had in fact completed (16,525,559,776 B = HF `x-linked-size`). **sha256 = 36e99f1f…c955f matches HF etag.** ✅
- Created `.venv\` (system pip broken on tqdm.exe lock) with gguf, numpy, requests.
- `scripts/gguf_dump.py` -> `results/gguf_dump.txt`, `results/tensors.json`. Key facts (now in THEORY.md):
  - arch `inkling2_loop`, 24 blocks, **loop.start=6, loop.end=18, loop.count=3** -> layers 6–17 run 3x.
  - Full-attention (non-SWA) layers: **5, 11, 17, 23** (11 and 17 are inside the loop). SWA window 1024.
  - Per layer experts: `ffn_gate_up_exps` Q4_K 364.5 MiB + `ffn_down_exps` Q5_0 222.75 MiB = **587.2 MiB** (4.59 MiB/expert).
  - Non-expert total 1.616 GiB (token_embd 469 MiB + output 469 MiB, both Q8_0; ~29–31 MiB/layer attention+loop ctrl).
  - Experts 13.76 GiB = 89.5% of file (not 94% as the plan said).


### 2026-11-21 — Tooling
- `scripts/server.ps1 start <name> <args>` / `stop` (graceful drain via /slots, kill by PID, logs in `logs\<name>.log`).
- `scripts/probe_mem.ps1 <name> <args>` — launch, scrape CUDA0 model/KV/compute buffer sizes, stop.
- cupy-cuda12x 14.2 installed in .venv (with CUDA wheels) for a GPU bandwidth/GEMM probe.

### 2026-11-21 — INFO: VRAM anatomy (b12615, `-ngl 99 --n-cpu-moe 24 -np 1 -fa on`, f16 KV)
- llama-server defaults now: `--fit on` (auto), `-np` auto=4, `--cache-ram 8192`, op-offload on, `--load-mode auto(mmap)`.
- CUDA0 model buffer (non-expert, all 24 layers + output) = **1185 MiB**. token_embd 469 MiB stays CPU_Mapped.
- KV @32K f16: non-SWA **1024 MiB (8 layer-instances = layers 5,23 + 11,17 ×3 loops)**; SWA 240 MiB (40 instances, 1536 cells); RS 3.7 MiB.
- **CUDA0 compute buffer is huge and dominated by op-offload** (staging copies of CPU expert weights for batched GPU matmul):

| config | compute buffer MiB |
| :--- | ---: |
| c32k ub512 op-offload on (default) | **10475** |
| c32k ub256 on | 8872 |
| c32k ub128 on | 8071 |
| c8k ub512 on | 8099 |
| c32k ub512 fa **off** | 12048 |
| c32k ub512 **--no-op-offload** | 3359 |
| c32k ub128 --no-op-offload | 840 |
| c32k ub64 --no-op-offload | 419 |
| c8k ub512 --no-op-offload | 983 |

- Without op-offload, compute ≈ 210 B × ubatch × n_ctx (dense f32 score tensor ~24 heads) — custom relative/log-scale
  attention on the full-attention layers is evidently not fully fused into FA.
- **With op-offload on, total allocation (~12.9 GB) > 8 GB VRAM: Windows WDDM sysmem fallback silently spills to shared RAM**
  (nvidia-smi shows only 7.7 GB used). Any config that over-allocates is invalid for measurement.
- Implication: with `--no-op-offload -ub 128`, fixed GPU cost ≈ 1185+1264+840 ≈ 3.3 GB -> ~5 expert layers (587 MiB each) fit in ~6.5 GB.
  q8_0 KV would free ~630 MiB (≈1 more layer). Trade-off to measure: op-offload prefill speed vs VRAM for experts.

### 2026-11-21 — INFO: GPU probe + ceilings (details in THEORY.md)
- `scripts/gpubw.py`: D2D read 422 GB/s, copy 375, PCIe H2D 14.1 GB/s, FP16 GEMM ~52 TFLOPS.
- `scripts/ceiling_inkling.py` -> `results/ceiling_512.txt`, `results/ceiling_32k.txt`.
  Decode ceiling: all experts CPU 20–23 tok/s; 5 looped layers on GPU 26.8–30.4; 6 looped 28.7–32.5 (bandwidth-only).
  Prefill: op-offload PCIe-bound ≈ 0.95 × ubatch tok/s; CPU-only ≈ 100–150 tok/s.
- Phase 0 complete except source build (deferred to Phase 3, not needed for prebuilt).

### 2026-11-21 — First real generation (advisor: blocking check before harness)
- `-ngl 99 --n-cpu-moe 24 -c 32768 -fa on -t 8 -np 1 -ub 128 --no-op-offload --cache-ram 0`: sane code + explanation,
  `finish=stop`, reasoning in `reasoning_content` (model reasons even at effort low). EOS warning harmless.
- Cold first request 12.3 tok/s (mmap page-in), **warm decode 15.4 tok/s** (= 67–77% of the 20–23 all-CPU ceiling).
- **Spill detector**: Windows counter `\GPU Adapter Memory(*)\Shared Usage`. Non-spilling config: ~155 MiB (pinned host bufs).
  Oversubscribed default (op-offload on): **6.3 GB shared**, single cold-ish sample 11.1 tok/s.

### 2026-11-21 — Phase 1 harness: `scripts/bench_inkling.py`
- Python launches llama-server directly (args list, no shell quoting issues), `-np 1 --cache-ram 0 -lv 4` always.
- Per request: temp 0, seed 1, `reasoning_effort: "low"`, non-streamed; numbers from server `timings`.
- Metrics: warm-up x2 (discarded), ttft_short, **c1_decode = median of 6 × 256-token panel requests** (coding/reasoning/chat),
  cold prefill ~512 & ~2048 (unique prefix, sized with `/tokenize`), warm repeat (cache_n), placement (buffer sizes from log),
  vram peak, shared-usage delta. Flags SPILL (>300 MiB shared growth => void), LOOP, TRUNC, PLACEMENT, REQ_FAIL.
  `--long` = needles at ~16K and ~30.7K (max_tokens 1536). Panel outputs -> `results/outputs/<name>.json`.
- **Target metric declared: c1 decode (primary).** Prefill reported as a trade-off, not gated by the 3% rule
  (op-offload on/off is a decode-vs-prefill trade).

### 2026-11-21 — Phase 1 baselines (ledger.jsonl)
| run | config | c1 decode | prefill 512 / 2048 cold | compute buf | shared Δ | verdict |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| B0-stock-fit | `-c 32768` only (auto-fit) | **8.16** | 47.3 / 49.6 | 10915 | 4390 MiB | INFO, SPILL — fit gave up: **0/49 layers on GPU** |
| B1-stock-ncmoe24 | `-ngl 99 --n-cpu-moe 24 -c 32768` | 15.48 | 57.6 / 75.7 | 10475 | 6254 MiB | INFO, SPILL |
| C0-a | B1 + `-fa on -t 8 -ub 128 --no-op-offload` | **15.68** | 49.7 / 45.2 | 840 | 110 MiB | **CONTROL** |
- Op-offload with a spilled staging buffer only gives 58–76 tok/s prefill (vs ~490 hoped) — the staging buffer lives in
  shared sysmem, so it is not a usable lever on 8 GB. Staging buffer size is ub-independent (~7 GB), so smaller ub can't save it.
- Prefill on CPU ~50 tok/s (≈300 GFLOPS effective on 6.1 GFLOP/token experts) — a 32K needle prefill costs ~10 min.
- C0 peak VRAM 4203 MiB (incl. ~1 GB desktop). Budget to ≤ ~7000 MiB peak → ~2.8 GB for experts → 4 layers safe, 5 borderline.

### 2026-11-21 — Phase 2 lever 1: loop-aware `-ot` placement (running)
- `-ot "blk\.(11|12|10|13)\.ffn_.*_exps\.weight=CUDA0,blk\.\d+\.ffn_.*_exps\.weight=CPU"` (replaces --n-cpu-moe). First match wins.
  Verified: CUDA0 model buffer 3534.37 MiB = 1185.4 + 4×587.25. ✅ Graph splits drop 98 -> 74.
- Queue: P-loop4 (11,12,10,13) → C0-b → P-loop5 (+9) → C0-c → P-outer4 (0,1,2,3) → C0-d. Progress in `logs\queue.txt`.

#### Results (c1 decode, 6 samples each; spread within ±0.1 except one 13.6 outlier in C0-d)
| run | experts on GPU | c1 median | vs neighbours | prefill 512/2048 | VRAM peak | verdict |
| :--- | :--- | ---: | :--- | ---: | ---: | :--- |
| C0-a | none | 15.68 | — | 49.7 / 45.2 | 4203 | control |
| P-loop4 | 11,12,10,13 | **19.11** | +21.9% vs 15.68/15.70 | 69.2 / 69.6 | 6551 | ACCEPT |
| C0-b | none | 15.70 | — | 49.6 / 45.4 | 4203 | control |
| P-loop5 | 11,12,10,13,9 | **20.12** | +28.2% vs 15.70/15.70 | 75.6 / 76.3 | **7137** | **ACCEPT → best** (headroom only ~1.0 GB) |
| C0-c | none | 15.70 | — | 49.4 / 44.9 | 4203 | control |
| P-outer4 | 0,1,2,3 | 16.76 | +6.8% | 57.6 / 57.1 | 6553 | REJECT vs loop4 (19.11) |
| C0-d | none | 15.70 | — | 53.0 / 53.4 | 5058 (!) | control; desktop VRAM grew ~855 MiB during this run |
- **Loop theory confirmed:** per GPU layer, looped = +0.85 tok/s/layer vs outer = +0.27 → 3.2×.
- vs ceiling: C0 15.7 / 20–23 = 68–78%; loop5 20.1 / 26.8–30.4 = 66–75%.
- Prefill also improves (+52%) since GPU-resident layers no longer need CPU GEMM.
- Risk: loop5 leaves ~1.0 GB VRAM; desktop VRAM fluctuates (C0-d saw +855 MiB) → spill risk. Next levers free VRAM.
- `best-config.json` = P-loop5.

### 2026-11-21 — Phase 2 lever 3 (KV dtype) + ubatch-64 (VRAM levers), on top of loop5
- KV-q8: loop5 + `-ctk q8_0 -ctv q8_0` (saves ~630 MiB @32K; gotcha says can slow decode on few-KV-layer models).
- UB64: loop5 + `-ub 64 -b 64` (compute buffer 840 → ~420 MiB; costs prefill).
- Queue: L5-a → KV-q8 → L5-b → UB64 → L5-c.

#### Results
| run | change vs loop5 | c1 median | prefill 512/2048 | KV / compute MiB | VRAM peak | verdict |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| L5-a | — | 20.16 | 73.7 / 74.7 | 1224 / 840 | 7280 | control |
| KV-q8 | `-ctk q8_0 -ctv q8_0` | 20.02 (−0.5%) | 73.8 / 75.3 | **650** / 840 | 6783 | ACCEPT as VRAM enabler (decode in noise; lossy-labelled) |
| L5-b | — | 20.07 | 74.2 / 75.5 | 1224 / 840 | 7278 | control |
| UB64 | `-ub 64 -b 64` | 20.12 (0%) | 68.5 / 70.3 (−7%) | 1224 / **419** | 6860 | ACCEPT as VRAM enabler (prefill trade, lossless) |
| L5-c | — | 20.06 | 74.2 / 74.8 | 1224 / 840 | 7281 | control |
- The skill's "q8_0 KV −40% decode" gotcha does **not** reproduce here at short context (needs a 32K check).
- Note desktop VRAM drifted up: L5 peak now 7280 (was 7137) → headroom < 0.9 GB. Freeing VRAM is needed regardless.
- **Greedy equivalence** (`scripts/greedy_cmp.py`, 6 panel prompts × 256 tok, temp 0):
  - Same config twice (L5-a vs L5-b): **6/6 identical** → harness is deterministic.
  - UB64 vs L5: **6/6 identical** → lossless.
  - KV-q8 vs L5: 1/6 identical; first divergence after 10–64 words, at benign word choices; all outputs coherent.
  - For scale, the lossless placement change C0 vs loop5 also diverges early (1/6 identical, 4–62 words) — kernel numerics.
    So q8 KV drift ≈ kernel-noise class. Must still pass the 16K/32K needle gate.

### 2026-11-21 — Phase 2 lever 1b: loop6 (add layer 14) using freed VRAM
- Queue: L5-d → L6-q8-ub64 → L5-e → L6-q8 (ub128) → L5-f.

#### Results
| run | config | c1 median | prefill 512/2048 | GPU model / KV / compute | VRAM peak | verdict |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| L5-d | loop5 f16 ub128 | 20.05 | 73.7 / 74.8 | 4122 / 1224 / 840 | 7278 | control |
| L6-q8-ub64 | loop6 (+14), q8 KV, ub64 | **21.11** (+5.0%) | 75.0 / 75.9 | 4709 / 650 / 420 | **6880** | **ACCEPT → best** (1.27 GB headroom) |
| L5-e | loop5 | 20.15 | 73.8 / 75.0 | | 7278 | control |
| L6-q8 | loop6, q8 KV, ub128 | 21.30 (+5.9%) | 79.2 / 81.5 | 4709 / 650 / 840 | 7301 | REJECT: headroom < 0.9 GB (rule ≥1.0) — keep as "max" variant if user frees desktop VRAM |
| L5-f | loop5 | 20.18 | 74.8 / 75.3 | | 7279 | control |
- 6th looped layer = +1.0 tok/s (matches +0.85–1.0/layer from earlier). loop7 would need another ~590 MiB → not safe with ~1 GB desktop use.
- `best-config.json` = L6-q8-ub64 (21.11 tok/s, +35% vs C0, +159% vs stock auto-fit).

### 2026-11-21 — Phase 2 lever 4: CPU threads / load mode (base L6-q8-ub64)
- Queue: L6-a → T6 (`-t 6`) → L6-b → T7 → L6-c → TB16 (`-tb 16`, prefill only) → L6-d → LM-none (`--load-mode none`, i.e. no mmap) → L6-e.

#### Results
| run | change | c1 median | prefill 512/2048 | startup | verdict |
| :--- | :--- | ---: | ---: | ---: | :--- |
| L6-a | — | 21.32 | 74.2 / 75.6 | 4.2 s | control |
| T6 | `-t 6` | 21.24 (−0.4%) | 69.0 / 70.6 | | REJECT (noise, prefill −7%) |
| L6-b | — | 21.31 | 74.9 / 76.4 | | control |
| T7 | `-t 7` | 21.13 (−0.9%) | 72.2 / 73.8 | | REJECT |
| L6-c | — | 21.29 | 74.9 / 75.8 | | control |
| TB16 | `-tb 16` | 21.25 (−0.2%, noise) | **90.3 / 93.6 (+21–23%)** | | **ACCEPT** (prefill lever; decode unaffected) |
| L6-d | — | 21.30 | 74.6 / 69.9 | | control |
| LM-none | `--load-mode none` | 21.18 (−0.6%) | 73.6 / 75.0 | **29.7 s** | REJECT (no gain, 7× slower startup) |
| L6-e | — | 21.20 | 73.9 / 75.1 | 9.7 s | control |
- LM-none's SPILL flag is a **false positive**: without mmap, weights live in pinned CUDA_Host memory, which Windows counts as
  "Shared Usage" (+11 GB). The spill detector is only valid for mmap configs; noted.
- Decode: 3 consecutive levers inside noise (T6, T7, LM-none) → **decode config plateau** at ~21.2–21.3 tok/s.
- Controls this block are 21.2–21.3 (earlier L6-q8-ub64 run was 21.11 — drift within ±1%).
- `best-config.json` = **BEST2** = L6-q8-ub64 + `-tb 16`.
- CUDA graphs: built with USE_GRAPHS=1; server logs "graphs reused" (ggml graph reuse, 0 reallocs). Capture status under CPU splits unknown → A/B with `GGML_CUDA_DISABLE_GRAPHS=1`.
- Harness now supports `"env": {...}` in configs.

### 2026-11-21 — Phase 2 levers 2 & 5: CUDA graphs, poll, n-gram speculation (base BEST2)
- Queue: B2-a → NOGRAPH (env GGML_CUDA_DISABLE_GRAPHS=1) → B2-b → POLL100 (`--poll 100`) → B2-c → SPEC-ngram (`--spec-type ngram-mod`) → B2-d.

#### Results
| run | change | c1 median | samples | prefill 512/2048 | verdict |
| :--- | :--- | ---: | :--- | ---: | :--- |
| B2-a | — | 21.06 | 20.7–21.16 | 89.8 / 92.3 | control |
| NOGRAPH | `GGML_CUDA_DISABLE_GRAPHS=1` | **13.89 (−34%)** | tight | 88.8 / 91.4 | INFO: **CUDA graphs ARE active and worth +52%** → keep default on |
| B2-b | — | 21.09 | | 89.5 / 93.2 | control |
| POLL100 | `--poll 100` | 21.12 (+0.1%) | | 90.0 / 92.8 | REJECT (noise) |
| B2-c | — | 21.09 | | 88.9 / 88.6 | control |
| SPEC-ngram | `--spec-type ngram-mod` | 21.20 (=B2-d) | one sample **24.73** | 91.2 / 93.1 | REJECT as default (no win on panel); **keep as optional coding/edit variant** |
| B2-d | — | 21.20 | | 89.8 / 92.6 | control |
- ngram-mod drafted only once in the whole panel: prompt 0 repeated text the warm-up had generated → 53/53 accepted, 24.7 tok/s.
  Zero cost when no match. Greedy: 5/6 identical vs B2-a; prompt 0 diverged after 38 words (batch-verify numerics).
- **Phase 2 decode PLATEAU** (T6, T7, LM-none, POLL100, SPEC all in noise). Best decode ≈ 21.1–21.3 tok/s.

### 2026-11-21 — Stability gates: long-context needle retrieval (running)
- G-BEST2 (BEST2, q8 KV) `--long`: 3 needles in ~16K and ~30.7K-token docs, max_tokens 1536.
- G-L5f16 (loop5, f16 KV, ub128, tb16) `--long`: f16-KV comparator.

#### Results — both PASS
| run | KV | 16K needles | 16K decode | 30.9K needles | 30.9K prefill | 30.9K decode | c1 |
| :--- | :--- | :---: | ---: | :---: | ---: | ---: | ---: |
| G-BEST2 | q8_0 | **3/3** stop | 20.25 | **3/3** stop | 87.2 | **19.64** | 21.16 |
| G-L5f16 | f16 (loop5) | 3/3 stop | 19.13 | 3/3 stop | 86.8 | 18.51 | 20.06 |
- q8 KV keeps full retrieval at 100% context, and decode at depth is *higher* than f16 (less KV bytes) → the "q8_0 −40%" gotcha does not apply here.
- A 30.9K-token prompt costs ~5.9 min of prefill (87 tok/s) — the main remaining UX problem.

### 2026-11-21 — Packaging (early Phase 4 item)
- `run-inkling.ps1` — BEST2 launcher (`-Coding` adds ngram-mod spec, `-MaxVram` = ub128 variant), warns if free VRAM is low.
  Smoke-tested: healthy, sane answer, 20.45 tok/s single cold sample, server stopped cleanly, VRAM back to 743 MiB.

---

## STATUS — Phase 2 complete (plateau), 2026-11-21. Phase 3 ON HOLD pending user check-in.

| Metric | Stock (B0 auto-fit) | Naive (B1 ncmoe24) | Control C0 | **Best (BEST2)** | Ceiling (bandwidth) | % of ceiling |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| c1 decode, short ctx (tok/s) | 8.16 (SPILL) | 15.48 (SPILL) | 15.68 | **21.1–21.3** | 28.7–32.5 (loop6) | **65–74%** |
| decode @ 30.9K ctx | — | — | — | **19.64** | ~27 | ~73% |
| cold prefill ~2K (tok/s) | 49.6 | 75.7 | 45.2 | **~92** | no validated ceiling (all-GPU compute bound ≈ 2,650) | ~3% of compute bound |
| TTFT short (ms) | 494 | 357 | 351 | **~285** | — | — |
| needles 16K / 31K | — | — | — | **3/3, 3/3** | — | — |
| VRAM peak (MiB, incl. ~1 GB desktop) | 7714 + 4.4 GB spilled | 7782 + 6.3 GB spilled | 4203 | **~6900–7000** | 8151 | ≥1.15 GB headroom |
All numbers: median of 6 × 256-token temp-0 panel requests via llama-server `timings`, warm, reasoning_effort low, `-np 1 --cache-ram 0`.

**BEST2** = `-ngl 99 -ot "blk\.(11|12|10|13|9|14)\.ffn_.*_exps\.weight=CUDA0,blk\.\d+\.ffn_.*_exps\.weight=CPU" --no-op-offload
-fa on -ctk q8_0 -ctv q8_0 -ub 64 -b 64 -t 8 -tb 16 -c 32768` (CUDA graphs on by default).

**What got the gains:** (1) not spilling (no op-offload) — the stock path silently oversubscribes VRAM; (2) loop-aware placement
(+28% for 5 looped layers; looped layer = 3.2× an outer one); (3) q8 KV + ub64 freeing ~1 GB for a 6th looped layer (+5%);
(4) `-tb 16` (+22% prefill). CUDA graphs (default) are worth +52% — never disable.

**Why it stops here (evidence):** remaining DDR traffic = 18 CPU-resident layers × passes × 36.7 MiB ≈ 1.1 GB/token at ~41–48 GB/s;
plus ~74 graph splits/token (CPU↔GPU hops on every CPU MoE layer). Every config lever since BEST2 landed in noise.
VRAM is full: a 7th looped layer needs ~590 MiB that the desktop (~1 GB) is using.

**Op-offload staging mechanism — measured (`probe_mem.ps1`, op-offload ON, ub64, q8 KV, np1):**
| experts on GPU | CPU-resident looped layers | CUDA0 compute buffer |
| :--- | ---: | ---: |
| none | 12 | 7478 MiB |
| 12 outer layers | 12 | 7467 MiB (unchanged) |
| BEST2 (6 looped) | 6 | 3954 MiB (= 7478 − 6 × 587.25, exact) |
| all 12 looped | 0 | 603 MiB |
→ The staging buffer keeps **one full 587 MiB expert copy alive per CPU-resident looped layer** (reused across the 3 passes);
outer layers share a slot. This is the concrete Phase 3a target.

**Remaining levers, ranked by payoff/effort** (payoffs are hypotheses until measured):
1. **Free desktop VRAM (user action, zero code):** close Chrome/Edge/OrcaSlicer/Antigravity or turn off their GPU acceleration while
   running → `-MaxVram` (ub128, +1%, +8% prefill) or loop7 (+~1 tok/s). Effort: trivial. Risk: spill if apps reopen.
2. **Phase 3 step 1 — toolchain + profile (recommended first ask):** install CUDA Toolkit 13.x + CMake + Ninja, build b12615 from
   source, profile decode with graphs off. Evidence so far points at thread-independent overhead (decode flat at -t 6/7/8, graphs
   worth +52%) — i.e. splits/syncs/launches — not DDR bandwidth or AVX2 kernels. Profile decides the kernel target.
3. **Phase 3a — op-offload staging fix:** stream CPU looped-layer experts through one reusable slot instead of keeping 6 copies.
   Hypothesis: with ~1 slot (~590 MiB) prefill could rise from ~90 toward PCIe-bound 200–450 tok/s (ub 256–512).
4. **Phase 3b — fuse Inkling's custom attention into FA:** compute ≈ 210 B × ub × ctx (dense f32 scores); fusing frees ~400–800 MiB
   (→ loop7, larger ub). Effort high.
5. **Phase 3c — CPU MoE path / split overhead:** only if the profile shows it.
6. **n-gram speculation** (`run-inkling.ps1 -Coding`): free; wins only on repeated/edited text.

**Not verified (say plainly):**
- Tool-call round-trip and reasoning-effort levels (none … max) gates — not run.
- Needle check matched codes anywhere in reasoning+content combined; long outputs were not saved. Tighten for the final gate run.
- RAM stability at DDR4-3600 (no TestMem5/MemTest86) and live DRAM voltage (WMI says 1.20 V; confirm 1.35 V in BIOS).
- Peak system RAM not recorded; tg128 / tg512 not measured separately (decode = 256-token median).
- Concurrency (cN) not measured (single-user target, `-np 1`).

### 2026-11-21 — Post-advisor fixes
- `run-inkling.ps1`: added `--cache-ram 0` (as benchmarked; default 8 GiB prompt cache was untested) and raised the VRAM warning
  threshold to footprint + 1 GB (6900 / 7300 MiB free). Re-tested: healthy, stops cleanly.
- `-tb 16` greedy check (L6-c vs TB16): **6/6 identical** → lossless. ✅

NEXT: user check-in — decide (a) desktop-VRAM habit / -MaxVram, (b) Phase 3 step 1 (toolchain install + source build + profile), (c) finish Phase 4 gates (tool call, effort levels, saved needle outputs) on the prebuilt.

---

## Phase 3 (user approved 2026-11-21: "Yes, start Phase 3")

### 2026-11-21 — Toolchain (no admin, no winget)
- CUDA 13.5 from NVIDIA redistributable zips (sha256-verified): nvcc 13.5.37, cudart, crt, nvvm, cccl, cublas 13.10, nvtx,
  profiler_api, cuobjdump, nvdisasm, cupti, VS integration, nsight_systems -> `toolchain\cuda-13.5\` (`scripts\get_cuda.py`).
- CMake 4.4.5 + Ninja 1.13.2 via pip in `.venv`. MSVC 14.44 (VS BuildTools 2022).
- Source: `llama.cpp\src` = tag b12615, commit a818a46f3 (== prebuilt), local branch `inkling-opt`. Private fork, not for upstream
  (AGENTS.md exempts private forks; still follow its style: ASCII-only, short comments).
- Official windows-cuda release builds ONLY ggml-cuda.dll with MSVC (`-DGGML_BACKEND_DL=ON -DGGML_CPU=OFF`); CPU variants come from a
  clang build. => `scripts\build.cmd <name>` builds ggml-base + ggml-cuda (sm_120a-real) and drops them into a copy of the prebuilt
  folder `llama.cpp\bin-<name>\`. CPU kernels stay byte-identical to the benchmarked prebuilt.

### 2026-11-21 — Code reading: op-offload staging (root cause)
- `ggml-backend.cpp` split pass: an input from another backend gets ONE copy per (src tensor, backend) via `tensor_id_copy(hash_id(src))`;
  later splits that read the same src reuse it (no re-upload). Loop passes 2-3 reuse the same physical weight tensors
  (`inkling2-loop.cpp: layers[il] = layers[layer_phys(il)]`), so the staging copy of each CPU looped layer stays live from pass 1
  to pass 3 -> allocator cannot reuse it -> +587 MiB per CPU looped layer. Matches the probe exactly.
- Fix idea (Phase 3a): for weight inputs with n_copies==1, make a fresh copy per split (re-upload per pass) and store per-split copy
  pointers (split->input_cpys) since copy_input / prefetch / graph_copy look copies up by hash.
- Op-offload uploads only the routed experts (`llama_context::sched_copy_experts` copy callback), not whole tensors.

### 2026-11-21 — DISCOVERY: built-in MoE cache (missed in Phase 2 flag scan)
- `--moe-cache-mib N` : VRAM cache (CRF/LRU) of experts of host-resident MoE layers. `--moe-cache-cpu-share F` (default 0):
  share of misses computed on CPU instead of uploaded. With share 1.0: hits computed on GPU concurrently with misses on CPU.
- `LLAMA_MOE_CACHE_TRACE` env dumps selected experts per lookup (for a cache simulator).
- This is "dynamic hot-expert pinning" — a config lever, test BEFORE writing sched patches.

NEXT: when vanilla build finishes: (1) A/B bin-vanilla vs prebuilt (speed + greedy); (2) MoE cache sweep on prebuilt:
loop5+587 MiB, loop4+1174, loop0+~3500, cpu_share 1.0 vs 0; (3) then decide on the sched patch.

### 2026-11-21 — Phase 3a patch written (branch inkling-opt, ggml/src/ggml-backend.cpp, +34/-7)
- Split pass: if a WEIGHT input (n_copies==1) already has a copy from an earlier split and is not yet an input of the current split,
  create a fresh copy for this split. `split->input_cpys[]` stores each split's own copy; graph-copy, copy_input and prefetch use it
  (`split_input_copy` macro; falls back to the old hash lookup when n_copies > 1, i.e. pipeline parallel untouched).
- Also fixes a likely CORRECTNESS issue with op-offload on looped models: upstream uploads the shared copy only in pass 1 with pass 1's
  routed experts (`sched_copy_experts` reads the split's first MUL_MAT_ID ids); passes 2-3 reuse it without upload -> experts routed
  only in pass 2/3 would read stale memory. Hidden when ubatch is large enough that all 128 experts are routed. Needs a test.
- Cost: looped CPU layers are uploaded 3x per ubatch instead of once (PCIe), in exchange for ~587 MiB less VRAM per CPU looped layer.

### 2026-11-21 — Vanilla source build OK
- After adding `cuda_nvml_dev` (nvml.h), `scripts\build.cmd vanilla` built ggml-base.dll (8:51, pre-patch) + ggml-cuda.dll (sm_120a, 151 objs)
  -> `llama.cpp\bin-vanilla\`. (Trailing "Access is denied" from the script is cosmetic; DLLs copied.)
- Harness: configs may set `"bin": "<folder under llama.cpp>"`; ledger records it.
- Running (one background job, serialized): P3-pre-a -> P3-vanilla -> P3-pre-b -> MC-L5-587 -> P3-pre-c -> MC-L4-1174 -> P3-pre-d ->
  MC-L0-3500 -> P3-pre-e, then `build.cmd patched` (ggml-base with the Phase 3a patch).
  MC-* = same VRAM budget as BEST2 but trading static looped layers for `--moe-cache-mib N --moe-cache-cpu-share 1.0`.

NEXT: read queue results (vanilla == prebuilt? MoE cache vs static placement), then test bin-patched (op-offload ON: staging size, correctness, prefill).

### 2026-11-21 — Results: vanilla build A/B + MoE cache sweep
| run | bin | c1 median | prefill 512/2048 | verdict |
| :--- | :--- | ---: | ---: | :--- |
| P3-pre-a | prebuilt | 20.86 (one 18.5 outlier) | 90.5 / 93.0 | control |
| P3-vanilla | bin-vanilla (MSVC+nvcc 13.5, sm_120a) | 21.06 | 90.1 / 93.2 | **EQUAL to prebuilt** -> toolchain valid |
| P3-pre-b | prebuilt | 21.04 | 89.8 / 93.4 | control |
| MC-L5-587 | loop5 + `--moe-cache-mib 587 --moe-cache-cpu-share 1.0` | 19.07 (−9.5%) | 76.6 / 85.1 | REJECT |
| P3-pre-c | | 21.11 | | control |
| MC-L4-1174 | loop4 + cache 1174 | 19.29 (−8.6%) | 78.2 / 80.4 | REJECT |
| P3-pre-d | | 21.13 | | control |
| MC-L0-3500 | no static + cache 3500 | 16.40 (−22%) | 59.8 / 52.0 | REJECT |
| P3-pre-e | | 21.10 | | control |
- Static loop-aware placement beats the dynamic MoE cache at equal VRAM (cache lookup splits + misses on CPU cost more than hits save).

### 2026-11-21 — Phase 3a patched build: staging buffer FIXED
- First patched build failed: my Python edit turned `\n` in a log string into a real newline (C2001). Fixed, rebuilt -> `bin-patched`.
- `scripts\server.ps1` honours `$env:LLAMA_BIN` (first sed attempt silently didn't apply -> first probe ran prebuilt; redone).
- CUDA0 compute buffer with op-offload ON (q8 KV, np1, 32K):
| placement / ub | vanilla | **patched** |
| :--- | ---: | ---: |
| BEST2 (6 looped GPU), ub64 | 3954 MiB | **603 MiB** |
| all experts CPU, ub64 | 7478 MiB | **603 MiB** |
| BEST2, ub512 | — | 3428 MiB (dense attention scores 210 B x ub x ctx dominate) |
- => per-split staging copies fit inside the existing compute peak. Op-offload no longer costs VRAM.
- Perplexity corpus: llama.cpp docs/*.md (611 KB) -> `results\ppl-corpus.txt`; `scripts\ppl.ps1` (c512, 24 chunks, BEST2 placement).
  ref (prebuilt, no op-offload, ub64): **PPL 4.5308 +/- 0.1346** (151 s).
- **Correctness (perplexity, c512 x 24 chunks, BEST2 placement, ub64):**
| run | PPL | s/pass | note |
| :--- | ---: | ---: | :--- |
| ref: prebuilt, --no-op-offload | 4.5308 +/- 0.1346 | 8.6 | reference |
| vanilla build, op-offload ON | **NaN (all 24 chunks)** | (470 s total) | **upstream bug reproduced**: passes 2-3 read un-uploaded expert copies |
| patched, op-offload ON | **4.5418 +/- 0.1350** | 12.6 | correct (+0.24%, GPU vs CPU numerics, << stderr) |
| patched, --no-op-offload | 4.5308 (identical to ref) | 7.0 | patch is a no-op on the default path |
- Prefill at ub64 with op-offload is slower than CPU (12.6 vs 7-8.6 s/pass): each ubatch streams ~17 GB of CPU experts over PCIe.
- Queued: ub512 vanilla-oo / patched-oo / ref (does vanilla become finite when every expert is routed in pass 1? prefill speed at ub512).
- **ub512 perplexity:** vanilla op-offload = **NaN again** (so not only "unrouted experts"; exact upstream mechanism unconfirmed,
  but the per-split copy is the only code difference and fixes it). patched-oo ub512 PPL 4.5515 (4.20 s/512-tok pass ~122 tok/s);
  ref no-offload ub512 PPL 4.5369 (5.08 s/pass ~101 tok/s). Op-offload ub512 = +21% prefill vs no-offload ub512, but ub512 needs a
  3.4 GB compute buffer (attention) -> incompatible with loop6 on 8 GB.

### 2026-11-21 — FINDING (corrected below): the relative-bias KV attention does not take the FA path
- `src/models/inkling2.cpp` passes a relative-position bias `kq_b` ([n_kv, n_tokens, n_head], gathered from a per-token table by
  `rel_pos`) to `build_attn`; `llama-graph.cpp:3202` `use_flash_attn = cparams.flash_attn && kq_b == nullptr` -> classic
  KQ + bias + softmax with fully materialized f32 scores in every layer. `-fa on` only changes cache layout, not the kernel.
- This is the 210 B x ub x ctx compute buffer. Options: (3b-i) query-chunked attention in the graph (peak = chunk x n_kv x heads,
  MoE still sees the full ubatch) — moderate, model-file change; (3b-ii) FA kernel with gathered bias — high effort.
- Decision per plan: profile decode first (primary metric) before choosing.
- CORRECTION: the decode profile shows `flash_attn_ext_vec` 56x/token, so FA *is* used by some attention in the graph (likely the
  attention-residual/bank mixing). The main KV attention with the relative bias `kq_b` is the non-FA path (llama-graph.cpp:3202).
  Exact attribution of the 56 FA calls not verified.

### 2026-11-21 — Phase 3 step 1: decode profile (nsys 2026.5.2, BEST2, prebuilt, llama-completion, 128 tok greedy)
- Files: `results\prof\decode-best2*.nsys-rep/.sqlite`, `scripts\prof_decode.py`. Graph-level trace (realistic):
| component | ms/token (profiled 51.8; unprofiled ~47) | share |
| :--- | ---: | ---: |
| GPU work (38.7 CUDA graph launches/token, p50 270 us) | 14.5 | 28% |
| GPU idle: CPU expert matmuls (18 CPU layer-passes x 2..3, ~1.1 GB/token) | ~31 | ~60% |
| GPU idle: ~141 CPU<->GPU hand-offs/token (p50 43 us) | ~6 | ~12% |
| memcpy (activations, 3.2 MB/token) | 0.35 | <1% |
- CPU experts run at ~37 GB/s effective = ~90% of measured DDR4 copy BW -> CPU kernel work (Phase 3c) has little headroom. Confirms
  the advisor's reading (decode flat across -t 6/7/8).
- GPU side: 14.5 ms vs ~6 ms bandwidth bound for its bytes. Node-trace kernel mix (inflated, proportions only): quantized matvec 48%
  (796/token, avg 11.9 us); **7,815 tiny kernels/token (avg ~1.2 us: bin_bcast, quantize_q8_1, rms_norm, concat, scale, unary, fwht,
  cpy, ssm_conv, ...) ~48%** -> the loop controllers / attention-residual / short-conv machinery is launch-granularity bound.
- Ceiling arithmetic for what's left: if GPU work -> ~7 ms (fusion) and hand-offs halved (~3 ms): 47 -> ~36 ms = ~27.5 tok/s (+30%).
  Each piece is substantial CUDA/graph engineering.

### 2026-11-21 — Phase 3a deliverable
- `patches\0001-sched-per-split-copies-for-shared-weights.patch` (git diff vs b12615; not committed). Built in `llama.cpp\bin-patched`.
- Effect: op-offload on looped models no longer reserves +587 MiB per CPU looped layer, and no longer produces NaN.
  BEST2 (no op-offload) is unaffected (bit-identical PPL), so `run-inkling.ps1` stays on the prebuilt.

NEXT: check in with user on Phase 3 direction (fusion vs hand-off latency vs query-chunked attention for prefill), see STATUS-P3.
- Greedy: P3-pre-b vs P3-vanilla **6/6 identical** -> source build is output-equivalent to the prebuilt.
- Gap buckets (measured, decode window): 31 long gaps/token (0.3-5 ms) = **32.4 ms/token** = one per CPU expert layer-pass (30)
  -> ~1.08 ms per 36.7 MiB pass (~35.6 GB/s, consistent with bandwidth-bound; decode flat across -t 6/7/8);
  101 short gaps/token (20-100 us) = **4.4 ms/token** hand-off/sync overhead.
- ggml-cuda SOFT_MAX fusion sites (3793, 4538-4599) are MoE top-k routing fusions, not attention. The 56 FA calls/token remain
  unattributed; whether the biased attention is fused at batch 1 is NOT verified.
- The node-trace "48% tiny kernels" is inflated by per-node tracing; share under graph replay unknown -> fusion payoff unquantified
  (upper-bound guess only).

### 2026-11-21 — NaN disambiguation (advisor confounds)
| run | binary | VRAM state | PPL |
| :--- | :--- | :--- | :--- |
| official-oo-ub64 | **official prebuilt b12615** | oversubscribed (staging 3954) | **NaN** -> not caused by my MSVC build |
| patched-oo-ub64-oversub11b | bin-patched | **deliberately oversubscribed**: model 7645 MiB, shared peak 1688 MiB (from 39) | **4.5418 finite** -> spill does not cause NaN |
- (First oversub attempt was invalid: ppl.ps1 prepended its own -ot and llama.cpp takes the first match. Fixed: caller -ot replaces default.)
- 4.5418 identical across placements is expected: at ub64 all expert matmuls run on GPU in both (op-offload), same kernels.
- Conclusion: **stock op-offload produces NaN for Inkling-2-loop on b12615 (official binary); the per-split-copy patch fixes it.**
  The exact upstream mechanism (why shared copies of looped weights end up invalid) is not confirmed.
- Upstream: AGENTS.md forbids AI-written issues/PRs; repro facts are here for the user to report personally if they wish.

---

## STATUS — Phase 3 step 1 complete (toolchain, profile, op-offload fix). 2026-11-21. Awaiting user direction.

**Delivered**
- No-admin CUDA 13.5 toolchain + MSVC/nvcc build of ggml-base/ggml-cuda (sm_120a) proven equivalent to the official binary
  (speed equal, 6/6 greedy identical).
- `patches/0001-sched-per-split-copies-for-shared-weights.patch` (+33/-7, ggml-backend.cpp): fixes op-offload on looped models:
  staging 3954/7478 MiB -> 603 MiB, PPL NaN -> 4.54 (ref 4.53). Bit-identical when op-offload is off.
- **BEST2 speed unchanged** (it runs with --no-op-offload); `run-inkling.ps1` stays on the official prebuilt.

**Measured decode anatomy (BEST2, ~47 ms/token)**: CPU expert passes 32.4 ms (30 x ~1.08 ms, ~36 GB/s, consistent with
DDR-bandwidth-bound) | GPU work 14.5 ms | CPU<->GPU hand-offs 4.4 ms (101 short gaps/token).

**Rejected in Phase 3**: MoE cache (`--moe-cache-mib`, cpu-share 1.0 only; hit rate not inspected): -9.5% / -8.6% / -22%.

**Prefill facts**: in perplexity runs, ub512 (no offload) ~101 tok/s vs ~60-73 at ub64; patched op-offload ub512 ~122 tok/s.
Batch size is the bigger prefill lever; the dense biased-attention buffer (210 B x ub x ctx) is what caps ub at 32K.

**Remaining options (payoffs are hypotheses)**
1. Finish Phase 4 gates on BEST2 (tool calls, reasoning-effort levels, tightened needle check with saved outputs, RAM stability). Low effort.
2. Query-chunked attention in inkling2.cpp so ub can be 512+ at 32K without the 3.4 GB buffer -> prefill maybe ~92 -> 120-150 tok/s
   while keeping loop6. Moderate effort (model graph change), needs PPL + needle gates.
3. GPU elementwise fusion for the loop-controller / attention-residual / conv chains (~8.8k kernels/token). Payoff unquantified
   under graphs (node trace inflates small kernels); upper bound maybe a few ms/token. High effort, open-ended.
4. Hand-off latency (4.4 ms/token): little config headroom (--poll tested); possible Windows HAGS on/off A/B (user's system setting).

**State left**: `llama.cpp\src` branch `inkling-opt` with uncommitted patch (also saved in patches\); build dirs `llama.cpp\build-*`,
`bin-vanilla`, `bin-patched`; `toolchain\cuda-13.5` (~2 GB) + `toolchain\dl` zips; nsys reports in results\prof. No servers running.

NEXT: user picks among options 1-4.

---

## Autonomous continuation (user, 2026-11-21: "go end to end - i wont be here to pick whats next")
Order: Phase 4 gates -> strict needles -> RAM smoke test -> chunked attention (implement, PPL/needle/speed A/B) -> final report.
Not doing: Windows system settings (HAGS etc.; user-only), upstream issues/PRs (AGENTS.md), commits (not requested).

### 2026-11-21 — Phase 4 gates on the shipped launcher (`run-inkling.ps1`, BEST2, prebuilt) — `scripts\gates.py`
- chat: PASS ("Paris"). reasoning_effort none/minimal/low/medium/high: all accepted, finish=stop, correct answer (24); completion
  tokens 101/131/131/95/101 (easy question; effort did not change length much — info only).
- tool call round trip: PASS — `get_weather {"city":"Oslo"}` (finish=tool_calls), then final answer uses the tool result
  ("Oslo is currently -3C with light snow."). Outputs in `results\gates\*.json`.
- Strict needles (codes must be in final content; outputs saved): running.

### 2026-11-21 — Phase 3b: query-chunked attention (implemented, not yet built/tested)
- Problem: Inkling materializes kq_b [n_kv, n_tok, 24] (96 B x ub x ctx) and then kq/softmax of the same size (non-FA path) ->
  210 B x ub x ctx compute buffer (3.4 GB at ub512/32K) which blocks large ubatches.
- Change (env-gated `LLAMA_ATTN_Q_CHUNK=N`, default off = original graph):
  - `llama-graph.h`: `kq_b_lazy_t {table, idx, n_chunk}` + `mutable const kq_b_lazy_t * kq_b_lazy` in llm_graph_context.
  - `llama-graph.cpp build_attn_mha`: new non-FA branch when kq_b_lazy set: for each chunk of N query tokens: KQ matmul on a q view,
    gather bias rows for the chunk (get_rows on table/idx views), pad, add, softmax with the chunk's mask rows, x V, concat on tokens.
  - `models/inkling2.cpp`: when enabled, pass the table + rel_pos lazily instead of building kq_b.
- Build now also swaps llama.dll (graph code lives there).
- Strict needles PASS: **3/3 in final content** at 16,394 tok (prefill 89.3, decode 20.19) and 30,734 tok (prefill 89.2, decode 19.69),
  finish=stop. Full outputs in `results\gates\needle_*.json`. **All Phase 4 gates pass on the shipped launcher.**

### 2026-11-21 — Chunked attention: debugging + redesign
- First build hung at `resolve_fused_ops` during graph reserve. Added a node-budget term in `llama_context::graph_max_nodes`
  (16 x n_layer_all x ceil(n_ubatch/chunk)) — still hung.
- Foreground run (bash, timeout) gave **rc=139 (segfault)**; the hidden-window runs had been stuck on Windows error reporting.
  Those crashed processes cannot be killed (Access denied) and keep DLLs locked in `bin-patched`, `bin-opt`, `bin-opt-vbase`
  (cleared by reboot). New builds go to fresh folders (`bin-opt2`).
- ROOT CAUSE (and a correction): Inkling's `build_attn(llm_graph_input_mem_hybrid_iswa_rel*)` **folds kq_b into a dense per-head
  mask** (`kq_mask = kq_b + mask`, cast to F16) and calls attention with kq_b = nullptr -> **Flash Attention IS used** (the 56 FA
  calls/token). The 210 B x ub x ctx buffer is that per-head mask (f32 bias + f32 sum + f16 copy), not attention scores.
  My first version passed kq_b = null into that add -> segfault.
- Redesign: hybrid build_attn skips the fold when `kq_b_lazy` is set; build_attn_mha FA branch loops over query chunks: gather bias rows
  for the chunk, add the chunk's mask rows, cast F16, `ggml_flash_attn_ext(q_chunk, k, v, mask_chunk)`, concat on tokens (dim 2).
  (The non-FA lazy branch is kept for -fa off.)
- Compute buffer @32K, BEST2 placement, --no-op-offload (bin-opt2):
| LLAMA_ATTN_Q_CHUNK | ub | CUDA0 compute |
| :--- | ---: | ---: |
| off | 512 | 3359 MiB |
| 64 | 512 | **905 MiB** |
| 128 | 512 | 1385 MiB |
| 64 | 1024 | 1122 MiB |
  (BEST2 at ub64 = 420 MiB.) Correctness PPL runs in progress.
- **Correctness (PPL, c512 x 24, BEST2 placement, ub512, bin-opt2):** chunk off 4.5369 | chunk64 **4.5333** (-0.08%, noise) |
  chunk64 + patched op-offload **4.5363**. Prompt speed in these runs: 8.66 / 8.69 / **4.28 ms/token** (115 / 115 / **234 tok/s**).
  -> chunking is correct; chunking + fixed op-offload doubles prefill at ub512.
- Compute buffer @32K with op-offload ON: chunk64/ub512 996, chunk32/ub512 **707**, chunk64/ub256 816, chunk32/ub256 **559** MiB
  (BEST2 ub64 no-offload: 420). VRAM rule (>=1 GB headroom over BEST2's ~6.9-7.0 GB peak): ub256/c32 fits; ub512/c32 ~0.88 GB.
- A/B queued: B2-f -> OPT-u256c32 -> B2-g -> OPT-u512c32 -> B2-h (OPT = bin-opt2, op-offload ON, LLAMA_ATTN_Q_CHUNK=32).

### 2026-11-22 — A/B: chunked attention + fixed op-offload vs BEST2
| run | config | c1 decode | prefill 512/2048 cold | TTFT short | compute | VRAM peak | verdict |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| B2-f | BEST2 | 21.11 | 90.3 / 93.5 | 273 | 420 | 6689 | control |
| OPT-u256c32 | bin-opt2, op-offload ON, ub256, chunk32 | **19.58 (-7.2%)** | **174.0 / 208.7 (x2.2)** | 288 | 559 | 6823 | TRADE (decode regresses) |
| B2-g | BEST2 | 21.10 | 90.6 / 89.7 | 271 | | 6689 | control |
| OPT-u512c32 | ub512, chunk32 | **19.46 (-7.8%)** | **252.3 / 308.9 (x3.3)** | 320 | 707 | 6980 | TRADE |
| B2-h | BEST2 | 21.09 | 90.4 / 93.2 | 275 | | 6690 | control |
- Greedy vs B2-f: 5/6 identical for both (prompt 5 diverges after 34 words; kernel-numerics class).
- Decode regression is unexpected (decode is batch 1: no op-offload, single chunk). Isolation queue: X1 = BEST2 on bin-opt2 (MSVC
  llama.dll), X2 = X1 + op-offload ON at ub64, X3 = prebuilt BEST2 with ub256 — each sandwiched by B2 controls.

### 2026-11-22 — Decode regression isolated and fixed
- Isolation (each sandwiched by BEST2 controls 21.09-21.12): X1 BEST2 on bin-opt2 (MSVC llama.dll) **21.09**; X2 + op-offload ON (ub64)
  **21.13**; X3 prebuilt ub256 no-offload **21.13** (prefill 104.5, but compute 1679 MiB, VRAM 7643 - not viable). => none regress;
  the chunked path itself caused it.
- nsys (graph trace, `scripts\prof_graphs.py`): BEST2 51.8 ms/tok, 102 memcpy/tok, 109 short gaps (4.45 ms) vs chunked 55.5 ms/tok,
  **193 memcpy/tok**, 203 short gaps (8.08 ms). Memcpy histogram: +47.6/tok H2D of 1024 B and 512 B = per-layer upload of
  row views of the host inputs (rel_pos I32 and the F16 mask): each layer made its own view tensor -> the scheduler uploads each view.
- Pinning mask nodes to the KV backend (bin-opt3) did not help (still 193). Fix (bin-opt4): `llm_graph_context::input_rows_view`
  caches row views of input tensors per graph (whole tensor when the chunk covers all rows), shared by all layers.
  -> **51.5 ms/tok, 102 memcpy/tok, 112 short gaps** = parity with BEST2.
- Running: PPL (chunk32, op-offload, ub512 + ub256) and A/B B2-m -> OPT4-u256c32 -> B2-n -> OPT4-u512c32 -> B2-o.
- Note: crashed processes keep bin-patched/bin-opt/bin-opt-vbase locked until reboot; bin-opt2/3 superseded by bin-opt4.

### 2026-11-22 — bin-opt4 results: decode parity + 2-3.4x prefill -> BEST3
| run | config | c1 decode | prefill 512/2048 cold | PPL (c512, ub=cfg) | compute | VRAM peak |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| B2-m | BEST2 | 21.12 | 90.0 / 93.4 | 4.5369* | 420 | 6465 |
| OPT4-u256c32 | op-offload, ub256, chunk32 | **21.14** | **178.7 / 215.8** | 4.5401 | 651 | 6677 |
| B2-n | BEST2 | 21.09 | 89.8 / 93.0 | | 420 | 6467 |
| OPT4-u512c32 | op-offload, ub512, chunk32 | **21.11** | **257.8 / 318.9** | **4.5351** | 716 | 6767 |
| B2-o | BEST2 | 21.09 | 89.7 / 93.0 | | 420 | 6467 |
(*ref PPL at ub512 no-offload.) Greedy vs B2-m: 5/6 identical for both (prompt 5 kernel-numerics divergence as before).
- **ACCEPT OPT4-u512c32 as BEST3**: decode within noise of all controls, prefill 2.9-3.4x, PPL within noise, +~300 MiB VRAM vs BEST2.
- Packaged: `llama.cpp\bin-inkling\` = official b12615 + patched ggml-base.dll (8315EBEF...), ggml-cuda.dll (C013DDC6..., unmodified
  source, MSVC/nvcc 13.5 sm_120a), llama.dll (91F68B1F...). Full source diff: `patches\0002-inkling-opt-full.patch`
  (ggml-backend.cpp, llama-context.cpp, llama-graph.cpp/.h, models/inkling2.cpp; +201/-28). `best-config.json` = BEST3.
- `run-inkling.ps1` rewritten: default = BEST3 (bin-inkling, op-offload on, ub512, LLAMA_ATTN_Q_CHUNK=32, warn if < 7200 MiB free);
  `-Safe` = BEST2 on the official binary; `-Coding` = + ngram-mod. (-MaxVram removed.)
- Running: full strict gate suite (`gates.py --long`) against the launcher's BEST3 server.
- Housekeeping slip: I deleted `results\gates\` (BEST2 gate outputs) before this run; BEST2 gate results are preserved only in
  the PLAN.md entries above.

### 2026-11-22 — BEST3 gates (launcher default) — ALL PASS
- chat PASS; reasoning_effort none/minimal/low/medium/high PASS (identical token counts to BEST2); tool call round trip PASS;
  needles: 16,394 tok 3/3 in content (prefill **309.6**, decode 20.33), 30,734 tok 3/3 in content (prefill **279.2**, decode 19.71).
  -> 30.7K-token prompt: ~110 s vs ~345 s on BEST2. Outputs in `results\gates\`.

### 2026-11-22 — RAM smoke test (`scripts\ramtest.py`, servers stopped)
- 20 GiB over 8 processes, 10.2 min, 328 write/verify rounds (incl. inverted patterns), 1640 GiB verified, **0 mismatches**. PASS.
  Smoke test only, not MemTest86.

---

## FINAL STATUS (2026-11-22) — see REPORT.md
- Default: **BEST3** via `run-inkling.ps1` (bin-inkling, patched): decode ~21.1 tok/s, prefill ~258-319 tok/s (279 at 30.7K),
  all gates pass. Fallback `-Safe` = BEST2 on the official binary.
- Phases: 0-2 done; 3 done (toolchain, profile, op-offload NaN/VRAM fix, chunked biased attention); 4 done (gates, launcher, report).
- Open (not pursued, payoff unquantified): GPU tiny-kernel fusion; hand-off latency; concurrency; >32K context.
NEXT: none required. Optional: reboot to clear stuck processes, MemTest86 overnight, confirm DRAM 1.35 V in BIOS.

### 2026-11-22 — Post-advisor checks on BEST3
- Patch file verified: `git apply -R --check patches/0002-inkling-opt-full.patch` clean.
- Short prompts (`scripts\shortprefill.py`, TTFT ms median of 3, unique prompts): BEST2 1459/1239/1886/3273/5706 for 45/82/146/266/507
  tokens; BEST3 (op-offload min batch 32) 1268/1267/1329/1547/1864 (82-tok case +2% vs BEST2); BEST3 + `GGML_OP_OFFLOAD_MIN_BATCH=128`
  **817/1037/1337/1541/1818** (never slower than BEST2); =256: 810/1084/1537/1550/1832. -> **BEST3 now uses min batch 128**
  (env set by launcher; batches < 128 take BEST2's validated CPU path). `results\shortprefill.jsonl`.
- Launcher modes (`scripts\test_launcher.ps1`): default healthy, sane code, 19.58 tok/s (single cold sample); -Safe healthy, 20.70;
  -Coding healthy, run1 20.66, repeat run2 20.60 with 324 drafted / 127 accepted (39%) -> no speedup, no slowdown.
  VRAM warning fired for BEST3 (7182 / 7126 MiB free < 7200): BEST3 keeps >= 1 GB headroom only while desktop VRAM < ~0.95 GB.
- No servers left running; VRAM idle 736 MiB. REPORT.md / THEORY.md updated with these caveats.

### 2026-11-22 — Cleanup (user request)
- Deleted (~8.6 GB): `toolchain\` (CUDA 13.5 redistributables, 3.87 GB), `llama.cpp\build-vanilla`, `build-patched`,
  `bin-vanilla`, `bin-patched`, `bin-opt`, `bin-opt-vbase`, `bin-opt2`, `bin-opt3`, `bin-opt4`. The stuck processes had exited, so no locks.
- Kept: `llama.cpp\b12615` (official, -Safe), `llama.cpp\bin-inkling` (BEST3, default), `llama.cpp\src` (branch inkling-opt,
  uncommitted, = patches\0002), `llama.cpp\dl` (release zips), `.venv` (incl. cmake/ninja), `models\`, `results\`, `logs\`, `patches\`.
- Launcher re-tested after cleanup: default BEST3 healthy 20.50 tok/s; -Safe healthy 20.49 tok/s; identical code output; no servers left.
- To rebuild later: `.venv\Scripts\python -I scripts\get_cuda.py` (re-downloads CUDA 13.5 components, ~1.9 GB of zips) then
  `scripts\build.cmd <name>` and copy ggml-base/ggml-cuda/llama DLLs into a fresh folder.

---


---
*(The notebook continues in the original project with harness/agent work, published separately as pi-learning.)*
