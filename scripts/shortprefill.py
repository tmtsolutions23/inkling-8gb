"""Short-prompt prefill / TTFT sweep (chat follow-ups, tool results): launches a config, sends unique cold prompts of ~N new tokens.
Usage: .venv\\Scripts\\python -I scripts/shortprefill.py configs/X.json [label]  -> appends to results/shortprefill.jsonl"""
import json, os, random, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_inkling as B  # noqa: E402

SIZES = (40, 80, 150, 300, 600)
cfg = json.load(open(sys.argv[1]))
label = sys.argv[2] if len(sys.argv) > 2 else cfg["name"]
srv = B.Server("sp-" + label, cfg["args"], cfg.get("env"), cfg.get("bin", "b12615"))
srv.start()
out = {"label": label, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": cfg.get("env", {}), "bin": cfg.get("bin", "b12615")}
try:
    B.chat("Say hello.", 8)
    for n in SIZES:
        pps, ms = [], []
        for rep in range(3):
            rng = random.Random(n * 100 + rep)
            words = " ".join(rng.choice(B.WORDS) for _ in range(int(n / 1.5)))
            prompt = f"[{time.time_ns()}] {words}\nCount the words above."
            r = B.chat(prompt, 1)
            t = r["timings"]
            pps.append(t["prompt_per_second"]); ms.append(t["prompt_ms"])
        out[str(n)] = {"prompt_n": t["prompt_n"], "pp_tok_s": round(statistics.median(pps), 1), "prompt_ms": round(statistics.median(ms), 1)}
        print(label, n, out[str(n)], flush=True)
finally:
    srv.stop()
with open(os.path.join(B.ROOT, "results", "shortprefill.jsonl"), "a") as f:
    f.write(json.dumps(out) + "\n")
