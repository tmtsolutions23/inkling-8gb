"""Inkling hill-climb harness for llama-server on Windows (adapted from skill/scripts/bench.py).

  .venv\\Scripts\\python -I scripts/bench_inkling.py run configs/NAME.json [--reps 6] [--long] [--no-prefill]
  .venv\\Scripts\\python -I scripts/bench_inkling.py show

Config JSON: {"name": "...", "args": ["-ngl","99", ...], "note": "..."}
Fixed per run: -np 1, --cache-ram 0 (no cross-request prompt cache), --metrics, temp 0, seed 1, reasoning_effort "low".
Metrics come from the server's own `timings`/`usage` (never streamed chunks):
  c1_decode (median of reps, 256 tok each, warm-up discarded), prefill_cold_<L> (unique prefix, sized with /tokenize),
  prefill_warm_<L> (same doc again: cache hit), ttft_short_ms, startup_s, vram_peak_mib, shared_delta_mib,
  long: needle retrieval at ~16K and ~31K tokens (with decode tok/s at that depth).
Flags: SPILL (WDDM shared GPU memory grew > 300 MiB -> run void), LOOP, TRUNC (completion < max_tokens), REQ_FAIL,
       PLACEMENT (CUDA0 model buffer differs from config's expect_gpu_model_mib).
Outputs of the temp-0 panel are saved to results/outputs/<name>.json for greedy-equivalence checks.
"""
import json
import os
import random
import re
import statistics
import string
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "llama.cpp", "b12615", "llama-server.exe")
MODEL = os.path.join(ROOT, "models", "Inkling-2-Mini-AD-Q4_K.gguf")
BASE = "http://127.0.0.1:8080"
LEDGER = os.path.join(ROOT, "ledger.jsonl")
EFFORT = "low"
PROMPTS = [  # fixed panel (coding / reasoning / chat). Never randomise.
    "Write a Python implementation of an LRU cache with O(1) get and put, with docstrings and tests.",
    "Explain, step by step, how TCP congestion control works, including slow start and fast recovery.",
    "Write a short story (about 600 words) about a lighthouse keeper who finds a message in a bottle.",
    "Compare B-trees and LSM-trees for a write-heavy database workload. Use a table and give a recommendation.",
    "Write a bash script that backs up a directory to a timestamped tar.gz and keeps only the last 7 backups.",
    "A train leaves at 9:40 and travels 210 km at 84 km/h, then waits 12 minutes, then 96 km at 64 km/h. When does it arrive? Show your work.",
]
WORDS = ("river stone paper cloud engine garden silver market winter orange signal harbor violet "
         "candle forest mirror thunder copper meadow lantern").split()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def http(path, body=None, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()


def shared_mib():
    v = ps("((Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples | Measure-Object CookedValue -Sum).Sum")
    try:
        return float(v) / 2**20
    except ValueError:
        return float("nan")


def vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop, self.vram, self.shared = threading.Event(), 0, 0.0

    def run(self):
        while not self.stop.wait(4):
            try:
                self.vram = max(self.vram, vram_mib())
                self.shared = max(self.shared, shared_mib())
            except Exception:
                pass


class Server:
    def __init__(self, name, args, env=None, bindir="b12615"):
        self.name, self.args, self.proc, self.env = name, args, None, env or {}
        self.exe = os.path.join(ROOT, "llama.cpp", bindir, "llama-server.exe")
        self.log = os.path.join(ROOT, "logs", f"{name}.log")

    def start(self):
        os.makedirs(os.path.dirname(self.log), exist_ok=True)
        cmd = [self.exe, "-m", MODEL, "--host", "127.0.0.1", "--port", "8080", "--no-webui", "--metrics", "--slots",
               "-np", "1", "--cache-ram", "0", "-lv", "4"] + self.args
        t0 = time.monotonic()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(self.log, "w"),
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                     env={**os.environ, **self.env})
        while time.monotonic() - t0 < 600:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode}; see {self.log}")
            try:
                if http("/health", timeout=2).get("status") == "ok":
                    return round(time.monotonic() - t0, 1)
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError("server not healthy in 600 s")

    def stop(self):
        if not self.proc or self.proc.poll() is not None:
            return
        t = time.monotonic()
        while time.monotonic() - t < 120:  # drain
            try:
                if not any(s.get("is_processing") for s in http("/slots", timeout=2)):
                    break
            except Exception:
                break
            time.sleep(1)
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def placement(self):
        txt = open(self.log, encoding="utf-8", errors="replace").read()
        gpu = [float(x) for x in re.findall(r"CUDA0 model buffer size =\s+([\d.]+) MiB", txt)]
        comp = [float(x) for x in re.findall(r"CUDA0 compute buffer size =\s+([\d.]+) MiB", txt)]
        kv = [float(x) for x in re.findall(r"CUDA0 KV buffer size =\s+([\d.]+) MiB", txt)]
        cpu_cpu = [float(x) for x in re.findall(r"CPU(?:_Mapped|_REPACK)? model buffer size =\s+([\d.]+) MiB", txt)]
        return {"gpu_model_mib": round(sum(gpu), 1), "gpu_compute_mib": round(sum(comp), 1),
                "gpu_kv_mib": round(sum(kv), 1), "cpu_model_mib": round(sum(cpu_cpu), 1)}


def chat(prompt, max_tokens, seed=1):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "seed": seed, "stream": False, "reasoning_effort": EFFORT}
    t0 = time.monotonic()
    r = http("/v1/chat/completions", body)
    m = r["choices"][0]["message"]
    return {"wall": time.monotonic() - t0, "timings": r.get("timings", {}), "usage": r.get("usage", {}),
            "finish": r["choices"][0].get("finish_reason"),
            "text": (m.get("reasoning_content") or "") + "\n<<content>>\n" + (m.get("content") or "")}


def ntok(text):
    return len(http("/tokenize", {"content": text})["tokens"])


def looping(text, n=8, thresh=0.5):
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(max(0, len(w) - n))]
    return len(grams) > 50 and 1 - len(set(grams)) / len(grams) > thresh


def document(target_tokens, seed, needles=0):
    """Unique filler doc sized with the real tokenizer to ~target_tokens (incl. question)."""
    rng = random.Random(seed)
    head = f"[doc {seed}-{time.time_ns()}-{rng.random()}] "  # defeats prefix reuse
    codes = ["".join(rng.choices(string.ascii_uppercase + string.digits, k=8)) for _ in range(needles)]
    q = ("\n\nList every secret code number and its value that appears in the document above. Answer only with the codes."
         if needles else "\n\nSummarise the document above in one sentence.")
    per_word = None
    n_words = int(target_tokens / 1.5)
    for _ in range(4):
        words = [rng.choice(WORDS) for _ in range(n_words)]
        for i, c in enumerate(codes):
            words.insert(int(len(words) * (i + 1) / (needles + 1)), f". The secret code number {i + 1} is {c}. ")
        text = head + " ".join(words) + q
        n = ntok(text)
        if abs(n - target_tokens) <= max(16, target_tokens * 0.01):
            break
        per_word = n / n_words
        n_words = int(n_words + (target_tokens - n) / per_word)
    return text, codes, n


def run(cfg_path, reps=6, long_ctx=False, prefill=True):
    cfg = json.load(open(cfg_path))
    name = cfg["name"]
    res = {"name": name, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": cfg["args"], "note": cfg.get("note", ""),
           "effort": EFFORT}
    flags = []
    sh0 = shared_mib()
    srv = Server(name, cfg["args"], cfg.get("env"), cfg.get("bin", "b12615"))
    res["env"] = cfg.get("env", {})
    res["bin"] = cfg.get("bin", "b12615")
    samp = Sampler()
    try:
        res["startup_s"] = srv.start()
        res.update(srv.placement())
        exp = cfg.get("expect_gpu_model_mib")
        if exp and abs(res["gpu_model_mib"] - exp) > 5:
            flags.append("PLACEMENT")
        samp.start()
        log(name, "healthy", res["startup_s"], "s", res)

        chat("Say hello.", 16)  # warm-up / page-in
        chat(PROMPTS[0], 64)
        r = chat("Reply with the single word: ready", 8)
        res["ttft_short_ms"] = round(r["timings"].get("prompt_ms", 0), 1)

        outs, dec = [], []
        for i in range(reps):
            r = chat(PROMPTS[i % len(PROMPTS)], 256)
            t = r["timings"]
            if r["usage"].get("completion_tokens", 0) < 256:
                flags.append("TRUNC")
            if looping(r["text"]):
                flags.append("LOOP")
            dec.append(round(t["predicted_per_second"], 2))
            outs.append({"prompt": i, "text": r["text"], "n": t.get("predicted_n")})
        res["c1_decode"] = round(statistics.median(dec), 2)
        res["c1_all"] = dec
        log("c1 decode median", res["c1_decode"], dec)
        os.makedirs(os.path.join(ROOT, "results", "outputs"), exist_ok=True)
        json.dump(outs, open(os.path.join(ROOT, "results", "outputs", f"{name}.json"), "w"), indent=1)

        if prefill:
            for L in (512, 2048):
                doc, _, n = document(L, seed=L)
                r = chat(doc, 1)
                t = r["timings"]
                res[f"prefill_cold_{L}"] = round(t["prompt_per_second"], 1)
                res[f"prefill_cold_{L}_n"] = t["prompt_n"]
                r2 = chat(doc, 1)  # same doc: slot KV reuse -> warm
                res[f"prefill_warm_{L}_cached"] = r2["timings"].get("cache_n")
                res[f"prefill_warm_{L}_ms"] = round(r2["timings"]["prompt_ms"], 1)
                log(f"prefill ~{L}: cold {res[f'prefill_cold_{L}']} tok/s over {t['prompt_n']} tok; "
                    f"warm {res[f'prefill_warm_{L}_ms']} ms (cached {res[f'prefill_warm_{L}_cached']})")

        if long_ctx:
            res["long"] = {}
            for L in (16384, 30720):
                doc, codes, n = document(L, seed=7 + L, needles=3)
                r = chat(doc, 1536)
                t = r["timings"]
                found = sum(c in r["text"] for c in codes)
                res["long"][str(L)] = {"prompt_tokens": t.get("prompt_n"), "prefill": round(t["prompt_per_second"], 1),
                                       "decode": round(t["predicted_per_second"], 2), "retrieval": f"{found}/{len(codes)}",
                                       "finish": r["finish"]}
                log("long", L, res["long"][str(L)])
        res["ok"] = True
    except Exception as e:
        res.update(ok=False, error=repr(e))
        flags.append("REQ_FAIL")
    finally:
        samp.stop.set()
        srv.stop()
    res["vram_peak_mib"] = samp.vram
    res["shared_delta_mib"] = round(samp.shared - sh0, 0)
    if samp.shared - sh0 > 300:
        flags.append("SPILL")
        res["ok"] = False
    res["flags"] = sorted(set(flags))
    with open(LEDGER, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("c1_all",)}, indent=1))
    return res


def show():
    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    keys = ["c1_decode", "prefill_cold_512", "prefill_cold_2048", "ttft_short_ms", "gpu_model_mib", "gpu_compute_mib",
            "vram_peak_mib", "shared_delta_mib"]
    print("name".ljust(30) + "".join(k[:14].rjust(15) for k in keys) + "  flags")
    for d in rows:
        print(d["name"][:30].ljust(30) + "".join(str(d.get(k, "-"))[:14].rjust(15) for k in keys) + "  " +
              ",".join(d.get("flags", [])))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "show"):
        sys.exit(__doc__)
    if a[0] == "show":
        show()
    else:
        reps = int(a[a.index("--reps") + 1]) if "--reps" in a else 6
        run(a[1], reps=reps, long_ctx="--long" in a, prefill="--no-prefill" not in a)
