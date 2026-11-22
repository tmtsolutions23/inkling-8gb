"""Phase 4 acceptance gates against a running llama-server (start it with run-inkling.ps1 or bench config).
  .venv\\Scripts\\python -I scripts/gates.py [--long]
Gates: chat sanity, reasoning_effort levels, tool-call round trip, (--long) strict needles at ~16K and ~30.7K.
Outputs saved to results/gates/<gate>.json. Exit code 1 if any gate fails."""
import json, os, random, string, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "http://127.0.0.1:8080"
OUT = os.path.join(ROOT, "results", sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "gates")
os.makedirs(OUT, exist_ok=True)
WORDS = ("river stone paper cloud engine garden silver market winter orange signal harbor violet "
         "candle forest mirror thunder copper meadow lantern").split()


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(messages, **kw):
    body = {"messages": messages, "temperature": 0, "seed": 1, "stream": False}
    body.update(kw)
    return post("/v1/chat/completions", body)


def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, name + ".json"), "w"), indent=1)


results = {}


def gate(name, ok, detail):
    results[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


# 1. chat sanity
r = chat([{"role": "user", "content": "What is the capital of France? Answer in one word."}], max_tokens=512, reasoning_effort="low")
c = r["choices"][0]["message"].get("content") or ""
save("chat", r)
gate("chat", "paris" in c.lower() and r["choices"][0]["finish_reason"] == "stop", repr(c.strip()[:80]))

# 2. reasoning effort levels: every level must be accepted, finish, and answer correctly
q = "A shop sells pens at 3 for $4. How many dollars do 18 pens cost? Give the final number on the last line."
effort = {}
for lvl in ("none", "minimal", "low", "medium", "high"):
    t0 = time.time()
    r = chat([{"role": "user", "content": q}], max_tokens=4096, reasoning_effort=lvl)
    m = r["choices"][0]["message"]
    effort[lvl] = {"finish": r["choices"][0]["finish_reason"], "reasoning_tokens_approx": len((m.get("reasoning_content") or "").split()),
                   "content": (m.get("content") or "")[-200:], "completion_tokens": r["usage"]["completion_tokens"],
                   "seconds": round(time.time() - t0, 1)}
save("effort", effort)
ok = all(v["finish"] == "stop" and "24" in v["content"] for v in effort.values())
gate("reasoning_effort", ok, {k: (v["completion_tokens"], v["finish"], "24" in v["content"]) for k, v in effort.items()})
toks = [effort[k]["completion_tokens"] for k in ("none", "low", "high")]
gate("effort_monotone_hint", toks[0] <= toks[2], f"completion tokens none/low/high = {toks} (info: higher effort should not think less)")

# 3. tool call round trip
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Oslo right now? Use the tool."}]
r1 = chat(msgs, tools=tools, max_tokens=2048, reasoning_effort="low")
m1 = r1["choices"][0]["message"]
calls = m1.get("tool_calls") or []
ok1 = bool(calls) and calls[0]["function"]["name"] == "get_weather" and "oslo" in calls[0]["function"]["arguments"].lower()
detail = {"finish": r1["choices"][0]["finish_reason"], "calls": [c["function"] for c in calls]}
ok2 = False
if ok1:
    msgs += [{"role": "assistant", "content": m1.get("content") or "", "tool_calls": calls},
             {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "name": "get_weather",
              "content": json.dumps({"city": "Oslo", "temp_c": -3, "conditions": "light snow"})}]
    r2 = chat(msgs, tools=tools, max_tokens=2048, reasoning_effort="low")
    c2 = r2["choices"][0]["message"].get("content") or ""
    ok2 = ("-3" in c2 or "3" in c2) and "snow" in c2.lower()
    detail["final"] = c2[:200]
    save("tool_round2", r2)
save("tool_round1", r1)
gate("tool_call", ok1 and ok2, detail)

# 4. strict needles: codes must appear in the final CONTENT (not just reasoning); outputs saved
if "--long" in sys.argv:
    sizes = [int(x) for x in sys.argv[sys.argv.index("--needles") + 1].split(",")] if "--needles" in sys.argv else [16384, 30720]
    for L in sizes:
        rng = random.Random(1000 + L)
        codes = ["".join(rng.choices(string.ascii_uppercase + string.digits, k=8)) for _ in range(3)]
        n_words = int(L / 1.5)
        for _ in range(4):
            words = [rng.choice(WORDS) for _ in range(n_words)]
            for i, cd in enumerate(codes):
                words.insert(int(len(words) * (i + 1) / 4), f". The secret code number {i + 1} is {cd}. ")
            doc = f"[doc {L}-{time.time_ns()}] " + " ".join(words) + \
                  "\n\nList every secret code number and its value that appears in the document above. Answer only with the codes."
            n = len(post("/tokenize", {"content": doc})["tokens"])
            if abs(n - L) <= max(16, L * 0.01):
                break
            n_words = int(n_words + (L - n) / (n / n_words))
        t0 = time.time()
        r = chat([{"role": "user", "content": doc}], max_tokens=2048, reasoning_effort="low")
        m = r["choices"][0]["message"]
        content = m.get("content") or ""
        found = [cd in content for cd in codes]
        save(f"needle_{L}", {"codes": codes, "prompt_tokens": r["usage"]["prompt_tokens"], "finish": r["choices"][0]["finish_reason"],
                             "content": content, "reasoning": m.get("reasoning_content"), "timings": r.get("timings"),
                             "seconds": round(time.time() - t0, 1)})
        gate(f"needle_{L}", all(found) and r["choices"][0]["finish_reason"] == "stop",
             f"{sum(found)}/3 in content, prompt {r['usage']['prompt_tokens']} tok, prefill {r['timings']['prompt_per_second']:.1f} tok/s, "
             f"decode {r['timings']['predicted_per_second']:.2f} tok/s, finish {r['choices'][0]['finish_reason']}")

save("summary", results)
sys.exit(0 if all(v["pass"] for k, v in results.items() if k != "effort_monotone_hint") else 1)
