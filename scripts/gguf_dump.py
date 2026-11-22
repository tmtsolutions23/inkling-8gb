"""Dump GGUF metadata + per-tensor sizes, grouped per layer (expert vs non-expert).
Usage: .venv\\Scripts\\python -I scripts/gguf_dump.py models/<file>.gguf [--json out.json]
"""
import json
import re
import sys
from collections import defaultdict

from gguf import GGUFReader


def val(field):
    try:
        return field.contents()
    except Exception:
        return "<?>"


def main():
    path = sys.argv[1]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    r = GGUFReader(path)

    print("=== metadata (non-tokenizer) ===")
    meta = {}
    for k, f in r.fields.items():
        if k.startswith("tokenizer.ggml.") and k not in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
            continue
        v = val(f)
        if isinstance(v, list) and len(v) > 64:
            v = f"<list len={len(v)}> {v[:8]}..."
        meta[k] = v if isinstance(v, (int, float, str, bool, list)) else str(v)
        print(f"{k} = {v}")

    tensors = []
    per_layer = defaultdict(lambda: defaultdict(int))
    types = defaultdict(int)
    for t in r.tensors:
        nbytes = int(t.n_bytes)
        m = re.match(r"blk\.(\d+)\.(.+)", t.name)
        layer = int(m.group(1)) if m else -1
        kind = "experts" if "_exps" in t.name else "other"
        per_layer[layer][kind] += nbytes
        types[t.tensor_type.name] += nbytes
        tensors.append({"name": t.name, "type": t.tensor_type.name,
                        "shape": [int(x) for x in t.shape], "bytes": nbytes, "layer": layer})

    print("\n=== tensors (first layer + non-block) ===")
    for t in tensors:
        if t["layer"] in (-1, 0, 6):
            print(f"{t['name']:48s} {t['type']:10s} {str(t['shape']):28s} {t['bytes']/2**20:9.2f} MiB")

    print("\n=== per layer (MiB) ===")
    tot_e = tot_o = 0
    for layer in sorted(per_layer):
        e, o = per_layer[layer]["experts"], per_layer[layer]["other"]
        tot_e += e
        tot_o += o
        print(f"layer {layer:3d}  experts {e/2**20:9.1f}  other {o/2**20:8.1f}")
    print(f"TOTAL experts {tot_e/2**30:.3f} GiB  other {tot_o/2**30:.3f} GiB  all {(tot_e+tot_o)/2**30:.3f} GiB")

    print("\n=== bytes by ggml type (GiB) ===")
    for k, v in sorted(types.items(), key=lambda x: -x[1]):
        print(f"{k:12s} {v/2**30:8.3f}")

    if out_json:
        with open(out_json, "w") as fh:
            json.dump({"meta": meta, "tensors": tensors}, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
