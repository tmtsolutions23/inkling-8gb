"""Derive an experiment config from best-config.json (or a given base) with one change.
  python -I scripts/mkcfg.py NAME --gpu-layers 11,12,10,13 [--base configs/X.json] [--set -t=6] [--add --mlock] [--drop -ub]
--gpu-layers L,...  : put those layers' routed experts on CUDA0 via -ot, all other experts on CPU (replaces --n-cpu-moe).
--set FLAG=VALUE    : set/replace a flag's value.   --add FLAG [VALUE]: append.   --drop FLAG: remove flag (+value).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NONVAL = {"--no-op-offload", "--mlock", "--no-mmap", "--swa-full", "--no-repack", "-nr", "--op-offload", "-cb", "-nocb",
          "--no-kv-offload", "-nkvo", "-kvu", "-no-kvu"}


def parse(args):
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a in NONVAL or i + 1 >= len(args) or (args[i + 1].startswith("-") and not args[i + 1].lstrip("-").isdigit()):
            out.append([a]); i += 1
        else:
            out.append([a, args[i + 1]]); i += 2
    return out


def main():
    a = sys.argv[1:]
    name = a[0]
    base = a[a.index("--base") + 1] if "--base" in a else os.path.join(ROOT, "best-config.json")
    cfg = json.load(open(base))
    pairs = parse(cfg["args"])
    changes = []
    i = 1
    while i < len(a):
        k = a[i]
        if k == "--gpu-layers":
            layers = [] if a[i + 1] == "none" else [int(x) for x in a[i + 1].split(",") if x != ""]
            pairs = [p for p in pairs if p[0] not in ("--n-cpu-moe", "-ncmoe", "-ot", "--override-tensor", "--cpu-moe", "-cmoe")]
            rx = "|".join(str(L) for L in layers)
            ot = (f"blk\\.({rx})\\.ffn_.*_exps\\.weight=CUDA0," if layers else "") + "blk\\.\\d+\\.ffn_.*_exps\\.weight=CPU"
            pairs.append(["-ot", ot])
            cfg["gpu_expert_layers"] = layers
            cfg["expect_gpu_model_mib"] = round(1185.4 + 587.25 * len(layers), 1)
            changes.append(f"experts on GPU: {layers}")
            i += 2
        elif k == "--set":
            f, v = a[i + 1].split("=", 1)
            hit = [p for p in pairs if p[0] == f]
            if hit:
                hit[0][1:] = [v]
            else:
                pairs.append([f, v])
            changes.append(f"{f} {v}")
            i += 2
        elif k == "--add":
            if i + 2 < len(a) and not a[i + 2].startswith("--"):
                pairs.append([a[i + 1], a[i + 2]]); changes.append(f"+{a[i+1]} {a[i+2]}"); i += 3
            else:
                pairs.append([a[i + 1]]); changes.append(f"+{a[i+1]}"); i += 2
        elif k == "--drop":
            pairs = [p for p in pairs if p[0] != a[i + 1]]; changes.append(f"-{a[i+1]}"); i += 2
        elif k == "--base":
            i += 2
        else:
            sys.exit(f"unknown option {k}")
    cfg["name"] = name
    cfg["args"] = [x for p in pairs for x in p]
    cfg["note"] = f"derived from {os.path.basename(base)}: " + "; ".join(changes)
    path = os.path.join(ROOT, "configs", f"{name}.json")
    json.dump(cfg, open(path, "w"), indent=1)
    print(path, cfg["args"])


if __name__ == "__main__":
    main()
