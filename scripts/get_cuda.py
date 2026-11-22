"""Download + merge CUDA 13.5 redistributable components (no admin) into toolchain/cuda-13.5.
Usage: .venv\\Scripts\\python -I scripts/get_cuda.py"""
import hashlib, json, os, shutil, sys, urllib.request, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
DEST = os.path.join(ROOT, "toolchain", "cuda-13.5")
DL = os.path.join(ROOT, "toolchain", "dl")
COMPS = ["cuda_nvcc", "cuda_cudart", "cuda_crt", "libnvvm", "cccl", "libcublas", "cuda_nvtx", "cuda_profiler_api",
         "cuda_cuobjdump", "cuda_nvml_dev", "cuda_nvdisasm", "cuda_cupti", "visual_studio_integration", "nsight_systems"]

man = json.load(urllib.request.urlopen(BASE + "redistrib_13.5.0.json"))
os.makedirs(DEST, exist_ok=True); os.makedirs(DL, exist_ok=True)
for c in COMPS:
    w = man.get(c, {}).get("windows-x86_64")
    if not w:
        print("skip (no windows build):", c); continue
    zpath = os.path.join(DL, os.path.basename(w["relative_path"]))
    if not os.path.exists(zpath) or os.path.getsize(zpath) != int(w["size"]):
        print(f"download {c} {int(w['size'])/2**20:.0f} MB", flush=True)
        urllib.request.urlretrieve(BASE + w["relative_path"], zpath)
    h = hashlib.sha256(open(zpath, "rb").read()).hexdigest()
    if h != w["sha256"]:
        sys.exit(f"SHA256 mismatch for {c}")
    with zipfile.ZipFile(zpath) as z:
        top = z.namelist()[0].split("/")[0]
        for m in z.infolist():
            rel = m.filename[len(top) + 1:]
            if not rel or m.is_dir():
                continue
            out = os.path.join(DEST, c if c == "nsight_systems" else "", rel) if c == "nsight_systems" else os.path.join(DEST, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print("ok", c, man[c]["version"], flush=True)
print("CUDA at", DEST)
