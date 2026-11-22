# One-click launcher for Inkling 2 Mini AD-Q4_K (RTX 5060 Ti 8 GB + 5800X3D + DDR4-3600).
#   .\run-inkling.ps1            # default: AGENT - patched build, 64K context (agents, chat), 5 GPU looped layers
#   .\run-inkling.ps1 -Fast      # BEST3 - 32K context, 6 GPU looped layers, ~5% faster decode
#   .\run-inkling.ps1 -Safe      # BEST2 - untouched official b12615 binary, 32K, ~1/3 the prefill speed
#   .\run-inkling.ps1 -Coding    # + n-gram speculation (no cost when it does not match; no measured gain)
#   .\run-inkling.ps1 -Lan       # also listen on the network (default: this PC only; Docker/podman containers can still reach it via host.docker.internal)
#   .\run-inkling.ps1 -NoKey     # do not require the API key (default: key in .secrets\inkling-api-key.txt, Authorization: Bearer <key>)
# Endpoints: http://127.0.0.1:8080/v1 (this PC), http://host.docker.internal:8080/v1 (containers). Model id: inkling-2-mini
# Measured (2026-11-22): AGENT ~20 tok/s decode, ~250 tok/s prefill at 30K; BEST3 ~21.1 / ~280; all gates pass (see REPORT.md).
param([switch]$Fast, [switch]$Safe, [switch]$Coding, [switch]$Lan, [switch]$NoKey, [int]$Port = 8080, [int]$Ctx = 0)

$root  = $PSScriptRoot
$model = Join-Path $root "models\Inkling-2-Mini-AD-Q4_K.gguf"
$keyFile = Join-Path $root ".secrets\inkling-api-key.txt"
$agent = -not ($Fast -or $Safe)

if ($Ctx -le 0) { $Ctx = if ($agent) { 65536 } else { 32768 } }
# loop-aware placement: experts of the middle (3x-looped) layers on GPU, all other experts in RAM (agent: 5 layers to fit 64K KV)
$gpuLayers = if ($agent) { "11|12|10|13|9" } else { "11|12|10|13|9|14" }
$bindHost = if ($Lan) { "0.0.0.0" } else { "127.0.0.1" }
$a = @(
  "-m", $model, "--alias", "inkling-2-mini", "--host", $bindHost, "--port", "$Port",
  "-c", "$Ctx", "-np", "1", "--cache-ram", "0",   # one conversation slot; its KV prefix is reused across turns/sessions
  "-ngl", "99",
  "-ot", "blk\.($gpuLayers)\.ffn_.*_exps\.weight=CUDA0,blk\.\d+\.ffn_.*_exps\.weight=CPU",
  "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
  "-t", "8", "-tb", "16",
  # default thinking effort for clients that do not send reasoning_effort (template default 0.9 thinks for minutes and
  # breaks clients with short timeouts); a request's own reasoning_effort still wins. Quotes escaped for PowerShell 5.1.
  "--chat-template-kwargs", '{\"reasoning_effort\":\"low\"}'
)
if ($Lan -and $NoKey) { throw "Refusing to listen on the network without an API key." }
if (-not $NoKey) {
  if (-not (Test-Path $keyFile)) { throw "API key file missing: $keyFile" }
  $a += @("--api-key-file", $keyFile)
}

if ($Safe) {
  # official binary: op-offload must stay off (its staging buffer spills VRAM and yields NaN on this looped model)
  $exe = Join-Path $root "llama.cpp\b12615\llama-server.exe"
  $a += @("--no-op-offload", "-ub", "64", "-b", "64")
  $need = 6900
  Remove-Item env:LLAMA_ATTN_Q_CHUNK, env:GGML_OP_OFFLOAD_MIN_BATCH -ErrorAction SilentlyContinue
  $mode = "BEST2 (official b12615, 32K)"
} else {
  # patched build: per-split staging copies (op-offload is correct and costs no VRAM) + query-chunked biased attention
  $exe = Join-Path $root "llama.cpp\bin-inkling\llama-server.exe"
  $a += @("-ub", "512", "-b", "512")
  $env:LLAMA_ATTN_Q_CHUNK = if ($agent) { "16" } else { "32" }
  $env:GGML_OP_OFFLOAD_MIN_BATCH = "128"   # batches < 128 tokens (chat follow-ups, spec verify) use the CPU path: never slower than BEST2
  $need = if ($agent) { 7500 } else { 7200 }
  $mode = if ($agent) { "AGENT (patched, 64K)" } else { "BEST3 (patched, 32K)" }
}
if ($Coding) { $a += @("--spec-type", "ngram-mod") }

$free = (nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits).Trim()
if ([int]$free -lt $need) {
  Write-Warning "Only $free MiB VRAM free (want >= $need). Close GPU-heavy apps (browsers, slicers) or buffers may spill into shared RAM and speed will drop sharply. (-Fast/-Safe need less.)"
}
Write-Host "Starting $mode on http://${bindHost}:$Port  coding=$Coding  key=$(if ($NoKey) {'none'} else {'required'})"
& $exe @a
