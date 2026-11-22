# Launch llama-server with given args, scrape model/KV/compute buffer sizes from the log, stop it.
#   .\scripts\probe_mem.ps1 <name> <args...>
param([Parameter(Mandatory)][string]$Name, [Parameter(ValueFromRemainingArguments)][string[]]$Extra)
$root = Split-Path -Parent $PSScriptRoot
$out = & (Join-Path $PSScriptRoot "server.ps1") start $Name @Extra -lv 4
$log = Join-Path $root "logs\$Name.log"
$lines = Select-String -Path $log -Pattern "CUDA0 model buffer|CUDA0 KV buffer|CUDA0 compute buffer|CUDA_Host compute buffer|CUDA0 RS buffer|graph: nodes" | ForEach-Object { ($_.Line -replace '^\S+ I ', '').Trim() }
& (Join-Path $PSScriptRoot "server.ps1") stop | Out-Null
"== $Name :: $($Extra -join ' ')"
"   $out"
$lines | ForEach-Object { "   $_" }
