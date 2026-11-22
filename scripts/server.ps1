# Start / stop a llama-server instance for experiments.
#   .\scripts\server.ps1 start <name> <extra args...>   -> logs\<name>.log, waits for /health
#   .\scripts\server.ps1 stop                           -> graceful stop (waits for idle slots, then kills by PID)
param([Parameter(Mandatory)][string]$Action, [string]$Name = "run", [Parameter(ValueFromRemainingArguments)][string[]]$Extra)

$root = Split-Path -Parent $PSScriptRoot
$bin = if ($env:LLAMA_BIN) { $env:LLAMA_BIN } else { "b12615" }
$exe = Join-Path $root "llama.cpp\$bin\llama-server.exe"
$model = Join-Path $root "models\Inkling-2-Mini-AD-Q4_K.gguf"
$pidFile = Join-Path $root "logs\server.pid"
$port = 8080

function Stop-Srv {
    if (Test-Path $pidFile) {
        $id = [int](Get-Content $pidFile)
        $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($proc) {
            # drain: wait until no slot is processing (max 120 s)
            $t = (Get-Date).AddSeconds(120)
            while ((Get-Date) -lt $t) {
                try { $s = Invoke-RestMethod "http://127.0.0.1:$port/slots" -TimeoutSec 2; if (-not ($s | Where-Object { $_.is_processing })) { break } } catch { break }
                Start-Sleep 1
            }
            Stop-Process -Id $id
            $proc.WaitForExit(30000) | Out-Null
        }
        Remove-Item $pidFile -ErrorAction SilentlyContinue
    }
}

if ($Action -eq "stop") { Stop-Srv; "stopped"; exit 0 }

Stop-Srv
New-Item -ItemType Directory -Force (Join-Path $root "logs") | Out-Null
$log = Join-Path $root "logs\$Name.log"
$args = @("-m", "`"$model`"","--host", "127.0.0.1", "--port", "$port", "--no-webui", "--metrics", "--slots") + $Extra
$t0 = Get-Date
$p = Start-Process -FilePath $exe -ArgumentList $args -RedirectStandardError $log -RedirectStandardOutput "$log.out" -PassThru -WindowStyle Hidden
$p.Id | Out-File $pidFile -Encoding ascii
$deadline = (Get-Date).AddSeconds(600); $ok = $false
while ((Get-Date) -lt $deadline) {
    try { $r = Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 2; if ($r.status -eq "ok") { $ok = $true; break } } catch {}
    if ($p.HasExited) { break }
    Start-Sleep -Milliseconds 500
}
$vram = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim()
"healthy=$ok startup_s=$([math]::Round(((Get-Date)-$t0).TotalSeconds,1)) pid=$($p.Id) vram_used_mib=$vram log=$log"
if (-not $ok) { Get-Content $log -Tail 30; exit 1 }
