# Smoke-test run-inkling.ps1 modes: start, request(s), report decode + output head, stop. Usage: .\scripts\test_launcher.ps1 [-Safe] [-Coding]
param([switch]$Safe, [switch]$Coding)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$flags = @(); if ($Safe) { $flags += "-Safe" }; if ($Coding) { $flags += "-Coding" }
$tag = "launcher" + ($flags -join "")
$p = Start-Process powershell -ArgumentList (@("-NoProfile","-ExecutionPolicy","Bypass","-File","`"$root\run-inkling.ps1`"") + $flags) -PassThru -WindowStyle Hidden `
     -RedirectStandardError "logs\$tag.err" -RedirectStandardOutput "logs\$tag.out"
$ok = $false; $t = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $t) { try { if ((Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 2).status -eq 'ok') { $ok = $true; break } } catch {}; Start-Sleep 1 }
"[$tag] healthy=$ok  " + ((Get-Content "logs\$tag.out" -ErrorAction SilentlyContinue) -join ' ')
if ($ok) {
  $prompt = "Write a Python function that parses an ISO-8601 date string and returns a datetime, with input validation and three unit tests."
  $runs = if ($Coding) { 2 } else { 1 }
  for ($i = 1; $i -le $runs; $i++) {
    $body = @{ messages=@(@{role="user";content=$prompt}); max_tokens=400; temperature=0; seed=1; reasoning_effort="low" } | ConvertTo-Json -Depth 5
    $r = Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600
    $c = ($r.choices[0].message.content + "")
    "[$tag] run $i decode=$([math]::Round($r.timings.predicted_per_second,2)) tok/s n=$($r.timings.predicted_n) finish=$($r.choices[0].finish_reason) draft_n=$($r.timings.draft_n) draft_acc=$($r.timings.draft_n_accepted)"
    "   content head: " + ($c.Substring(0, [Math]::Min(120, $c.Length)) -replace "`n", " ")
  }
}
Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $_.WorkingSet64 -gt 10MB } | Stop-Process -ErrorAction SilentlyContinue
Start-Sleep 2; Stop-Process -Id $p.Id -ErrorAction SilentlyContinue
