# Serialized experiment queue: runs configs in order, one GPU job at a time.
param([Parameter(ValueFromRemainingArguments)][string[]]$Names)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
foreach ($n in $Names) {
  $flags = @()
  if ($n -match '^(.*?)(\+long)$') { $n = $Matches[1]; $flags += "--long" }
  "=== $(Get-Date -Format HH:mm:ss) START $n" | Tee-Object -Append logs\queue.txt
  & .\.venv\Scripts\python.exe -I scripts\bench_inkling.py run "configs\$n.json" @flags *> "logs\bench-$n.txt"
  "=== $(Get-Date -Format HH:mm:ss) END $n" | Tee-Object -Append logs\queue.txt
}
& .\.venv\Scripts\python.exe -I scripts\bench_inkling.py show
