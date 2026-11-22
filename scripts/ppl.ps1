# Perplexity on results\ppl-corpus.txt (llama.cpp docs) with BEST2 placement.
#   .\scripts\ppl.ps1 <name> <bin-folder> [extra args...]    -> logs\ppl-<name>.log, prints final estimate
param([Parameter(Mandatory)][string]$Name, [Parameter(Mandatory)][string]$Bin, [Parameter(ValueFromRemainingArguments)][string[]]$Extra)
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "llama.cpp\$Bin\llama-perplexity.exe"
$model = Join-Path $root "models\Inkling-2-Mini-AD-Q4_K.gguf"
$a = @("-m", $model, "-f", (Join-Path $root "results\ppl-corpus.txt"), "-c", "512", "-b", "512", "--chunks", "24",
       "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-t", "8", "-tb", "16", "-lv", "4")
# default placement = BEST2 unless the caller passes its own -ot (first match wins in llama.cpp, so never pass two)
if ($Extra -notcontains "-ot") { $a += @("-ot", "blk\.(11|12|10|13|9|14)\.ffn_.*_exps\.weight=CUDA0,blk\.\d+\.ffn_.*_exps\.weight=CPU") }
$a += $Extra
$log = Join-Path $root "logs\ppl-$Name.log"
$t0 = Get-Date
& $exe @a *> $log
$fin = Select-String -Path $log -Pattern "Final estimate" | ForEach-Object { $_.Line }
"$Name [$Bin $($Extra -join ' ')] $fin  ($([math]::Round(((Get-Date)-$t0).TotalSeconds))s)"
Select-String -Path $log -Pattern "CUDA0 compute buffer|prompt eval time" | Select-Object -Last 2 | ForEach-Object { "   " + $_.Line.Trim() }
