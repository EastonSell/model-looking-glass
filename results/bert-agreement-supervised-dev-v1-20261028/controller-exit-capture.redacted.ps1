$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path '..\..').Path
$repo = (Resolve-Path '.').Path
$base = '<REDACTED ABSOLUTE BASE INTERPRETER>'
$site = Join-Path $taskRoot '.venv\Lib\site-packages'
$cache = Join-Path $taskRoot 'hf-cache\hub'
$prefix = Join-Path $taskRoot 'bert-supervised-dev-prefix-b0dde0dd-01'
$custody = Join-Path $taskRoot 'bert-cpu-study-custody-v1'
$output = Join-Path $taskRoot 'bert-supervised-dev-b0dde0dd-01'
$exitRecord = Join-Path $taskRoot 'bert-supervised-dev-b0dde0dd-01-exit.json'
foreach ($path in @($prefix, $custody, $output, $exitRecord)) {
    if (Test-Path -LiteralPath $path) { throw 'Attempt path already exists' }
}
$null = New-Item -ItemType Directory -Path $prefix
$null = New-Item -ItemType Directory -Path $custody
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $base -I -S -B -X "pycache_prefix=$prefix" (Join-Path $repo 'scripts\bert_agreement_bootstrap.py') `
    --bootstrap-site $site --repo $repo --cache $cache --output $output `
    --commit 'b0dde0dd66712771a3a5b314dd4f2431c73f852c' --split development --family 1 `
    --custody-root $custody
$controllerExit = $LASTEXITCODE
@{schema='bert-agreement-local-controller-exit-v1'; exit_code=$controllerExit} |
    ConvertTo-Json -Compress | Set-Content -LiteralPath $exitRecord -Encoding ascii
exit $controllerExit
