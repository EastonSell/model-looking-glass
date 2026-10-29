param(
    [Parameter(Mandatory=$true)][ValidateSet('rehearsal','baseline')][string]$Mode,
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-z0-9-]+$')][string]$RunName,
    [Parameter(Mandatory=$true)][string]$TaskRoot,
    [Parameter(Mandatory=$true)][string]$BasePython,
    [Parameter(Mandatory=$true)][string]$Venv,
    [Parameter(Mandatory=$true)][string]$Cache,
    [Parameter(Mandatory=$true)][string]$PropagationReport,
    [ValidateRange(1,1800)][int]$Timeout=900
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-NewJson([string]$Path, [object]$Value) {
    $json = ($Value | ConvertTo-Json -Depth 12 -Compress) + "`n"
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json)
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
}
function Hash-Of([string]$Path) { return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant() }

$task = (Resolve-Path -LiteralPath $TaskRoot).Path
$python = (Resolve-Path -LiteralPath $BasePython).Path
$venvPath = (Resolve-Path -LiteralPath $Venv).Path
$cachePath = (Resolve-Path -LiteralPath $Cache).Path
$reportPath = (Resolve-Path -LiteralPath $PropagationReport).Path
$source = Join-Path $task 'src'
$sources = Join-Path $source 'rehearsal-sources-v5.json'
$study = Join-Path $task 'cpu-study-root'
$stage = Join-Path $task 'staging'
$attempts = Join-Path $task 'attempts'
$external = Join-Path $attempts ($RunName + '-operator')
$observation = Join-Path $attempts ($RunName + '-observer')
$output = Join-Path $attempts ($RunName + '-owner')
$prefix = Join-Path $external 'prefix'
$expectedSource = 'd949eabc63475e04681978e8168e16871b00ab998a0eab648112c50cdf348aed'
$expectedObserver = '756dfb3d262cd9f2ec7890fcc878eabdfb9853223c5013c82a5cbedf4d476d6d'
$expectedReport = '25f4e43477355b357bc13e71e44f60c2135b490ea14359bb87b5448a33784d26'
$revision = 'd09b8ef5c0589fc644730faaa74e263ff90a8389'
$operatorHash = Hash-Of $PSCommandPath
if ((Hash-Of $sources) -ne $expectedSource -or
    (Hash-Of (Join-Path $source 'scripts/observe_france_baseline_v2.py')) -ne $expectedObserver -or
    (Hash-Of $reportPath) -ne $expectedReport) { throw 'Pinned source or public currentness bytes differ' }
$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
if ($report.schema -ne 'france-v4-propagation-pending-addendum-v1' -or
    $report.status -ne 'no_collision_in_named_bundle' -or $report.matches.Count -ne 0 -or
    $report.pins_sha256.france -ne 'c8dd418bea948c8ca2b4d1e39e696374a28bd2f911bb16707415c38170fbba82') {
    throw 'Public currentness report differs'
}
if ([IO.DriveInfo]::new('C').AvailableFreeSpace -lt $(if($Mode -eq 'baseline'){2GB}else{256MB}) -or
    [IO.DriveInfo]::new('E').AvailableFreeSpace -lt $(if($Mode -eq 'baseline'){3GB}else{512MB})) {
    throw 'Fresh workload-specific disk capacity is insufficient'
}
New-Item -ItemType Directory -Path $external -ErrorAction Stop | Out-Null
$phase = 'prefix'; $proc = $null
try {
    New-Item -ItemType Directory -Path $prefix -ErrorAction Stop | Out-Null
    & "$env:SystemRoot\System32\icacls.exe" $prefix /deny '*S-1-1-0:(WD,AD)' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not deny prefix creation' }
    $probe = @'
import pathlib, sys
try:
    pathlib.Path(sys.argv[1]).write_bytes(b'x')
except PermissionError:
    raise SystemExit(0)
raise SystemExit(1)
'@
    & $python -I -S -B -c $probe (Join-Path $prefix 'creation-probe')
    if ($LASTEXITCODE -ne 0 -or (Get-ChildItem -Force -LiteralPath $prefix | Measure-Object).Count -ne 0) {
        throw 'The observer prefix is not empty and creation-protected'
    }
    $env:CUDA_VISIBLE_DEVICES = '-1'
    $env:TEMP = $external; $env:TMP = $external
    $argsList = @('-I','-S','-B','-X',("pycache_prefix=" + $prefix),
        (Join-Path $source 'scripts/observe_france_baseline_v2.py'), '--source', $source,
        '--sources', $sources, '--sources-sha256', $expectedSource, '--source-revision', $revision,
        '--venv', $venvPath, '--dependency', (Join-Path $venvPath 'Lib/site-packages'),
        '--study-root', $study, '--observer-prefix', $prefix, '--observation', $observation,
        '--output', $output, '--mode', $Mode, '--cache', $cachePath, '--stage', $stage, '--timeout', "$Timeout",
        '--operator-start', (Join-Path $external 'operator-start.json'), '--operator-sha256', $operatorHash)
    if (@($argsList | Where-Object { $_ -match '[\s"]' }).Count -ne 0) {
        throw 'Operator arguments requiring shell quoting are unsupported'
    }
    $phase = 'start_observer'
    $started = [DateTime]::UtcNow.ToString('o')
    $proc = Start-Process -FilePath $python -ArgumentList $argsList -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $external 'stdout.log') -RedirectStandardError (Join-Path $external 'stderr.log')
    $startPartial = Join-Path $external 'operator-start.partial'
    $startFinal = Join-Path $external 'operator-start.json'
    Write-NewJson $startPartial @{ pid=$proc.Id; started_utc=$started;
        source_sha256=$expectedSource; source_revision=$revision; operator_sha256=$operatorHash;
        propagation_report_sha256=$expectedReport;
        scope='Direct-process observation; source revision and public currentness require separate checks.' }
    [IO.File]::Move($startPartial, $startFinal)
    $phase = 'wait_observer'
    $reaped = $proc.WaitForExit(($Timeout + 90) * 1000)
    if (-not $reaped) {
        $proc.Kill(); $directReaped = $proc.WaitForExit(10000)
        Write-NewJson (Join-Path $external 'operator-incomplete.json') @{ status='incomplete'; phase='timeout';
            pid=$proc.Id; direct_reaped=$directReaped; descendant_cleanup='unverified';
            scope='Preserve all files and do not retry uncertain work.' }
        exit 1
    }
    $exitCode = $proc.ExitCode
    if ($exitCode -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $observation 'outer-exit.json')) -or
        (Get-ChildItem -Force -LiteralPath $prefix | Measure-Object).Count -ne 0) {
        throw 'Observer exit, receipt or prefix differs'
    }
    $outer = Get-Content -Raw -LiteralPath (Join-Path $observation 'outer-exit.json') | ConvertFrom-Json
    if ($outer.status -ne 'reaped' -or $outer.exit_code -ne 0 -or $outer.timed_out -ne $false) {
        throw 'Observer did not record a successful direct owner exit'
    }
    & "$env:SystemRoot\System32\icacls.exe" $prefix /remove:d '*S-1-1-0' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Prefix protection cleanup was not confirmed' }
    Write-NewJson (Join-Path $external 'operator-exit.json') @{ status='reaped'; pid=$proc.Id; exit_code=$exitCode;
        timed_out=$false; prefix_empty=$true; outer_exit_sha256=(Hash-Of (Join-Path $observation 'outer-exit.json'));
        operator_sha256=$operatorHash;
        finished_utc=[DateTime]::UtcNow.ToString('o'); scope='Operator observation; not an execution attestation.' }
    exit 0
} catch {
    $primary = $_
    $directReaped = $false
    if ($null -ne $proc) {
        try { if (-not $proc.HasExited) { $proc.Kill() }; $directReaped = $proc.WaitForExit(10000) } catch { }
    }
    try { Write-NewJson (Join-Path $external 'operator-incomplete.json') @{ status='incomplete'; phase=$phase;
        error_type=$primary.Exception.GetType().Name; direct_reaped=$directReaped; descendant_cleanup='unverified';
        scope='Preserve all files and do not retry uncertain work.' } } catch { }
    throw $primary
}
