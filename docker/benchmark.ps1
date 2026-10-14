# Runs the viproc benchmark suite (python/benchmarks/suite.py) in the Linux
# image and in the Windows build (Wine image), then merges both into one
# comparison report. Builds the images from this checkout first (Docker's
# cache makes that quick when nothing changed; $env:VIPROC_NO_BUILD = 1
# skips it). Docker Desktop in Linux
# container mode; for the native MSVC image see docker/README.md.
#
#   docker\benchmark.ps1 [--quick] [SUITE_ARGS...]
#
# The reports are copied out of the containers (docker cp, no bind mount)
# into .\reports, relative to the current directory:
#   reports\report.html (comparison), reports\<image>\report.html (details,
#   also report.md and results.json)
# Output directory: $env:VIPROC_BENCH_OUT (default reports). Images:
# $env:VIPROC_IMAGES = "linux wine" (default).
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$out = if ($env:VIPROC_BENCH_OUT) { $env:VIPROC_BENCH_OUT } else { 'reports' }
$images = if ($env:VIPROC_IMAGES) { $env:VIPROC_IMAGES -split '\s+' } else { @('linux', 'wine') }
New-Item -ItemType Directory -Force -Path $out | Out-Null
$outAbs = (Resolve-Path $out).Path

function Invoke-Docker([string[]]$arguments) {
    & docker @arguments
    if ($LASTEXITCODE -ne 0) { throw "docker $($arguments -join ' ') failed ($LASTEXITCODE)" }
}

# Runs $image $arguments in a new container, then copies its /work/reports/$sub
# to $out\$sub. $copyIn: a host directory copied to /work/in first (or $null).
function Invoke-AndCopy([string]$sub, [string]$image, [string[]]$arguments, $copyIn) {
    $cid = (& docker create -e VIPROC_REPORTS_COPIED=1 $image @arguments)
    if ($LASTEXITCODE -ne 0) { throw "docker create $image failed ($LASTEXITCODE)" }
    try {
        if ($copyIn) { Invoke-Docker @('cp', "$copyIn\.", "${cid}:/work/in") }
        Invoke-Docker @('start', '-a', $cid)
        $dest = Join-Path $outAbs $sub
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        # No stale report may survive a run that does not write one.
        foreach ($f in 'report.html', 'report.md', 'results.json') {
            Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $dest $f)
        }
        Invoke-Docker @('cp', "${cid}:/work/reports/$sub/.", $dest)
    } finally {
        & docker rm -f $cid *> $null
    }
}

$results = @()
foreach ($img in $images) {
    if (-not $env:VIPROC_NO_BUILD) {
        Write-Host "== building viproc:$img (cached when unchanged)"
        Invoke-Docker @('build', '-f', "$repo\docker\$img\Dockerfile", '-t', "viproc:$img", $repo)
    }
    Write-Host "== benchmark suite on viproc:$img"
    Invoke-AndCopy $img "viproc:$img" (@('benchmark', 'suite', '--out', "/work/reports/$img") + $args) $null
    $results += "/work/in/$img/results.json"
}

if ($results.Count -gt 1) {
    Write-Host "== comparison"
    Invoke-AndCopy '.' 'viproc:linux' (@('python', '/src/python/benchmarks/suite.py', '--merge') +
                                       $results + @('--out', '/work/reports')) $outAbs
}
Write-Host "== reports in $outAbs\ (report.html: comparison; <image>\report.html: details)"
