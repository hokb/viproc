# Runs the viproc benchmark suite (python/benchmarks/suite.py) in the Linux
# image and in the Windows build (Wine image), then merges both into one
# comparison report. Builds missing images first. Docker Desktop in Linux
# container mode; for the native MSVC image see docker/README.md.
#
#   docker\benchmark.ps1 [--quick] [SUITE_ARGS...]
#
# Results: viproc-benchmark\<platform>\report.html (details) and
# viproc-benchmark\report.html (comparison), relative to the current
# directory. Images: $env:VIPROC_IMAGES = "linux wine" (default).
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$out = if ($env:VIPROC_BENCH_OUT) { $env:VIPROC_BENCH_OUT } else { 'viproc-benchmark' }
$images = if ($env:VIPROC_IMAGES) { $env:VIPROC_IMAGES -split '\s+' } else { @('linux', 'wine') }
New-Item -ItemType Directory -Force -Path $out | Out-Null
$outAbs = (Resolve-Path $out).Path

function Invoke-Docker([string[]]$arguments) {
    & docker @arguments
    if ($LASTEXITCODE -ne 0) { throw "docker $($arguments -join ' ') failed ($LASTEXITCODE)" }
}

$results = @()
foreach ($img in $images) {
    & docker image inspect "viproc:$img" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "== building viproc:$img"
        Invoke-Docker @('build', '-f', "$repo\docker\$img\Dockerfile", '-t', "viproc:$img", $repo)
    }
    Write-Host "== benchmark suite on viproc:$img"
    Invoke-Docker (@('run', '--rm', '-v', "${outAbs}:/work", "viproc:$img",
                     'benchmark', 'suite', '--out', "/work/$img") + $args)
    $results += "/work/$img/results.json"
}

if ($results.Count -gt 1) {
    Invoke-Docker (@('run', '--rm', '-v', "${outAbs}:/work", 'viproc:linux', 'python',
                     '/src/python/benchmarks/suite.py', '--merge') + $results + @('--out', '/work'))
}
Write-Host "== reports in $out\ (report.html: comparison; <platform>\report.html: details)"
