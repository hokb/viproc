# Development loop (Windows): pull the checkout, build and install viproc into
# .venv (pip + scikit-build-core, incremental build in build\pip-<tag>), run
# the Python tests. Called by dev.cmd; Linux: dev.sh.
param(
    [switch]$NoPull,   # build the checkout as it is
    [switch]$NoTest,   # skip pytest
    [string]$Python    # interpreter for a new .venv (default: py -3, else python)
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $exe $($arguments -join ' ')" }
}

try {
    if (-not $NoPull) {
        Step 'git pull'
        Run git @('pull', '--ff-only')
    }

    $venvPython = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Path $venvPython)) {
        Step 'create .venv'
        if ($Python) { $base = @($Python) }
        elseif ($env:VIPROC_PYTHON) { $base = @($env:VIPROC_PYTHON) }
        elseif (Get-Command py -ErrorAction SilentlyContinue) { $base = @('py', '-3') }
        else { $base = @('python') }
        $exe = $base[0]; $pre = @($base | Select-Object -Skip 1)
        # No double quotes in arguments: Windows PowerShell 5.1 does not escape them.
        $check = 'import sys, struct; sys.exit(0 if sys.version_info >= (3, 12) and struct.calcsize(''P'') == 8 else 1)'
        & $exe @pre -c $check
        if ($LASTEXITCODE -ne 0) {
            throw "$($base -join ' ') is not a 64-bit Python >= 3.12; pass -Python <path to python.exe>"
        }
        Run $exe ($pre + @('-m', 'venv', '.venv'))
    }

    Step 'install build requirements'
    # Installed into .venv once, so builds need no isolated environment (faster).
    Run $venvPython @('-m', 'pip', 'install', '-q', '--upgrade', 'pip', 'scikit-build-core', 'numpy>=2.0', 'pytest', 'cmake>=3.24')

    Step 'build and install viproc (Release)'
    # scikit-build-core picks the Visual Studio generator for x64 Python: no
    # developer prompt needed, and the architecture matches the interpreter.
    Run $venvPython @('-m', 'pip', 'install', '--no-build-isolation', '--no-deps', '.')
    Run $venvPython @('-c', 'import viproc, numpy; print(''viproc'', viproc.__version__, ''numpy'', numpy.__version__, viproc.__file__)')

    if (-not $NoTest) {
        Step 'pytest'
        Run $venvPython @('-m', 'pytest', '-q', 'python\tests')
    }

    Write-Host "`nDone. Run programs with: .venv\Scripts\python prog.py" -ForegroundColor Green
    Write-Host "  (or .venv\Scripts\activate, then python prog.py)"
} catch {
    Write-Host "`nERROR: $_" -ForegroundColor Red
    exit 1
}
