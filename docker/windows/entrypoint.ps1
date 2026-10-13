# Command line of the viproc Windows image (docker/README.md); the same
# commands as docker/entrypoint.sh in the Linux and Wine images.
#
# The image holds the sources in $env:VIPROC_SRC and the build in
# $env:VIPROC_BUILD. Every command first brings the build up to date (MSVC
# environment of the Build Tools, Ninja), so a checkout mounted over the
# sources is rebuilt incrementally before it runs. No param() block: all
# arguments, including ones starting with '-', are passed on unchanged.
$ErrorActionPreference = 'Stop'

$Command = if ($args.Count -gt 0) { $args[0] } else { 'test' }
$Rest = @($args | Select-Object -Skip 1) # always an array, also for one element

function Enter-Msvc {
    Import-Module "$env:VS_INSTALL\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    Enter-VsDevShell -VsInstallPath $env:VS_INSTALL -SkipAutomaticLocation `
        -DevCmdArguments '-arch=x64 -host_arch=x64 -no_logo' | Out-Null
}

function Invoke-Quiet([string]$exe, [string[]]$arguments) {
    # Windows PowerShell turns redirected stderr lines into errors; only the
    # exit code decides here.
    $ErrorActionPreference = 'Continue'
    $out = & $exe @arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        $out | Write-Host
        exit $LASTEXITCODE
    }
}

function Update-Build {
    Enter-Msvc
    if (-not (Test-Path "$env:VIPROC_BUILD\CMakeCache.txt")) {
        $type = if ($env:VIPROC_BUILD_TYPE) { $env:VIPROC_BUILD_TYPE } else { 'Release' }
        Invoke-Quiet cmake @('-S', $env:VIPROC_SRC, '-B', $env:VIPROC_BUILD, '-G', 'Ninja',
            "-DCMAKE_BUILD_TYPE=$type", '-DPython3_EXECUTABLE=C:\Python\python.exe')
    }
    Invoke-Quiet cmake @('--build', $env:VIPROC_BUILD)
}

function Exit-With([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    exit $LASTEXITCODE
}

switch ($Command) {
    'test' {
        Update-Build
        Exit-With ctest (@('--test-dir', $env:VIPROC_BUILD, '--output-on-failure') + $Rest)
    }
    'pytest' {
        Update-Build
        Exit-With python (@('-m', 'pytest', "$env:VIPROC_SRC\python\tests") + $Rest)
    }
    'python' {
        Update-Build
        Exit-With python $Rest
    }
    'benchmark' {
        Update-Build
        # "benchmark --quick" runs the suite with --quick
        $named = $Rest.Count -gt 0 -and -not $Rest[0].StartsWith('-')
        $name = if ($named) { $Rest[0] } else { 'suite' }
        $more = if ($named) { @($Rest | Select-Object -Skip 1) } else { $Rest }
        if ($name -eq 'suite' -and $more -notcontains '--out' -and $more -notcontains '--merge') {
            $more = @('--out', "C:\work\reports\$env:VIPROC_PLATFORM") + $more
        }
        Exit-With python (@("$env:VIPROC_SRC\python\benchmarks\$name.py") + $more)
    }
    'build' {
        Update-Build
    }
    'info' {
        Update-Build
        Write-Host "image: $env:VIPROC_PLATFORM, build type $env:VIPROC_BUILD_TYPE"
        # single quotes only: Windows PowerShell does not escape double
        # quotes in arguments of native commands
        Exit-With python @('-c', @'
import platform, sys
import numpy, viproc
print('platform:', platform.platform())
print('python:', sys.version.split()[0], 'numpy:', numpy.__version__)
print('temporary elision:', viproc._viproc.elision_mode())
'@)
    }
    { $_ -in 'help', '-h', '--help' } {
        @"
usage: docker run [--rm -it] [-v DIR:C:\work] IMAGE [COMMAND [ARGS...]]

commands:
  test [CTEST_ARGS...]     build, then run all test suites with ctest (default)
  pytest [PYTEST_ARGS...]  build, then run the Python bridge tests
  python [ARGS...]         Python with viproc importable (interactive with -it)
  benchmark [NAME [ARGS]]  run python/benchmarks/NAME.py (default: suite, the full
                           benchmark suite; reports go to C:\work\reports\<platform>:
                           mount a directory on C:\work to keep them)
  build                    bring the build up to date
  info                     platform, Python, NumPy and viproc configuration
  help                     this text
  anything else            executed as a command (e.g. cmd, powershell)

image: $env:VIPROC_PLATFORM, build type $env:VIPROC_BUILD_TYPE, sources $env:VIPROC_SRC, build $env:VIPROC_BUILD
"@ | Write-Host
    }
    default {
        Enter-Msvc
        Exit-With $Command $Rest
    }
}
