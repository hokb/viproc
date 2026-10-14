# escape=`
# viproc on Windows, native: MSVC build of the runtime and the Python
# bridge with the official Windows CPython. Runs the test suites by default
# and doubles as a command line (docker/README.md).
#
# Needs a Windows host with Windows containers (Docker Desktop in Windows
# container mode, or Docker Engine on Windows Server). The base image's
# Windows version must suit the host: WINDOWS_VERSION=ltsc2022 (Windows
# Server 2022 / Windows 11), ltsc2025 (Windows Server 2025).
#
#   docker build -f docker/windows/Dockerfile -t viproc:windows .
#   docker run --rm viproc:windows                 # all test suites (ctest)
#   docker run --rm -it viproc:windows python      # Python with viproc
ARG WINDOWS_VERSION=ltsc2022
FROM mcr.microsoft.com/windows/servercore:${WINDOWS_VERSION}

ARG PYTHON_VERSION=3.13.16
ARG NUMPY_VERSION=2.5.4
ARG VS_VERSION=17
ARG BUILD_TYPE=Release

SHELL ["powershell", "-NoProfile", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# MSVC, Windows SDK, CMake and Ninja: Visual Studio Build Tools, C++ workload
# with its recommended components.
RUN Invoke-WebRequest "https://aka.ms/vs/$env:VS_VERSION/release/vs_buildtools.exe" -OutFile C:\vs_buildtools.exe; `
    $p = Start-Process C:\vs_buildtools.exe -Wait -PassThru -ArgumentList @( `
        '--quiet', '--wait', '--norestart', '--nocache', '--installPath', 'C:\BuildTools', `
        '--add', 'Microsoft.VisualStudio.Workload.VCTools', '--includeRecommended'); `
    if ($p.ExitCode -notin 0, 3010) { throw "Build Tools setup failed: $($p.ExitCode)" }; `
    Remove-Item C:\vs_buildtools.exe

# Windows CPython from its NuGet package (the same one the Wine image uses).
RUN Invoke-WebRequest "https://api.nuget.org/v3-flatcontainer/python/$env:PYTHON_VERSION/python.$env:PYTHON_VERSION.nupkg" -OutFile C:\python.zip; `
    Expand-Archive C:\python.zip C:\python-pkg; `
    Move-Item C:\python-pkg\tools C:\Python; `
    Remove-Item -Recurse C:\python.zip, C:\python-pkg; `
    [Environment]::SetEnvironmentVariable('PATH', 'C:\Python;C:\Python\Scripts;' + $env:PATH, 'Machine')

RUN python -m pip install --no-cache-dir --no-warn-script-location "numpy==$env:NUMPY_VERSION" pytest

ENV VIPROC_PLATFORM=windows `
    VIPROC_SRC=C:\src `
    VIPROC_BUILD=C:\build `
    VIPROC_BUILD_TYPE=${BUILD_TYPE} `
    VS_INSTALL=C:\BuildTools `
    PYTHONPATH=C:\build\python `
    PYTHONPYCACHEPREFIX=C:\pycache `
    OPENBLAS_NUM_THREADS=1

COPY . C:\src
RUN C:\src\docker\windows\entrypoint.ps1 build

WORKDIR C:\work
ENTRYPOINT ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "C:\\src\\docker\\windows\\entrypoint.ps1"]
CMD ["test"]
