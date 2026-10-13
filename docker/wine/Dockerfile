# syntax=docker/dockerfile:1
# viproc on Windows, run under Wine on a Linux host: the Windows build
# (llvm-mingw, x64, UCRT) of the runtime and the Python bridge, tested with
# the official Windows CPython and NumPy's Windows wheels. Runs the test
# suites by default and doubles as a command line (docker/README.md).
#
#   docker build -f docker/wine/Dockerfile -t viproc:wine .
#   docker run --rm viproc:wine                 # all test suites (ctest)
#   docker run --rm -it viproc:wine python      # Windows Python with viproc
#
# For a native Windows build with MSVC, see docker/windows/Dockerfile.
FROM ubuntu:26.04

ARG PYTHON_VERSION=3.13.16
ARG NUMPY_VERSION=2.5.4
ARG LLVM_MINGW_VERSION=20250924
# Source of the Universal CRT redistributable (ucrtbase.dll): Wine's own
# ucrtbase lacks functions NumPy needs (crealf, ...).
ARG WINDOWS_SDK_VERSION=10.0.28000.2705
ARG BUILD_TYPE=Release

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates curl unzip xz-utils cmake ninja-build python3-pip wine wine64 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/lib/x86_64-linux-gnu/wine:$PATH" \
    WINEPREFIX=/opt/wineprefix \
    WINEDEBUG=-all \
    WINEDLLOVERRIDES="mscoree,mshtml=;ucrtbase=n,b" \
    WINEPATH="Z:\\opt\\winpython" \
    XDG_RUNTIME_DIR=/tmp/xdg

# Toolchain, Windows Python (NuGet package "python"), UCRT. Behind a
# TLS-inspecting proxy, pass its CA: --secret id=ca,src=ca.pem (not kept).
RUN --mount=type=secret,id=ca,required=false \
    set -eu; \
    if [ -s /run/secrets/ca ]; then export CURL_CA_BUNDLE=/run/secrets/ca; fi; \
    curl -fsSL "https://github.com/mstorsjo/llvm-mingw/releases/download/${LLVM_MINGW_VERSION}/llvm-mingw-${LLVM_MINGW_VERSION}-ucrt-ubuntu-22.04-x86_64.tar.xz" \
        | tar -xJ -C /opt; \
    mv /opt/llvm-mingw-${LLVM_MINGW_VERSION}-ucrt-ubuntu-22.04-x86_64 /opt/llvm-mingw; \
    curl -fsSL -o /tmp/python.nupkg "https://api.nuget.org/v3-flatcontainer/python/${PYTHON_VERSION}/python.${PYTHON_VERSION}.nupkg"; \
    unzip -q /tmp/python.nupkg "tools/*" -d /tmp/python; \
    mv /tmp/python/tools /opt/winpython; \
    curl -fsSL -o /tmp/sdk.nupkg "https://api.nuget.org/v3-flatcontainer/microsoft.windows.sdk.cpp/${WINDOWS_SDK_VERSION}/microsoft.windows.sdk.cpp.${WINDOWS_SDK_VERSION}.nupkg"; \
    unzip -q -j /tmp/sdk.nupkg "c/Redist/*/ucrt/DLLs/x64/ucrtbase.dll" -d /tmp/ucrt; \
    mkdir -p -m 700 "$XDG_RUNTIME_DIR"; \
    wineboot --init; \
    wineserver --wait; \
    cp /tmp/ucrt/ucrtbase.dll "$WINEPREFIX/drive_c/windows/system32/ucrtbase.dll"; \
    rm -rf /tmp/python.nupkg /tmp/python /tmp/sdk.nupkg /tmp/ucrt

# `python` runs the Windows interpreter under Wine.
RUN printf '#!/bin/sh\nexec wine /opt/winpython/python.exe "$@"\n' > /usr/local/bin/python \
    && chmod +x /usr/local/bin/python

# NumPy and pytest: Windows wheels, fetched on the Linux side, installed
# offline (no networking inside Wine). The download evaluates environment
# markers for Linux: colorama, pytest's Windows dependency, is named.
RUN --mount=type=secret,id=ca,required=false \
    set -eu; \
    if [ -s /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pyver=$(echo ${PYTHON_VERSION} | cut -d. -f1-2); \
    python3 -m pip download --no-cache-dir -q -d /tmp/wheels --only-binary=:all: \
        --platform win_amd64 --python-version "$pyver" "numpy==${NUMPY_VERSION}" pytest \
        colorama; \
    python -m pip install --no-cache-dir --no-warn-script-location --no-index \
        --find-links /tmp/wheels "numpy==${NUMPY_VERSION}" pytest; \
    wineserver --wait; \
    rm -rf /tmp/wheels

COPY . /src
RUN cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
       -DCMAKE_TOOLCHAIN_FILE=/src/docker/wine/mingw-toolchain.cmake \
       -DPython3_EXECUTABLE=/usr/local/bin/python \
       -DPython3_INCLUDE_DIR=/opt/winpython/include \
       -DPython3_LIBRARY=/opt/winpython/libs/python$(echo ${PYTHON_VERSION} | cut -d. -f1-2 | tr -d .).lib \
       -DPython3_NumPy_INCLUDE_DIR=/opt/winpython/Lib/site-packages/numpy/_core/include \
    && cmake --build /build \
    && wineserver --wait

ENV VIPROC_PLATFORM=windows-wine \
    VIPROC_SRC=/src \
    VIPROC_BUILD=/build \
    PYTHONPATH=/build/python \
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    OPENBLAS_NUM_THREADS=1
WORKDIR /work
ENTRYPOINT ["/src/docker/entrypoint.sh"]
CMD ["test"]
