# syntax=docker/dockerfile:1
# viproc on Linux: builds the runtime and the Python bridge, runs the test
# suites by default and doubles as a command line (docker/README.md).
#
#   docker build -f docker/linux/Dockerfile -t viproc:linux .
#   docker run --rm viproc:linux                 # all test suites (ctest)
#   docker run --rm -it viproc:linux python      # Python with viproc
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-bookworm

ARG NUMPY_VERSION=2.5.4
ARG BUILD_TYPE=Release

# The base image brings GCC; CMake and Ninja come from PyPI. Behind a
# TLS-inspecting proxy, pass its CA: --secret id=ca,src=ca.pem (not kept).
RUN --mount=type=secret,id=ca,required=false \
    if [ -s /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi \
    && pip install --no-cache-dir "cmake>=3.24" ninja "numpy==${NUMPY_VERSION}" pytest

COPY . /src
# The command line, with LF line endings even from a Windows checkout made
# before .gitattributes (CRLF breaks the "#!/bin/bash" line).
RUN sed 's/\r$//' /src/docker/entrypoint.sh > /usr/local/bin/viproc \
    && chmod 755 /usr/local/bin/viproc
RUN cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    && cmake --build /build

ENV VIPROC_PLATFORM=linux \
    VIPROC_SRC=/src \
    VIPROC_BUILD=/build \
    PYTHONPATH=/build/python \
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    OPENBLAS_NUM_THREADS=1
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/viproc"]
CMD ["test"]
