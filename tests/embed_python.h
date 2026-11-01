// Starts an embedded Python interpreter for the test executables.
#pragma once

#include "numpy2x_python.h"

#include <cstdio>

// Initializes Python as if `NUMPY2X_PYTHON_EXECUTABLE` (the interpreter CMake
// found) had been started: its prefix, standard library and site-packages
// (including a virtual environment's) are used. Without this, an embedded
// interpreter derives its paths from the test executable's location, which
// fails on Windows.
inline bool init_embedded_python() {
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    PyStatus status = PyStatus_Ok();
#ifdef NUMPY2X_PYTHON_EXECUTABLE
    status = PyConfig_SetBytesString(&config, &config.program_name, NUMPY2X_PYTHON_EXECUTABLE);
#endif
    if (!PyStatus_Exception(status)) {
        status = Py_InitializeFromConfig(&config);
    }
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        std::fprintf(stderr, "cannot initialize Python: %s\n",
                     status.err_msg != nullptr ? status.err_msg : "unknown error");
        return false;
    }
    return true;
}
