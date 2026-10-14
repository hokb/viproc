/* Includes <Python.h> (C and C++). Use this instead of including Python.h
 * directly.
 *
 * MSVC debug builds define _DEBUG, which makes pyconfig.h request the debug
 * Python library (python3XX_d.lib, via #pragma comment(lib)) and switch to the
 * debug ABI (Py_DEBUG). Regular Python installations ship neither, and an
 * extension built that way would not load into a release interpreter. So
 * _DEBUG is hidden while Python.h is processed (same approach as pybind11). */
#ifndef VIPROC_PYTHON_H
#define VIPROC_PYTHON_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif

#if defined(_MSC_VER) && defined(_DEBUG) && !defined(Py_DEBUG)
#ifdef __cplusplus
/* Fix the C++ standard library's debug settings before _DEBUG goes away. */
#include <yvals.h>
#endif
#define VIPROC_RESTORE_DEBUG
#undef _DEBUG
#endif

#include <Python.h>

#ifdef VIPROC_RESTORE_DEBUG
#define _DEBUG 1
#undef VIPROC_RESTORE_DEBUG
#endif

#endif /* VIPROC_PYTHON_H */
