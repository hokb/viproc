cmake_minimum_required(VERSION 3.24)
project(numpy2x VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Everything ends up in the Python extension module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "numpy2x needs a 64-bit build, but the compiler targets 32 bit. On Windows "
                        "use dev.cmd (it builds for the interpreter's architecture), or the \"x64 "
                        "Native Tools Command Prompt\" (vcvarsall.bat x64). pip install . from a "
                        "32-bit developer prompt builds 32 bit too (VSCMD_ARG_TGT_ARCH=x86).")
endif()

option(NUMPY2X_BUILD_TESTS "Build numpy2x tests" ON)
# Test executables embed Python; the extension module itself must not link libpython.
set(NUMPY2X_PYTHON_COMPONENTS Interpreter Development.Module NumPy)
if(NUMPY2X_BUILD_TESTS)
    list(APPEND NUMPY2X_PYTHON_COMPONENTS Development.Embed)
endif()
find_package(Python3 3.12 REQUIRED COMPONENTS ${NUMPY2X_PYTHON_COMPONENTS})
# numpy2x reproduces the calling patterns of the NumPy it runs on; only 2.5+ is tested.
if(Python3_NumPy_VERSION AND Python3_NumPy_VERSION VERSION_LESS 2.5)
    message(FATAL_ERROR "numpy2x needs NumPy >= 2.5, found ${Python3_NumPy_VERSION}")
endif()

option(NUMPY2X_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(NUMPY2X_ELEMENT_POOL "Cache element memory in numpy2x's pool (OFF for AddressSanitizer runs)" ON)

add_library(numpy2x_warnings INTERFACE)
target_compile_options(numpy2x_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<BOOL:${NUMPY2X_WARNINGS_AS_ERRORS}>,$<CXX_COMPILER_ID:GNU,Clang>>:-Werror>)

# Runtime: async execution / glue layer between user ops and the NumPy ndarray layer.
add_library(numpy2x
    src/runtime/version.cpp
    src/runtime/layout.cpp
    src/runtime/countable_array.cpp
    src/runtime/cpu.cpp
    src/runtime/workers.cpp
    src/runtime/element_pool.cpp
    src/runtime/task.cpp
    src/runtime/runtime.cpp
    src/runtime/assign.cpp
    src/runtime/index.cpp
    src/runtime/offload.cpp)
if(NOT NUMPY2X_ELEMENT_POOL)
    target_compile_definitions(numpy2x PRIVATE NUMPY2X_NO_ELEMENT_POOL)
endif()
target_include_directories(numpy2x PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_definitions(numpy2x PRIVATE
    NUMPY2X_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    NUMPY2X_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    NUMPY2X_VERSION_PATCH=${PROJECT_VERSION_PATCH})
target_include_directories(numpy2x PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
find_package(Threads REQUIRED)
target_link_libraries(numpy2x PUBLIC Threads::Threads PRIVATE numpy2x_warnings)
if(WIN32)
    target_link_libraries(numpy2x PRIVATE advapi32) # large-page privilege (element pool)
endif()

# Kernel adapters that run tasks with NumPy's loops (needs CPython + NumPy).
# Links only Python's headers (Python3::Module): an extension module must not
# link libpython; test executables that embed Python add Python3::Python.
add_library(numpy2x_numpy src/kernels/numpy_ufunc.cpp src/kernels/numpy_reduce.cpp
                         src/kernels/numpy_matmul.cpp src/kernels/numpy_gufunc.cpp
                         src/kernels/numpy_sort.cpp)
target_link_libraries(numpy2x_numpy PUBLIC numpy2x Python3::Module Python3::NumPy PRIVATE numpy2x_warnings)

# Public C ABI (include/numpy2x/numpy2x.h).
add_library(numpy2x_capi src/capi/capi.cpp)
target_link_libraries(numpy2x_capi PUBLIC numpy2x_numpy PRIVATE numpy2x_warnings)

# Python package: numpy2x/*.py plus the _numpy2x extension module, assembled
# in ${CMAKE_BINARY_DIR}/python (put that directory on PYTHONPATH).
set(NUMPY2X_PYTHON_DIR ${CMAKE_BINARY_DIR}/python)
Python3_add_library(_numpy2x MODULE WITH_SOABI python/numpy2x/_numpy2x.c)
target_link_libraries(_numpy2x PRIVATE numpy2x_capi)
target_compile_options(_numpy2x PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<BOOL:${NUMPY2X_WARNINGS_AS_ERRORS}>,$<C_COMPILER_ID:GNU,Clang>>:-Werror>)
# The generator expression keeps multi-config generators (Visual Studio) from
# appending a per-configuration subdirectory, where Python would not find it.
set_target_properties(_numpy2x PROPERTIES
    LINKER_LANGUAGE CXX
    LIBRARY_OUTPUT_DIRECTORY $<1:${NUMPY2X_PYTHON_DIR}/numpy2x>
    RUNTIME_OUTPUT_DIRECTORY $<1:${NUMPY2X_PYTHON_DIR}/numpy2x>)
file(GLOB NUMPY2X_PY_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/python/numpy2x/*.py)
# _numpy2x_hook.py (import numpy -> numpy2x for python -m numpy2x and NUMPY2X=1)
# sits next to the package: numpy2x.pth imports it at every interpreter start,
# without importing numpy2x.
set(NUMPY2X_HOOK ${CMAKE_SOURCE_DIR}/python/_numpy2x_hook.py)
add_custom_target(numpy2x_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${NUMPY2X_PYTHON_DIR}/numpy2x
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NUMPY2X_PY_SOURCES} ${NUMPY2X_PYTHON_DIR}/numpy2x
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NUMPY2X_HOOK} ${NUMPY2X_PYTHON_DIR}
    DEPENDS ${NUMPY2X_PY_SOURCES} ${NUMPY2X_HOOK}
    COMMENT "Copying the numpy2x Python package")
add_dependencies(_numpy2x numpy2x_python)

# Wheel contents (pip install . via scikit-build-core, pyproject.toml): the
# package as assembled above.
if(SKBUILD)
    install(TARGETS _numpy2x LIBRARY DESTINATION numpy2x RUNTIME DESTINATION numpy2x)
    install(FILES ${NUMPY2X_PY_SOURCES} DESTINATION numpy2x)
    install(FILES ${NUMPY2X_HOOK} python/numpy2x.pth DESTINATION .)
endif()

if(NUMPY2X_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
