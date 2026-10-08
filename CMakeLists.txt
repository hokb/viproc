cmake_minimum_required(VERSION 3.24)
project(viproc VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Everything ends up in the Python extension module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(VIPROC_BUILD_TESTS "Build viproc tests" ON)
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.Embed NumPy)

option(VIPROC_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

add_library(viproc_warnings INTERFACE)
target_compile_options(viproc_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<BOOL:${VIPROC_WARNINGS_AS_ERRORS}>,$<CXX_COMPILER_ID:GNU,Clang>>:-Werror>)

# Runtime: async execution / glue layer between user ops and the NumPy ndarray layer.
add_library(viproc
    src/runtime/version.cpp
    src/runtime/layout.cpp
    src/runtime/buffer.cpp
    src/runtime/task.cpp
    src/runtime/runtime.cpp
    src/runtime/assign.cpp)
target_include_directories(viproc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_definitions(viproc PRIVATE
    VIPROC_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    VIPROC_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    VIPROC_VERSION_PATCH=${PROJECT_VERSION_PATCH})
target_include_directories(viproc PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
find_package(Threads REQUIRED)
target_link_libraries(viproc PUBLIC Threads::Threads PRIVATE viproc_warnings)

# Kernel adapters that run tasks with NumPy's loops (needs CPython + NumPy).
# Links only Python's headers (Python3::Module): an extension module must not
# link libpython; test executables that embed Python add Python3::Python.
add_library(viproc_numpy src/kernels/numpy_ufunc.cpp)
target_link_libraries(viproc_numpy PUBLIC viproc Python3::Module Python3::NumPy PRIVATE viproc_warnings)

# Public C ABI (include/viproc/viproc.h).
add_library(viproc_capi src/capi/capi.cpp)
target_link_libraries(viproc_capi PUBLIC viproc_numpy PRIVATE viproc_warnings)

# Python package: viproc/*.py plus the _viproc extension module, assembled
# in ${CMAKE_BINARY_DIR}/python (put that directory on PYTHONPATH).
set(VIPROC_PYTHON_DIR ${CMAKE_BINARY_DIR}/python)
Python3_add_library(_viproc MODULE WITH_SOABI python/viproc/_viproc.c)
target_link_libraries(_viproc PRIVATE viproc_capi)
target_compile_options(_viproc PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<BOOL:${VIPROC_WARNINGS_AS_ERRORS}>,$<C_COMPILER_ID:GNU,Clang>>:-Werror>)
set_target_properties(_viproc PROPERTIES
    LINKER_LANGUAGE CXX
    LIBRARY_OUTPUT_DIRECTORY ${VIPROC_PYTHON_DIR}/viproc)
file(GLOB VIPROC_PY_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/python/viproc/*.py)
add_custom_target(viproc_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${VIPROC_PYTHON_DIR}/viproc
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${VIPROC_PY_SOURCES} ${VIPROC_PYTHON_DIR}/viproc
    DEPENDS ${VIPROC_PY_SOURCES}
    COMMENT "Copying the viproc Python package")
add_dependencies(_viproc viproc_python)

if(VIPROC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
