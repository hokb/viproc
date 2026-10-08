cmake_minimum_required(VERSION 3.24)
project(viproc VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VIPROC_BUILD_TESTS "Build viproc tests" ON)
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development.Embed NumPy)

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
    src/runtime/runtime.cpp)
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
add_library(viproc_numpy src/kernels/numpy_ufunc.cpp)
target_link_libraries(viproc_numpy PUBLIC viproc Python3::Python Python3::NumPy PRIVATE viproc_warnings)

if(VIPROC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
