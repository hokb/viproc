cmake_minimum_required(VERSION 3.24)
project(viproc VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VIPROC_BUILD_TESTS "Build viproc tests" ON)
option(VIPROC_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

add_library(viproc_warnings INTERFACE)
target_compile_options(viproc_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<BOOL:${VIPROC_WARNINGS_AS_ERRORS}>,$<CXX_COMPILER_ID:GNU,Clang>>:-Werror>)

# Runtime: async execution / glue layer between user ops and the NumPy ndarray layer.
add_library(viproc src/runtime/version.cpp)
target_include_directories(viproc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_definitions(viproc PRIVATE
    VIPROC_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    VIPROC_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    VIPROC_VERSION_PATCH=${PROJECT_VERSION_PATCH})
target_link_libraries(viproc PRIVATE viproc_warnings)

if(VIPROC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
