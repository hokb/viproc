add_executable(test_version test_version.c)
target_link_libraries(test_version PRIVATE viproc)
add_test(NAME version COMMAND test_version)

# Verifies that NumPy's inner loops can be called from worker threads without the GIL.
add_executable(test_numpy_loops test_numpy_loops.cpp)
target_link_libraries(test_numpy_loops PRIVATE Python3::Python Python3::NumPy viproc_warnings)
target_include_directories(test_numpy_loops PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME numpy_loops COMMAND test_numpy_loops)

# Runtime core: ordering, continuations, copy-on-write, errors, look-ahead limit.
add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE viproc viproc_warnings)
add_test(NAME runtime COMMAND test_runtime)

# End-to-end: runtime + NumPy loops, bit-identical to eager NumPy.
add_executable(test_numpy_runtime test_numpy_runtime.cpp)
target_link_libraries(test_numpy_runtime PRIVATE viproc_numpy Python3::Python viproc_warnings)
add_test(NAME numpy_runtime COMMAND test_numpy_runtime)

# Python bridge (pytest), against the package assembled in the build tree.
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pytest"
                RESULT_VARIABLE VIPROC_NO_PYTEST OUTPUT_QUIET ERROR_QUIET)
if(VIPROC_NO_PYTEST)
    message(WARNING "pytest is not installed for ${Python3_EXECUTABLE}: the Python bridge "
                    "tests are not registered. Install it with: \"${Python3_EXECUTABLE}\" "
                    "-m pip install pytest (then re-run CMake).")
else()
    add_test(NAME python COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_SOURCE_DIR}/python/tests)
    set_tests_properties(python PROPERTIES ENVIRONMENT "PYTHONPATH=${VIPROC_PYTHON_DIR}")
endif()

# Test executables that embed Python start it as the interpreter CMake found
# (see embed_python.h). On Windows they also need the Python DLL on PATH.
foreach(t test_numpy_loops test_numpy_runtime)
    target_compile_definitions(${t} PRIVATE VIPROC_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
endforeach()
if(WIN32 AND Python3_RUNTIME_LIBRARY_DIRS)
    foreach(t numpy_loops numpy_runtime)
        set_tests_properties(${t} PROPERTIES ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:${Python3_RUNTIME_LIBRARY_DIRS}")
    endforeach()
endif()
