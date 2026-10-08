add_executable(test_version test_version.c)
target_link_libraries(test_version PRIVATE viproc)
add_test(NAME version COMMAND test_version)

# Verifies that NumPy's inner loops can be called from worker threads without the GIL.
add_executable(test_numpy_loops test_numpy_loops.cpp)
target_link_libraries(test_numpy_loops PRIVATE Python3::Python Python3::NumPy viproc_warnings)
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
add_test(NAME python COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_SOURCE_DIR}/python/tests)
set_tests_properties(python PROPERTIES ENVIRONMENT "PYTHONPATH=${VIPROC_PYTHON_DIR}")
