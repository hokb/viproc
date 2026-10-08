add_executable(test_version test_version.c)
target_link_libraries(test_version PRIVATE viproc)
add_test(NAME version COMMAND test_version)

# Verifies that NumPy's inner loops can be called from worker threads without the GIL.
add_executable(test_numpy_loops test_numpy_loops.cpp)
target_link_libraries(test_numpy_loops PRIVATE Python3::Python Python3::NumPy viproc_warnings)
add_test(NAME numpy_loops COMMAND test_numpy_loops)
