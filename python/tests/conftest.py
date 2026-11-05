"""Shared pytest configuration.

The suite runs at the default optimization level O1 (bit-identical to
NumPy). `VIPROC_OPT=2 pytest python/tests` runs it at O2, which may compute
correct but deviating results (docs/design/goals.md, "Optimization levels"):
tests marked `o1` check what only O1 promises (bit-identity of NumPy's
iterator paths, temporary elision at issue) and are skipped there.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "o1: checks a property of optimization level O1 only")


def pytest_runtest_setup(item):
    if item.get_closest_marker("o1") is not None:
        import viproc

        if viproc.opt_level() >= 2:
            pytest.skip("O1 property (bit-identical iteration, elision at issue)")
