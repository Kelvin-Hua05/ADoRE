"""Smoke test for the package scaffold only (no research functionality exists yet)."""
import adore


def test_import_and_version():
    assert isinstance(adore.__version__, str)
    assert adore.__version__.count(".") == 2
