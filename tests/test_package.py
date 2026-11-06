import adore_doe


def test_import_and_version():
    assert isinstance(adore_doe.__version__, str) and adore_doe.__version__
    for name in ("core", "screening"):
        assert hasattr(adore_doe, name)


def test_core_import_does_not_need_matplotlib():
    """ADR-008: matplotlib is optional; importing the package must work without it and
    plotting must fail with an actionable message."""
    import os
    import subprocess
    import sys
    src = os.path.dirname(os.path.dirname(adore_doe.__file__))
    code = ("import sys; sys.modules['matplotlib'] = None; sys.modules['matplotlib.pyplot'] = None\n"
            "import adore_doe, adore_doe.screening\n"
            "try:\n    from adore_doe.screening import plots\n"
            "except ImportError as e:\n    print('MSG', e)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": src})
    assert out.returncode == 0, out.stderr
    assert "adore-doe[plot]" in out.stdout
