import os
import subprocess
import sys

import numpy as np
import pytest

import adore_doe as ad
from adore_doe.core import (ExperimentDataset, SeedManager, check_seed, derive_seed, make_rng,
                            resolve_seed)


def test_check_seed_accepts_only_nonnegative_ints():
    assert check_seed(None) is None and check_seed(0) == 0 and check_seed(np.int64(5)) == 5
    for bad in (-1, 1.5, "7", True, [1], np.float64(2.0)):
        with pytest.raises(ad.DesignError):
            check_seed(bad)


def test_resolve_seed_returns_recorded_integer():
    assert resolve_seed(42) == 42
    s = resolve_seed(None)
    assert isinstance(s, int) and 0 <= s < 2**32
    a = make_rng(s).random(5)
    assert np.array_equal(a, make_rng(s).random(5))                 # recorded seed reproduces run


def test_make_rng_accepts_int_sequence_generator_and_never_uses_global_state():
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    make_rng(3).random(10)
    assert np.array_equal(before, np.random.get_state()[1])         # global RNG untouched
    g = np.random.default_rng(1)
    assert make_rng(g) is g
    assert np.array_equal(make_rng(np.random.SeedSequence(5)).random(3),
                          np.random.default_rng(np.random.SeedSequence(5)).random(3))


def test_derive_seed_deterministic_and_independent():
    assert derive_seed(42, "candidates") == derive_seed(42, "candidates")
    names = ["candidates", "moment", "starts", "randomize", "x"]
    seeds = {derive_seed(42, n) for n in names}
    assert len(seeds) == len(names)                                  # different names differ
    assert derive_seed(42, "a") != derive_seed(43, "a")              # different root differs
    assert derive_seed(42, "a", "b") != derive_seed(42, "b", "a")    # order matters


def test_derive_seed_pinned_values_are_stable():
    # Pinned: an accidental change of the derivation would silently change every reproducible
    # design.  (SeedSequence streams are stable across NumPy versions by NumPy's own policy.)
    assert derive_seed(42, "candidates") == 2341742572
    assert derive_seed(42, "moment") == 2342660455
    assert derive_seed(7, "a", "b") == 2518928298


def test_child_seeds_identical_across_processes():
    """Python's hash() is salted per process; ours must not be."""
    code = ("import sys; sys.path.insert(0, sys.argv[1])\n"
            "from adore_doe.core import derive_seed\n"
            "print(derive_seed(42, 'candidates'), derive_seed(7, 'a', 'b'))")
    src = os.path.dirname(os.path.dirname(ad.__file__))
    outs = {subprocess.run([sys.executable, "-c", code, src], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout.strip()
            for h in (0, 1, 12345)}
    assert len(outs) == 1 and outs.pop() == f"{derive_seed(42, 'candidates')} {derive_seed(7, 'a', 'b')}"


def test_seed_manager_streams_and_record():
    m = SeedManager(11)
    r1 = m.rng("candidates").random(4)
    r2 = m.rng("moment").random(4)
    assert not np.array_equal(r1, r2)
    m2 = SeedManager(11)
    assert np.array_equal(m2.rng("candidates").random(4), r1)       # same root -> same stream
    rec = m.as_dict()
    assert rec["root"] == 11 and set(rec) == {"root", "candidates", "moment"}
    assert rec["candidates"] == derive_seed(11, "candidates")
    assert SeedManager(None).root >= 0 and "SeedManager" in repr(m)
    # drawing from one stream does not change another
    a = SeedManager(5); a.rng("x").random(100)
    assert np.array_equal(a.rng("y").random(3), SeedManager(5).rng("y").random(3))


def test_dataset_randomize_uses_seed_manager_and_records():
    f = [ad.Factor("A", 0, 1), ad.Factor("B", 0, 1)]
    d1 = ExperimentDataset(f, np.zeros((6, 2))).randomize(seed=9)
    d2 = ExperimentDataset(f, np.zeros((6, 2))).randomize(seed=9)
    assert np.array_equal(d1.run_order, d2.run_order) and d1.info["seed"] == 9
    d3 = ExperimentDataset(f, np.zeros((6, 2))).randomize()           # seed=None: recorded
    d4 = ExperimentDataset(f, np.zeros((6, 2))).randomize(seed=d3.info["seed"])
    assert np.array_equal(d3.run_order, d4.run_order)
    with pytest.raises(ad.DesignError):
        ExperimentDataset(f, np.zeros((6, 2))).randomize(seed=-3)
    assert np.array_equal(d1.run_order, np.random.default_rng(9).permutation(6) + 1)  # Phase-1 compatible
