import numpy as np
import pandas as pd
import pytest

import adore_doe as yp
from adore_doe.screening.pb import hadamard, supported_runs


def make_factors(k, cls=yp.Factor):
    return [cls(f"X{i+1}", low=i, high=i + 10 * (i + 1)) for i in range(k)]


# ---------------------------------------------------------------- Factor
def test_factor_roundtrip_and_levels():
    f = yp.Factor("T", low=60, high=100, unit="C")
    assert f.to_coded(60) == -1 and f.to_coded(100) == 1 and f.to_coded(80) == 0
    x = np.array([60.0, 70.0, 80.0, 95.0])
    assert np.allclose(f.from_coded(f.to_coded(x)), x)
    assert f.from_coded(-1) == 60 and f.from_coded(0) == 80


def test_factor_invalid():
    for kw in [dict(low=5, high=5), dict(low=6, high=5), dict(low="a", high=1),
               dict(low=0, high=float("inf"))]:
        with pytest.raises(yp.DesignError):
            yp.Factor("A", **kw)
    with pytest.raises(yp.DesignError):
        yp.Factor("", low=0, high=1)
    with pytest.raises(yp.DesignError):
        yp.Factor("A", 0, 1, factor_type="ordinal")


def test_categorical_factor():
    f = yp.Factor("Cat", "old", "new", factor_type="categorical")
    assert f.to_coded("old") == -1 and f.to_coded("new") == 1
    assert list(f.from_coded(np.array([-1.0, 1.0]))) == ["old", "new"]
    with pytest.raises(yp.DesignError):
        f.to_coded("other")
    with pytest.raises(yp.DesignError):
        f.center
    with pytest.raises(yp.DesignError):
        yp.screening.plackett_burman([f, yp.Factor("B", 0, 1)], runs=4, center_points=1)


# ---------------------------------------------------------------- Hadamard / PB validity
def test_supported_hadamard_orthogonal():
    for n in supported_runs(68):
        H = hadamard(n)
        assert np.array_equal(H @ H.T, n * np.eye(n))
        assert np.all(H[:, 0] == 1) and np.all(H[0] == 1)


@pytest.mark.parametrize("runs", [0, 2, 5, 6, 10, 14, 30, 50, 52])
def test_invalid_runs_rejected(runs):
    with pytest.raises(yp.DesignError):
        yp.screening.plackett_burman(make_factors(3), runs=runs)


def test_too_many_factors_for_runs():
    with pytest.raises(yp.DesignError, match="at most"):
        yp.screening.plackett_burman(make_factors(8), runs=8)


@pytest.mark.parametrize("runs,k", [(4, 3), (8, 7), (8, 3), (12, 11), (12, 5), (16, 15),
                                    (20, 19), (24, 23), (28, 27), (36, 20)])
def test_pb_properties(runs, k):
    d = yp.screening.plackett_burman(make_factors(k), runs=runs)
    C = d.coded
    assert C.shape == (runs, k)
    assert set(np.unique(C)) == {-1.0, 1.0}
    assert np.allclose(C.sum(axis=0), 0)                    # balanced
    full = np.column_stack([C, d.dummy_coded])
    assert np.allclose(full.T @ full, runs * np.eye(runs - 1))   # orthogonal incl. dummies
    assert d.info["n_dummy_columns"] == runs - 1 - k
    assert d.info["design_type"] == "plackett_burman"


def test_default_runs_smallest_valid():
    assert yp.screening.plackett_burman(make_factors(3)).n_runs == 4
    assert yp.screening.plackett_burman(make_factors(4)).n_runs == 8
    assert yp.screening.plackett_burman(make_factors(8)).n_runs == 12
    assert yp.screening.plackett_burman(make_factors(12)).n_runs == 16


def test_duplicate_factor_names_rejected():
    with pytest.raises(yp.DesignError):
        yp.screening.plackett_burman([yp.Factor("A", 0, 1), yp.Factor("A", 0, 2)], runs=4)


def test_pb12_aliasing_is_partial_one_third():
    """Documented aliasing: in PB-12 an interaction column correlates +-1/3 with main effects."""
    d = yp.screening.plackett_burman(make_factors(11), runs=12)
    C = d.coded
    ab = C[:, 0] * C[:, 1]
    r = np.abs(C.T @ ab) / 12
    assert np.isclose(r.max(), 1 / 3)
    assert np.isclose(np.sort(r)[0], 0.0)    # not aliased with its parents


def test_pb8_aliasing_is_complete():
    d = yp.screening.plackett_burman(make_factors(7), runs=8)
    C = d.coded
    ab = C[:, 0] * C[:, 1]
    assert np.isclose(np.abs(C.T @ ab).max() / 8, 1.0)


# ---------------------------------------------------------------- center points & actual values
def test_center_points_and_actual_values():
    fs = [yp.Factor("A", 0.1, 0.3), yp.Factor("B", 10, 20), yp.Factor("C", 1, 5)]
    d = yp.screening.plackett_burman(fs, runs=8, center_points=3)
    assert d.n_runs == 11 and d.center_points == 3
    assert list(d.point_type[:8]) == ["factorial"] * 8
    cen = d.coded[d.point_type == "center"]
    assert np.all(cen == 0)
    act = d.actual_df()
    assert np.allclose(act.iloc[-1].to_numpy(), [0.2, 15, 3])
    assert set(act["A"].round(10)) == {0.1, 0.3, 0.2}
    assert np.allclose(d.coded_df().to_numpy(), d.coded)
    with pytest.raises(yp.DesignError):
        yp.screening.plackett_burman(fs, runs=8, center_points=-1)


# ---------------------------------------------------------------- randomisation
def test_randomisation_seeded_and_std_order_kept():
    a = yp.screening.plackett_burman(make_factors(5), runs=12, randomize=True, seed=42)
    b = yp.screening.plackett_burman(make_factors(5), runs=12, randomize=True, seed=42)
    c = yp.screening.plackett_burman(make_factors(5), runs=12, randomize=True, seed=43)
    assert np.array_equal(a.run_order, b.run_order)
    assert not np.array_equal(a.run_order, c.run_order)
    assert np.array_equal(a.coded, c.coded)                       # design rows untouched
    assert sorted(a.run_order) == list(range(1, 13))
    assert not np.array_equal(a.run_order, np.arange(1, 13))
    assert a.info["seed"] == 42 and a.info["randomized"]
    assert np.array_equal(a.to_dataframe(order="std")["StdOrder"], np.arange(1, 13))
    assert np.array_equal(a.to_dataframe(order="run")["RunOrder"], np.arange(1, 13))
    plain = yp.screening.plackett_burman(make_factors(5), runs=12)
    assert np.array_equal(plain.run_order, np.arange(1, 13))
    d = yp.screening.plackett_burman(make_factors(5), runs=12, randomize=True)  # no seed: recorded
    assert isinstance(d.info["seed"], int)
    e = yp.screening.plackett_burman(make_factors(5), runs=12, randomize=True, seed=d.info["seed"])
    assert np.array_equal(d.run_order, e.run_order)


# ---------------------------------------------------------------- responses
def test_multiple_responses_and_run_order_alignment():
    d = yp.screening.plackett_burman(make_factors(3), runs=4, randomize=True, seed=1)
    d.add_response("Y1", [1, 2, 3, 4])
    d.add_response("Y2", [10, 20, 30, 40], order="run")       # given in execution order
    assert d.response_names == ["Y1", "Y2"]
    y2 = d.get_response("Y2")
    for std_i, ro in enumerate(d.run_order):                  # value at run position ro
        assert y2[std_i] == 10 * ro
    with pytest.raises(yp.DesignError):
        d.add_response("Y3", [1, 2, 3])
    with pytest.raises(yp.DesignError):
        d.add_response("X1", [1, 2, 3, 4])                    # collides with factor name
    with pytest.raises(yp.DesignError):
        d.add_response("Y1", [1, 2, 3, 4], overwrite=False)
    with pytest.raises(yp.DesignError):
        d.add_response("bad", [1, 2, np.inf, 4])
    with pytest.raises(yp.DesignError):
        d.add_response("bad", ["a", "b", "c", "d"])


def test_validate_reports_missing():
    d = yp.screening.plackett_burman(make_factors(3), runs=4)
    d.add_response("Y", [1, None, 3, 4])
    issues = d.validate()
    assert any("missing" in s for s in issues)
    with pytest.raises(yp.DesignError):
        d.validate(raise_on_error=True)
    d.add_response("Y", [1, 2, 3, 4])
    assert d.validate() == []


# ---------------------------------------------------------------- CSV round trip
def test_csv_roundtrip(tmp_path):
    fs = [yp.Factor("A", 0.1, 0.3, unit="g"), yp.Factor("B", 10, 20),
          yp.Factor("Cat", "old", "new", factor_type="categorical")]
    d = yp.screening.plackett_burman(fs, runs=8, randomize=True, seed=7)
    d.add_response("Yield", np.arange(8) * 1.5)
    d.add_response("Purity", [1, 2, np.nan, 4, 5, 6, 7, 8])
    p = d.to_csv(tmp_path / "design.csv")
    df = pd.read_csv(p, skiprows=1)
    assert list(df["RunOrder"]) == list(range(1, 9))          # exported in execution order
    d2 = yp.ExperimentDataset.from_csv(p)
    assert np.allclose(d2.coded, d.coded)
    assert np.array_equal(d2.run_order, d.run_order)
    assert np.allclose(d2.dummy_coded, d.dummy_coded)
    assert d2.factor_names == d.factor_names and d2.factors[0].unit == "g"
    assert d2.factors[2].factor_type == "categorical"
    assert np.allclose(d2.get_response("Yield"), d.get_response("Yield"))
    assert np.isnan(d2.get_response("Purity")[2])
    assert d2.info["seed"] == 7 and d2.info["design_type"] == "plackett_burman"
    assert list(d2.actual_df()["Cat"]) == list(d.actual_df()["Cat"])


def test_csv_inconsistent_or_headerless(tmp_path):
    d = yp.screening.plackett_burman([yp.Factor("A", 0, 10), yp.Factor("B", 0, 10)], runs=4)
    p = d.to_csv(tmp_path / "d.csv")
    txt = p.read_text().splitlines()
    # tamper with a coded value -> must be detected
    hdr = txt[1].split(",")
    j = hdr.index("A[coded]")
    row = txt[2].split(",")
    row[j] = "0.5"
    txt[2] = ",".join(row)
    bad = tmp_path / "bad.csv"
    bad.write_text("\n".join(txt))
    with pytest.raises(yp.DesignError, match="disagree"):
        yp.ExperimentDataset.from_csv(bad)
    # header-less CSV (plain table) needs explicit factors
    plain = tmp_path / "plain.csv"
    d.to_dataframe(order="std").to_csv(plain, index=False)
    with pytest.raises(yp.DesignError):
        yp.ExperimentDataset.from_csv(plain)
    d3 = yp.ExperimentDataset.from_csv(plain, factors=d.factors)
    assert np.allclose(d3.coded, d.coded)
    miss = tmp_path / "miss.csv"
    pd.DataFrame({"StdOrder": [1, 2]}).to_csv(miss, index=False)
    with pytest.raises(yp.DesignError, match="missing"):
        yp.ExperimentDataset.from_csv(miss, factors=d.factors)
