import numpy as np
import pandas as pd
import pytest

import adore_doe as ad
from adore_doe.core import Experiment, ExperimentDataset


def facs(k=3):
    return [ad.Factor(f"X{i+1}", 10 * i, 10 * i + 20, unit="u") for i in range(k)]


def base(n=6, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return ExperimentDataset(facs(k), rng.uniform(-1, 1, (n, k)))


# ---------------------------------------------------------------- coding / validation
def test_coding_roundtrip_exact_and_bounds():
    f = ad.Factor("T", 60, 100)
    x = np.linspace(60, 100, 9)
    c = f.to_coded(x)
    assert c[0] == -1 and c[-1] == 1 and c[4] == 0
    assert np.allclose(f.from_coded(c), x, rtol=0, atol=1e-12)
    g = ad.Factor("S", 0.1, 0.3)
    assert np.allclose(g.from_coded(g.to_coded(np.array([0.1, 0.13, 0.3]))), [0.1, 0.13, 0.3], atol=1e-15)


def test_response_validation():
    r = ad.Response("C", weight=2.0, r2_threshold=0.9, direction="minimize")
    assert r.weight == 2.0
    for bad in (dict(name=""), dict(name="C", weight=-1), dict(name="C", direction="up"),
                dict(name="C", direction="target"), dict(name="C", r2_threshold=1.5)):
        with pytest.raises(ad.DesignError):
            ad.Response(**bad)


def test_dataset_input_validation():
    with pytest.raises(ad.DesignError):
        ExperimentDataset([ad.Factor("A", 0, 1), ad.Factor("A", 0, 2)], np.zeros((3, 2)))
    with pytest.raises(ad.DesignError):
        ExperimentDataset(facs(2), np.zeros((3, 3)))
    with pytest.raises(ad.DesignError):
        ExperimentDataset([ad.Factor("StdOrder", 0, 1)], np.zeros((3, 1)))
    with pytest.raises(ad.DesignError):
        ExperimentDataset([], np.zeros((3, 0)))
    with pytest.raises(ad.DesignError):
        ExperimentDataset(facs(2), np.zeros((3, 2)), run_order=[1, 1, 2])


# ---------------------------------------------------------------- responses share X
def test_all_responses_share_the_same_X():
    d = base()
    d.add_response("C", np.arange(6.0))
    d.add_response("R", np.arange(6.0) ** 2)
    assert d.response_names == ["C", "R"]
    assert d.coded.shape == (6, 3)                       # one X for every response
    df = d.to_dataframe()
    assert list(df.columns[-2:]) == ["C", "R"] and len(df) == 6
    with pytest.raises(ad.DesignError):
        d.add_response("bad", [1, 2, 3])                  # wrong length: cannot have its own X
    d.add_response("R2", [1, None, 3, 4, 5, 6])
    assert list(d.complete_mask(["C", "R"])) == [True] * 6
    assert list(d.complete_mask()) == [True, False, True, True, True, True]
    assert d.get_response("R2")[1] != d.get_response("R2")[1]       # NaN kept as NaN


def test_append_runs_keeps_shared_X_and_alignment():
    d = base(4)
    d.add_response("C", [1, 2, 3, 4])
    d.add_response("R", [5, 6, 7, 8])
    ids = d.append_runs([[0, 0, 0], [0.5, 0.5, 0.5]], source="AL-iter-1", values={"C": [9, 10]})
    assert list(ids) == [5, 6] and d.n_runs == 6
    assert list(d.source[-2:]) == ["AL-iter-1"] * 2 and d.stage[-1] == 1
    assert np.isnan(d.get_response("R")[4:]).all() and list(d.get_response("C")[4:]) == [9, 10]
    assert list(d.run_order) == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ad.DesignError):
        d.append_runs([[0, 0]])
    with pytest.raises(ad.DesignError):
        d.append_runs([[0, 0, 0]], values={"nope": [1]})


def test_replicates_and_unique_points():
    c = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0], [-1, 0, 1]], float)
    d = ExperimentDataset(facs(), c)
    assert d.n_unique_points == 3 and d.n_replicated_runs == 2
    g = d.replicate_groups()
    assert g[0] == g[2] == g[3] and len(set(g)) == 3


def test_run_order_vs_std_order_and_experiment_view():
    d = base(5).randomize(seed=3)
    d.add_response("C", [10, 20, 30, 40, 50])
    run_tbl = d.to_dataframe(order="run")
    assert list(run_tbl["RunOrder"]) == [1, 2, 3, 4, 5]
    assert sorted(run_tbl["StdOrder"]) == [1, 2, 3, 4, 5]
    e = d.experiment(2)
    assert isinstance(e, Experiment) and e.run_id == 2 and e.responses["C"] == 20
    assert e.run_order == d.run_order[1]
    assert np.isclose(e.actual["X1"], d.factors[0].from_coded(e.coded["X1"]))
    with pytest.raises(IndexError):
        d.experiment(9)


def test_validation_source_split():
    d = base(5)
    d.append_runs(np.zeros((2, 3)), source="validation")
    assert d.training_mask.sum() == 5 and d.validation().n_runs == 2
    assert d.training().n_runs == 5


def test_subset_copy_independent():
    d = base(5)
    d.add_response("C", np.arange(5.0))
    s = d.subset(np.array([True, False, True, False, True]))
    assert s.n_runs == 3 and list(s.get_response("C")) == [0, 2, 4]
    assert sorted(s.run_order) == [1, 2, 3]
    c = d.copy()
    c.add_response("C", np.zeros(5))
    assert list(d.get_response("C")) == [0, 1, 2, 3, 4]


def test_csv_roundtrip_with_sources_and_specs(tmp_path):
    d = base(5, seed=1).randomize(seed=2)
    d.add_response("C", np.arange(5.0), spec=ad.Response("C", weight=0.7, r2_threshold=0.9,
                                                         direction="maximize"))
    d.add_response("R", [1, np.nan, 3, 4, 5])
    d.append_runs(np.zeros((1, 3)), source="validation", values={"C": [9.0]})
    p = d.to_csv(tmp_path / "ds.csv")
    d2 = ExperimentDataset.from_csv(p)
    assert np.allclose(d2.coded, d.coded) and np.array_equal(d2.run_order, d.run_order)
    assert list(d2.source) == list(d.source) and list(d2.stage) == list(d.stage)
    assert np.allclose(d2.get_response("C"), d.get_response("C"), equal_nan=True)
    assert np.isnan(d2.get_response("R")[1])
    assert d2.response_specs["C"].weight == 0.7 and d2.response_specs["C"].direction == "maximize"
    assert d2.factors[0].unit == "u"


def test_csv_errors(tmp_path):
    d = base(4)
    p = d.to_csv(tmp_path / "d.csv")
    lines = p.read_text().splitlines()
    hdr = lines[1].split(",")
    row = lines[2].split(",")
    row[hdr.index("X1[coded]")] = "0.9999"
    lines[2] = ",".join(row)
    bad = tmp_path / "bad.csv"
    bad.write_text("\n".join(lines))
    with pytest.raises(ad.DesignError, match="disagree"):
        ExperimentDataset.from_csv(bad)
    plain = tmp_path / "plain.csv"
    d.to_dataframe().to_csv(plain, index=False)
    with pytest.raises(ad.DesignError):
        ExperimentDataset.from_csv(plain)
    assert np.allclose(ExperimentDataset.from_csv(plain, factors=d.factors).coded, d.coded)


def test_validate_flags_outside_and_missing():
    d = ExperimentDataset(facs(2), [[0.0, 2.0], [0, 0]])
    assert any("outside" in s for s in d.validate())
    d.info["allow_outside"] = True
    assert d.validate() == []
    d.add_response("C", [1, None])
    assert any("missing" in s for s in d.validate())
    with pytest.raises(ad.DesignError):
        d.validate(raise_on_error=True)
