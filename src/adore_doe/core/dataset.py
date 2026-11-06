"""ExperimentDataset: the single hand-over object between modules (ADR-005).

* one coded design matrix X (n x k, coded in [-1, +1] unless `info['allow_outside']`), shared
  by EVERY response; responses are separate columns (NaN = not measured yet);
* `std_order` (= run_id, 1..n) identifies a row and never changes; `run_order` is the
  execution position (a permutation of 1..n);
* `source` labels each run ('initial', 'AL-iter-3', 'validation', ...), `stage` is an int;
* `point_type` ('factorial', 'center', 'axial', 'design', ...);
* `info` carries design metadata (design type, seed, model spec, ...).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..exceptions import DesignError
from .factor import Factor, Response

_RESERVED = {"StdOrder", "RunOrder", "PointType", "Source", "Stage"}
_META_PREFIX = "# adore_doe-dataset "


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


@dataclass
class Experiment:
    """Read-only view of one run."""
    run_id: int
    run_order: int
    coded: dict
    actual: dict
    responses: dict
    point_type: str
    source: str
    stage: int


class ExperimentDataset:
    def __init__(self, factors, coded, point_type=None, run_order=None, dummy=None,
                 dummy_names=None, responses=None, info=None, source=None, stage=None):
        self.factors: list[Factor] = list(factors)
        names = [f.name for f in self.factors]
        if not names:
            raise DesignError("At least one factor is required.")
        if len(set(names)) != len(names):
            raise DesignError("Duplicate factor names.")
        bad = [n for n in names if n in _RESERVED or n.endswith("[coded]") or n.startswith("dummy:")]
        if bad:
            raise DesignError(f"Reserved factor name(s): {bad}")
        self._coded = np.asarray(coded, dtype=float)
        if self._coded.ndim != 2 or self._coded.shape[1] != len(self.factors):
            raise DesignError("coded must have shape (n_runs, n_factors).")
        n = self._coded.shape[0]
        self.info: dict = dict(info or {})
        self.point_type = (np.array(["factorial"] * n, dtype=object) if point_type is None
                           else np.asarray(point_type, dtype=object))
        self.run_order = np.arange(1, n + 1) if run_order is None else np.asarray(run_order, dtype=int)
        if sorted(self.run_order.tolist()) != list(range(1, n + 1)):
            raise DesignError("run_order must be a permutation of 1..n_runs.")
        self.source = (np.array([self.info.get("source", "initial")] * n, dtype=object)
                       if source is None else np.asarray(source, dtype=object))
        self.stage = np.zeros(n, dtype=int) if stage is None else np.asarray(stage, dtype=int)
        if not (len(self.point_type) == len(self.source) == len(self.stage) == n):
            raise DesignError("point_type/source/stage must have one entry per run.")
        self._dummy = None if dummy is None else np.asarray(dummy, dtype=float)
        self.dummy_names = list(dummy_names or ([] if dummy is None else
                                                [f"d{i+1}" for i in range(self._dummy.shape[1])]))
        self._responses: dict[str, np.ndarray] = {}
        self.response_specs: dict[str, Response] = {}
        for k, v in (responses or {}).items():
            self.add_response(k, v)

    # ------------------------------------------------------------------ info
    @property
    def n_runs(self) -> int:
        return self._coded.shape[0]

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    @property
    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    @property
    def std_order(self) -> np.ndarray:
        return np.arange(1, self.n_runs + 1)

    run_id = std_order

    @property
    def center_points(self) -> int:
        return int(np.sum(self.point_type == "center"))

    @property
    def coded(self) -> np.ndarray:
        return self._coded.copy()

    @property
    def dummy_coded(self) -> np.ndarray:
        return np.zeros((self.n_runs, 0)) if self._dummy is None else self._dummy.copy()

    @property
    def response_names(self) -> list[str]:
        return list(self._responses)

    def factor(self, name: str) -> Factor:
        for f in self.factors:
            if f.name == name:
                return f
        raise KeyError(name)

    # ------------------------------------------------------- sources / roles
    @property
    def validation_mask(self) -> np.ndarray:
        return np.asarray(self.source) == "validation"

    @property
    def training_mask(self) -> np.ndarray:
        return ~self.validation_mask

    def training(self) -> "ExperimentDataset":
        return self.subset(self.training_mask)

    def validation(self) -> "ExperimentDataset":
        return self.subset(self.validation_mask)

    # ----------------------------------------------------------- replication
    def replicate_groups(self, decimals: int = 9) -> np.ndarray:
        """Group id per run; runs with identical coded settings share an id (replicates)."""
        keys = [tuple(np.round(r, decimals) + 0.0) for r in self._coded]
        ids: dict = {}
        return np.array([ids.setdefault(k, len(ids)) for k in keys])

    @property
    def n_unique_points(self) -> int:
        return int(len(set(self.replicate_groups())))

    @property
    def n_replicated_runs(self) -> int:
        """Runs that repeat an already-used setting (n_runs - n_unique_points)."""
        return self.n_runs - self.n_unique_points

    # ------------------------------------------------------------ dataframes
    def _row_index(self, order: str) -> np.ndarray:
        if order == "std":
            return np.arange(self.n_runs)
        if order == "run":
            return np.argsort(self.run_order, kind="stable")
        raise ValueError("order must be 'std' or 'run'.")

    def coded_df(self, order: str = "std") -> pd.DataFrame:
        idx = self._row_index(order)
        return pd.DataFrame(self._coded[idx], columns=self.factor_names, index=self.std_order[idx])

    def actual_df(self, order: str = "std") -> pd.DataFrame:
        idx = self._row_index(order)
        cols = {f.name: f.from_coded(self._coded[:, j]) for j, f in enumerate(self.factors)}
        return pd.DataFrame(cols, index=self.std_order).iloc[idx]

    def to_dataframe(self, order: str = "std", coded: bool = False,
                     include_dummy: bool = False) -> pd.DataFrame:
        idx = self._row_index(order)
        out = pd.DataFrame({"StdOrder": self.std_order[idx], "RunOrder": self.run_order[idx],
                            "PointType": self.point_type[idx], "Source": self.source[idx],
                            "Stage": self.stage[idx]})
        body = (self.coded_df(order) if coded else self.actual_df(order)).reset_index(drop=True)
        out = pd.concat([out, body], axis=1)
        if include_dummy and self._dummy is not None:
            for j, nm in enumerate(self.dummy_names):
                out[f"dummy:{nm}"] = self._dummy[idx, j]
        for k, v in self._responses.items():
            out[k] = v[idx]
        return out

    def experiment(self, run_id: int) -> Experiment:
        i = int(run_id) - 1
        if not 0 <= i < self.n_runs:
            raise IndexError(run_id)
        act = self.actual_df().iloc[i]
        return Experiment(int(run_id), int(self.run_order[i]),
                          dict(zip(self.factor_names, self._coded[i])), act.to_dict(),
                          {k: float(v[i]) for k, v in self._responses.items()},
                          str(self.point_type[i]), str(self.source[i]), int(self.stage[i]))

    # ------------------------------------------------------------ responses
    @staticmethod
    def _check_rname(name):
        if not isinstance(name, str) or not name:
            raise DesignError("Response name must be a non-empty string.")
        if name in _RESERVED or name.endswith("[coded]") or name.startswith("dummy:"):
            raise DesignError(f"Response name {name!r} is reserved.")

    def add_response(self, name, values, order: str = "std", overwrite: bool = True,
                     spec: Response | None = None):
        """Add/update a response (aligned with std order, or execution order if order='run').
        None/NaN = not measured yet.  All responses share the same X."""
        self._check_rname(name)
        if name in self.factor_names:
            raise DesignError(f"Response name {name!r} collides with a factor name.")
        if name in self._responses and not overwrite:
            raise DesignError(f"Response {name!r} already exists (overwrite=False).")
        try:
            v = np.array([np.nan if x is None else x for x in values], dtype=float)
        except (TypeError, ValueError) as e:
            raise DesignError(f"Response {name!r} contains non-numeric values.") from e
        if v.shape != (self.n_runs,):
            raise DesignError(f"Response {name!r} has {v.size} values, dataset has {self.n_runs} runs.")
        if np.any(np.isinf(v)):
            raise DesignError(f"Response {name!r} contains infinite values.")
        if order == "run":
            out = np.empty_like(v)
            out[np.argsort(self.run_order, kind="stable")] = v
            v = out
        elif order != "std":
            raise ValueError("order must be 'std' or 'run'.")
        self._responses[name] = v
        if spec is not None:
            if spec.name != name:
                raise DesignError("spec.name must equal the response name.")
            self.response_specs[name] = spec
        else:
            self.response_specs.setdefault(name, Response(name))
        return self

    def get_response(self, name: str) -> np.ndarray:
        if name not in self._responses:
            raise KeyError(f"No response {name!r}; available: {self.response_names}")
        return self._responses[name].copy()

    def drop_response(self, name: str):
        self._responses.pop(name)
        self.response_specs.pop(name, None)
        return self

    def complete_mask(self, names=None) -> np.ndarray:
        """Runs for which all given responses (default: all) have been measured."""
        names = self.response_names if names is None else list(names)
        if not names:
            return np.zeros(self.n_runs, dtype=bool)
        return np.all([~np.isnan(self._responses[n]) for n in names], axis=0)

    # ---------------------------------------------------------- growing / copying
    def append_runs(self, coded, source: str = "sequential", stage: int | None = None,
                    point_type: str = "design", values: dict | None = None) -> np.ndarray:
        """Append new runs (executed after the existing ones).  `values` maps response name ->
        values for the new rows (missing responses become NaN).  Returns the new run_ids."""
        if self._dummy is not None:
            raise DesignError("Cannot append runs to a design with dummy columns (screening design).")
        c = np.atleast_2d(np.asarray(coded, dtype=float))
        if c.ndim != 2 or c.shape[1] != self.n_factors:
            raise DesignError(f"coded must have {self.n_factors} columns.")
        if not np.all(np.isfinite(c)):
            raise DesignError("coded contains NaN/inf.")
        m, n0 = c.shape[0], self.n_runs
        values = values or {}
        for k, v in values.items():
            if k not in self._responses:
                raise DesignError(f"Unknown response {k!r}; add it with add_response first.")
            if len(v) != m:
                raise DesignError(f"values[{k!r}] must have {m} entries.")
        self._coded = np.vstack([self._coded, c])
        self.point_type = np.concatenate([self.point_type, np.array([point_type] * m, dtype=object)])
        self.source = np.concatenate([self.source, np.array([source] * m, dtype=object)])
        st = (int(self.stage.max()) + 1) if stage is None else int(stage)
        self.stage = np.concatenate([self.stage, np.full(m, st, dtype=int)])
        self.run_order = np.concatenate([self.run_order, np.arange(n0 + 1, n0 + m + 1)])
        for k in self._responses:
            new = np.array([np.nan if x is None else x for x in values[k]], float) if k in values \
                else np.full(m, np.nan)
            self._responses[k] = np.concatenate([self._responses[k], new])
        return np.arange(n0 + 1, n0 + m + 1)

    def subset(self, mask) -> "ExperimentDataset":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_runs,):
            raise ValueError("mask must have one entry per run.")
        idx = np.where(mask)[0]
        ro = np.argsort(np.argsort(self.run_order[idx], kind="stable"), kind="stable") + 1
        d = ExperimentDataset(
            [Factor.from_dict(f.to_dict()) for f in self.factors], self._coded[idx],
            point_type=self.point_type[idx], run_order=ro,
            dummy=None if self._dummy is None else self._dummy[idx], dummy_names=self.dummy_names,
            info=json.loads(json.dumps(self.info, default=_jsonable)),
            source=self.source[idx], stage=self.stage[idx])
        for k, v in self._responses.items():
            d.add_response(k, v[idx], spec=self.response_specs.get(k))
        return d

    def copy(self) -> "ExperimentDataset":
        return self.subset(np.ones(self.n_runs, dtype=bool))

    # --------------------------------------------------------- randomising
    def randomize(self, seed: int | None = None):
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**32))
        self.run_order = np.random.default_rng(seed).permutation(self.n_runs) + 1
        self.info["randomized"] = True
        self.info["seed"] = int(seed)
        return self

    # ----------------------------------------------------------- validation
    def validate(self, raise_on_error: bool = False) -> list[str]:
        issues: list[str] = []
        c = self._coded
        if not np.all(np.isfinite(c)):
            issues.append("Coded matrix contains NaN/inf.")
        if not self.info.get("allow_outside") and np.any(np.abs(c) > 1 + 1e-9):
            issues.append("Coded values outside [-1, 1] found.")
        for j, f in enumerate(self.factors):
            if f.is_numeric and not f.low < f.high:
                issues.append(f"Factor {f.name!r}: low >= high.")
            if not f.is_numeric and np.any(np.isclose(c[:, j], 0.0)):
                issues.append(f"Categorical factor {f.name!r} has center-level runs.")
        for k, v in self._responses.items():
            nmiss = int(np.isnan(v).sum())
            if nmiss:
                issues.append(f"Response {k!r}: {nmiss} missing value(s).")
        if raise_on_error and issues:
            raise DesignError("; ".join(issues))
        return issues

    # ------------------------------------------------------------------ CSV
    def to_csv(self, path, order: str = "run", include_coded: bool = True) -> Path:
        path = Path(path)
        df = self.to_dataframe(order=order, coded=False)
        idx = self._row_index(order)
        if include_coded:
            cd = self.coded_df(order).reset_index(drop=True)
            for nm in self.factor_names:
                df[f"{nm}[coded]"] = cd[nm].to_numpy()
        if self._dummy is not None:
            for j, nm in enumerate(self.dummy_names):
                df[f"dummy:{nm}"] = self._dummy[idx, j]
        resp = self.response_names
        df = df[[c for c in df.columns if c not in resp] + resp]
        meta = {"factors": [f.to_dict() for f in self.factors], "info": self.info,
                "dummy_names": self.dummy_names, "responses": resp,
                "response_specs": {k: vars(v) for k, v in self.response_specs.items()}}
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(_META_PREFIX + json.dumps(meta, default=_jsonable) + "\n")
            df.to_csv(fh, index=False)
        return path

    @classmethod
    def from_csv(cls, path, factors: list[Factor] | None = None) -> "ExperimentDataset":
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            first = fh.readline()
        meta, skip = None, 0
        if first.startswith(_META_PREFIX):
            meta, skip = json.loads(first[len(_META_PREFIX):]), 1
        if factors is None:
            if meta is None:
                raise DesignError("CSV has no adore_doe header; pass `factors=` explicitly.")
            factors = [Factor.from_dict(d) for d in meta["factors"]]
        df = pd.read_csv(path, skiprows=skip)
        need = {"StdOrder", "RunOrder", "PointType"} | {f.name for f in factors}
        miss = need - set(df.columns)
        if miss:
            raise DesignError(f"CSV is missing required column(s): {sorted(miss)}")
        if df["StdOrder"].duplicated().any():
            raise DesignError("Duplicate StdOrder values in CSV.")
        df = df.sort_values("StdOrder").reset_index(drop=True)
        if sorted(df["StdOrder"].tolist()) != list(range(1, len(df) + 1)):
            raise DesignError("StdOrder must be 1..N.")
        coded = np.empty((len(df), len(factors)))
        for j, f in enumerate(factors):
            col = f"{f.name}[coded]"
            from_actual = np.asarray(f.to_coded(df[f.name].to_numpy()), dtype=float)
            if col in df.columns:
                coded[:, j] = df[col].to_numpy(dtype=float)
                if not np.allclose(coded[:, j], from_actual, atol=1e-8):
                    raise DesignError(f"Factor {f.name!r}: actual and coded columns disagree.")
            else:
                coded[:, j] = from_actual
        dnames = [c[len("dummy:"):] for c in df.columns if c.startswith("dummy:")]
        dummy = df[[f"dummy:{n}" for n in dnames]].to_numpy(dtype=float) if dnames else None
        known = need | {"Source", "Stage"} | {c for c in df.columns
                                              if c.endswith("[coded]") or c.startswith("dummy:")}
        resp_cols = (meta or {}).get("responses") or [c for c in df.columns if c not in known]
        ds = cls(factors, coded, point_type=df["PointType"].to_numpy(dtype=object),
                 run_order=df["RunOrder"].to_numpy(dtype=int), dummy=dummy, dummy_names=dnames,
                 info=(meta or {}).get("info", {}),
                 source=df["Source"].to_numpy(dtype=object) if "Source" in df else None,
                 stage=df["Stage"].to_numpy(dtype=int) if "Stage" in df else None)
        specs = (meta or {}).get("response_specs", {})
        for r in resp_cols:
            if r in df.columns:
                ds.add_response(r, df[r].to_numpy(dtype=float),
                                spec=Response(**specs[r]) if r in specs else None)
        return ds

    # ----------------------------------------------------------------- repr
    def summary(self) -> str:
        i = self.info
        lines = [f"ExperimentDataset: {i.get('design_type', 'custom')}",
                 f"  runs={self.n_runs}  unique_points={self.n_unique_points}  "
                 f"replicated_runs={self.n_replicated_runs}  factors={self.n_factors}  "
                 f"center_points={self.center_points}  dummy_columns={len(self.dummy_names)}",
                 f"  randomized={i.get('randomized', False)}  seed={i.get('seed')}",
                 f"  responses={self.response_names}"]
        lines += [f"  - {f!r}" for f in self.factors]
        if i.get("aliasing_note"):
            lines.append(f"  note: {i['aliasing_note']}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<ExperimentDataset {self.info.get('design_type', 'custom')} runs={self.n_runs} "
                f"factors={self.n_factors} responses={self.response_names}>")

    def __len__(self) -> int:
        return self.n_runs
