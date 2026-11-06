"""Factor definition with coded <-> actual value conversion.

Two-level coding (numeric factors)::

    coded  = (x - (high + low)/2) / ((high - low)/2)     # low -> -1, high -> +1
    x      = (high + low)/2 + coded * (high - low)/2     # center -> 0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..exceptions import DesignError

_TYPES = ("numeric", "categorical")


@dataclass
class Factor:
    name: str
    low: Any
    high: Any
    unit: str | None = None
    factor_type: str = "numeric"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DesignError("Factor name must be a non-empty string.")
        if self.factor_type not in _TYPES:
            raise DesignError(f"factor_type must be one of {_TYPES}, got {self.factor_type!r}.")
        if self.factor_type == "numeric":
            try:
                self.low, self.high = float(self.low), float(self.high)
            except (TypeError, ValueError) as e:
                raise DesignError(f"Factor {self.name!r}: low/high must be numeric.") from e
            if not (np.isfinite(self.low) and np.isfinite(self.high)):
                raise DesignError(f"Factor {self.name!r}: low/high must be finite.")
            if not self.low < self.high:
                raise DesignError(
                    f"Factor {self.name!r}: require low < high (got {self.low}, {self.high})."
                )
        else:
            if self.low == self.high:
                raise DesignError(f"Factor {self.name!r}: the two levels must differ.")

    # ------------------------------------------------------------------ props
    @property
    def is_numeric(self) -> bool:
        return self.factor_type == "numeric"

    @property
    def center(self) -> float:
        if not self.is_numeric:
            raise DesignError(f"Categorical factor {self.name!r} has no center level.")
        return (self.high + self.low) / 2.0

    @property
    def half_range(self) -> float:
        if not self.is_numeric:
            raise DesignError(f"Categorical factor {self.name!r} has no numeric range.")
        return (self.high - self.low) / 2.0

    # ------------------------------------------------------------- conversion
    def to_coded(self, x):
        """Actual value(s) -> coded value(s) in [-1, +1] (extrapolation allowed for numeric)."""
        if self.is_numeric:
            return (np.asarray(x, dtype=float) - self.center) / self.half_range
        arr = np.atleast_1d(np.asarray(x, dtype=object))
        out = np.empty(arr.shape, dtype=float)
        for i, v in enumerate(arr):
            if v == self.low:
                out[i] = -1.0
            elif v == self.high:
                out[i] = 1.0
            else:
                raise DesignError(
                    f"Factor {self.name!r}: value {v!r} is not one of the levels "
                    f"({self.low!r}, {self.high!r})."
                )
        return out if np.ndim(x) else out[0]

    def from_coded(self, c):
        """Coded value(s) -> actual value(s)."""
        if self.is_numeric:
            return self.center + np.asarray(c, dtype=float) * self.half_range
        arr = np.atleast_1d(np.asarray(c, dtype=float))
        out = np.empty(arr.shape, dtype=object)
        for i, v in enumerate(arr):
            if np.isclose(v, -1.0):
                out[i] = self.low
            elif np.isclose(v, 1.0):
                out[i] = self.high
            else:
                raise DesignError(
                    f"Categorical factor {self.name!r} only has coded levels -1/+1 (got {v})."
                )
        return out if np.ndim(c) else out[0]

    # -------------------------------------------------------------- serialise
    def to_dict(self) -> dict:
        return {"name": self.name, "low": self.low, "high": self.high, "unit": self.unit,
                "factor_type": self.factor_type, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, d: dict) -> "Factor":
        return cls(**d)

    def __repr__(self) -> str:
        u = f" {self.unit}" if self.unit else ""
        return f"Factor({self.name!r}, {self.low}..{self.high}{u}, {self.factor_type})"


@dataclass
class Response:
    """A measured/simulated output.  `weight` and `r2_threshold` are used by later stages
    (aggregation / stopping), `direction` by optimisation."""
    name: str
    weight: float = 1.0
    r2_threshold: float | None = None
    direction: str | None = None          # None | 'maximize' | 'minimize' | 'target'
    target: float | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DesignError("Response name must be a non-empty string.")
        if not (np.isfinite(self.weight) and self.weight >= 0):
            raise DesignError(f"Response {self.name!r}: weight must be finite and >= 0.")
        if self.direction not in (None, "maximize", "minimize", "target"):
            raise DesignError(f"Response {self.name!r}: invalid direction {self.direction!r}.")
        if self.direction == "target" and self.target is None:
            raise DesignError(f"Response {self.name!r}: direction='target' needs `target`.")
        if self.r2_threshold is not None and not self.r2_threshold <= 1.0:
            raise DesignError(f"Response {self.name!r}: r2_threshold must be <= 1.")
