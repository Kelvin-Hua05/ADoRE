"""Model matrix specification f(x) in CODED coordinates (research_methodology §3, §4.2 row 1).

kind='linear'      : 1, x_i
kind='interaction' : 1, x_i, x_i x_j (i<j)
kind='quadratic'   : 1, x_i, x_i x_j (i<j), x_i^2          p = (k+1)(k+2)/2
Each term is stored as a tuple of exponents (one per factor); the design-matrix column of a
term is prod_j z_j ** e_j.  The order of `terms` is the order of the columns of X.
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field

import numpy as np

from ..exceptions import DesignError

KINDS = ("linear", "interaction", "quadratic")


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    k: int
    terms: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise DesignError(f"model must be one of {KINDS}, got {self.kind!r}.")
        if not isinstance(self.k, (int, np.integer)) or self.k < 1:
            raise DesignError("k (number of factors) must be a positive integer.")
        k = int(self.k)
        zero = (0,) * k
        t = [zero]
        for i in range(k):
            t.append(tuple(1 if j == i else 0 for j in range(k)))
        if self.kind in ("interaction", "quadratic"):
            for i in range(k):
                for j in range(i + 1, k):
                    t.append(tuple(1 if m in (i, j) else 0 for m in range(k)))
        if self.kind == "quadratic":
            for i in range(k):
                t.append(tuple(2 if j == i else 0 for j in range(k)))
        object.__setattr__(self, "terms", tuple(t))

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    def names(self, factor_names=None) -> list[str]:
        fn = list(factor_names) if factor_names is not None else list(string.ascii_uppercase[: self.k])
        if len(fn) != self.k:
            raise DesignError("factor_names must have k entries.")
        out = []
        for e in self.terms:
            parts = [f"{fn[j]}" if p == 1 else f"{fn[j]}^{p}" for j, p in enumerate(e) if p]
            out.append("*".join(parts) if parts else "Intercept")
        return out

    def expand(self, Z) -> np.ndarray:
        """Coded points (n x k) -> model matrix (n x p)."""
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if Z.shape[1] != self.k:
            raise DesignError(f"points must have {self.k} columns, got {Z.shape[1]}.")
        cols = [np.prod(np.power(Z, np.array(e)), axis=1) for e in self.terms]
        return np.column_stack(cols)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "k": int(self.k), "terms": [list(e) for e in self.terms]}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        m = cls(d["kind"], d["k"])
        if [list(e) for e in m.terms] != [list(e) for e in d["terms"]]:
            raise DesignError("model terms do not match the stored specification.")
        return m


def as_model(model, k: int) -> ModelSpec:
    if isinstance(model, ModelSpec):
        if model.k != k:
            raise DesignError(f"ModelSpec has k={model.k} but {k} factors were given.")
        return model
    return ModelSpec(model, k)
