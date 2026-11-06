"""Two-level (fractional) factorial designs.

Fractional factorial 2^(k-p)
----------------------------
Base factors A.. (the first k-p factors) form a full 2^(k-p) factorial in standard (Yates)
order (first factor varies fastest).  Each remaining factor is generated as a product of base
columns ("E=ABCD").  Each generator gives a defining word (e.g. ABCDE); the defining relation
I = {all products of the words} is computed explicitly; resolution = shortest word length and
the alias structure is derived from it (nothing is looked up).

Because a main-effects-only analysis pools every non-modelled contrast into the error, the
remaining distinct contrast columns (interactions, labelled by their lowest-order alias) are
stored as `dummy` columns with `info['dummy_kind']='interaction'`: they feed the Lenth /
half-normal (Daniel) analysis and the error df.  If interactions are real this inflates sigma
-- see docs.
"""
from __future__ import annotations

import itertools
import string

import numpy as np

from ..core.dataset import ExperimentDataset
from ..core.factor import Factor
from ..exceptions import DesignError

# standard generator sets; the resolution is *computed*, these are only defaults.
DEFAULT_GENERATORS = {
    (3, 1): ["C=AB"], (4, 1): ["D=ABC"], (5, 1): ["E=ABCD"], (5, 2): ["D=AB", "E=AC"],
    (6, 1): ["F=ABCDE"], (6, 2): ["E=ABC", "F=BCD"], (6, 3): ["D=AB", "E=AC", "F=BC"],
    (7, 1): ["G=ABCDEF"], (7, 2): ["F=ABCD", "G=ABDE"], (7, 3): ["E=ABC", "F=BCD", "G=ACD"],
    (7, 4): ["D=AB", "E=AC", "F=BC", "G=ABC"], (8, 2): ["G=ABCD", "H=ABEF"],
    (8, 4): ["E=BCD", "F=ACD", "G=ABC", "H=ABD"],
}
_ROMAN = {2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}


def _letters(n: int) -> str:
    if n > 26:
        raise DesignError("At most 26 factors are supported for letter-coded generators.")
    return string.ascii_uppercase[:n]


def _parse_generators(k: int, gens: list[str]):
    p = len(gens)
    letters, base = _letters(k), k - p
    out = {}
    for g in gens:
        try:
            lhs, rhs = [s.strip().upper() for s in g.split("=")]
        except ValueError as e:
            raise DesignError(f"Bad generator {g!r}; use e.g. 'E=ABCD'.") from e
        if len(lhs) != 1 or lhs not in letters[base:]:
            raise DesignError(f"Generator {g!r}: left side must be one of {letters[base:]}.")
        if not rhs or any(c not in letters[:base] for c in rhs) or len(set(rhs)) != len(rhs):
            raise DesignError(f"Generator {g!r}: right side must use distinct base factors {letters[:base]}.")
        out[lhs] = rhs
    if sorted(out) != list(letters[base:]):
        raise DesignError(f"Generators must define exactly the factors {letters[base:]}.")
    return out


def _defining_relation(k, gmap):
    """All defining words (frozensets of factor indices), excluding identity."""
    letters = _letters(k)
    words = [frozenset([letters.index(l)] + [letters.index(c) for c in r]) for l, r in gmap.items()]
    rel = set()
    for r in range(1, len(words) + 1):
        for comb in itertools.combinations(words, r):
            w = frozenset()
            for x in comb:
                w = w.symmetric_difference(x)
            rel.add(w)
    return sorted(rel, key=lambda w: (len(w), sorted(w)))


def _word(s, letters):
    return "".join(letters[i] for i in sorted(s))


def resolution_of(k: int, generators: list[str]) -> int:
    rel = _defining_relation(k, _parse_generators(k, generators)) if generators else []
    return min((len(w) for w in rel), default=k + 1)


def alias_table(k: int, generators: list[str], max_order: int = 2) -> dict:
    """{effect: [aliased effects]} for all effects of order <= max_order (alias order <= max_order+2)."""
    letters = _letters(k)
    rel = _defining_relation(k, _parse_generators(k, generators)) if generators else []
    table = {}
    for r in range(1, max_order + 1):
        for comb in itertools.combinations(range(k), r):
            e = frozenset(comb)
            al = sorted({_word(e.symmetric_difference(w), letters) for w in rel},
                        key=lambda s: (len(s), s))
            table[_word(e, letters)] = al
    return table


def fractional_factorial(factors: list[Factor], p: int = 0, generators: list[str] | None = None,
                         center_points: int = 0, randomize: bool = False,
                         seed: int | None = None) -> ExperimentDataset:
    """2^(k-p) design with k = len(factors).  p=0 (default) is the full factorial.

    `generators` e.g. ["E=ABCD"]; if omitted and p>0 a standard set is used (error if none is
    tabulated for (k, p)).  Resolution and the alias structure are computed from the generators.
    """
    factors = list(factors)
    k = len(factors)
    if k < 2:
        raise DesignError("At least two factors are required.")
    if not all(isinstance(f, Factor) for f in factors):
        raise DesignError("`factors` must be Factor objects.")
    if len({f.name for f in factors}) != k:
        raise DesignError("Duplicate factor names.")
    if center_points and any(not f.is_numeric for f in factors):
        raise DesignError("Center points need numeric factors.")
    if not isinstance(center_points, (int, np.integer)) or center_points < 0:
        raise DesignError("center_points must be a non-negative integer.")
    if generators is not None:
        p = len(generators)
    if not isinstance(p, (int, np.integer)) or p < 0 or p >= k:
        raise DesignError(f"Invalid fraction p={p} for k={k} factors.")
    if p > 0 and generators is None:
        if (k, p) not in DEFAULT_GENERATORS:
            raise DesignError(f"No default generators for 2^({k}-{p}); pass `generators=`.")
        generators = DEFAULT_GENERATORS[(k, p)]
    base = k - p
    n = 2 ** base
    if n < k + 1:
        raise DesignError(f"2^({k}-{p}) has {n} runs: too few to estimate {k} main effects.")
    idx = np.arange(n)
    B = np.column_stack([np.where((idx >> j) & 1, 1.0, -1.0) for j in range(base)])
    gmap = _parse_generators(k, generators) if p else {}
    letters = _letters(k)
    cols = [B[:, j] for j in range(base)]
    for l in letters[base:]:
        cols.append(np.prod([B[:, letters.index(c)] for c in gmap[l]], axis=0))
    X = np.column_stack(cols)
    rel = _defining_relation(k, gmap) if p else []
    res = min((len(w) for w in rel), default=None)
    # interaction contrast columns (lowest-order representative of every distinct alias class)
    classes = []
    allsets = [frozenset(c) for r in range(1, k + 1) for c in itertools.combinations(range(k), r)]
    covered = set(rel)                      # defining words = identity class: not a contrast
    for e in allsets:                       # ascending order => lowest-order representative first
        if e in covered:
            continue
        cls = {e} | {e.symmetric_difference(w) for w in rel}
        covered |= cls
        classes.append(e)
    inter = [e for e in classes if len(e) >= 2]
    dummy = np.column_stack([np.prod(X[:, sorted(e)], axis=1) for e in inter]) if inter else None
    dnames = [_word(e, letters) for e in inter]
    coded = np.vstack([X, np.zeros((center_points, k))])
    ptype = np.array(["factorial"] * n + ["center"] * int(center_points), dtype=object)
    if dummy is not None:
        dummy = np.vstack([dummy, np.zeros((center_points, dummy.shape[1]))])
    note = ("Full factorial: all interactions estimable; main-effects analysis pools them into error."
            if p == 0 else
            f"Resolution {_ROMAN.get(res, res)}: " + {
                3: "main effects are aliased with two-factor interactions.",
                4: "main effects are clear of 2fi, but 2fi are aliased with each other.",
            }.get(res, "main effects and two-factor interactions are not aliased with each other (res>=V)."))
    info = {"design_type": "fractional_factorial" if p else "full_factorial", "k": k, "p": p,
            "generators": generators or [], "resolution": res, "factorial_runs": n,
            "total_runs": n + int(center_points), "center_points": int(center_points),
            "defining_relation": ["I"] + [_word(w, letters) for w in rel],
            "dummy_kind": "interaction", "dummy_label": "Interaction contrast (pooled as error)",
            "randomized": False, "seed": None, "aliasing_note": note}
    ds = ExperimentDataset(factors, coded, point_type=ptype, dummy=dummy, dummy_names=dnames, info=info)
    if randomize:
        ds.randomize(seed)
    elif seed is not None:
        ds.info["seed"] = int(seed)
    return ds
