"""Plackett-Burman designs built from normalised Hadamard matrices.

A PB design with N runs (N a multiple of 4) is an N x (N-1) two-level orthogonal array:
the columns of a normalised Hadamard matrix of order N with the all-ones column removed.
Hadamard matrices are built by

* Sylvester doubling                        (N = 2^m: 4, 8, 16, 32, 64, ...),
* Paley construction I,  q prime, q = 3 (mod 4), N = q + 1        (12, 20, 24, 44, 48, 60, 68, ...),
* Paley construction II, q prime, q = 1 (mod 4), N = 2(q + 1)     (28, 36, ...),
* Kronecker product with H2                 (40, 56, ...).

For N = 12, 20, 24, 44 the Paley-I construction is the same family as the classical cyclic PB
designs (not compared entry-by-entry here). Every constructed matrix is verified: H H^T = N I.
N = 52 (and a few larger orders) is not available with these constructions.

ALIASING: for N = 2^m the main effects are completely aliased with certain two-factor
interactions (resolution III). For N = 12, 20, 24, ... each two-factor interaction is
*partially* aliased with many main effects (coefficient +-1/3 for N=12), so a single
interaction can bias several main-effect estimates. PB designs are for main-effect
screening only; interactions are NOT estimable.
"""
from __future__ import annotations

import numpy as np

from ..exceptions import DesignError
from ..core.factor import Factor
from ..core.dataset import ExperimentDataset

ALIASING_NOTE = (
    "Plackett-Burman: main-effect screening only (resolution III). Two-factor interactions "
    "are aliased with main effects (fully for N=2^m, partially for N=12,20,24,...); "
    "a significant main effect may be caused by an interaction."
)


def _is_prime(q: int) -> bool:
    if q < 2:
        return False
    return all(q % p for p in range(2, int(q ** 0.5) + 1))


def _legendre_matrix(q: int) -> np.ndarray:
    squares = {(x * x) % q for x in range(1, q)}
    chi = np.array([0] + [1 if a in squares else -1 for a in range(1, q)])
    idx = (np.arange(q)[:, None] - np.arange(q)[None, :]) % q
    return chi[idx]


def _paley1(q: int) -> np.ndarray:           # q prime, q % 4 == 3 -> order q + 1
    Q = _legendre_matrix(q)
    S = np.zeros((q + 1, q + 1), dtype=int)
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = Q
    return S + np.eye(q + 1, dtype=int)


def _paley2(q: int) -> np.ndarray:           # q prime, q % 4 == 1 -> order 2(q + 1)
    Q = _legendre_matrix(q)
    C = np.zeros((q + 1, q + 1), dtype=int)
    C[0, 1:] = 1
    C[1:, 0] = 1
    C[1:, 1:] = Q
    I = np.eye(q + 1, dtype=int)
    return np.block([[C + I, C - I], [C - I, -C - I]])


def _hadamard_raw(n: int) -> np.ndarray | None:
    if n == 1:
        return np.ones((1, 1), dtype=int)
    if n & (n - 1) == 0:                      # Sylvester
        H = np.ones((1, 1), dtype=int)
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        return H
    if _is_prime(n - 1) and (n - 1) % 4 == 3:
        return _paley1(n - 1)
    if n % 2 == 0 and _is_prime(n // 2 - 1) and (n // 2 - 1) % 4 == 1:
        return _paley2(n // 2 - 1)
    if n % 2 == 0 and n % 4 == 0:
        h = _hadamard_raw(n // 2)
        if h is not None:
            return np.kron(np.array([[1, 1], [1, -1]]), h)
    return None


def hadamard(n: int) -> np.ndarray:
    """Normalised Hadamard matrix of order n (first row and column all +1), verified."""
    if n < 4 or n % 4:
        raise DesignError(f"Hadamard order must be a multiple of 4 (got {n}).")
    H = _hadamard_raw(n)
    if H is None:
        raise DesignError(f"No Hadamard construction implemented for order {n}.")
    H = H * H[:, [0]]          # make first column +1
    H = H * H[[0], :]          # make first row +1
    if not np.array_equal(H @ H.T, n * np.eye(n, dtype=int)):
        raise DesignError(f"Internal error: Hadamard matrix of order {n} failed verification.")
    return H


def supported_runs(max_runs: int = 68) -> list[int]:
    """Run sizes (multiples of 4) for which a PB design can be constructed."""
    out = []
    for n in range(4, max_runs + 1, 4):
        try:
            hadamard(n)
            out.append(n)
        except DesignError:
            pass
    return out


def plackett_burman(factors: list[Factor], runs: int | None = None, center_points: int = 0,
                    randomize: bool = False, seed: int | None = None) -> ExperimentDataset:
    """Build a Plackett-Burman screening design.

    Parameters
    ----------
    factors : list of numeric two-level :class:`Factor` (1 <= k <= runs - 1).
    runs : number of factorial runs N (a valid PB size). ``None`` -> smallest valid N >= k + 1.
        Invalid sizes raise :class:`DesignError`; they are never silently altered.
    center_points : number of added center-point runs (all coded 0; numeric factors only).
    randomize : randomise the execution order (``run_order``); the std order is kept.
    seed : seed for the randomisation (stored in ``design.info['seed']``).

    Unassigned columns (N - 1 - k of them) are kept as *dummy columns*; they carry no factor
    and give an error estimate under the no-interaction assumption.
    """
    factors = list(factors)
    if not factors:
        raise DesignError("At least one factor is required.")
    if not all(isinstance(f, Factor) for f in factors):
        raise DesignError("`factors` must be a list of Factor objects.")
    names = [f.name for f in factors]
    if len(set(names)) != len(names):
        raise DesignError("Duplicate factor names.")
    k = len(factors)
    if not isinstance(center_points, (int, np.integer)) or center_points < 0:
        raise DesignError("center_points must be a non-negative integer.")
    if center_points and any(not f.is_numeric for f in factors):
        raise DesignError("Center points need numeric factors; categorical factors have no center.")

    valid = supported_runs()
    if runs is None:
        cand = [n for n in valid if n - 1 >= k]
        if not cand:
            raise DesignError(f"No supported PB size for {k} factors (supported: {valid}).")
        runs = cand[0]
    else:
        if not isinstance(runs, (int, np.integer)) or runs < 4 or runs % 4:
            raise DesignError(f"Invalid PB run count {runs!r}: must be a multiple of 4 "
                              f"(supported: {valid}).")
        if runs not in valid and runs > max(valid):
            hadamard(int(runs))           # may still succeed for larger orders
        elif runs not in valid:
            raise DesignError(f"No PB construction implemented for runs={runs} "
                              f"(supported: {valid}).")
        if k > runs - 1:
            raise DesignError(f"{k} factors need at least {k + 1} runs; got runs={runs} "
                              f"(a PB design with N runs holds at most N-1 factors).")

    H = hadamard(int(runs))
    cols = H[:, 1:].astype(float)                 # drop the constant column
    n_dummy = cols.shape[1] - k
    n_tot = runs + int(center_points)
    coded = np.zeros((n_tot, k))
    coded[:runs] = cols[:, :k]
    dummy = None
    dnames: list[str] = []
    if n_dummy:
        dummy = np.zeros((n_tot, n_dummy))
        dummy[:runs] = cols[:, k:]
        dnames = [f"d{i + 1}" for i in range(n_dummy)]
    ptype = np.array(["factorial"] * runs + ["center"] * int(center_points), dtype=object)

    info = {"design_type": "plackett_burman", "factorial_runs": int(runs), "total_runs": int(n_tot),
            "n_factors": k, "center_points": int(center_points), "n_dummy_columns": n_dummy,
            "randomized": False, "seed": None, "aliasing_note": ALIASING_NOTE}
    dm = ExperimentDataset(factors, coded, point_type=ptype, dummy=dummy, dummy_names=dnames, info=info)
    if randomize:
        dm.randomize(seed)
    elif seed is not None:
        dm.info["seed"] = int(seed)
    return dm
