"""Explicit seed management (roadmap v0.1 "顯式 seed 管理").

Rules used everywhere in ADoRE-DoE
----------------------------------
* every random procedure takes a ``seed`` argument and never touches NumPy's global RNG;
* ``seed=None`` means "pick one for me" -- the chosen integer is returned/recorded
  (``dataset.info["seed"]``) so the run can be reproduced afterwards;
* one *root* seed is split into independent, named *child* streams (``candidates``, ``moment``,
  ``starts``, ``randomize``, ...) so that changing one component never shifts another;
* child seeds are derived from the name with SHA-256 (NOT Python's ``hash``, which is salted
  per process), so they are identical across processes, machines and Python versions.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..exceptions import DesignError


def check_seed(seed):
    """Validate a seed: ``None`` or a non-negative integer (bool / float / negative rejected)."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise DesignError(f"seed must be None or a non-negative integer, got {seed!r}.")
    if seed < 0:
        raise DesignError(f"seed must be non-negative, got {seed}.")
    return int(seed)


def resolve_seed(seed=None) -> int:
    """Return a concrete integer seed (generate one from OS entropy if ``seed`` is None)."""
    s = check_seed(seed)
    return s if s is not None else int(np.random.SeedSequence().entropy % (2**32))


def make_rng(seed=None) -> np.random.Generator:
    """Generator from an int / None / SeedSequence; an existing Generator is passed through."""
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    return np.random.default_rng(resolve_seed(seed))


def _key(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")


def derive_seed(seed: int, *names: str) -> int:
    """Deterministic child seed of ``seed`` for the stream called ``names`` (32-bit int)."""
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=tuple(_key(n) for n in names))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


class SeedManager:
    """Root seed + named independent child streams; remembers what was handed out."""

    def __init__(self, seed=None):
        self.root = resolve_seed(seed)
        self._children: dict[str, int] = {}

    def child_seed(self, name: str) -> int:
        s = derive_seed(self.root, name)
        self._children[name] = s
        return s

    def rng(self, name: str) -> np.random.Generator:
        return np.random.default_rng(self.child_seed(name))

    def as_dict(self) -> dict:
        """JSON-friendly record for ``dataset.info['seeds']`` (root + every child used)."""
        return {"root": self.root, **self._children}

    def __repr__(self) -> str:
        return f"SeedManager(root={self.root}, children={sorted(self._children)})"
