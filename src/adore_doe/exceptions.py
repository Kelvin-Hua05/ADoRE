"""Exceptions and warnings used across YourPkg."""


class DesignError(ValueError):
    """Invalid design specification (factors, runs, CSV content, ...)."""


class RankDeficientError(DesignError):
    """The model matrix is not full column rank: some parameters are not estimable."""


class InferenceError(RuntimeError):
    """Classical t / p-value inference is impossible for the current design/model."""


class InferenceWarning(UserWarning):
    """Statistical inference is limited or approximate (e.g. low df, missing runs)."""
