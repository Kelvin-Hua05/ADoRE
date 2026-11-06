"""ADoRE-DoE - Adaptive Design of Experiments with Residual Enhancement."""
from . import core, screening
from .core import Experiment, ExperimentDataset, Factor, Response
from .exceptions import (DesignError, InferenceError, InferenceWarning,
                         RankDeficientError)

__version__ = "0.1.0.dev0"
__all__ = ["Factor", "Response", "ExperimentDataset", "Experiment", "core", "screening",
           "DesignError", "RankDeficientError", "InferenceError", "InferenceWarning"]
