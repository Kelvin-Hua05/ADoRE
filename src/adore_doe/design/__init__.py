"""RSM Design (Phase 2): CCD (ADR-010), I-optimal (primary) / D-optimal (comparison) designs,
augmentation, candidate sets, linear-constrained regions, rank / conditioning checks, FDS plot.
See docs/rsm_design.md."""
from .candidates import CandidateSet, generate_candidates
from .ccd import ccd, ccd_run_count
from .criteria import (DesignCheck, check_design, d_efficiency, i_criterion, log_det,
                       prediction_variance, require_full_rank)
from .fds import FDSResult, fds, plot_fds
from .model import ModelSpec, as_model
from .moment import (IntegrationDomain, MomentResult, exact_uniform_moment,
                     i_criterion_convergence)
from .optimal import Augmentation, augment, candidate_scores, d_optimal, i_optimal
from .region import Region

__all__ = ["ccd", "ccd_run_count", "i_optimal", "d_optimal", "augment", "Augmentation",
           "candidate_scores", "ModelSpec", "as_model", "Region", "IntegrationDomain", "MomentResult",
           "exact_uniform_moment", "i_criterion_convergence", "generate_candidates", "CandidateSet",
           "DesignCheck", "check_design", "require_full_rank", "i_criterion", "log_det",
           "d_efficiency", "prediction_variance", "fds", "FDSResult", "plot_fds"]
