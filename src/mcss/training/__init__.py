"""Offline-only differentiable training kernels for the dynamic carrier."""

from mcss.training.grounded import (
    GroundedPairResult,
    build_grounded_state,
    grounded_measurement_loss,
    paired_context_measurement_loss,
)
from mcss.training.write_unroll import unroll_observed_episode

__all__ = (
    "GroundedPairResult",
    "build_grounded_state",
    "grounded_measurement_loss",
    "paired_context_measurement_loss",
    "unroll_observed_episode",
)
