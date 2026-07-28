"""Generic sequence-solver contracts and isolated candidate implementations.

The shared shadow manifests remain plan-only. Geometry-specific candidate
implementations may execute inside a safe directory, but cannot consume legacy
poses or write accepted outputs until an explicit Stage 4 promotion step.
"""

from .golden import (
    DEFAULT_SEQUENCE_PROBLEM_GOLDEN,
    build_canonical_sequence_problem_summary,
    verify_sequence_problem_summary,
)
from .diagnostics import build_sequence_solver_shadow_diagnostics
from .diagnostics_golden import (
    DEFAULT_SEQUENCE_DIAGNOSTICS_GOLDEN,
    build_canonical_sequence_solver_diagnostics_summary,
    verify_sequence_solver_diagnostics_summary,
)
from .candidate import (
    SANDBOX_MANIFEST_NAME,
    SPHERE_SANDBOX_ARTIFACTS,
    build_candidate_sandbox_manifest,
    default_candidate_dir,
    validate_candidate_sandbox_manifest,
    write_candidate_sandbox_manifest,
)
from .candidate_golden import (
    DEFAULT_CANDIDATE_SANDBOX_GOLDEN,
    build_canonical_candidate_sandbox_summary,
    verify_candidate_sandbox_summary,
)
from .problem import build_sequence_problem_shadow
from .sphere_sequence import (
    SPHERE_ATTEMPT_NAME,
    SPHERE_CANDIDATE_NAME,
    SPHERE_RESIDUAL_NAME,
    SphereSequenceParameters,
    solve_sphere_sequence_candidate,
)
from .sphere_golden import (
    DEFAULT_SPHERE_SEQUENCE_GOLDEN,
    build_sphere_sequence_regression_summary,
    verify_sphere_sequence_regression,
)
from .projected_periodic_sequence import (
    PeriodicKinematicContract,
    ProjectedPeriodicGeometryProvider,
    ProjectedPeriodicObservation,
    ProjectedPeriodicParameters,
    ProjectedPeriodicSolution,
    solve_projected_periodic_sequence,
)
from .projected_periodic_golden import (
    DEFAULT_PROJECTED_PERIODIC_GOLDEN,
    build_projected_periodic_regression_summary,
    verify_projected_periodic_regression,
)
from .rigid_correspondence import RigidCorrespondenceInitializer
from .line_diagnostics import (
    build_line_contact_diagnostics,
    validate_line_contact_diagnostics,
)
from .chair_diagnostics import (
    build_chair_contact_diagnostics,
    validate_chair_contact_diagnostics,
)
from .chair_factor_candidate import (
    CHAIR_FACTOR_ATTEMPT_NAME,
    build_chair_factor_executor_candidate,
    prepare_chair_factor_executor_candidate,
    validate_chair_factor_executor_candidate,
)
from .validation import validate_sequence_problem_shadow

__all__ = [
    "DEFAULT_SEQUENCE_DIAGNOSTICS_GOLDEN",
    "DEFAULT_CANDIDATE_SANDBOX_GOLDEN",
    "DEFAULT_SEQUENCE_PROBLEM_GOLDEN",
    "DEFAULT_SPHERE_SEQUENCE_GOLDEN",
    "DEFAULT_PROJECTED_PERIODIC_GOLDEN",
    "CHAIR_FACTOR_ATTEMPT_NAME",
    "SANDBOX_MANIFEST_NAME",
    "SPHERE_SANDBOX_ARTIFACTS",
    "SPHERE_ATTEMPT_NAME",
    "SPHERE_CANDIDATE_NAME",
    "SPHERE_RESIDUAL_NAME",
    "ProjectedPeriodicGeometryProvider",
    "PeriodicKinematicContract",
    "ProjectedPeriodicObservation",
    "ProjectedPeriodicParameters",
    "ProjectedPeriodicSolution",
    "RigidCorrespondenceInitializer",
    "SphereSequenceParameters",
    "build_candidate_sandbox_manifest",
    "build_canonical_candidate_sandbox_summary",
    "build_canonical_sequence_problem_summary",
    "build_canonical_sequence_solver_diagnostics_summary",
    "build_sequence_problem_shadow",
    "build_sequence_solver_shadow_diagnostics",
    "build_sphere_sequence_regression_summary",
    "build_projected_periodic_regression_summary",
    "build_line_contact_diagnostics",
    "build_chair_contact_diagnostics",
    "build_chair_factor_executor_candidate",
    "default_candidate_dir",
    "validate_candidate_sandbox_manifest",
    "validate_line_contact_diagnostics",
    "validate_chair_contact_diagnostics",
    "validate_chair_factor_executor_candidate",
    "validate_sequence_problem_shadow",
    "verify_candidate_sandbox_summary",
    "verify_sequence_problem_summary",
    "verify_sequence_solver_diagnostics_summary",
    "verify_sphere_sequence_regression",
    "verify_projected_periodic_regression",
    "write_candidate_sandbox_manifest",
    "prepare_chair_factor_executor_candidate",
    "solve_sphere_sequence_candidate",
    "solve_projected_periodic_sequence",
]
