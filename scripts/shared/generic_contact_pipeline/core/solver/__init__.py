"""Generic sequence-solver shadow contracts.

This package describes the solver problem that future branches may execute.
The current branch intentionally stays in shadow mode: it builds deterministic
problem manifests and validates provenance without consuming legacy poses or
writing accepted outputs.
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
from .validation import validate_sequence_problem_shadow

__all__ = [
    "DEFAULT_SEQUENCE_DIAGNOSTICS_GOLDEN",
    "DEFAULT_CANDIDATE_SANDBOX_GOLDEN",
    "DEFAULT_SEQUENCE_PROBLEM_GOLDEN",
    "SANDBOX_MANIFEST_NAME",
    "build_candidate_sandbox_manifest",
    "build_canonical_candidate_sandbox_summary",
    "build_canonical_sequence_problem_summary",
    "build_canonical_sequence_solver_diagnostics_summary",
    "build_sequence_problem_shadow",
    "build_sequence_solver_shadow_diagnostics",
    "default_candidate_dir",
    "validate_candidate_sandbox_manifest",
    "validate_sequence_problem_shadow",
    "verify_candidate_sandbox_summary",
    "verify_sequence_problem_summary",
    "verify_sequence_solver_diagnostics_summary",
    "write_candidate_sandbox_manifest",
]
