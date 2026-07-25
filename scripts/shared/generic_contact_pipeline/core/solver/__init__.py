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
from .problem import build_sequence_problem_shadow
from .validation import validate_sequence_problem_shadow

__all__ = [
    "DEFAULT_SEQUENCE_PROBLEM_GOLDEN",
    "build_canonical_sequence_problem_summary",
    "build_sequence_problem_shadow",
    "validate_sequence_problem_shadow",
    "verify_sequence_problem_summary",
]
