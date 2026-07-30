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
from .factor_residuals import FactorResidualEvaluator
from .residual_inputs import (
    AudioAlignmentFactorInput,
    ContactFactorInput,
    GaugeFactorInput,
    JointLimitFactorInput,
    LineReprojectionFactorInput,
    MaskSilhouetteFactorInput,
    MetricDepthFactorInput,
    PointReprojectionFactorInput,
    SupportPlaneFactorInput,
    PeriodicPhaseFactorInput,
    PosePriorFactorInput,
    ResidualInputRequest,
    WorldSpaceContactSample,
    build_audio_event_alignment_residual_inputs,
    build_gauge_constraint_residual_inputs,
    build_geometry_sequence_residual_input_bundle,
    build_geometry_sequence_residual_dependencies,
    build_joint_limit_residual_inputs,
    build_line_reprojection_residual_inputs,
    build_mask_silhouette_residual_inputs,
    build_metric_depth_measurement_residual_inputs,
    build_point_reprojection_residual_inputs,
    build_support_plane_residual_inputs,
    build_metric_depth_residual_inputs,
    build_periodic_phase_prior_residual_inputs,
    build_pose_prior_residual_inputs,
    build_residual_input_bundle,
    build_sequence_temporal_residual_inputs,
    build_state_regularization_residual_inputs,
    build_world_space_contact_residual_inputs,
    build_world_space_contact_relative_velocity_residual_inputs,
    build_world_space_contact_twist_gauge_residual_inputs,
    build_world_space_contact_sample_residual_inputs,
)
from .candidate import (
    CHAIR_SANDBOX_ARTIFACTS,
    GENERIC_OBJECT_SANDBOX_ARTIFACTS,
    SANDBOX_MANIFEST_NAME,
    SPHERE_SANDBOX_ARTIFACTS,
    build_candidate_sandbox_manifest,
    default_candidate_dir,
    validate_candidate_sandbox_manifest,
    write_candidate_sandbox_manifest,
    verify_materialized_generic_object_candidate,
)
from .candidate_golden import (
    DEFAULT_CANDIDATE_SANDBOX_GOLDEN,
    DEFAULT_MATERIALIZED_CANDIDATE_GOLDEN,
    build_canonical_candidate_sandbox_summary,
    build_materialized_candidate_summary,
    verify_materialized_candidate_summary,
    verify_candidate_sandbox_summary,
)
from .attempt_artifacts import (
    ISOLATED_ATTEMPT_FILENAMES,
    IsolatedAttemptState,
    load_isolated_attempt_state,
    update_isolated_attempt_evidence,
    write_isolated_sequence_attempt,
)
from .contact_correspondence import (
    RigidContactHypothesis,
    RigidContactHypothesisLedger,
    apply_rigid_contact_hypotheses,
    build_typed_rigid_contact_hypotheses,
    rigid_contact_hypothesis_ledger_record,
)
from .capability_initializers import (
    InitializationRequest,
    InitializationResult,
    initialize_from_capabilities,
)
from .problem import build_sequence_problem_shadow
from .problem_factory import (
    SequenceFactorInputs,
    SequenceProblemFactory,
    SequenceProblemPreparation,
    sequence_problem_preparation_record,
)
from .legacy_production_problem import (
    LegacyObjectProblemPreparation,
    legacy_object_problem_preparation_record,
    prepare_legacy_articulated_object_problem,
)
from .capability_production_problem import (
    CapabilityObjectProblemPreparation,
    capability_object_problem_preparation_record,
    prepare_capability_object_problem,
)
from .publisher import (
    AcceptedObjectOutputPublisher,
    ObjectPublicationGate,
    ObjectPublicationResult,
    evaluate_object_publication_gate,
    object_publication_record,
)
from .optimization import (
    GenericSequenceSolveResult,
    SequenceOptimizationParameters,
    SequenceOptimizationProblem,
    build_runtime_residual_blocks,
    build_runtime_residual_vector,
    solve_sequence_optimization,
)
from .parameterization import StateSpecParameterization
from .render_evidence import render_line_reprojection_evidence
from .sparsity import ResidualRowDependency, build_factor_frame_jacobian_sparsity
from .problem_contract import SequenceProblemContract, build_sequence_problem_contract, sequence_problem_contract_record
from .runtime import (
    GenericExecutorAttemptLedger,
    GenericExecutorPrepareResult,
    GenericExecutorRuntimePlan,
    GenericSequenceExecutor,
    attempt_ledger_record,
    build_generic_executor_runtime_plan,
    prepare_result_record,
    runtime_plan_record,
)
from .residual_boundary import (
    GenericResidualBoundary,
    GenericResidualDryRunLedger,
    GenericResidualExecutionPlan,
    ResidualDryRunRecord,
    ResidualExecutionPlanRecord,
    ResidualBoundaryRecord,
    build_generic_residual_dry_run,
    build_generic_residual_boundary,
    build_generic_residual_execution_plan,
    residual_boundary_ledger_record,
    residual_boundary_record,
    residual_dry_run_ledger_record,
    residual_dry_run_record,
    residual_execution_plan_ledger_record,
    residual_execution_plan_record,
    runtime_configured_factor_ids,
)
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
    verify_materialized_sphere_sequence_candidate,
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
    verify_materialized_projected_periodic_candidate,
)
from .rigid_correspondence import RigidCorrespondenceInitializer, StateSpecRigidCorrespondenceInitializer
from .line_diagnostics import (
    LINE_CONTACT_ATTEMPT_NAME,
    LINE_CONTACT_CANDIDATE_NAME,
    LINE_CONTACT_RESIDUAL_NAME,
    LINE_CONTACT_SANDBOX_ARTIFACTS,
    build_line_contact_diagnostics,
    prepare_line_contact_candidate,
    validate_line_contact_diagnostics,
    verify_materialized_line_contact_candidate,
)
from .validation import validate_sequence_problem_shadow

__all__ = [
    "DEFAULT_SEQUENCE_DIAGNOSTICS_GOLDEN",
    "DEFAULT_CANDIDATE_SANDBOX_GOLDEN",
    "DEFAULT_MATERIALIZED_CANDIDATE_GOLDEN",
    "DEFAULT_SEQUENCE_PROBLEM_GOLDEN",
    "DEFAULT_SPHERE_SEQUENCE_GOLDEN",
    "DEFAULT_PROJECTED_PERIODIC_GOLDEN",
    "CHAIR_FACTOR_ATTEMPT_NAME",
    "CHAIR_FACTOR_CANDIDATE_NAME",
    "CHAIR_FACTOR_RESIDUAL_TABLE_NAME",
    "CHAIR_FACTOR_RESIDUALS_NAME",
    "CHAIR_SANDBOX_ARTIFACTS",
    "GENERIC_OBJECT_SANDBOX_ARTIFACTS",
    "AudioAlignmentFactorInput",
    "AcceptedObjectOutputPublisher",
    "CapabilityObjectProblemPreparation",
    "ContactFactorInput",
    "FactorResidualEvaluator",
    "GaugeFactorInput",
    "GenericExecutorPrepareResult",
    "GenericExecutorRuntimePlan",
    "GenericExecutorAttemptLedger",
    "GenericResidualBoundary",
    "GenericSequenceExecutor",
    "GenericSequenceSolveResult",
    "JointLimitFactorInput",
    "InitializationRequest",
    "InitializationResult",
    "ISOLATED_ATTEMPT_FILENAMES",
    "IsolatedAttemptState",
    "LineReprojectionFactorInput",
    "MaskSilhouetteFactorInput",
    "MetricDepthFactorInput",
    "PointReprojectionFactorInput",
    "LegacyObjectProblemPreparation",
    "SupportPlaneFactorInput",
    "PeriodicPhaseFactorInput",
    "ObjectPublicationGate",
    "evaluate_object_publication_gate",
    "ObjectPublicationResult",
    "PosePriorFactorInput",
    "ResidualBoundaryRecord",
    "ResidualRowDependency",
    "ResidualInputRequest",
    "WorldSpaceContactSample",
    "LINE_CONTACT_ATTEMPT_NAME",
    "LINE_CONTACT_CANDIDATE_NAME",
    "LINE_CONTACT_RESIDUAL_NAME",
    "LINE_CONTACT_SANDBOX_ARTIFACTS",
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
    "RigidContactHypothesis",
    "RigidContactHypothesisLedger",
    "StateSpecRigidCorrespondenceInitializer",
    "SequenceProblemContract",
    "SequenceOptimizationParameters",
    "SequenceOptimizationProblem",
    "SequenceFactorInputs",
    "SequenceProblemFactory",
    "SequenceProblemPreparation",
    "StateSpecParameterization",
    "SphereSequenceParameters",
    "build_candidate_sandbox_manifest",
    "build_audio_event_alignment_residual_inputs",
    "build_world_space_contact_relative_velocity_residual_inputs",
    "build_world_space_contact_twist_gauge_residual_inputs",
    "build_canonical_candidate_sandbox_summary",
    "build_materialized_candidate_summary",
    "build_canonical_sequence_problem_summary",
    "build_canonical_sequence_solver_diagnostics_summary",
    "build_generic_executor_runtime_plan",
    "build_generic_residual_boundary",
    "build_gauge_constraint_residual_inputs",
    "build_geometry_sequence_residual_input_bundle",
    "build_geometry_sequence_residual_dependencies",
    "build_joint_limit_residual_inputs",
    "build_line_reprojection_residual_inputs",
    "build_mask_silhouette_residual_inputs",
    "build_metric_depth_measurement_residual_inputs",
    "build_point_reprojection_residual_inputs",
    "build_support_plane_residual_inputs",
    "build_metric_depth_residual_inputs",
    "build_periodic_phase_prior_residual_inputs",
    "build_pose_prior_residual_inputs",
    "build_residual_input_bundle",
    "build_runtime_residual_vector",
    "build_typed_rigid_contact_hypotheses",
    "build_runtime_residual_blocks",
    "build_factor_frame_jacobian_sparsity",
    "build_sequence_temporal_residual_inputs",
    "build_state_regularization_residual_inputs",
    "build_world_space_contact_residual_inputs",
    "build_world_space_contact_sample_residual_inputs",
    "build_sequence_problem_shadow",
    "build_sequence_problem_contract",
    "initialize_from_capabilities",
    "solve_sequence_optimization",
    "sequence_problem_preparation_record",
    "load_isolated_attempt_state",
    "legacy_object_problem_preparation_record",
    "capability_object_problem_preparation_record",
    "prepare_capability_object_problem",
    "update_isolated_attempt_evidence",
    "write_isolated_sequence_attempt",
    "build_sequence_solver_shadow_diagnostics",
    "build_sphere_sequence_regression_summary",
    "build_projected_periodic_regression_summary",
    "build_line_contact_diagnostics",
    "build_chair_contact_diagnostics",
    "build_chair_factor_executor_candidate",
    "build_chair_factor_residual_coverage",
    "default_candidate_dir",
    "prepare_line_contact_candidate",
    "validate_candidate_sandbox_manifest",
    "validate_line_contact_diagnostics",
    "validate_chair_contact_diagnostics",
    "validate_chair_factor_executor_candidate",
    "validate_chair_factor_residual_coverage",
    "verify_materialized_chair_factor_candidate",
    "verify_materialized_generic_object_candidate",
    "verify_materialized_candidate_summary",
    "verify_materialized_line_contact_candidate",
    "verify_materialized_sphere_sequence_candidate",
    "verify_materialized_projected_periodic_candidate",
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
    "sequence_problem_contract_record",
    "runtime_plan_record",
    "render_line_reprojection_evidence",
    "runtime_configured_factor_ids",
    "apply_rigid_contact_hypotheses",
    "rigid_contact_hypothesis_ledger_record",
    "prepare_result_record",
    "prepare_legacy_articulated_object_problem",
    "object_publication_record",
    "attempt_ledger_record",
    "residual_boundary_ledger_record",
    "residual_boundary_record",
]


_COMPATIBILITY_EXPORTS = {
    "build_chair_contact_diagnostics": (".chair_diagnostics", "build_chair_contact_diagnostics"),
    "validate_chair_contact_diagnostics": (".chair_diagnostics", "validate_chair_contact_diagnostics"),
    "CHAIR_FACTOR_CANDIDATE_NAME": (".chair_factor_candidate", "CHAIR_FACTOR_CANDIDATE_NAME"),
    "CHAIR_FACTOR_ATTEMPT_NAME": (".chair_factor_candidate", "CHAIR_FACTOR_ATTEMPT_NAME"),
    "CHAIR_FACTOR_RESIDUAL_TABLE_NAME": (".chair_factor_candidate", "CHAIR_FACTOR_RESIDUAL_TABLE_NAME"),
    "CHAIR_FACTOR_RESIDUALS_NAME": (".chair_factor_candidate", "CHAIR_FACTOR_RESIDUALS_NAME"),
    "build_chair_factor_executor_candidate": (".chair_factor_candidate", "build_chair_factor_executor_candidate"),
    "build_chair_factor_residual_coverage": (".chair_factor_candidate", "build_chair_factor_residual_coverage"),
    "prepare_chair_factor_executor_candidate": (".chair_factor_candidate", "prepare_chair_factor_executor_candidate"),
    "validate_chair_factor_executor_candidate": (".chair_factor_candidate", "validate_chair_factor_executor_candidate"),
    "validate_chair_factor_residual_coverage": (".chair_factor_candidate", "validate_chair_factor_residual_coverage"),
    "verify_materialized_chair_factor_candidate": (".chair_factor_candidate", "verify_materialized_chair_factor_candidate"),
}


def __getattr__(name: str):
    """Lazy-load legacy chair evidence APIs outside the production import graph."""
    target = _COMPATIBILITY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
