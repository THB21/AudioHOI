from __future__ import annotations

from pathlib import Path

from .config import CaseProfile


def stage_paths(profile: CaseProfile) -> dict[str, Path]:
    base = profile.result_dir
    return {
        "hoi_profile": base / "hoi_profile.json",
        "hoi_profile_resolved": base / "hoi_profile_resolved.yaml",
        "llm_prior_trace": base / "llm_prior_trace.json",
        "prompt_context": base / "prompt_context.json",
        "stage0_metrics": base / "stage0_metrics.json",
        "stage0_inputs_manifest": base / "stage0_inputs_manifest.json",
        "object_observations": base / "object_observations.csv",
        "object_local_points": base / "object_local_points.csv",
        "object_local_segments": base / "object_local_segments.csv",
        "stage1_metrics": base / "stage1_metrics.json",
        "contact_candidates": base / "contact_candidates.csv",
        "contact_state": base / "contact_state_frames.csv",
        "stage2_metrics": base / "stage2_metrics.json",
        "object_pose_init": base / "object_pose_init.csv",
        "stage3_metrics": base / "stage3_metrics.json",
        "object_pose": base / "object_pose.csv",
        "object_contact_points": base / "object_contact_points.csv",
        "object_phase": base / "object_phase.csv",
        "stage4_metrics": base / "stage4_metrics.json",
        "render_manifest": profile.render_dir / "outputs.json",
        "stage5_metrics": base / "stage5_metrics.json",
        "compare_report": base / "stage6_compare_report.json",
        "migration_audit": base / "migration_audit.json",
        "migration_audit_csv": base / "migration_audit.csv",
        "llm_csv_audit_queries": base / "llm_csv_audit_queries.json",
        "llm_csv_audit_results": base / "llm_csv_audit_results.json",
        "llm_csv_audit_summary": base / "llm_csv_audit_summary.md",
        "llm_csv_audit_failures": base / "llm_csv_audit_failures.csv",
        "loss_dir": base / "loss_analysis",
        "loss_summary": base / "loss_analysis" / "loss_summary.json",
        "loss_trace": base / "loss_analysis" / "loss_trace.csv",
        "loss_residuals": base / "loss_analysis" / "per_frame_residuals.csv",
        "pipeline_manifest": base / "pipeline_manifest.json",
        "vlm_dir": base / "vlm",
        "vlm_queries": base / "vlm" / "vlm_queries.csv",
        "vlm_results": base / "vlm" / "vlm_results.csv",
        "vlm_gates": base / "vlm" / "vlm_gates.csv",
        "vlm_summary": base / "vlm" / "vlm_summary.md",
    }


REQUIRED_RENDER_FILES = [
    "object_only/overlay.mp4",
    "object_only/camera3d.mp4",
    "object_only/side_yz.mp4",
    "with_human/overlay.mp4",
    "with_human/camera3d.mp4",
    "with_human/side_yz.mp4",
]
