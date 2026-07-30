from __future__ import annotations

import json

from ...core.base.config import CaseProfile
from ...core.base.io import write_json
from ...core.base.schema import stage_paths
from ...core.preprocess import run_case_ingestion


def run(profile: CaseProfile) -> dict[str, object]:
    """Prepare and hash-validate every required input before object solving."""

    result = run_case_ingestion(profile)
    task_records = [record.as_record() for record in result.tasks]
    prepared_inputs = {
        record.task_id: {
            "status": record.status,
            "runtime_env": record.runtime_env,
            "outputs": dict(record.output_paths),
            "output_hashes": dict(record.output_hashes),
            "error": record.error,
            "human_state_role": record.human_state_role,
        }
        for record in result.tasks
    }
    checks = {
        "case_ingestion_accepted": result.status == "accepted",
        "frame_count_positive": result.frame_count > 0,
        "fps_positive": result.fps > 0.0,
        "required_tasks_complete": all(
            (not record.required) or record.status in {"generated", "reused"}
            for record in result.tasks
        ),
        "gvhmr_read_only": all(
            record.human_state_role == "read_only_observed"
            for record in result.tasks
            if record.task_id == "gvhmr"
        ),
    }
    manifest_payload = json.loads(result.manifest_path.read_text())
    compatibility_manifest = {
        "schema_version": 2,
        "stage": "stage0_preprocess",
        "case_name": profile.case_name,
        "sample_dir": str(profile.sample_dir),
        "result_dir": str(profile.result_dir),
        "render_dir": str(profile.render_dir),
        "prepared_inputs": prepared_inputs,
        "checks": checks,
        "case_ingestion_manifest": str(result.manifest_path),
        "case_ingestion_manifest_sha256": manifest_payload.get("canonical_sha256", ""),
        "status": result.status,
    }
    metrics = {
        "schema_version": 2,
        "stage": "stage0_preprocess",
        "case_name": profile.case_name,
        "status": result.status,
        "frame_count": result.frame_count,
        "fps": result.fps,
        "checks": checks,
        "tasks": task_records,
        "generated_tasks": sum(record.status == "generated" for record in result.tasks),
        "reused_tasks": sum(record.status == "reused" for record in result.tasks),
        "disabled_tasks": sum(record.status == "disabled" for record in result.tasks),
        "failed_tasks": [record.task_id for record in result.tasks if record.status == "failed"],
        "object_only_boundary": True,
        "human_state_optimized": False,
    }
    paths = stage_paths(profile)
    write_json(paths["stage0_inputs_manifest"], compatibility_manifest)
    write_json(paths["stage0_metrics"], metrics)
    if result.status != "accepted":
        failed = next((record for record in result.tasks if record.status == "failed"), None)
        detail = f"{failed.task_id}: {failed.error}" if failed else "unknown task failure"
        raise RuntimeError(f"case ingestion failed for {profile.case_name}: {detail}")
    return metrics
