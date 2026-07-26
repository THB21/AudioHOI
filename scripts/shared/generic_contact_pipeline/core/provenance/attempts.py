from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..base.io import REPO, repo_relative_value, write_json
from ..base.schema import stage_paths
from ..contracts.stage_artifacts import ContractValidationError, validate_stage_contracts
from ..plugins.registry import stage_plugin_audit
from .artifact_store import store_stage_artifacts


STAGE_ARTIFACT_KEYS: dict[str, tuple[str, ...]] = {
    "stage-1": ("hoi_profile", "hoi_profile_resolved", "llm_prior_trace", "prompt_context"),
    "stage0": ("stage0_inputs_manifest", "stage0_metrics"),
    "stage1": (
        "object_observations", "object_correspondence", "object_surface_points",
        "object_semantic_points", "line_correspondence", "line_observations",
        "object_local_points", "object_local_segments", "stage1_metrics",
    ),
    "stage2": (
        "contact_candidates", "contact_events", "human_sites", "support_geometry",
        "anchor_state", "contact_state", "stage2_metrics",
    ),
    "stage3": ("object_pose_init", "stage3_metrics"),
    "stage4": (
        "object_pose_pre_smooth", "motion_regime", "physical_smooth_residuals",
        "pose_jump_audit", "optimizer_decisions", "sphere_candidate", "sphere_residuals",
        "sphere_attempt", "object_pose",
        "object_contact_points", "object_phase", "stage4_metrics",
    ),
    "stage5": ("render_manifest", "stage5_metrics"),
    "stage6": ("compare_report",),
    "stage6.5": ("llm_csv_audit_queries", "llm_csv_audit_results", "llm_csv_audit_failures"),
    "stage7": ("loss_summary", "loss_trace", "loss_residuals"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_stage_artifacts(profile: Any, stage_name: str) -> dict[str, dict[str, object]]:
    paths = stage_paths(profile)
    candidates = [paths[key] for key in STAGE_ARTIFACT_KEYS.get(stage_name, ())]
    if stage_name in {"stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"}:
        candidates.extend(
            [
                paths["vlm_dir"] / stage_name / "vlm_gates.csv",
                paths["stage_audit_dir"] / stage_name / "stage_audit_gates.csv",
            ]
        )
    records: dict[str, dict[str, object]] = {}
    for path in candidates:
        rel = str(repo_relative_value(path))
        if path.is_file():
            records[rel] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    return records


@dataclass
class StageAttempt:
    profile: Any
    stage_name: str
    trigger: str
    parent_attempt_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    attempt_id: str = field(init=False)
    started_at: str = field(init=False)
    before: dict[str, dict[str, object]] = field(init=False)
    finalized: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        stage_dir = self.profile.result_dir / "provenance" / "stages" / self.stage_name
        attempts_dir = stage_dir / "attempts"
        existing = [int(path.stem) for path in attempts_dir.glob("*.json") if path.stem.isdigit()]
        index = max(existing, default=0) + 1
        self.attempt_id = f"{index:06d}"
        self.started_at = _now()
        self.before = snapshot_stage_artifacts(self.profile, self.stage_name)
        write_json(
            stage_dir / "active_attempt.json",
            {
                "schema_version": 1,
                "stage": self.stage_name,
                "attempt_id": self.attempt_id,
                "status": "running",
                "trigger": self.trigger,
                "parent_attempt_id": self.parent_attempt_id,
                "started_at": self.started_at,
            },
        )

    def finish(
        self,
        *,
        status: str = "completed",
        result_summary: dict[str, object] | None = None,
        error: BaseException | None = None,
    ) -> dict[str, object]:
        stage_dir = self.profile.result_dir / "provenance" / "stages" / self.stage_name
        after = snapshot_stage_artifacts(self.profile, self.stage_name)
        stored_artifacts = store_stage_artifacts(self.profile.result_dir, after)
        contract_audit = validate_stage_contracts(self.profile, self.stage_name)
        try:
            plugin_audit = stage_plugin_audit(self.profile, self.stage_name)
            plugin_audit["status"] = "resolved"
        except Exception as exc:
            plugin_audit = {
                "schema_version": 1,
                "stage": self.stage_name,
                "status": "not_configured",
                "error": str(exc),
                "plugins": [],
            }
        contract_failed = contract_audit["status"] == "fail" and status != "failed"
        final_status = "contract_failed" if contract_failed else status
        changed = sorted(
            path for path in set(self.before) | set(after) if self.before.get(path) != after.get(path)
        )
        record: dict[str, object] = {
            "schema_version": 4,
            "case_name": self.profile.case_name,
            "result_name": self.profile.result_name,
            "stage": self.stage_name,
            "attempt_id": self.attempt_id,
            "status": final_status,
            "trigger": self.trigger,
            "parent_attempt_id": self.parent_attempt_id,
            "started_at": self.started_at,
            "finished_at": _now(),
            "metadata": self.metadata,
            "artifacts_before": self.before,
            "artifacts_after": after,
            "stored_artifacts": stored_artifacts,
            "contract_audit": contract_audit,
            "plugin_audit": plugin_audit,
            "changed_artifacts": changed,
            "result_summary": result_summary or {},
        }
        if error is not None:
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            }
        record = repo_relative_value(record)
        attempt_path = stage_dir / "attempts" / f"{self.attempt_id}.json"
        if attempt_path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable attempt record {attempt_path}")
        write_json(attempt_path, record)
        self.finalized = True
        write_json(
            stage_dir / "active_attempt.json",
            {
                "schema_version": 1,
                "stage": self.stage_name,
                "attempt_id": self.attempt_id,
                "status": final_status,
                "record": attempt_path,
                "finished_at": record["finished_at"],
            },
        )
        if contract_failed:
            raise ContractValidationError(
                f"{self.profile.case_name}:{self.stage_name} artifact contract failed: "
                + "; ".join(str(error) for error in contract_audit["errors"])
            )
        return record
