from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline import run_pipeline
from scripts.shared.generic_contact_pipeline.core.base.config import CaseProfile
from scripts.shared.generic_contact_pipeline.core.provenance.artifact_store import verify_attempt_artifacts


def main() -> None:
    with TemporaryDirectory() as tmp:
        sample_dir = Path(tmp) / "sample"
        sample_dir.mkdir()
        profile = CaseProfile(
            {
                "case_name": "fake_case",
                "sample_dir": str(sample_dir),
                "result_name": "fake_result",
                "object_family": "test",
            }
        )
        calls = {"stage": 0, "audit": 0, "repair": 0}

        def fake_stage(current_profile):
            calls["stage"] += 1
            current_profile.result_dir.mkdir(parents=True, exist_ok=True)
            (current_profile.result_dir / "object_pose.csv").write_text("frame,x\n1,1\n")
            return {"status": "completed"}

        def fake_audit(current_profile, stage_name, *, llm_mode):
            calls["audit"] += 1
            return {
                "decision": "rerun" if calls["audit"] == 1 else "pass",
                "rerun_stage": calls["audit"] == 1,
            }

        def fake_repair(current_profile, stage_name, result, audit):
            calls["repair"] += 1
            (current_profile.result_dir / "object_pose.csv").write_text("frame,x\n1,2\n")
            return {**result, "repair": "completed"}

        args = Namespace(
            result_name="",
            ablation_flag=[],
            llm_mode="none",
            vlm_mode="none",
            vlm_blocking=False,
            export_vlm_trace=False,
            run_final_evaluator=False,
        )
        with (
            patch.object(run_pipeline, "load_case_profile", return_value=profile),
            patch.object(run_pipeline, "STAGES", [("stage4", fake_stage)]),
            patch.object(run_pipeline, "_run_stage_vlm", return_value=None),
            patch.object(run_pipeline, "write_stage_audit", side_effect=fake_audit),
            patch.object(run_pipeline, "_repair_after_stage_audit", side_effect=fake_repair),
        ):
            manifest = run_pipeline.run_case("fake_case", "stage4", "stage4", args=args)

        assert calls == {"stage": 1, "audit": 2, "repair": 1}, calls
        attempts = sorted(
            (profile.result_dir / "provenance/stages/stage4/attempts").glob("*.json")
        )
        assert [path.name for path in attempts] == ["000001.json", "000002.json"]
        second = json.loads(attempts[1].read_text())
        assert second["trigger"] == "stage_audit_rerun"
        assert second["parent_attempt_id"] == "000001"
        assert second["stored_artifacts"]
        assert verify_attempt_artifacts(profile.result_dir) == []
        assert manifest["ablation_mechanisms"]["all_requested_flags_have_consumers"]
    print("run_case provenance smoke: pass")


if __name__ == "__main__":
    main()
