from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.shared.generic_contact_pipeline.core.base.config import (
    CaseProfile,
    load_case_profile,
    with_runtime_overrides,
)
from scripts.shared.generic_contact_pipeline.core.contracts.stage_artifacts import (
    ContractValidationError,
    LineAnchorState,
    LineLocalCoordinate,
    PointAnchorState,
    PointLocalCoordinate,
    adapt_contact_candidate,
    adapt_line_anchor,
    adapt_point_anchor,
    validate_stage_contracts,
)
from scripts.shared.generic_contact_pipeline.core.provenance.attempts import StageAttempt


class TypedContractsTest(unittest.TestCase):
    def test_five_canonical_cases_pass_stage1_through_stage4_contracts(self) -> None:
        for case_name in ("basketball", "football", "mug", "chair", "stick"):
            profile = with_runtime_overrides(
                load_case_profile(case_name), result_name="benchmark_vlm_qwen"
            )
            for stage_name in ("stage1", "stage2", "stage3", "stage4"):
                audit = validate_stage_contracts(profile, stage_name)
                self.assertEqual(audit["status"], "pass", audit)

    def test_contact_and_anchor_adapters_preserve_point_vs_line_semantics(self) -> None:
        point_candidate = adapt_contact_candidate(
            {
                "frame": "1", "time": "0.0", "contact_active": "1",
                "human_part": "palm", "human_side": "left", "object_part": "handle",
                "stable_local_x": "0.1", "stable_local_y": "", "stable_local_z": "0.3",
            }
        )
        line_candidate = adapt_contact_candidate(
            {
                "frame": "1", "time": "0.0", "contact_active": "1",
                "human_part": "palm", "human_side": "right", "object_part": "main_body",
                "object_local_s": "0.25",
            }
        )
        self.assertIsInstance(point_candidate.local_coordinate, PointLocalCoordinate)
        self.assertIsNone(point_candidate.local_coordinate.y)
        self.assertIsInstance(line_candidate.local_coordinate, LineLocalCoordinate)

        point_anchor = adapt_point_anchor(
            {
                "frame": "1", "time": "0", "contact_id": "c1", "human_part": "floor",
                "human_side": "", "object_part": "bottom", "stable_local_x": "",
                "stable_local_y": "", "stable_local_z": "",
            }
        )
        line_anchor = adapt_line_anchor(
            {
                "frame": "1", "time": "0", "human_side": "left",
                "observed_object_local_s": "0.2", "stable_object_local_s": "0.3",
            }
        )
        self.assertIsInstance(point_anchor, PointAnchorState)
        self.assertIsNone(point_anchor.human_side)
        self.assertIsInstance(line_anchor, LineAnchorState)

    def test_contract_failure_is_persisted_and_blocks_attempt_completion(self) -> None:
        with TemporaryDirectory() as tmp:
            sample = Path(tmp) / "sample"
            sample.mkdir()
            profile = CaseProfile(
                {
                    "case_name": "contract_fixture",
                    "sample_dir": str(sample),
                    "result_name": "result",
                }
            )
            profile.result_dir.mkdir(parents=True)
            (profile.result_dir / "object_pose.csv").write_text(
                "frame,time,tx,ty,tz,qw,qx,qy\n1,0,0,0,0,1,0,0\n"
            )

            attempt = StageAttempt(profile, "stage4", trigger="contract_test")
            with self.assertRaisesRegex(ContractValidationError, "missing required column qz"):
                attempt.finish()

            self.assertTrue(attempt.finalized)
            attempt_path = profile.result_dir / "provenance/stages/stage4/attempts/000001.json"
            payload = json.loads(attempt_path.read_text())
            self.assertEqual(payload["status"], "contract_failed")
            self.assertEqual(payload["contract_audit"]["status"], "fail")
            self.assertTrue(payload["stored_artifacts"])
            active = json.loads(
                (profile.result_dir / "provenance/stages/stage4/active_attempt.json").read_text()
            )
            self.assertEqual(active["status"], "contract_failed")


if __name__ == "__main__":
    unittest.main()
