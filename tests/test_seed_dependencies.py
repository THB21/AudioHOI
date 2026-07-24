from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides
from scripts.shared.generic_contact_pipeline.core.provenance.seed_dependencies import audit_seed_dependencies
from scripts.shared.generic_contact_pipeline.core.semantics.provenance import resolve_mug_m17_phase
from scripts.shared.generic_contact_pipeline.core.semantics.static_tail import enforce_declared_static_tail


EXPECTATIONS = REPO / "tests/golden/solved_seed_dependency_expectations.json"
FRESH_RESULT_NAME = "__seed_audit_fresh_result__"


def _summary(audit: dict[str, object]) -> dict[str, object]:
    dependencies = {}
    for item in audit["dependencies"]:
        dependencies[item["dependency_id"]] = {
            key: item[key]
            for key in (
                "selected_candidate_id",
                "resolution",
                "rerun_readiness",
                "solved_seed_dependency",
            )
        }
    return {
        "rerun_readiness": audit["rerun_readiness"],
        "dependencies": dependencies,
    }


class SeedDependencyAuditTest(unittest.TestCase):
    def test_existing_and_fresh_seed_selection_matches_frozen_expectations(self) -> None:
        expected = json.loads(EXPECTATIONS.read_text())
        for context, result_name in (
            ("existing", "benchmark_vlm_qwen"),
            ("fresh", FRESH_RESULT_NAME),
        ):
            for case_name in ("mug", "chair"):
                profile = with_runtime_overrides(load_case_profile(case_name), result_name=result_name)
                self.assertEqual(_summary(audit_seed_dependencies(profile)), expected[context][case_name])

    def test_audit_is_read_only_for_fresh_result_directories(self) -> None:
        fresh_dirs = []
        for case_name in ("mug", "chair"):
            profile = with_runtime_overrides(load_case_profile(case_name), result_name=FRESH_RESULT_NAME)
            fresh_dirs.append(profile.result_dir)
            self.assertFalse(profile.result_dir.exists())
            audit_seed_dependencies(profile)
        self.assertTrue(all(not path.exists() for path in fresh_dirs))

    def test_cli_reports_both_contexts(self) -> None:
        tool = REPO / "scripts/shared/generic_contact_pipeline/tools/audit_seed_dependencies.py"
        completed = subprocess.run(
            [sys.executable, str(tool)],
            cwd=REPO,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(set(payload["contexts"]), {"existing", "fresh"})
        self.assertEqual(set(payload["contexts"]["existing"]), {"mug", "chair"})

    def test_mug_phase_resolver_has_no_historical_or_identity_fallback(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            profile = SimpleNamespace(result_dir=result_dir)
            with self.assertRaisesRegex(FileNotFoundError, "observation-derived mug phase"):
                resolve_mug_m17_phase(profile)
            self.assertFalse(result_dir.exists())
            phase = result_dir / "observation_seed/axial_phase.csv"
            phase.parent.mkdir(parents=True)
            phase.write_text("frame,m17_phase_rad\n1,0.0\n")
            selected, info = resolve_mug_m17_phase(profile)
            self.assertEqual(selected, phase)
            self.assertFalse(info["historical_solved_seed_used"])

    def test_table_freeze_postcondition_survives_sequence_smoothing(self) -> None:
        profile = SimpleNamespace(refinement_policies=lambda: ["table_freeze"])
        rows = [
            {
                "frame": str(frame), "m45_static_frame": "2",
                "tx": str(frame), "ty": "0", "tz": "3", "qw": "1", "qx": "0", "qy": "0", "qz": "0",
                "source": "smoothed",
            }
            for frame in range(1, 5)
        ]
        frozen, report = enforce_declared_static_tail(profile, rows)
        self.assertEqual([row["tx"] for row in frozen], ["1", "2", "2", "2"])
        self.assertEqual(report["frozen_rows"], 2)
        self.assertTrue(all("table_freeze_postcondition" in row["source"] for row in frozen[2:]))


if __name__ == "__main__":
    unittest.main()
