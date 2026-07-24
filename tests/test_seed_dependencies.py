from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides
from scripts.shared.generic_contact_pipeline.core.provenance.seed_dependencies import audit_seed_dependencies


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


if __name__ == "__main__":
    unittest.main()
