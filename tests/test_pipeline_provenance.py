from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.shared.generic_contact_pipeline.core.provenance.attempts import StageAttempt
from scripts.shared.generic_contact_pipeline.core.provenance.artifact_store import (
    ArtifactStore,
    verify_attempt_artifacts,
)
from scripts.shared.generic_contact_pipeline.core.provenance.golden import (
    CANONICAL_CASES,
    DEFAULT_GOLDEN_MANIFEST,
    artifact_record,
    sync_golden_inputs,
    verify_golden_manifest,
)


class PipelineProvenanceTest(unittest.TestCase):
    def test_artifact_store_deduplicates_content_and_detects_corruption(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "result"
            first = root / "first.csv"
            second = root / "second.csv"
            first.write_text("frame,value\n1,same\n")
            second.write_bytes(first.read_bytes())
            store = ArtifactStore(result_dir)

            first_ref = store.put(first, canonical_path="first.csv")
            second_ref = store.put(second, canonical_path="second.csv")

            self.assertEqual(first_ref["sha256"], second_ref["sha256"])
            self.assertEqual(first_ref["blob_path"], second_ref["blob_path"])
            blobs = list((result_dir / "provenance/artifact_store/sha256").glob("*/*"))
            self.assertEqual(len(blobs), 1)
            self.assertEqual(store.verify_reference(first_ref), [])

            blob = blobs[0]
            blob.chmod(0o644)
            blob.write_text("corrupt\n")
            self.assertTrue(any("sha256 expected" in error for error in store.verify_reference(first_ref)))
            with self.assertRaisesRegex(RuntimeError, "blob is corrupt"):
                store.put(first)

    def test_five_case_golden_manifest_has_stage_pose_gate_and_decoded_render_hashes(self) -> None:
        payload = json.loads(DEFAULT_GOLDEN_MANIFEST.read_text())

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(tuple(payload["cases"]), CANONICAL_CASES)
        for case_name, case in payload["cases"].items():
            self.assertEqual(
                set(case["stages"]),
                {"stage-1", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5"},
                case_name,
            )
            self.assertTrue(case["inputs"], case_name)
            self.assertIn(case["recorded_execution"]["vlm_mode"], {"none", "qwen"})
            self.assertIn(case["recorded_execution"]["llm_mode"], {"none", "mistral"})
            self.assertTrue(case["contact_and_gate_state"], case_name)
            self.assertEqual(len(case["outputs"]["pose"]["sha256"]), 64, case_name)
            renders = case["outputs"]["decoded_renders"]
            self.assertEqual(len(renders), 6, case_name)
            self.assertTrue(all(len(item["decoded"]["sha256_rgb24"]) == 64 for item in renders))

    def test_artifact_record_includes_csv_contract_and_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.csv"
            path.write_text("frame,value\n1,a\n2,b\n")

            record = artifact_record(path)

            self.assertEqual(record["columns"], ["frame", "value"])
            self.assertEqual(record["rows"], 2)
            self.assertEqual(len(record["sha256"]), 64)

    def test_golden_verifier_detects_changed_and_unresolved_input_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository_file = root / "repository.csv"
            repository_file.write_text("frame,value\n1,a\n")
            external_file = root / "input_root" / "generated.bin"
            external_file.parent.mkdir()
            external_file.write_bytes(b"canonical input")
            manifest = {
                "schema_version": 1,
                "cases": {
                    case: (
                        {
                            "repository": artifact_record(repository_file),
                            "external": artifact_record(
                                external_file,
                                logical_path="generated.bin",
                                source_scope="input_root",
                            ),
                        }
                        if case == "basketball"
                        else {}
                    )
                    for case in CANONICAL_CASES
                },
            }

            unresolved = verify_golden_manifest(manifest, verify_decoded_renders=False)
            self.assertTrue(any("requires --input-root" in error for error in unresolved))
            self.assertEqual(
                verify_golden_manifest(
                    manifest,
                    verify_decoded_renders=False,
                    input_root=external_file.parent,
                ),
                [],
            )

            repository_file.write_text("frame,value\n1,changed\n")
            changed = verify_golden_manifest(
                manifest,
                verify_decoded_renders=False,
                input_root=external_file.parent,
            )
            self.assertTrue(any("sha256 expected" in error for error in changed))

    def test_golden_input_sync_is_dry_run_first_and_never_overwrites(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            source_file = source / "sample/generated/input.csv"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("frame,value\n1,canonical\n")
            record = artifact_record(source_file, logical_path="sample/generated/input.csv")
            manifest = {
                "cases": {
                    case: ({"inputs": [record]} if case == "basketball" else {"inputs": []})
                    for case in CANONICAL_CASES
                }
            }

            dry_run = sync_golden_inputs(
                manifest,
                source_root=source,
                destination_root=destination,
            )
            self.assertEqual(dry_run["would_copy"], ["sample/generated/input.csv"])
            self.assertFalse((destination / "sample/generated/input.csv").exists())

            applied = sync_golden_inputs(
                manifest,
                source_root=source,
                destination_root=destination,
                apply=True,
            )
            self.assertEqual(applied["copied"], ["sample/generated/input.csv"])

            target = destination / "sample/generated/input.csv"
            target.write_text("different\n")
            conflict = sync_golden_inputs(
                manifest,
                source_root=source,
                destination_root=destination,
                apply=True,
            )
            self.assertTrue(any("non-golden hash" in error for error in conflict["errors"]))
            self.assertEqual(target.read_text(), "different\n")

    def test_stage_rerun_creates_distinct_immutable_attempt_records(self) -> None:
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            profile = SimpleNamespace(
                case_name="test_case",
                result_name="test_result",
                result_dir=result_dir,
                render_dir=Path(tmp) / "renders",
            )
            result_dir.mkdir(parents=True)
            (result_dir / "object_pose.csv").write_text("frame,x\n1,0\n")

            first = StageAttempt(profile, "stage4", trigger="scheduled_pipeline_stage")
            (result_dir / "object_pose.csv").write_text("frame,x\n1,1\n")
            first_record = first.finish(status="completed_rerun_requested")

            second = StageAttempt(
                profile,
                "stage4",
                trigger="stage_audit_rerun",
                parent_attempt_id=first.attempt_id,
            )
            (result_dir / "object_pose.csv").write_text("frame,x\n1,2\n")
            second_record = second.finish()

            attempts = sorted((result_dir / "provenance/stages/stage4/attempts").glob("*.json"))
            self.assertEqual([path.name for path in attempts], ["000001.json", "000002.json"])
            self.assertEqual(second_record["parent_attempt_id"], first_record["attempt_id"])
            self.assertNotEqual(
                first_record["artifacts_after"],
                second_record["artifacts_after"],
            )
            self.assertEqual(set(first_record["stored_artifacts"]), set(first_record["artifacts_after"]))
            self.assertEqual(set(second_record["stored_artifacts"]), set(second_record["artifacts_after"]))
            self.assertEqual(verify_attempt_artifacts(result_dir), [])
            second_payload = json.loads(attempts[1].read_text())
            second_payload["stored_artifacts"] = {}
            attempts[1].write_text(json.dumps(second_payload))
            self.assertTrue(
                any("missing store references" in error for error in verify_attempt_artifacts(result_dir))
            )
            active = json.loads((result_dir / "provenance/stages/stage4/active_attempt.json").read_text())
            self.assertEqual(active["attempt_id"], "000002")
            self.assertEqual(active["status"], "completed")

    def test_run_case_orchestration_records_audit_rerun_as_second_attempt(self) -> None:
        runtime = Path("/home/yang/miniconda3/envs/audiohoi/bin/python")
        if not runtime.exists():
            self.skipTest("audiohoi runtime is not installed")
        helper = Path(__file__).parent / "runtime" / "run_case_provenance_smoke.py"
        completed = subprocess.run(
            [str(runtime), str(helper)],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("run_case provenance smoke: pass", completed.stdout)


if __name__ == "__main__":
    unittest.main()
