from __future__ import annotations

import json
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.shared.generic_contact_pipeline.core.base.io import read_csv, write_csv
from scripts.shared.generic_contact_pipeline.core.base.ablation import validate_ablation_flags
from scripts.shared.generic_contact_pipeline.core.base.schema import stage_paths
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.schemas import EvaluationPaths
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.overlay_metrics import compute_overlay_metrics
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.hoi_contact_metrics import compute_hoi_contact_metrics
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.part_metrics import compute_part_metrics
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.ablation_registry import (
    DEFAULT_VARIANTS,
    MATERIALIZED_DEFAULT_VARIANTS,
    MethodVariant,
    validate_method_result_mapping,
)
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.ablation_runner import run_ablation_evaluation
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.summary_writer import (
    write_unified_final_summary,
    run_unified_final_evaluation,
)
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.utils import normalize_artifact_value, repo_rel
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.final_result_sources import (
    load_final_result_profiles,
    validate_final_result_profile,
)
from scripts.shared.generic_contact_pipeline.tools.run_ablation_evaluation import DEFAULT_METHODS


class FinalHoiEvaluationTest(unittest.TestCase):
    def test_canonical_final_result_sources_are_frame_aligned_and_exclude_unpaired_mug(self) -> None:
        profiles = load_final_result_profiles()

        self.assertEqual([profile.case_name for profile in profiles], ["basketball", "football"])
        validations = [validate_final_result_profile(profile) for profile in profiles]
        missing = sorted(
            {artifact for row in validations for artifact in row["missing_artifacts"]}
        )
        if missing:
            self.skipTest(
                "requires generated repository data not present in a clean checkout: "
                + ", ".join(missing)
            )
        self.assertTrue(all(row["hard_metrics_ready"] for row in validations))
        self.assertTrue(all(row["gate_trace_ready"] for row in validations))
        self.assertTrue(all(row["final_pose_frame_aligned"] for row in validations))
        self.assertTrue(all(row["source_pose_frame_aligned"] for row in validations))

    def test_final_hoi_artifact_paths_are_repo_relative(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        absolute = repo_root / "samples_known_object" / "11_stick" / "results" / "benchmark_vlm_qwen" / "object_pose.csv"

        self.assertEqual(
            repo_rel(absolute),
            "samples_known_object/11_stick/results/benchmark_vlm_qwen/object_pose.csv",
        )
        self.assertEqual(
            normalize_artifact_value({"path": str(absolute), "items": [absolute]})["items"][0],
            "samples_known_object/11_stick/results/benchmark_vlm_qwen/object_pose.csv",
        )

    def _profile(self, tmp: str, *, case_name: str = "stick", result_name: str = "benchmark_vlm_qwen") -> SimpleNamespace:
        root = Path(tmp)
        sample = root / "sample"
        result = sample / "results" / result_name
        render = sample / "results" / "renders" / result_name
        sample.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            case_name=case_name,
            sample_dir=sample,
            result_dir=result,
            result_name=result_name,
            render_dir=render,
            data={
                "case_name": case_name,
                "result_name": result_name,
                "sample_dir": str(sample),
                "object_family": "rigid_staff",
            },
            component=lambda key: "",
        )

    def _write_artifacts(self, profile: SimpleNamespace) -> None:
        paths = stage_paths(profile)
        write_csv(
            paths["object_pose"],
            [
                {"frame": "1", "time": "0.0", "tx": "0", "ty": "0", "tz": "4", "qw": "1", "qx": "0", "qy": "0", "qz": "0", "length_m": "1.86"},
                {"frame": "2", "time": "0.1", "tx": "0.1", "ty": "0", "tz": "4", "qw": "0.999", "qx": "0", "qy": "0.045", "qz": "0", "length_m": "1.86"},
                {"frame": "3", "time": "0.2", "tx": "0.2", "ty": "0", "tz": "4", "qw": "0.996", "qx": "0", "qy": "0.090", "qz": "0", "length_m": "1.86"},
                {"frame": "4", "time": "0.3", "tx": "0.22", "ty": "0", "tz": "4", "qw": "0.991", "qx": "0", "qy": "0.134", "qz": "0", "length_m": "1.86"},
            ],
        )
        write_csv(
            paths["object_pose_pre_smooth"],
            [
                {"frame": "1", "time": "0.0", "tx": "0.02", "ty": "0", "tz": "4", "qw": "1", "qx": "0", "qy": "0", "qz": "0"},
                {"frame": "2", "time": "0.1", "tx": "0.13", "ty": "0", "tz": "4", "qw": "0.999", "qx": "0", "qy": "0.045", "qz": "0"},
                {"frame": "3", "time": "0.2", "tx": "0.24", "ty": "0", "tz": "4", "qw": "0.996", "qx": "0", "qy": "0.090", "qz": "0"},
                {"frame": "4", "time": "0.3", "tx": "0.25", "ty": "0", "tz": "4", "qw": "0.991", "qx": "0", "qy": "0.134", "qz": "0"},
            ],
        )
        write_csv(
            paths["object_observations"],
            [
                {"frame": "1", "mask_iou": "0.80", "observation_conf": "0.75"},
                {"frame": "2", "mask_iou": "0.90", "observation_conf": "0.85"},
                {"frame": "3", "mask_iou": "0.70", "observation_conf": "0.65"},
            ],
        )
        write_csv(
            paths["line_correspondence"],
            [
                {"frame": "1", "line_observation_trusted": "1.0", "endpoint_track_conf": "0.9"},
                {"frame": "2", "line_observation_trusted": "0.8", "endpoint_track_conf": "0.7"},
            ],
        )
        write_csv(paths["pose_jump_audit"], [{"frame": "2", "smoothness_spike": "1", "static_tail_drift_m": "0.03"}])
        write_csv(paths["physical_smooth_residuals"], [{"frame": "1", "velocity_norm": "0.1", "acceleration_norm": "0.2", "anchor_residual_enabled": "1"}])
        write_csv(
            paths["anchor_state"],
            [
                {"frame": "1", "anchor_drift_m": "0.02", "anchor_update_allowed": "1", "pose_anchor_allowed": "1"},
                {"frame": "2", "anchor_drift_m": "0.04", "anchor_update_allowed": "0", "pose_anchor_allowed": "1"},
            ],
        )
        write_csv(
            paths["optimizer_decisions"],
            [
                {
                    "frame": "1",
                    "visual_residual_enabled": "1",
                    "contact_anchor_residual_enabled": "1",
                    "velocity_residual_enabled": "1",
                    "acceleration_residual_enabled": "1",
                    "static_freeze_residual_enabled": "1",
                    "boundary_freeze_interpolation_enabled": "0",
                    "static_tail_freeze_enabled": "0",
                    "feedback_reoptimized": "1",
                    "feedback_reweight_reason": "stage_audit_residual_reweight",
                }
            ],
        )
        write_csv(
            profile.result_dir / "vlm_trace" / "04_gating" / "gate_timeline.csv",
            [
                {"frame": "1", "stage": "stage2", "constraint": "contact_anchor", "active": "1", "source_gate": "pass"},
                {"frame": "2", "stage": "stage4", "constraint": "temporal_smooth", "active": "1", "source_gate": "unclear"},
            ],
        )
        write_csv(
            paths["stage_audit_dir"] / "stage_audit_gates.csv",
            [{"case_name": profile.case_name, "stage": "stage4", "check_type": "optimizer", "pass_gate": "pass", "residual_reweight": "1"}],
        )
        write_csv(
            profile.sample_dir / "results" / "human_audio_semantics" / "contact_records.csv",
            [{"frame": "2", "refined_frame": "2", "time": "0.1", "source": "audio", "relevant": "1"}],
        )
        hoi_dir = profile.sample_dir / "results" / "hoi_eval"
        hoi_dir.mkdir(parents=True, exist_ok=True)
        (hoi_dir / "hoi_interaction_metrics.json").write_text(
            json.dumps(
                {
                    "case": profile.case_name,
                    "object_type": "capsule",
                    "n_frames": 3,
                    "pen_frame_ratio": 0.1,
                    "pen_depth_mean_mm": 2.0,
                    "pen_depth_max_mm": 7.0,
                    "non_collision_ratio": 0.9,
                    "contact_frame_ratio": 0.67,
                    "contact_gap_mm": 12.0,
                    "part_correct_ratio": 0.8,
                    "contact_ratio_audio_windows": 0.75,
                    "accel_at_events": 0.4,
                    "accel_in_flight": 0.1,
                    "object_jerk": 0.02,
                }
            )
            + "\n"
        )

    def test_unified_final_evaluation_merges_object_and_tom_hoi_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            self._write_artifacts(profile)

            summary = run_unified_final_evaluation(profile)

            eval_dir = profile.result_dir / "evaluation"
            self.assertTrue((eval_dir / "final_evaluation_summary.json").exists())
            self.assertTrue((eval_dir / "final_evaluation_detailed.csv").exists())
            self.assertTrue((eval_dir / "object_6d_metrics.json").exists())
            self.assertTrue((eval_dir / "overlay_metrics.json").exists())
            self.assertTrue((eval_dir / "penetration_floating_metrics.json").exists())
            self.assertTrue((eval_dir / "temporal_plausibility_metrics.json").exists())
            self.assertTrue((eval_dir / "gate_impact_metrics.json").exists())
            self.assertEqual(summary["case"], "stick")
            self.assertTrue(summary["metrics"]["se3_valid"])
            self.assertAlmostEqual(summary["metrics"]["overlay_hard_score"], 0.8)
            self.assertAlmostEqual(summary["metrics"]["penetration_frame_ratio"], 0.1)
            self.assertAlmostEqual(summary["metrics"]["contact_gap_mm"], 12.0)
            self.assertGreater(summary["metrics"]["tradeoff_score"], 0.0)
            rows = read_csv(eval_dir / "final_evaluation_detailed.csv")
            self.assertEqual(rows[0]["case"], "stick")
            self.assertIn("overlay_hard_score", rows[0])
            self.assertIn("tradeoff_score", rows[0])
            self.assertIn("contact_proxy", rows[0])
            self.assertIn("contact_proxy_source", rows[0])
            self.assertIn("translation_spike_count", rows[0])
            self.assertIn("non_event_spike_count", rows[0])
            self.assertIn("gate_active_count", rows[0])
            self.assertIn("optimizer_reweighted_frames", rows[0])
            self.assertAlmostEqual(float(rows[0]["contact_proxy"]), 0.7866278610665535)
            self.assertEqual(rows[0]["contact_proxy_source"], "contact_gap_mm_exp_decay_sigma_50")
            self.assertEqual(rows[0]["gate_active_count"], "2")
            self.assertEqual(rows[0]["optimizer_reweighted_frames"], "1")
            manifest = json.loads((profile.result_dir / "pipeline_manifest.json").read_text())
            self.assertIn("final_hoi_evaluation", manifest)
            self.assertEqual(
                manifest["final_hoi_evaluation"]["summary_json"],
                str(eval_dir / "final_evaluation_summary.json"),
            )
            self.assertEqual(
                manifest["final_hoi_evaluation"]["pipeline_qa_summary_csv"],
                str(profile.result_dir / "vlm_trace" / "06_evaluation" / "pipeline_qa_summary.csv"),
            )
            self.assertIn("temporal_plausibility_metrics_json", manifest["final_hoi_evaluation"])
            self.assertIn("gate_impact_metrics_json", manifest["final_hoi_evaluation"])

    def test_unified_final_evaluation_exports_vlm_llm_qa_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            self._write_artifacts(profile)

            summary = run_unified_final_evaluation(profile)

            eval_dir = profile.result_dir / "vlm_trace" / "06_evaluation"
            for name in [
                "vlm_eval_queries.csv",
                "vlm_eval_raw_responses.jsonl",
                "vlm_eval_parsed_scores.csv",
                "vlm_eval_summary.json",
                "llm_eval_summary.md",
                "qa_audit_report.html",
            ]:
                self.assertTrue((eval_dir / name).exists(), name)
            self.assertIn("qa.vlm_eval_queries_csv", summary["artifacts"])
            self.assertIn("qa.qa_audit_report_html", summary["artifacts"])
            parsed = read_csv(eval_dir / "vlm_eval_parsed_scores.csv")
            self.assertGreater(len(parsed), 0)
            self.assertIn("question", parsed[0])
            self.assertIn("raw_answer", parsed[0])
            self.assertIn("affected_constraint", parsed[0])

    def test_unified_final_evaluation_aggregates_pipeline_vlm_llm_qa(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            self._write_artifacts(profile)
            paths = stage_paths(profile)
            write_csv(
                paths["vlm_dir"] / "stage2" / "vlm_queries.csv",
                [
                    {
                        "query_id": "q_contact",
                        "frame": "1",
                        "query_type": "contact_relation_check",
                        "question": "What is the highlighted object contact region closest to?",
                        "input_image_path": "evidence/contact.png",
                    }
                ],
            )
            write_csv(
                paths["vlm_dir"] / "stage2" / "vlm_results.csv",
                [
                    {
                        "query_id": "q_contact",
                        "frame": "1",
                        "query_type": "contact_relation_check",
                        "normalized_label": "hand_on_contact_region",
                        "raw_response": "{\"label\":\"hand_on_contact_region\"}",
                    }
                ],
            )
            write_csv(
                paths["vlm_dir"] / "stage2" / "vlm_gates.csv",
                [
                    {
                        "query_id": "q_contact",
                        "frame": "1",
                        "query_type": "contact_relation_check",
                        "pass_gate": "pass",
                        "allow_contact_residual": "1",
                        "is_effective": "1",
                        "repair_action": "accept_candidate",
                    }
                ],
            )
            write_csv(
                paths["stage_audit_dir"] / "stage4" / "llm_audit_results.csv",
                [
                    {
                        "check_id": "llm_optimizer",
                        "stage": "stage4",
                        "question": "Did the optimizer use gate-controlled residuals?",
                        "raw_answer": "pass: optimizer decisions are consistent",
                        "parsed_label": "pass",
                        "pass_gate": "pass",
                        "affected_constraint": "sequence_optimizer",
                        "changed_optimizer_behavior": "1",
                    }
                ],
            )

            summary = run_unified_final_evaluation(profile)

            eval_dir = profile.result_dir / "vlm_trace" / "06_evaluation"
            for name in ["pipeline_qa_summary.csv", "pipeline_qa_summary.json", "pipeline_qa_summary.md"]:
                self.assertTrue((eval_dir / name).exists(), name)
            rows = read_csv(eval_dir / "pipeline_qa_summary.csv")
            self.assertGreaterEqual(len(rows), 2)
            by_source = {row["source_type"]: row for row in rows}
            self.assertEqual(by_source["vlm"]["question"], "What is the highlighted object contact region closest to?")
            self.assertEqual(by_source["vlm"]["parsed_label"], "hand_on_contact_region")
            self.assertEqual(by_source["vlm"]["pass_gate"], "pass")
            self.assertEqual(by_source["vlm"]["changed_optimizer_behavior"], "1")
            self.assertEqual(by_source["llm"]["affected_constraint"], "sequence_optimizer")
            self.assertIn("qa.pipeline_qa_summary_csv", summary["artifacts"])

    def test_ablation_registry_rejects_same_result_directory_for_distinct_methods(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            variants = [
                MethodVariant(method="full_audio_vlm_llm", result_name="same_result", ablation_flags=[]),
                MethodVariant(method="no_audio", result_name="same_result", ablation_flags=["no_audio"]),
            ]

            with self.assertRaisesRegex(ValueError, "same result directory"):
                validate_method_result_mapping([profile], variants)

    def test_ablation_registry_defaults_to_materialized_benchmark_results(self) -> None:
        by_method = {variant.method: variant for variant in DEFAULT_VARIANTS}

        self.assertEqual(by_method["full_audio_vlm_llm"].result_name, "clean_ablation_full_audio_vlm_llm")
        self.assertEqual(by_method["no_vlm_llm"].result_name, "clean_ablation_no_vlm_llm")
        self.assertEqual(by_method["no_vlm_llm"].vlm, "none")
        self.assertEqual(by_method["no_vlm_llm"].llm, "none")
        self.assertEqual(by_method["no_vlm_llm"].ablation_flags, [])
        self.assertEqual(by_method["no_audio"].result_name, "clean_ablation_no_audio")
        self.assertEqual(by_method["no_audio"].ablation_flags, ["disable_audio_events"])
        self.assertEqual(by_method["no_llm"].result_name, "benchmark_no_llm")
        self.assertEqual(by_method["no_contact_anchor"].result_name, "benchmark_no_anchor")
        self.assertFalse(by_method["no_contact_anchor"].mechanism_supported)
        self.assertEqual(by_method["no_contact_anchor"].ablation_flags, [])
        self.assertEqual(by_method["audio_enabled"].result_name, "benchmark_audio_enabled")

    def test_runtime_ablation_flags_must_have_algorithm_consumers(self) -> None:
        self.assertEqual(validate_ablation_flags(["disable_audio_events"]), ["disable_audio_events"])
        with self.assertRaisesRegex(ValueError, "[Nn]o current pipeline component consumes"):
            validate_ablation_flags(["no_contact_anchor"])

    def test_ablation_cli_default_methods_are_materialized_required_variants(self) -> None:
        self.assertEqual(
            DEFAULT_METHODS,
            ["full_audio_vlm_llm", "no_audio", "no_vlm_llm"],
        )
        self.assertEqual([variant.method for variant in MATERIALIZED_DEFAULT_VARIANTS], DEFAULT_METHODS)
        self.assertNotIn("object_only", [variant.method for variant in MATERIALIZED_DEFAULT_VARIANTS])

    def test_run_pipeline_post_run_final_evaluator_uses_unified_final_hoi(self) -> None:
        source = Path("scripts/shared/generic_contact_pipeline/run_pipeline.py").read_text()

        self.assertIn("from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.summary_writer import run_unified_final_evaluation", source)
        self.assertIn('post_run["final_hoi_evaluator"] = run_unified_final_evaluation(profile)', source)
        self.assertNotIn('post_run["final_evaluator"] = run_final_evaluator(profile, method="vlm_gated")', source)

    def test_run_pipeline_has_current_ablation_evaluation_entrypoint(self) -> None:
        source = Path("scripts/shared/generic_contact_pipeline/run_pipeline.py").read_text()

        self.assertIn("from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.ablation_runner import run_ablation_evaluation", source)
        self.assertIn("from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.ablation_registry import MATERIALIZED_DEFAULT_VARIANTS", source)
        self.assertIn("--run-ablation-evaluation", source)
        self.assertIn("run_ablation_evaluation(", source)
        self.assertIn("variants=MATERIALIZED_DEFAULT_VARIANTS", source)
        self.assertIn("require_existing=True", source)

    def test_unified_final_summary_writes_cross_case_table(self) -> None:
        with TemporaryDirectory() as tmp:
            stick = self._profile(tmp, case_name="stick")
            mug = self._profile(tmp, case_name="mug")
            self._write_artifacts(stick)
            self._write_artifacts(mug)

            result = write_unified_final_summary([stick, mug], output_dir=Path(tmp) / "summary")

            table = read_csv(Path(result["table"]))
            self.assertEqual(result["rows"], 2)
            self.assertEqual({row["case"] for row in table}, {"stick", "mug"})
            self.assertIn("overlay_hard_score", table[0])
            self.assertTrue((Path(tmp) / "summary" / "final_evaluation_human_readable.md").exists())
            readable = (Path(tmp) / "summary" / "final_evaluation_human_readable.md").read_text()
            header = next(line for line in readable.splitlines() if line.startswith("| Case |"))
            self.assertEqual(
                header,
                "| Case | Object 6DoF | Visual Overlay | Contact/Anchor | Physical | Temporal |",
            )
            self.assertNotIn("Audio Benefit", readable)
            self.assertNotIn("Final Pass", readable)
            self.assertNotIn("None", readable)
            manifest = json.loads((Path(tmp) / "summary" / "final_evaluation_summary_manifest.json").read_text())
            self.assertIn("cases", manifest)
            self.assertEqual({case["case"] for case in manifest["cases"]}, {"stick", "mug"})
            for case in manifest["cases"]:
                self.assertIn("result_dir", case)
                self.assertIn("evaluation_summary_json", case)
                self.assertIn("pipeline_qa_summary_csv", case)

    def test_ablation_runner_marks_missing_variants_and_writes_delta_table(self) -> None:
        with TemporaryDirectory() as tmp:
            full = self._profile(tmp, result_name="full_result")
            no_vlm_llm = self._profile(tmp, result_name="no_vlm_llm_result")
            self._write_artifacts(full)
            self._write_artifacts(no_vlm_llm)
            (full.result_dir / "pipeline_manifest.json").write_text(
                json.dumps({"vlm_mode": "qwen", "llm_mode": "mistral", "profile": {}}) + "\n"
            )
            (no_vlm_llm.result_dir / "pipeline_manifest.json").write_text(
                json.dumps({"vlm_mode": "none", "llm_mode": "none", "profile": {"ablation_flags": ["no_vlm", "no_llm"]}}) + "\n"
            )
            variants = [
                MethodVariant(method="full_audio_vlm_llm", result_name="full_result", ablation_flags=[], audio=True, vlm="qwen", llm="mistral"),
                MethodVariant(method="no_vlm_llm", result_name="no_vlm_llm_result", ablation_flags=["no_vlm", "no_llm"], audio=True, vlm="none", llm="none"),
                MethodVariant(method="no_audio", result_name="missing_result", ablation_flags=["disable_audio_events"], audio=False, vlm="qwen", llm="mistral"),
            ]

            result = run_ablation_evaluation([full], variants=variants, output_dir=Path(tmp) / "ablation")

            table = read_csv(Path(result["table"]))
            rows = {row["method"]: row for row in table}
            self.assertEqual(rows["full_audio_vlm_llm"]["method_status"], "ok")
            self.assertEqual(rows["no_vlm_llm"]["method_status"], "ok")
            self.assertEqual(rows["no_audio"]["method_status"], "missing_result")
            self.assertEqual(rows["full_audio_vlm_llm"]["audio"], "True")
            self.assertEqual(rows["no_vlm_llm"]["vlm"], "none")
            self.assertEqual(rows["no_vlm_llm"]["llm"], "none")
            self.assertEqual(rows["no_vlm_llm"]["ablation_flags"], "no_vlm|no_llm")
            self.assertIn("contact_proxy", rows["full_audio_vlm_llm"])
            self.assertIn("gate_active_count", rows["full_audio_vlm_llm"])
            self.assertIn("pose_delta_translation_max_m", rows["full_audio_vlm_llm"])
            self.assertEqual(rows["full_audio_vlm_llm"]["gate_active_count"], "2")
            self.assertEqual(rows["full_audio_vlm_llm"]["optimizer_reweighted_frames"], "1")
            self.assertIn("pose_sha256", rows["full_audio_vlm_llm"])
            self.assertNotEqual(rows["full_audio_vlm_llm"]["pose_sha256"], "")
            self.assertEqual(rows["no_vlm_llm"]["same_pose_as_baseline"], "True")
            self.assertEqual(rows["no_vlm_llm"]["metrics_identical_to_baseline"], "False")
            self.assertTrue((Path(tmp) / "ablation" / "ablation_delta_table.csv").exists())
            self.assertTrue((Path(tmp) / "ablation" / "ablation_report.md").exists())
            self.assertTrue((Path(tmp) / "ablation" / "ablation_method_registry.csv").exists())
            self.assertTrue((Path(tmp) / "ablation" / "ablation_method_registry_manifest.json").exists())
            strict_result = run_ablation_evaluation([full], variants=variants[:2], output_dir=Path(tmp) / "strict_ablation", require_existing=True)
            strict_manifest = json.loads((Path(tmp) / "strict_ablation" / "ablation_method_registry_manifest.json").read_text())
            self.assertEqual(strict_result["missing_results"], 0)
            self.assertTrue(strict_manifest["require_existing"])
            registry = read_csv(Path(tmp) / "ablation" / "ablation_method_registry.csv")
            registry_methods = {row["method"] for row in registry}
            self.assertEqual(registry_methods, {"full_audio_vlm_llm", "no_vlm_llm", "no_audio"})
            self.assertTrue(all(row["case"] == "stick" for row in registry))
            self.assertTrue(all(row["result_name"] in {"full_result", "no_vlm_llm_result", "missing_result"} for row in registry))
            report = (Path(tmp) / "ablation" / "ablation_report.md").read_text()
            self.assertIn("| case | method | status | result | audio | VLM | LLM | flags | same pose | same metrics | contact proxy | overlay IoU | overlay source | gate status | gate source | gate events | gates active | reweight | pose delta max | final pass |", report)
            self.assertIn("| stick | no_vlm_llm | ok | no_vlm_llm_result | True | none | none | no_vlm|no_llm | True | False |", report)

    def test_overlay_metrics_use_mask_pair_iou_when_observed_and_render_masks_exist(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            observed_dir = profile.sample_dir / "results" / "segmentation" / "masks"
            rendered_dir = profile.render_dir / "object_masks"
            observed_dir.mkdir(parents=True, exist_ok=True)
            rendered_dir.mkdir(parents=True, exist_ok=True)
            (observed_dir / "00001_mask.pgm").write_bytes(b"P5\n2 2\n255\n\xff\xff\x00\x00")
            (rendered_dir / "00001_mask.pgm").write_bytes(b"P5\n2 2\n255\n\xff\x00\xff\x00")

            block = compute_overlay_metrics(EvaluationPaths.from_profile(profile))

            self.assertEqual(block.metrics["overlay_hard_metric_source"], "mask_pair_iou")
            self.assertAlmostEqual(block.metrics["overlay_hard_score"], 1.0 / 3.0)
            self.assertAlmostEqual(block.metrics["overlay_mask_coverage"], 0.5)
            self.assertAlmostEqual(block.metrics["overlay_render_false_coverage"], 0.5)
            rows = read_csv(profile.result_dir / "evaluation" / "overlay_mask_metrics.csv")
            self.assertEqual(rows[0]["frame"], "1")
            self.assertAlmostEqual(float(rows[0]["iou"]), 1.0 / 3.0)

    def test_overlay_metrics_generate_evaluation_render_mask_when_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            observed_dir = profile.sample_dir / "results" / "segmentation" / "masks"
            observed_dir.mkdir(parents=True, exist_ok=True)
            (observed_dir / "00001_mask.pgm").write_bytes(b"P5\n3 3\n255\n\x00\xff\x00\xff\xff\xff\x00\xff\x00")
            write_csv(
                profile.result_dir / "object_pose.csv",
                [{"frame": "1", "u_proj": "1", "v_proj": "1", "radius_proj_px": "1"}],
            )

            block = compute_overlay_metrics(EvaluationPaths.from_profile(profile))

            self.assertEqual(block.metrics["overlay_hard_metric_source"], "generated_eval_proxy_render_mask_iou")
            self.assertAlmostEqual(block.metrics["overlay_hard_score"], 1.0)
            self.assertTrue((profile.result_dir / "evaluation" / "render_masks" / "00001_mask.pgm").exists())

    def test_overlay_metrics_generate_sphere_mask_using_observation_radius(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            observed_dir = profile.sample_dir / "results" / "segmentation" / "masks"
            observed_dir.mkdir(parents=True, exist_ok=True)
            (observed_dir / "00001_mask.pgm").write_bytes(b"P5\n3 3\n255\n\x00\xff\x00\xff\xff\xff\x00\xff\x00")
            write_csv(profile.result_dir / "object_pose.csv", [{"frame": "1", "u_proj": "1", "v_proj": "1"}])
            write_csv(profile.result_dir / "object_observations.csv", [{"frame": "1", "radius_px": "1"}])

            block = compute_overlay_metrics(EvaluationPaths.from_profile(profile))

            self.assertEqual(block.metrics["overlay_hard_metric_source"], "generated_eval_proxy_render_mask_iou")
            self.assertAlmostEqual(block.metrics["overlay_hard_score"], 1.0)

    def test_overlay_metrics_generate_bbox_proxy_mask_when_projection_is_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            observed_dir = profile.sample_dir / "results" / "segmentation" / "masks"
            observed_dir.mkdir(parents=True, exist_ok=True)
            (observed_dir / "00001_mask.pgm").write_bytes(b"P5\n3 3\n255\n\xff\xff\x00\xff\xff\x00\x00\x00\x00")
            write_csv(profile.result_dir / "object_pose.csv", [{"frame": "1", "tx": "0", "ty": "0", "tz": "4"}])
            write_csv(
                profile.result_dir / "object_observations.csv",
                [{"frame": "1", "mask_bbox_x1": "0", "mask_bbox_y1": "0", "mask_bbox_x2": "1", "mask_bbox_y2": "1"}],
            )

            block = compute_overlay_metrics(EvaluationPaths.from_profile(profile))

            self.assertEqual(block.metrics["overlay_hard_metric_source"], "generated_eval_proxy_render_mask_iou")
            self.assertAlmostEqual(block.metrics["overlay_hard_score"], 1.0)

    def test_overlay_metrics_generate_full_urdf_render_mask_when_urdf_exists(self) -> None:
        if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("PIL") is None:
            self.skipTest("full URDF mask rasterization requires the audiohoi runtime dependencies")
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            urdf = profile.sample_dir / "articraft" / "model.urdf"
            urdf.parent.mkdir(parents=True, exist_ok=True)
            urdf.write_text(
                """
<robot name="box">
  <link name="body">
    <visual name="box">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><box size="0.20 0.20 0.20" /></geometry>
    </visual>
  </link>
</robot>
""".strip()
            )
            observed_dir = profile.sample_dir / "results" / "segmentation" / "masks"
            observed_dir.mkdir(parents=True, exist_ok=True)
            width, height = 1280, 720
            pixels = bytearray(width * height)
            for y in range(323, 398):
                for x in range(603, 678):
                    pixels[y * width + x] = 255
            (observed_dir / "00001_mask.pgm").write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + bytes(pixels))
            write_csv(
                profile.result_dir / "object_pose.csv",
                [{"frame": "1", "tx": "0", "ty": "0", "tz": "4", "qw": "1", "qx": "0", "qy": "0", "qz": "0"}],
            )

            block = compute_overlay_metrics(EvaluationPaths.from_profile(profile))

            self.assertEqual(block.metrics["overlay_hard_metric_source"], "generated_eval_full_geometry_mask_iou")
            self.assertGreater(float(block.metrics["overlay_hard_score"]), 0.5)
            self.assertTrue((profile.result_dir / "evaluation" / "render_masks" / "00001_mask.pgm").exists())

    def test_hoi_contact_metrics_export_part_pair_rows_and_intervals(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            write_csv(
                profile.result_dir / "object_contact_points.csv",
                [
                    {
                        "frame": "1",
                        "time": "0.0",
                        "contact_active": "1",
                        "human_part": "palm",
                        "human_side": "left",
                        "object_part": "main_body",
                        "object_local_id": "left_palm_line_anchor",
                        "object_local_s": "0.40",
                        "stable_object_local_s": "0.39",
                        "local_s_drift": "0.01",
                        "contact_conf": "0.9",
                        "contact_depth_offset_m": "0.02",
                        "palm_to_line_px": "5.0",
                        "source": "unit_contact",
                    },
                    {
                        "frame": "2",
                        "time": "0.1",
                        "contact_active": "1",
                        "human_part": "palm",
                        "human_side": "left",
                        "object_part": "main_body",
                        "object_local_id": "left_palm_line_anchor",
                        "object_local_s": "0.42",
                        "stable_object_local_s": "0.39",
                        "local_s_drift": "0.03",
                        "contact_conf": "0.8",
                        "contact_depth_offset_m": "0.01",
                        "palm_to_line_px": "7.0",
                        "source": "unit_contact",
                    },
                ],
            )
            write_csv(
                profile.result_dir / "anchor_state.csv",
                [
                    {
                        "frame": "1",
                        "human_side": "left",
                        "human_part": "palm",
                        "object_part": "main_body",
                        "contact_observed": "1",
                        "contact_persistent": "1",
                        "anchor_update_allowed": "1",
                        "pose_anchor_allowed": "1",
                        "anchor_action": "update",
                    },
                    {
                        "frame": "2",
                        "human_side": "left",
                        "human_part": "palm",
                        "object_part": "main_body",
                        "contact_observed": "1",
                        "contact_persistent": "1",
                        "anchor_update_allowed": "0",
                        "pose_anchor_allowed": "1",
                        "anchor_action": "propagate",
                    },
                ],
            )

            block = compute_hoi_contact_metrics(EvaluationPaths.from_profile(profile))

            eval_dir = profile.result_dir / "evaluation"
            pairs = read_csv(eval_dir / "hoi_contact_pairs.csv")
            intervals = read_csv(eval_dir / "hoi_contact_intervals.csv")
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0]["human_part"], "palm")
            self.assertEqual(pairs[0]["object_part"], "main_body")
            self.assertEqual(pairs[0]["observed"], "1")
            self.assertEqual(pairs[1]["anchor_update_allowed"], "0")
            self.assertEqual(len(intervals), 1)
            self.assertEqual(intervals[0]["start_frame"], "1")
            self.assertEqual(intervals[0]["end_frame"], "2")
            self.assertAlmostEqual(block.metrics["contact_anchor_drift_mean"], 0.02)
            self.assertEqual(block.metrics["hoi_contact_pair_rows"], 2)

    def test_part_metrics_export_human_and_object_part_coverage(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            profile.data["vlm"] = {"parts": ["handle", "cup_body", "rim", "bottom", "background", "unclear"]}
            (profile.sample_dir / "results" / "gvhmr").mkdir(parents=True, exist_ok=True)
            (profile.sample_dir / "results" / "gvhmr" / "result.pkl").write_bytes(b"placeholder")
            (profile.sample_dir / "results" / "hands").mkdir(parents=True, exist_ok=True)
            write_csv(profile.sample_dir / "results" / "hands" / "hand_keypoints_3d.csv", [{"frame": "1", "hand_side": "left"}])
            write_csv(
                profile.result_dir / "evaluation" / "hoi_contact_pairs.csv",
                [
                    {"frame": "1", "human_part": "palm", "human_side": "left_hand", "object_part": "handle", "observed": "1"},
                    {"frame": "2", "human_part": "mouth", "human_side": "", "object_part": "rim", "observed": "1"},
                ],
            )
            write_csv(
                profile.result_dir / "object_surface_points.csv",
                [
                    {"frame": "1", "part": "handle"},
                    {"frame": "1", "part": "rim"},
                ],
            )

            block = compute_part_metrics(EvaluationPaths.from_profile(profile), profile.data)

            eval_dir = profile.result_dir / "evaluation"
            human_parts = read_csv(eval_dir / "human_parts.csv")
            object_parts = read_csv(eval_dir / "object_parts.csv")
            part_metrics = read_csv(eval_dir / "part_metrics.csv")
            self.assertTrue(any(row["part"] == "left_hand" and row["available"] == "1" for row in human_parts))
            self.assertTrue(any(row["part"] == "mouth" and row["contact_evidence_rows"] == "1" for row in human_parts))
            self.assertTrue(any(row["part"] == "handle" and row["contact_evidence_rows"] == "1" for row in object_parts))
            self.assertEqual(part_metrics[0]["human_part_contact_coverage"], "0.6666666666666666")
            self.assertEqual(block.metrics["object_part_count"], 4)

    def test_part_metrics_normalize_object_part_aliases_and_preserve_raw_vocab(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp, case_name="mug")
            profile.data["vlm"] = {"parts": ["handle", "cup_body", "rim", "bottom"]}
            write_csv(
                profile.result_dir / "evaluation" / "hoi_contact_pairs.csv",
                [
                    {"frame": "1", "human_part": "palm", "human_side": "right", "object_part": "handle_loop", "observed": "1"},
                    {"frame": "2", "human_part": "palm", "human_side": "right", "object_part": "body_shell", "observed": "1"},
                    {"frame": "3", "human_part": "mouth", "human_side": "", "object_part": "rim_ring", "observed": "1"},
                ],
            )
            write_csv(
                profile.result_dir / "object_surface_points.csv",
                [
                    {"frame": "1", "part": "body_shell"},
                    {"frame": "1", "part": "rim_ring"},
                    {"frame": "1", "part": "handle_loop"},
                ],
            )

            block = compute_part_metrics(EvaluationPaths.from_profile(profile), profile.data)

            eval_dir = profile.result_dir / "evaluation"
            object_parts = {row["part"]: row for row in read_csv(eval_dir / "object_parts.csv")}
            vocab_map = read_csv(eval_dir / "object_part_vocab_map.csv")
            self.assertEqual(set(object_parts), {"handle", "cup_body", "rim", "bottom"})
            self.assertEqual(object_parts["handle"]["contact_evidence_rows"], "1")
            self.assertEqual(object_parts["handle"]["surface_point_rows"], "1")
            self.assertIn("handle_loop", object_parts["handle"]["raw_parts"])
            self.assertEqual(object_parts["cup_body"]["contact_evidence_rows"], "1")
            self.assertIn("body_shell", object_parts["cup_body"]["raw_parts"])
            self.assertEqual(object_parts["rim"]["contact_evidence_rows"], "1")
            self.assertIn("rim_ring", object_parts["rim"]["raw_parts"])
            self.assertTrue(any(row["raw_part"] == "handle_loop" and row["canonical_part"] == "handle" for row in vocab_map))
            self.assertEqual(block.metrics["object_part_count"], 4)
            self.assertEqual(block.metrics["object_part_available_count"], 3)
            self.assertAlmostEqual(block.metrics["object_part_contact_coverage"], 0.75)


if __name__ == "__main__":
    unittest.main()
