from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.shared.generic_contact_pipeline.core.base.io import read_csv, write_csv
from scripts.shared.generic_contact_pipeline.core.base.schema import stage_paths
from scripts.shared.generic_contact_pipeline.core.evaluation.benchmark import run_benchmark
from scripts.shared.generic_contact_pipeline.core.evaluation.final_evaluator import run_final_evaluator
from scripts.shared.generic_contact_pipeline.core.evaluation.final_summary import write_final_summary
from scripts.shared.generic_contact_pipeline.core.evaluation.vlm_trace import export_vlm_trace


class VlmTraceFinalEvaluatorBenchmarkTest(unittest.TestCase):
    def _profile(self, tmp: str, *, case_name: str = "stick", result_name: str = "unit_result") -> SimpleNamespace:
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
                "object_family": "line_object",
                "render_backend": "generic_urdf",
            },
            component=lambda key: "generic_urdf" if key == "render_backend" else "",
        )

    def _write_minimal_artifacts(self, profile: SimpleNamespace) -> None:
        paths = stage_paths(profile)
        write_csv(
            paths["object_observations"],
            [
                {"frame": "1", "time": "0.0", "mask_iou": "0.80", "overlay_error_px": "8"},
                {"frame": "2", "time": "0.1", "mask_iou": "0.82", "overlay_error_px": "9"},
                {"frame": "3", "time": "0.2", "mask_iou": "0.81", "overlay_error_px": "7"},
            ],
        )
        write_csv(paths["object_correspondence"], [{"frame": "1", "track_id": "0", "confidence": "0.7"}])
        write_csv(paths["object_surface_points"], [{"frame": "1", "point_id": "0"}])
        write_csv(paths["object_semantic_points"], [{"frame": "1", "semantic_part": "main_body"}])
        write_csv(
            paths["contact_candidates"],
            [
                {"frame": "1", "contact_observed": "1", "contact_confidence": "0.9", "contact_part": "hand"},
                {"frame": "2", "contact_observed": "1", "contact_confidence": "0.8", "contact_part": "hand"},
                {"frame": "3", "contact_observed": "0", "contact_confidence": "0.1", "contact_part": "nothing"},
            ],
        )
        write_csv(
            paths["anchor_state"],
            [
                {"frame": "1", "contact_observed": "1", "contact_persistent": "1", "anchor_update_allowed": "1", "pose_anchor_allowed": "1", "anchor_drift_m": "0.01"},
                {"frame": "2", "contact_observed": "1", "contact_persistent": "1", "anchor_update_allowed": "1", "pose_anchor_allowed": "1", "anchor_drift_m": "0.02"},
                {"frame": "3", "contact_observed": "0", "contact_persistent": "0", "anchor_update_allowed": "0", "pose_anchor_allowed": "0", "anchor_drift_m": "0.04"},
            ],
        )
        pose_rows = [
            {"frame": "1", "time": "0.0", "tx": "0.0", "ty": "0.0", "tz": "4.0", "qw": "1", "qx": "0", "qy": "0", "qz": "0", "length_m": "2.0"},
            {"frame": "2", "time": "0.1", "tx": "0.1", "ty": "0.0", "tz": "4.0", "qw": "1", "qx": "0", "qy": "0", "qz": "0", "length_m": "2.01"},
            {"frame": "3", "time": "0.2", "tx": "0.1", "ty": "0.0", "tz": "4.0", "qw": "1", "qx": "0", "qy": "0", "qz": "0", "length_m": "2.0"},
        ]
        write_csv(paths["object_pose_init"], pose_rows)
        write_csv(paths["object_pose_pre_smooth"], pose_rows)
        write_csv(paths["object_pose"], pose_rows)
        write_csv(paths["object_contact_points"], [{"frame": "1", "contact_active": "1"}, {"frame": "2", "contact_active": "1"}])
        write_csv(
            paths["physical_smooth_residuals"],
            [
                {"frame": "1", "velocity_norm": "0.0", "acceleration_norm": "0.0", "penetration_depth_m": "0.0", "floating_gap_m": "0.0"},
                {"frame": "2", "velocity_norm": "0.1", "acceleration_norm": "0.1", "penetration_depth_m": "0.0", "floating_gap_m": "0.0"},
                {"frame": "3", "velocity_norm": "0.0", "acceleration_norm": "0.0", "penetration_depth_m": "0.0", "floating_gap_m": "0.03"},
            ],
        )
        write_csv(paths["pose_jump_audit"], [{"frame": "2", "visual_spike": "0", "contact_spike": "0", "smoothness_spike": "1", "static_tail_drift_m": "0.02"}])
        write_csv(paths["optimizer_decisions"], [{"frame": "1", "sequence_se3_optimizer_enabled": "1", "smooth_weight": "1.0", "accel_weight": "1.0"}])
        write_csv(paths["vlm_dir"] / "stage4" / "vlm_queries.csv", [{"query_id": "q1", "frame": "1", "query_type": "overlay_alignment_check"}])
        write_csv(paths["vlm_dir"] / "stage4" / "vlm_results.csv", [{"query_id": "q1", "frame": "1", "normalized_label": "aligned", "raw_response": "looks aligned"}])
        write_csv(paths["vlm_dir"] / "stage4" / "vlm_gates.csv", [{"query_id": "q1", "frame": "1", "pass_gate": "pass", "is_effective": "1", "allow_pose_refine": "1"}])
        write_csv(paths["stage_audit_dir"] / "stage4" / "stage_audit_gates.csv", [{"frame": "1", "check_type": "llm_csv:optimizer_gate", "pass_gate": "pass"}])
        write_csv(paths["render_manifest"], [{"path": str(profile.render_dir / "with_human" / "overlay.mp4")}])
        (profile.render_dir / "with_human").mkdir(parents=True, exist_ok=True)
        (profile.render_dir / "with_human" / "overlay.mp4").write_bytes(b"unit mp4 placeholder")
        (profile.sample_dir / "video.mp4").write_bytes(b"unit video placeholder")
        write_csv(profile.sample_dir / "annotations" / "evaluation" / "contact_labels.csv", [{"frame": "1", "contact_observed": "1"}, {"frame": "2", "contact_observed": "1"}, {"frame": "3", "contact_observed": "0"}])

    def test_export_vlm_trace_writes_stage_directories_and_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            self._write_minimal_artifacts(profile)

            result = export_vlm_trace(profile)

            trace_root = profile.result_dir / "vlm_trace"
            for name in ["00_input", "01_observation", "02_contact_candidates", "03_vlm_reasoning", "04_gating", "05_optimization", "06_evaluation"]:
                self.assertTrue((trace_root / name).is_dir(), name)
            manifest = json.loads((trace_root / "trace_manifest.json").read_text())
            self.assertEqual(manifest["case_name"], "stick")
            self.assertIn("object_observations", manifest["artifacts"])
            self.assertTrue((trace_root / "trace_completeness.json").exists())
            self.assertIn("missing_required", manifest["trace_completeness"])
            self.assertEqual(result["trace_root"], str(trace_root))

    def test_final_evaluator_writes_three_layer_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            self._write_minimal_artifacts(profile)

            result = run_final_evaluator(profile, method="vlm_gated")

            eval_dir = profile.result_dir / "vlm_trace" / "06_evaluation"
            self.assertTrue((eval_dir / "metric_scores.json").exists())
            self.assertTrue((eval_dir / "failure_flags.json").exists())
            self.assertTrue((eval_dir / "vlm_score.json").exists())
            self.assertTrue((eval_dir / "vlm_framewise_judgment.csv").exists())
            self.assertTrue((eval_dir / "llm_audit_report.json").exists())
            summary = json.loads((eval_dir / "evaluation_summary.json").read_text())
            self.assertEqual(summary["case"], "stick")
            self.assertEqual(summary["method"], "vlm_gated")
            self.assertGreater(summary["metrics"]["contact_f1"], 0.99)
            self.assertIn("final_pass", result)

    def test_final_evaluator_writes_vlm_llm_qa_summary_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            self._write_minimal_artifacts(profile)

            run_final_evaluator(profile, method="vlm_gated", llm_mode="none")

            eval_dir = profile.result_dir / "vlm_trace" / "06_evaluation"
            self.assertTrue((eval_dir / "vlm_eval_queries.csv").exists())
            self.assertTrue((eval_dir / "vlm_eval_raw_responses.jsonl").exists())
            self.assertTrue((eval_dir / "vlm_eval_parsed_scores.csv").exists())
            self.assertTrue((eval_dir / "vlm_eval_summary.json").exists())
            self.assertTrue((eval_dir / "llm_eval_summary.md").exists())
            self.assertTrue((eval_dir / "qa_audit_report.html").exists())
            parsed = read_csv(eval_dir / "vlm_eval_parsed_scores.csv")
            self.assertGreater(len(parsed), 0)
            for field in [
                "stage",
                "frame",
                "evidence_path",
                "question",
                "raw_answer",
                "visibility",
                "contact_correctness",
                "support_consistency",
                "penetration_absence",
                "temporal_plausibility",
                "overall_plausibility",
                "failure_stage_hint",
                "affected_constraint",
                "changed_optimizer_behavior",
            ]:
                self.assertIn(field, parsed[0])

    def test_final_evaluator_keeps_pipeline_vlm_gates_separate_from_final_judge(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            self._write_minimal_artifacts(profile)
            paths = stage_paths(profile)
            write_csv(
                paths["vlm_dir"] / "stage5" / "vlm_results.csv",
                [
                    {"query_id": "v1", "frame": "1", "query_type": "post_render_sanity_check", "normalized_label": "pass", "pass_gate": "pass", "raw_response": "render is aligned"},
                    {"query_id": "v2", "frame": "2", "query_type": "temporal_motion_check", "normalized_label": "inconsistent", "pass_gate": "reject", "raw_response": "object jumps"},
                ],
            )
            write_csv(
                paths["vlm_dir"] / "stage5" / "vlm_gates.csv",
                [
                    {"query_id": "v1", "frame": "1", "query_type": "post_render_sanity_check", "pass_gate": "pass", "is_effective": "1"},
                    {"query_id": "v2", "frame": "2", "query_type": "temporal_motion_check", "pass_gate": "reject", "is_effective": "1"},
                ],
            )

            run_final_evaluator(profile, method="vlm_gated")

            eval_dir = profile.result_dir / "vlm_trace" / "06_evaluation"
            score = json.loads((eval_dir / "vlm_score.json").read_text())
            framewise = read_csv(eval_dir / "vlm_framewise_judgment.csv")
            self.assertEqual(score["mode"], "dry-run_metric_grounded")
            self.assertEqual(score["pipeline_vlm_gate_score"]["mode"], "existing_vlm_artifacts")
            self.assertAlmostEqual(score["pipeline_vlm_gate_score"]["score"], 0.5)
            self.assertIn("fail", {row["label"] for row in framewise})

    def test_benchmark_outputs_fixed_columns_for_methods(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            self._write_minimal_artifacts(profile)

            result = run_benchmark([profile], methods=["vlm_gated"])

            table = read_csv(profile.result_dir / "benchmark" / "benchmark_table.csv")
            self.assertEqual(len(table), 1)
            self.assertIn("Result Name", table[0])
            self.assertIn("Audio", table[0])
            self.assertIn("VLM", table[0])
            self.assertIn("LLM", table[0])
            self.assertIn("Contact F1 ↑", table[0])
            self.assertIn("Contact Proxy ↑", table[0])
            self.assertIn("Overlay Proxy ↑", table[0])
            self.assertIn("Static Drift ↓", table[0])
            self.assertIn("Failure Stage", table[0])
            self.assertTrue((profile.result_dir / "benchmark" / "benchmark_report.md").exists())
            self.assertTrue((profile.result_dir / "benchmark" / "benchmark_deltas.csv").exists())
            self.assertEqual(result["rows"], 1)

    def test_final_summary_outputs_only_current_final_result(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp, result_name="benchmark_vlm_qwen")
            self._write_minimal_artifacts(profile)

            result = write_final_summary([profile], output_dir=Path(tmp) / "final_summary", rerun_evaluator=True)

            table = read_csv(Path(result["table"]))
            self.assertEqual(result["rows"], 1)
            self.assertEqual(table[0]["Case"], "stick")
            self.assertEqual(table[0]["Final Result"], "benchmark_vlm_qwen")
            self.assertEqual(table[0]["SE3 Pose"], "yes")
            self.assertEqual(table[0]["Translation"], "yes")
            self.assertEqual(table[0]["Rotation"], "yes")
            self.assertIn("Contact Proxy", table[0])
            self.assertNotIn("Method", table[0])
            self.assertTrue((Path(tmp) / "final_summary" / "final_result_evaluation_summary.md").exists())
            self.assertTrue((Path(tmp) / "final_summary" / "final_result_evaluation_summary.html").exists())

    def test_benchmark_rejects_same_result_for_distinct_methods(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            self._write_minimal_artifacts(profile)

            with self.assertRaisesRegex(ValueError, "same result"):
                run_benchmark(
                    [profile],
                    methods=["baseline_no_vlm", "vlm_gated"],
                    method_result_names={"vlm_gated": "unit_result"},
                )

    def test_benchmark_can_map_methods_to_distinct_result_names(self) -> None:
        with TemporaryDirectory() as tmp:
            base = self._profile(tmp, result_name="baseline_result")
            gated = self._profile(tmp, result_name="gated_result")
            self._write_minimal_artifacts(base)
            self._write_minimal_artifacts(gated)
            write_csv(stage_paths(gated)["pose_jump_audit"], [])

            result = run_benchmark(
                [base],
                methods=["baseline_no_vlm", "vlm_gated"],
                method_result_names={"vlm_gated": "gated_result"},
            )

            table = read_csv(base.result_dir / "benchmark" / "benchmark_table.csv")
            self.assertEqual(result["rows"], 2)
            self.assertEqual(table[0]["Method"], "baseline_no_vlm")
            self.assertEqual(table[1]["Method"], "vlm_gated")
            self.assertEqual(table[0]["Result Name"], "baseline_result")
            self.assertEqual(table[1]["Result Name"], "gated_result")
            self.assertEqual(table[0]["Jump ↓"], "1")
            self.assertEqual(table[1]["Jump ↓"], "0")

    def test_benchmark_supports_case_specific_result_mapping_and_missing_variants(self) -> None:
        with TemporaryDirectory() as tmp:
            stick_base = self._profile(tmp, case_name="stick", result_name="baseline_result")
            stick_gated = self._profile(tmp, case_name="stick", result_name="stick_qwen")
            mug_base = self._profile(tmp, case_name="mug", result_name="baseline_result")
            mug_gated = self._profile(tmp, case_name="mug", result_name="mug_qwen")
            for profile in [stick_base, stick_gated, mug_base, mug_gated]:
                self._write_minimal_artifacts(profile)

            result = run_benchmark(
                [stick_base, mug_base],
                methods=["baseline_no_vlm", "vlm_gated", "no_audio"],
                method_result_names={"stick:vlm_gated": "stick_qwen", "mug:vlm_gated": "mug_qwen"},
            )

            table = read_csv(stick_base.result_dir / "benchmark" / "benchmark_table.csv")
            self.assertEqual(result["rows"], 6)
            rows = {(row["Case"], row["Method"]): row for row in table}
            self.assertEqual(rows[("stick", "vlm_gated")]["Result Name"], "stick_qwen")
            self.assertEqual(rows[("mug", "vlm_gated")]["Result Name"], "mug_qwen")
            self.assertEqual(rows[("stick", "no_audio")]["Contact F1 ↑"], "missing:missing_result_mapping")

    def test_oracle_contact_method_uses_manual_labels_as_contact_upper_bound(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            self._write_minimal_artifacts(profile)
            paths = stage_paths(profile)
            write_csv(paths["contact_candidates"], [{"frame": "1", "contact_observed": "0"}, {"frame": "2", "contact_observed": "0"}, {"frame": "3", "contact_observed": "0"}])

            summary = run_final_evaluator(profile, method="oracle_contact")

            self.assertEqual(summary["metrics"]["contact_label_status"], "oracle")
            self.assertEqual(summary["metrics"]["contact_f1"], 1.0)

    def test_metric_fallbacks_fill_proxy_columns_without_manual_labels(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            self._write_minimal_artifacts(profile)
            paths = stage_paths(profile)
            labels = profile.sample_dir / "annotations" / "evaluation" / "contact_labels.csv"
            labels.unlink()
            write_csv(
                paths["object_observations"],
                [
                    {"frame": "1", "observation_conf": "0.8", "proxy_jitter_px": "5", "support_gap_px": "10"},
                    {"frame": "2", "observation_conf": "0.6", "proxy_jitter_px": "7", "support_gap_px": "0"},
                ],
            )
            write_csv(
                paths["contact_candidates"],
                [
                    {"frame": "1", "contact_active": "1", "contact_conf": "0.9", "contact_depth_offset_m": "-0.02"},
                    {"frame": "2", "contact_active": "0", "contact_conf": "0.2", "contact_depth_offset_m": "0.03"},
                ],
            )
            write_csv(
                paths["anchor_state"],
                [
                    {
                        "frame": "1",
                        "stable_local_x": "0.0",
                        "stable_local_y": "0.0",
                        "stable_local_z": "0.0",
                        "observed_local_x": "0.03",
                        "observed_local_y": "0.04",
                        "observed_local_z": "0.0",
                    },
                    {"frame": "2", "stable_local_s": "0.20", "observed_local_s": "0.25"},
                ],
            )

            summary = run_final_evaluator(profile, method="vlm_gated", llm_mode="none")
            metrics = summary["metrics"]

            self.assertEqual(metrics["contact_label_status"], "needs_manual_labels")
            self.assertIsNone(metrics["contact_f1"])
            self.assertGreater(metrics["contact_proxy_score"], 0.0)
            self.assertGreater(metrics["anchor_drift_m_mean"], 0.0)
            self.assertGreater(metrics["penetration_rate"], 0.0)
            self.assertGreater(metrics["floating_rate"], 0.0)
            self.assertGreater(metrics["overlay_proxy_score"], 0.0)

    def test_oracle_contact_without_labels_is_unavailable_in_benchmark(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = self._profile(tmp)
            self._write_minimal_artifacts(profile)
            (profile.sample_dir / "annotations" / "evaluation" / "contact_labels.csv").unlink()

            run_benchmark([profile], methods=["oracle_contact"])

            table = read_csv(profile.result_dir / "benchmark" / "benchmark_table.csv")
            self.assertEqual(table[0]["Contact F1 ↑"], "no_manual_labels")
            self.assertNotEqual(table[0]["Contact Proxy ↑"], table[0]["Contact F1 ↑"])


if __name__ == "__main__":
    unittest.main()
