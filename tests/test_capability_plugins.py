from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.shared.generic_contact_pipeline.core.base.config import CaseProfile, load_case_profile
from scripts.shared.generic_contact_pipeline.core.plugins.registry import (
    REGISTRY,
    CapabilityRegistry,
    PluginResolutionError,
    PluginSpec,
    resolve_pipeline_plugins,
    stage_plugin_audit,
)


class CapabilityPluginsTest(unittest.TestCase):
    def test_five_case_plugin_matrix_resolves_with_declared_capabilities(self) -> None:
        expected = {
            "basketball": ("mask_track_center", "hand_floor", "translation3", ["generic_sphere_sequence", "backproject_xy"]),
            "football": ("mask_track_center", "foot_floor", "translation3", ["generic_sphere_sequence", "backproject_xy"]),
            "mug": ("rigid_body_plus_parts", "palm_handle_rim_body", "rigid6_plus_phase", ["stable_grasp_anchor", "anchor_depth", "table_freeze"]),
            "chair": ("semantic_graph_tracks", "two_hand_toprail_endpoint", "semantic_graph_6d", ["small_se3", "anchor_propagate_freeze", "sequence_se3_optimizer"]),
            "stick": ("mask_track_center", "persistent_two_palm_line", "translation3", ["line_contact_lock", "backproject_xy"]),
        }
        for case_name, selectors in expected.items():
            resolved = resolve_pipeline_plugins(load_case_profile(case_name))
            self.assertEqual(resolved.observation.name, selectors[0])
            self.assertEqual(resolved.contact.name, selectors[1])
            self.assertEqual(resolved.pose.name, selectors[2])
            self.assertEqual([spec.name for spec in resolved.refinement], selectors[3])
            if case_name in {"basketball", "football"}:
                self.assertEqual(resolved.refinement[0].role, "mainline_implementation")
            self.assertIn("pose.se3", resolved.final_capabilities)
            audit = stage_plugin_audit(load_case_profile(case_name), "stage4")
            expected_active = list(selectors[3])
            if case_name == "stick":
                expected_active.append("generic_line_physical_smooth")
            if "sequence_se3_optimizer" not in expected_active:
                expected_active.append("sequence_se3_optimizer")
            self.assertEqual([item["name"] for item in audit["plugins"]], expected_active)

    def test_registry_rejects_duplicate_and_unknown_plugins(self) -> None:
        spec = PluginSpec("observation", "fixture", "fixture.module", "build", (), ("fixture",))
        registry = CapabilityRegistry([spec])
        with self.assertRaisesRegex(ValueError, "Duplicate plugin"):
            registry.register(spec)
        with self.assertRaisesRegex(PluginResolutionError, "Unknown pose plugin"):
            registry.get("pose", "missing")

    def test_resolution_rejects_missing_declared_capability_before_execution(self) -> None:
        profile = CaseProfile(
            {
                "case_name": "invalid_line_case",
                "sample_dir": "samples_known_object/11_stick",
                "observation_model": "mask_track_center",
                "contact_policy": "persistent_two_palm_line",
                "pose_model": "translation3",
                "refinement_policy": [],
            }
        )
        with self.assertRaisesRegex(PluginResolutionError, "geometry.line_object"):
            resolve_pipeline_plugins(profile)

    def test_every_registered_plugin_points_to_an_explicit_repository_module(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        plugin_ids = [spec.plugin_id for spec in REGISTRY.all()]
        self.assertEqual(len(plugin_ids), len(set(plugin_ids)))
        for spec in REGISTRY.all():
            module_path = repo / (spec.module.replace(".", "/") + ".py")
            self.assertTrue(module_path.is_file(), spec.plugin_id)
            self.assertTrue(spec.requires or spec.kind == "observation")
            self.assertTrue(spec.provides)


if __name__ == "__main__":
    unittest.main()
