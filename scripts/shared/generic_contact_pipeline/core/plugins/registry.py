from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable


class PluginResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PluginSpec:
    kind: str
    name: str
    module: str
    entrypoint: str
    requires: tuple[str, ...]
    provides: tuple[str, ...]
    role: str = "compatibility_adapter"

    @property
    def plugin_id(self) -> str:
        return f"{self.kind}:{self.name}"

    def load(self) -> Callable[..., object]:
        module = importlib.import_module(self.module)
        entrypoint = getattr(module, self.entrypoint, None)
        if not callable(entrypoint):
            raise PluginResolutionError(
                f"{self.plugin_id} entrypoint {self.module}:{self.entrypoint} is not callable"
            )
        return entrypoint

    def describe(self) -> dict[str, object]:
        return asdict(self) | {"plugin_id": self.plugin_id}


class CapabilityRegistry:
    def __init__(self, specs: Iterable[PluginSpec] = ()):
        self._specs: dict[tuple[str, str], PluginSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: PluginSpec) -> None:
        key = (spec.kind, spec.name)
        if key in self._specs:
            raise ValueError(f"Duplicate plugin registration: {spec.plugin_id}")
        self._specs[key] = spec

    def get(self, kind: str, name: str) -> PluginSpec:
        try:
            return self._specs[(kind, name)]
        except KeyError as exc:
            available = sorted(spec.name for key, spec in self._specs.items() if key[0] == kind)
            raise PluginResolutionError(
                f"Unknown {kind} plugin {name!r}; available: {available}"
            ) from exc

    def all(self) -> list[PluginSpec]:
        return [self._specs[key] for key in sorted(self._specs)]


OBSERVATION_MODULE = "scripts.shared.generic_contact_pipeline.components.observation.policies"
CONTACT_MODULE = "scripts.shared.generic_contact_pipeline.components.contact.policies"
POSE_MODULE = "scripts.shared.generic_contact_pipeline.components.pose.models"
REFINEMENT_MODULE = "scripts.shared.generic_contact_pipeline.components.refinement.policies"


REGISTRY = CapabilityRegistry(
    [
        PluginSpec("observation", "mask_track_center", f"{OBSERVATION_MODULE}.mask_track_center", "build", (), ("observation.point_reference",)),
        PluginSpec("observation", "rigid_body_parts", f"{OBSERVATION_MODULE}.rigid_body_parts", "build", (), ("observation.rigid_parts", "geometry.local_xyz")),
        PluginSpec("observation", "rigid_body_plus_parts", f"{OBSERVATION_MODULE}.rigid_body_plus_parts", "build", (), ("observation.rigid_parts", "geometry.local_xyz")),
        PluginSpec("observation", "semantic_graph_tracks", f"{OBSERVATION_MODULE}.semantic_graph_tracks", "build", (), ("observation.semantic_graph", "geometry.semantic_segments")),
        PluginSpec("contact", "hand_floor", f"{CONTACT_MODULE}.hand_floor", "build", ("observation.point_reference",), ("contact.candidates", "anchor.support")),
        PluginSpec("contact", "foot_floor", f"{CONTACT_MODULE}.foot_floor", "build", ("observation.point_reference",), ("contact.candidates", "anchor.support")),
        PluginSpec("contact", "palm_handle", f"{CONTACT_MODULE}.palm_handle", "build", ("observation.rigid_parts",), ("contact.candidates", "anchor.local_xyz")),
        PluginSpec("contact", "palm_handle_rim_body", f"{CONTACT_MODULE}.palm_handle_rim_body", "build", ("observation.rigid_parts",), ("contact.candidates", "anchor.local_xyz")),
        PluginSpec("contact", "persistent_two_palm_line", f"{CONTACT_MODULE}.persistent_two_palm_line", "build", ("observation.point_reference", "geometry.line_object"), ("contact.candidates", "anchor.local_s")),
        PluginSpec("contact", "two_hand_endpoint", f"{CONTACT_MODULE}.two_hand_endpoint", "build", ("observation.semantic_graph",), ("contact.candidates", "anchor.local_xyz")),
        PluginSpec("contact", "two_hand_toprail_endpoint", f"{CONTACT_MODULE}.two_hand_toprail_endpoint", "build", ("observation.semantic_graph",), ("contact.candidates", "anchor.local_xyz")),
        PluginSpec("pose", "translation3", f"{POSE_MODULE}.translation3", "build", ("observation.point_reference", "contact.candidates"), ("pose.se3",)),
        PluginSpec("pose", "rigid6", f"{POSE_MODULE}.rigid6", "build", ("observation.rigid_parts", "contact.candidates"), ("pose.se3",)),
        PluginSpec("pose", "rigid6_plus_phase", f"{POSE_MODULE}.rigid6_plus_phase", "build", ("observation.rigid_parts", "contact.candidates"), ("pose.se3", "articulation.phase")),
        PluginSpec("pose", "semantic_graph_6d", f"{POSE_MODULE}.semantic_graph_6d", "build", ("observation.semantic_graph", "contact.candidates"), ("pose.se3", "articulation.support")),
        PluginSpec("refinement", "anchor_depth", f"{REFINEMENT_MODULE}.anchor_depth", "apply", ("pose.se3", "contact.candidates"), ("pose.seed_refined",)),
        PluginSpec("refinement", "generic_sphere_sequence", f"{REFINEMENT_MODULE}.generic_sphere_sequence", "apply", ("pose.se3", "contact.candidates"), ("pose.seed_refined",), role="mainline_implementation"),
        PluginSpec("refinement", "anchor_propagate_freeze", f"{REFINEMENT_MODULE}.anchor_propagate_freeze", "apply", ("pose.se3", "anchor.local_xyz"), ("pose.seed_refined",)),
        PluginSpec("refinement", "backproject_xy", f"{REFINEMENT_MODULE}.backproject_xy", "apply", ("pose.se3",), ("pose.seed_refined",)),
        PluginSpec("refinement", "line_contact_lock", f"{REFINEMENT_MODULE}.line_contact_lock", "apply", ("pose.se3", "anchor.local_s", "geometry.line_object"), ("pose.seed_refined",)),
        PluginSpec("refinement", "small_se3", f"{REFINEMENT_MODULE}.small_se3", "apply", ("pose.se3", "anchor.local_xyz"), ("pose.seed_refined",)),
        PluginSpec("refinement", "stable_grasp_anchor", f"{REFINEMENT_MODULE}.stable_grasp_anchor", "apply", ("pose.se3", "anchor.local_xyz"), ("pose.seed_refined",)),
        PluginSpec("refinement", "table_freeze", f"{REFINEMENT_MODULE}.table_freeze", "apply", ("pose.se3",), ("pose.seed_refined",)),
        PluginSpec("refinement", "generic_line_physical_smooth", f"{REFINEMENT_MODULE}.generic_line_physical_smooth", "apply", ("pose.se3", "geometry.line_object"), ("pose.sequence_refined",), role="mainline_implementation"),
        PluginSpec("refinement", "sequence_se3_optimizer", "scripts.shared.generic_contact_pipeline.components.refinement.sequence_se3_optimizer", "smooth_quaternion_pose_sequence", ("pose.se3",), ("pose.sequence_refined",), role="mainline_marker_and_implementation"),
    ]
)


@dataclass(frozen=True)
class ResolvedPipelinePlugins:
    observation: PluginSpec
    contact: PluginSpec
    pose: PluginSpec
    refinement: tuple[PluginSpec, ...]
    implicit_mainline: tuple[PluginSpec, ...]
    intrinsic_capabilities: tuple[str, ...]
    final_capabilities: tuple[str, ...]

    def for_stage(self, stage_name: str) -> tuple[PluginSpec, ...]:
        if stage_name == "stage1":
            return (self.observation,)
        if stage_name == "stage2":
            return (self.contact,)
        if stage_name == "stage3":
            return (self.pose,)
        if stage_name == "stage4":
            combined = (*self.refinement, *self.implicit_mainline)
            by_id: dict[str, PluginSpec] = {}
            for spec in combined:
                by_id.setdefault(spec.plugin_id, spec)
            return tuple(by_id.values())
        return ()

    def describe(self) -> dict[str, object]:
        return {
            "observation": self.observation.describe(),
            "contact": self.contact.describe(),
            "pose": self.pose.describe(),
            "refinement": [spec.describe() for spec in self.refinement],
            "implicit_mainline": [spec.describe() for spec in self.implicit_mainline],
            "intrinsic_capabilities": list(self.intrinsic_capabilities),
            "final_capabilities": list(self.final_capabilities),
        }


def _require(spec: PluginSpec, available: set[str]) -> None:
    missing = sorted(set(spec.requires) - available)
    if missing:
        raise PluginResolutionError(
            f"{spec.plugin_id} missing required capabilities {missing}; available={sorted(available)}"
        )


def resolve_pipeline_plugins(profile: Any) -> ResolvedPipelinePlugins:
    intrinsic = {"input.video", "input.audio"}
    if profile.data.get("line_object"):
        intrinsic.add("geometry.line_object")
    available = set(intrinsic)

    observation = REGISTRY.get("observation", profile.component("observation_model"))
    _require(observation, available)
    available.update(observation.provides)

    contact = REGISTRY.get("contact", profile.component("contact_policy"))
    _require(contact, available)
    available.update(contact.provides)

    pose = REGISTRY.get("pose", profile.component("pose_model"))
    _require(pose, available)
    available.update(pose.provides)

    refinement: list[PluginSpec] = []
    for name in profile.refinement_policies():
        spec = REGISTRY.get("refinement", name)
        _require(spec, available)
        available.update(spec.provides)
        refinement.append(spec)

    implicit_mainline = [REGISTRY.get("refinement", "sequence_se3_optimizer")]
    if profile.data.get("line_object"):
        implicit_mainline.insert(0, REGISTRY.get("refinement", "generic_line_physical_smooth"))
    for spec in implicit_mainline:
        _require(spec, available)
        available.update(spec.provides)

    return ResolvedPipelinePlugins(
        observation=observation,
        contact=contact,
        pose=pose,
        refinement=tuple(refinement),
        implicit_mainline=tuple(implicit_mainline),
        intrinsic_capabilities=tuple(sorted(intrinsic)),
        final_capabilities=tuple(sorted(available)),
    )


def invoke_selected_plugin(
    profile: Any,
    kind: str,
    name: str,
    *args: object,
    **kwargs: object,
) -> tuple[object, dict[str, object]]:
    resolved = resolve_pipeline_plugins(profile)
    selected = {
        "observation": (resolved.observation,),
        "contact": (resolved.contact,),
        "pose": (resolved.pose,),
        "refinement": (*resolved.refinement, *resolved.implicit_mainline),
    }.get(kind, ())
    spec = next((candidate for candidate in selected if candidate.name == name), None)
    if spec is None:
        raise PluginResolutionError(
            f"{kind}:{name} is not selected by case profile {profile.case_name}"
        )
    result = spec.load()(profile, *args, **kwargs)
    return result, spec.describe()


def stage_plugin_audit(profile: Any, stage_name: str) -> dict[str, object]:
    resolved = resolve_pipeline_plugins(profile)
    return {
        "schema_version": 1,
        "stage": stage_name,
        "plugins": [spec.describe() for spec in resolved.for_stage(stage_name)],
        "final_capabilities": list(resolved.final_capabilities),
    }
