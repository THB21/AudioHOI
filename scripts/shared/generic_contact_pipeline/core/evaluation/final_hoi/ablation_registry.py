from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...base.config import CaseProfile, with_runtime_overrides


@dataclass(frozen=True)
class MethodVariant:
    method: str
    result_name: str
    ablation_flags: list[str] = field(default_factory=list)
    audio: bool | None = None
    vlm: str | None = None
    llm: str | None = None
    required: bool = True
    mechanism: str = ""
    mechanism_supported: bool = True


DEFAULT_VARIANTS = [
    MethodVariant("full_audio_vlm_llm", "clean_ablation_full_audio_vlm_llm", [], audio=True, vlm="qwen", llm="mistral", mechanism="reference_full_pipeline"),
    MethodVariant("no_audio", "clean_ablation_no_audio", ["disable_audio_events"], audio=False, vlm="qwen", llm="mistral", mechanism="disable_audio_events runtime consumers"),
    MethodVariant("no_vlm_llm", "clean_ablation_no_vlm_llm", [], audio=True, vlm="none", llm="none", mechanism="vlm_mode=none and llm_mode=none"),
    MethodVariant("audio_enabled", "benchmark_audio_enabled", [], audio=True, vlm="qwen", llm="mistral", required=False, mechanism="reference_audio_enabled_pipeline"),
    MethodVariant("no_vlm", "benchmark_baseline_no_vlm", [], audio=True, vlm="none", llm="mistral", mechanism="vlm_mode=none"),
    MethodVariant("no_llm", "benchmark_no_llm", [], audio=True, vlm="qwen", llm="none", mechanism="llm_mode=none"),
    MethodVariant(
        "no_contact_anchor", "benchmark_no_anchor", [], audio=True, vlm="qwen", llm="mistral",
        required=False, mechanism="unverified legacy result; no runtime consumer", mechanism_supported=False,
    ),
    MethodVariant(
        "object_only", "benchmark_object_only", [], audio=True, vlm="qwen", llm="mistral",
        required=False, mechanism="render/evaluation label; not a solver intervention", mechanism_supported=False,
    ),
]

MATERIALIZED_DEFAULT_METHODS = [
    "full_audio_vlm_llm",
    "no_audio",
    "no_vlm_llm",
]

MATERIALIZED_DEFAULT_VARIANTS = [
    variant for variant in DEFAULT_VARIANTS if variant.method in MATERIALIZED_DEFAULT_METHODS
]


def resolve_variant_profile(profile: CaseProfile, variant: MethodVariant) -> CaseProfile:
    return with_runtime_overrides(profile, result_name=variant.result_name, ablation_flags=variant.ablation_flags)


def validate_method_result_mapping(
    profiles: list[CaseProfile],
    variants: list[MethodVariant],
    *,
    allow_same_result_debug: bool = False,
    require_existing: bool = False,
) -> dict[str, dict[str, Path]]:
    mapping: dict[str, dict[str, Path]] = {}
    for profile in profiles:
        seen: dict[Path, str] = {}
        per_case: dict[str, Path] = {}
        for variant in variants:
            variant_profile = resolve_variant_profile(profile, variant)
            result_dir = variant_profile.result_dir.resolve()
            if not allow_same_result_debug and result_dir in seen:
                raise ValueError(
                    f"{profile.case_name}: methods {seen[result_dir]!r} and {variant.method!r} map to the same result directory {result_dir}"
                )
            if require_existing and variant.required and not result_dir.exists():
                raise FileNotFoundError(f"{profile.case_name}:{variant.method} missing result directory {result_dir}")
            seen[result_dir] = variant.method
            per_case[variant.method] = result_dir
        mapping[profile.case_name] = per_case
    return mapping
