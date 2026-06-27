from __future__ import annotations

import json
import os
import subprocess
import sys
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ..base.config import CaseProfile
from ..base.io import REPO, repo_path, write_json
from ..base.schema import stage_paths
from ..gates.vlm_provider import load_vlm_provider


DEFAULT_QUERY_POLICY = {
    "stage0": ["target_mask_check"],
    "stage1": ["keypart_identity_check", "track_stability_check"],
    "stage2": ["contact_relation_check"],
    "stage3": ["overlay_alignment_check"],
    "stage4": ["anchor_update_check", "temporal_motion_check"],
    "stage5": ["post_render_sanity_check"],
}


REQUIRED_LIST_FIELDS = [
    "object_parts",
    "human_parts",
    "interaction_edges",
    "support_priors",
    "motion_priors",
]


def seed_profile_path(profile: CaseProfile) -> Path:
    return profile.sample_dir / "annotations" / "hoi_profile_seed.json"


def default_profile(profile: CaseProfile) -> dict[str, Any]:
    vlm = profile.data.get("vlm", {}) if isinstance(profile.data.get("vlm"), dict) else {}
    parts = list(vlm.get("parts") or ["main_body", "contact_region", "support_region"])
    contacts = list(vlm.get("contacts") or ["hand", "foot", "floor", "table"])
    clean_parts = [p for p in parts if p not in {"background", "unclear"}]
    human_parts = sorted({c for c in contacts if c not in {"nothing_nearby", "unclear", "floor", "table", "body"}})
    if not human_parts:
        human_parts = ["hand"]
    return {
        "case_name": profile.case_name,
        "object_category": str(vlm.get("target") or profile.case_name),
        "object_family": str(profile.data.get("object_family", "object")),
        "object_parts": [{"name": part, "role": "semantic_or_contact_candidate"} for part in clean_parts],
        "human_parts": human_parts,
        "interaction_edges": [
            {"human_part": hp, "object_part": clean_parts[0] if clean_parts else "main_body", "relation": "candidate_contact"}
            for hp in human_parts
        ],
        "support_priors": ["floor_support"] if "floor" in contacts else [],
        "motion_priors": ["smooth_temporal_motion"],
        "vlm_query_policy": deepcopy(DEFAULT_QUERY_POLICY),
        "source": "default_from_case_config",
    }


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _short_text(text: str, limit: int = 2400) -> str:
    text = " ".join(text.replace("\x00", " ").split())
    return text[:limit]


def _add_prompt_entry(entries: list[dict[str, Any]], *, kind: str, path: Path, text: str, record_id: str = "") -> None:
    clean = _short_text(text)
    if not clean:
        return
    digest = hashlib.sha256(clean.encode("utf-8", errors="ignore")).hexdigest()
    if any(item.get("sha256") == digest for item in entries):
        return
    entries.append(
        {
            "kind": kind,
            "path": str(path.relative_to(REPO) if path.is_absolute() and path.is_relative_to(REPO) else path),
            "record_id": record_id,
            "text_preview": clean,
            "sha256": digest,
        }
    )


def _read_text_if_small(path: Path, *, limit_bytes: int = 80_000) -> str:
    try:
        if not path.exists() or not path.is_file() or path.stat().st_size > limit_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def collect_prompt_context(profile: CaseProfile, *, max_entries: int = 18) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    missing: list[str] = []

    metadata_path = profile.sample_dir / "metadata.json"
    metadata = read_json_if_exists(metadata_path)
    if metadata:
        for key in ("prompt", "detection_text", "name", "video", "image"):
            value = metadata.get(key)
            if value:
                _add_prompt_entry(entries, kind=f"sample_metadata.{key}", path=metadata_path, text=str(value))
    else:
        missing.append(str(metadata_path))

    prompt_candidates: list[Path] = []
    prompt_candidates.extend(profile.sample_dir.glob("*prompt*.txt"))
    prompt_candidates.extend((profile.sample_dir / "articraft").glob("*prompt*.txt"))
    prompt_candidates.extend((profile.sample_dir / "articraft" / "clean_mug_reference_pack").glob("*prompt*.txt"))
    prompt_candidates.extend((profile.sample_dir / "articraft" / "generated_record").glob("**/prompt.txt"))
    prompt_candidates.extend((profile.sample_dir / "annotations").glob("**/prompt.txt"))
    for path in sorted({p for p in prompt_candidates if p.exists()}):
        _add_prompt_entry(entries, kind="prompt_txt", path=path, text=_read_text_if_small(path))
        if len(entries) >= max_entries:
            break

    record_candidates: list[Path] = []
    record_candidates.extend((profile.sample_dir / "articraft" / "generated_record").glob("**/record*.json"))
    record_candidates.extend((profile.sample_dir / "proxy").glob("*.json"))
    for path in sorted({p for p in record_candidates if p.exists()}):
        data = read_json_if_exists(path)
        if not data:
            continue
        record_id = str(data.get("record_id") or "")
        display = data.get("display") if isinstance(data.get("display"), dict) else {}
        for key in ("title", "prompt_preview"):
            if display.get(key):
                _add_prompt_entry(entries, kind=f"record_display.{key}", path=path, text=str(display[key]), record_id=record_id)
        if data.get("prompt"):
            _add_prompt_entry(entries, kind="record.prompt", path=path, text=str(data["prompt"]), record_id=record_id)
        artifacts = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
        prompt_rel = artifacts.get("prompt_txt")
        if prompt_rel:
            prompt_path = path.parent / str(prompt_rel)
            text = _read_text_if_small(prompt_path)
            if text:
                _add_prompt_entry(entries, kind="record_artifact.prompt_txt", path=prompt_path, text=text, record_id=record_id)
            else:
                missing.append(str(prompt_path))
        if len(entries) >= max_entries:
            break

    selected = entries[:max_entries]
    return {
        "case_name": profile.case_name,
        "result_name": profile.result_name,
        "prompt_entry_count": len(selected),
        "entries": selected,
        "missing_referenced_prompts": missing[:20],
        "usage_rule": "Prompt context is semantic evidence for LLM/VLM query design only; it must not supply continuous pose, coordinates, or loss weights.",
    }


def attach_prompt_context(resolved: dict[str, Any], prompt_context: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(resolved)
    entries = prompt_context.get("entries", []) if isinstance(prompt_context, dict) else []
    data["source_prompt_count"] = len(entries) if isinstance(entries, list) else 0
    data["source_prompts"] = [
        {
            "kind": item.get("kind", ""),
            "path": item.get("path", ""),
            "record_id": item.get("record_id", ""),
            "text_preview": item.get("text_preview", ""),
        }
        for item in entries[:8]
        if isinstance(item, dict)
    ]
    return data


def normalize_part_list(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, str):
            name = item.strip()
            role = "semantic_or_contact_candidate"
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            role = str(item.get("role", "semantic_or_contact_candidate")).strip()
        else:
            continue
        if name and name not in {"background", "unclear"}:
            out.append({"name": name, "role": role or "semantic_or_contact_candidate"})
    return out


def normalize_string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        value = str(item).strip()
        if value and value not in out:
            out.append(value)
    return out


def normalize_edges(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        human = str(item.get("human_part", "")).strip()
        obj = str(item.get("object_part", "")).strip()
        rel = str(item.get("relation", "contact")).strip()
        if human and obj:
            out.append({"human_part": human, "object_part": obj, "relation": rel or "contact"})
    return out


def normalize_query_policy(raw: Any) -> dict[str, list[str]]:
    base = deepcopy(DEFAULT_QUERY_POLICY)
    if not isinstance(raw, dict):
        return base
    for stage, queries in raw.items():
        values = normalize_string_list(queries)
        if values:
            base[str(stage)] = values
    return base


def validate_and_resolve(profile: CaseProfile, raw: dict[str, Any] | None, *, source: str) -> tuple[dict[str, Any], list[str]]:
    base = default_profile(profile)
    warnings: list[str] = []
    data = deepcopy(base)
    if raw:
        for key, value in raw.items():
            data[key] = value
    data["case_name"] = profile.case_name
    data["object_family"] = str(data.get("object_family") or profile.data.get("object_family", "object"))
    data["object_category"] = str(data.get("object_category") or data.get("object_family") or profile.case_name)
    data["object_parts"] = normalize_part_list(data.get("object_parts"))
    data["human_parts"] = normalize_string_list(data.get("human_parts"))
    data["interaction_edges"] = normalize_edges(data.get("interaction_edges"))
    data["support_priors"] = normalize_string_list(data.get("support_priors"))
    data["motion_priors"] = normalize_string_list(data.get("motion_priors"))
    data["vlm_query_policy"] = normalize_query_policy(data.get("vlm_query_policy"))
    for field in REQUIRED_LIST_FIELDS:
        if not data.get(field):
            warnings.append(f"{field} missing or empty; filled from case config defaults")
            data[field] = base[field]
    if not data.get("object_parts"):
        warnings.append("object_parts still empty after fallback")
    data["source"] = source
    data["schema_version"] = 1
    data["rules"] = {
        "llm_outputs_pose": False,
        "vlm_outputs_pose": False,
        "vlm_outputs_loss_weights": False,
        "optimizer_is_only_continuous_solver": True,
    }
    return data, warnings


def load_resolved_hoi_profile(profile: CaseProfile) -> dict[str, Any] | None:
    return read_json_if_exists(stage_paths(profile)["hoi_profile"])


def write_resolved_profile(profile: CaseProfile, resolved: dict[str, Any], trace: dict[str, Any]) -> dict[str, Path]:
    paths = stage_paths(profile)
    prompt_context = collect_prompt_context(profile)
    write_json(paths["prompt_context"], prompt_context)
    write_json(paths["hoi_profile"], resolved)
    paths["hoi_profile_resolved"].parent.mkdir(parents=True, exist_ok=True)
    paths["hoi_profile_resolved"].write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(paths["llm_prior_trace"], trace)
    return {
        "hoi_profile": paths["hoi_profile"],
        "hoi_profile_resolved": paths["hoi_profile_resolved"],
        "llm_prior_trace": paths["llm_prior_trace"],
        "prompt_context": paths["prompt_context"],
    }


def build_profile_from_seed(profile: CaseProfile) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_path = seed_profile_path(profile)
    seed = read_json_if_exists(seed_path)
    source = "seed_profile" if seed is not None else "case_config_default"
    resolved, warnings = validate_and_resolve(profile, seed, source=source)
    prompt_context = collect_prompt_context(profile)
    resolved = attach_prompt_context(resolved, prompt_context)
    trace = {
        "case_name": profile.case_name,
        "mode": "seed",
        "seed_profile": str(seed_path),
        "seed_exists": seed is not None,
        "prompt_context": str(stage_paths(profile)["prompt_context"]),
        "prompt_entry_count": prompt_context.get("prompt_entry_count", 0),
        "warnings": warnings,
        "fallback_used": seed is None,
    }
    return resolved, trace


def build_profile_from_none(profile: CaseProfile) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved, warnings = validate_and_resolve(profile, None, source="case_config_default")
    prompt_context = collect_prompt_context(profile)
    resolved = attach_prompt_context(resolved, prompt_context)
    trace = {
        "case_name": profile.case_name,
        "mode": "none",
        "prompt_context": str(stage_paths(profile)["prompt_context"]),
        "prompt_entry_count": prompt_context.get("prompt_entry_count", 0),
        "warnings": warnings,
        "fallback_used": True,
    }
    return resolved, trace


def resolve_hoi_profile(profile: CaseProfile, mode: str = "seed") -> tuple[dict[str, Any], dict[str, Any]]:
    if mode == "none":
        return build_profile_from_none(profile)
    if mode == "seed":
        return build_profile_from_seed(profile)
    if mode == "qwen":
        seed_resolved, seed_trace = build_profile_from_seed(profile)
        provider = load_vlm_provider()
        script = REPO / "scripts/shared/generic_contact_pipeline/stages/gates/stage_llm_qwen_profile.py"
        cmd = [
            str(provider.python),
            str(script),
            "--case",
            profile.case_name,
            "--result-name",
            profile.result_name,
            "--provider",
            provider.name,
            "--model-id",
            provider.model_id,
            "--local-dir",
            str(provider.local_dir) if provider.local_dir else "",
            "--device-map",
            provider.device_map,
        ]
        cmd.append("--load-4bit" if provider.load_4bit else "--no-load-4bit")
        env = os.environ.copy()
        env.update(provider.env)
        try:
            proc = subprocess.run(cmd, cwd=REPO, env=env, text=True, capture_output=True, check=False, timeout=900)
            raw_path = stage_paths(profile)["llm_prior_trace"].parent / "llm_prior_qwen_raw.json"
            raw = read_json_if_exists(raw_path)
            parsed = raw.get("parsed_json") if isinstance(raw, dict) else None
            if proc.returncode == 0 and isinstance(parsed, dict):
                resolved, warnings = validate_and_resolve(profile, parsed, source="qwen_llm_schema_validated")
                prompt_context = collect_prompt_context(profile)
                resolved = attach_prompt_context(resolved, prompt_context)
                trace = {
                    "case_name": profile.case_name,
                    "mode": "qwen",
                    "qwen_status": "ok",
                    "qwen_raw_path": str(raw_path),
                    "prompt_context": str(stage_paths(profile)["prompt_context"]),
                    "prompt_entry_count": prompt_context.get("prompt_entry_count", 0),
                    "warnings": warnings,
                    "fallback_used": False,
                    "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-2000:],
                }
                return resolved, trace
            seed_trace.update(
                {
                    "mode": "qwen",
                    "qwen_status": "failed_or_invalid_json",
                    "qwen_returncode": proc.returncode,
                    "qwen_raw_path": str(raw_path),
                    "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-2000:],
                    "fallback_used": True,
                }
            )
            return seed_resolved, seed_trace
        except Exception as exc:
            seed_trace.update(
                {
                    "mode": "qwen",
                    "qwen_status": "exception",
                    "qwen_error": repr(exc),
                    "fallback_used": True,
                }
            )
            return seed_resolved, seed_trace
    if mode == "mistral":
        seed_resolved, seed_trace = build_profile_from_seed(profile)
        script = REPO / "scripts/shared/generic_contact_pipeline/stages/gates/stage_llm_mistral_profile.py"
        cmd = [
            sys.executable,
            str(script),
            "--case",
            profile.case_name,
            "--result-name",
            profile.result_name,
        ]
        try:
            proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=False, timeout=240)
            raw_path = stage_paths(profile)["llm_prior_trace"].parent / "llm_prior_mistral_raw.json"
            raw = read_json_if_exists(raw_path)
            parsed = raw.get("parsed_json") if isinstance(raw, dict) else None
            if proc.returncode == 0 and isinstance(parsed, dict):
                resolved, warnings = validate_and_resolve(profile, parsed, source="mistral_llm_schema_validated")
                prompt_context = collect_prompt_context(profile)
                resolved = attach_prompt_context(resolved, prompt_context)
                trace = {
                    "case_name": profile.case_name,
                    "mode": "mistral",
                    "mistral_status": "ok",
                    "mistral_raw_path": str(raw_path),
                    "prompt_context": str(stage_paths(profile)["prompt_context"]),
                    "prompt_entry_count": prompt_context.get("prompt_entry_count", 0),
                    "warnings": warnings,
                    "fallback_used": False,
                    "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-2000:],
                }
                return resolved, trace
            seed_trace.update(
                {
                    "mode": "mistral",
                    "mistral_status": "failed_or_invalid_json",
                    "mistral_returncode": proc.returncode,
                    "mistral_raw_path": str(raw_path),
                    "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-2000:],
                    "fallback_used": True,
                }
            )
            return seed_resolved, seed_trace
        except Exception as exc:
            seed_trace.update(
                {
                    "mode": "mistral",
                    "mistral_status": "exception",
                    "mistral_error": repr(exc),
                    "fallback_used": True,
                }
            )
            return seed_resolved, seed_trace
    raise ValueError(f"Unknown LLM mode {mode!r}; expected none, seed, qwen, or mistral")


def profile_choices_for_vlm(profile: CaseProfile) -> dict[str, Any] | None:
    resolved = load_resolved_hoi_profile(profile)
    if resolved is None:
        return None
    parts = [str(item.get("name")) for item in resolved.get("object_parts", []) if isinstance(item, dict) and item.get("name")]
    parts = [p for p in parts if p not in {"background", "unclear"}]
    contacts = []
    for edge in resolved.get("interaction_edges", []):
        if not isinstance(edge, dict):
            continue
        human = str(edge.get("human_part", "")).strip()
        obj = str(edge.get("object_part", "")).strip()
        rel = str(edge.get("relation", "contact")).strip()
        if human and obj:
            contacts.append(f"{human}_on_{obj}" if rel in {"hold", "touch", "contact", "candidate_contact"} else f"{human}_{rel}_{obj}")
    for support in resolved.get("support_priors", []):
        support = str(support).strip()
        if support:
            contacts.append(support)
    contacts.extend(["no_contact", "unclear"])
    contact_choices = []
    for item in contacts:
        if item not in contact_choices:
            contact_choices.append(item)
    part_choices = parts + ["background", "unclear"]
    stable_refs = parts[:6] + ["unclear"]
    return {
        "target": str(resolved.get("object_category") or profile.case_name),
        "parts": part_choices,
        "contacts": contact_choices,
        "stable_refs": stable_refs,
        "source": str(resolved.get("source", "hoi_profile")),
        "vlm_query_policy": resolved.get("vlm_query_policy", {}),
    }
