#!/usr/bin/env python3
"""Materialize fair object-stage VLM/audio evidence ablations.

The tool contains no solver or pose-editing logic.  It freezes the shared input
contract and delegates every executable variant to the same pipeline entrypoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.evaluation.final_hoi.ablation_registry import (  # noqa: E402
    SUITCASE_EVIDENCE_VARIANTS,
)


CASE_NAME = "suitcase_drag"
OUTPUT_ROOT = REPO / "output/suitcase_evidence_ablations"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _shared_contract() -> dict[str, str]:
    profile = load_case_profile(CASE_NAME)
    normalized = copy.deepcopy(profile.data)
    normalized.pop("result_name", None)
    normalized.pop("ablation_flags", None)
    descriptor = REPO / str(profile.data["geometry_asset_descriptor"])
    problem = dict(profile.data.get("generic_object_problem", {}))
    initializer_artifacts = [
        profile.sample_dir / "results" / str(name)
        for name in problem.get("initializer_artifacts", ())
    ]
    initializer_payload = {
        "kind": problem.get("initializer"),
        "artifacts": {
            str(path.relative_to(REPO)): _sha256_file(path) if path.is_file() else "missing"
            for path in initializer_artifacts
        },
    }
    solver_budget = {
        "factor_runtime": profile.data.get("factor_runtime", {}),
        "generic_solver": profile.data.get("generic_solver", {}),
        "generic_object_problem": {
            key: value
            for key, value in problem.items()
            if key not in {"interaction_state_artifact", "contact_artifact"}
        },
    }
    return {
        "shared_config_sha256": _sha256_json(normalized),
        "shared_initializer_sha256": _sha256_json(initializer_payload),
        "shared_geometry_sha256": _sha256_file(descriptor),
        "solver_budget_sha256": _sha256_json(solver_budget),
    }


def _variant_record(variant, shared: dict[str, str]) -> dict[str, object]:
    return {
        "variant": variant.method,
        "case": CASE_NAME,
        "result_name": variant.result_name,
        "vlm_mode": variant.vlm,
        "llm_mode": "none",
        "ablation_flags": sorted(variant.ablation_flags),
        **shared,
        "canonical_write_allowed": False,
        "human_state_optimized": False,
        "status": "planned",
    }


def _command(record: dict[str, object], from_stage: str, to_stage: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.shared.generic_contact_pipeline.run_pipeline",
        "--case",
        str(record["case"]),
        "--from-stage",
        from_stage,
        "--to-stage",
        to_stage,
        "--result-name",
        str(record["result_name"]),
        "--vlm-mode",
        str(record["vlm_mode"]),
        "--llm-mode",
        "none",
    ]
    for flag in record["ablation_flags"]:
        command.extend(("--ablation-flag", str(flag)))
    return command


def build_matrix(selected: set[str] | None = None) -> dict[str, object]:
    shared = _shared_contract()
    variants = [
        _variant_record(variant, shared)
        for variant in SUITCASE_EVIDENCE_VARIANTS
        if selected is None or variant.method in selected
    ]
    if not variants:
        raise ValueError("no evidence ablation variants selected")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "object_stage_only": True,
        "canonical_write_allowed": False,
        "variants": variants,
    }


def execute_matrix(matrix: dict[str, object], from_stage: str, to_stage: str) -> None:
    for record in matrix["variants"]:
        command = _command(record, from_stage, to_stage)
        record["command"] = command
        record["started_at"] = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(command, cwd=REPO, check=False)
        record["ended_at"] = datetime.now(timezone.utc).isoformat()
        record["returncode"] = completed.returncode
        record["status"] = "complete" if completed.returncode == 0 else "failed"
        _write_json_atomic(OUTPUT_ROOT / "run_matrix.json", matrix)
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--variant", action="append", choices=[v.method for v in SUITCASE_EVIDENCE_VARIANTS])
    parser.add_argument("--from-stage", default="stage0")
    parser.add_argument("--to-stage", default="stage4")
    args = parser.parse_args()
    matrix = build_matrix(set(args.variant) if args.variant else None)
    _write_json_atomic(OUTPUT_ROOT / "run_matrix.json", matrix)
    if args.execute:
        execute_matrix(matrix, args.from_stage, args.to_stage)
    print(json.dumps(matrix, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
