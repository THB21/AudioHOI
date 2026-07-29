#!/usr/bin/env python3
"""Materialize a safe generic object-problem preparation ledger."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides
from scripts.shared.generic_contact_pipeline.core.base.io import REPO
from scripts.shared.generic_contact_pipeline.core.solver import (
    legacy_object_problem_preparation_record,
    prepare_legacy_articulated_object_problem,
)


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--result-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--body-models-root",
        type=Path,
        default=REPO / "third-party/GVHMR/inputs/checkpoints/body_models",
    )
    args = parser.parse_args()
    profile = with_runtime_overrides(load_case_profile(args.case), result_name=args.result_name)
    prepared = prepare_legacy_articulated_object_problem(
        profile=profile,
        result_dir=profile.result_dir,
        repository_root=REPO,
        body_models_root=args.body_models_root,
    )
    _write_atomic(args.output, legacy_object_problem_preparation_record(prepared))
    print(args.output)


if __name__ == "__main__":
    main()
