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
    AcceptedObjectOutputPublisher,
    GenericSequenceExecutor,
    ObjectPublicationGate,
    SequenceOptimizationParameters,
    CapabilityObjectProblemPreparation,
    capability_object_problem_preparation_record,
    object_publication_record,
    evaluate_object_publication_gate,
    prepare_capability_object_problem,
    write_isolated_sequence_attempt,
    update_isolated_attempt_evidence,
)
from scripts.shared.generic_contact_pipeline.core.factors import factor_arbitration_ledger_record


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
    parser.add_argument(
        "--ablation-flag",
        action="append",
        default=[],
        help="Apply the same typed evidence ablation flags as the production pipeline.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument(
        "--vlm-arbitration",
        choices=("off", "required"),
        default="off",
        help="Require evaluated discrete VLM factor gates, or explicitly disable their solver influence.",
    )
    parser.add_argument(
        "--allow-accepted-write",
        action="store_true",
        help="Allow a passing hard gate to atomically replace canonical object_pose.csv.",
    )
    parser.add_argument(
        "--body-models-root",
        type=Path,
        default=REPO / "third-party/GVHMR/inputs/checkpoints/body_models",
    )
    args = parser.parse_args()
    profile = with_runtime_overrides(
        load_case_profile(args.case),
        result_name=args.result_name,
        ablation_flags=args.ablation_flag,
    )
    prepared = prepare_capability_object_problem(
        profile=profile,
        result_dir=profile.result_dir,
        repository_root=REPO,
        body_models_root=args.body_models_root,
        factor_arbitration_mode=args.vlm_arbitration,
    )
    _write_atomic(args.output, capability_object_problem_preparation_record(prepared))
    if args.solve:
        if args.candidate_dir is None:
            raise SystemExit("--solve requires --candidate-dir")
        result = GenericSequenceExecutor().solve(
            prepared.preparation.problem,
            SequenceOptimizationParameters(max_function_evaluations=args.max_nfev),
        )
        attempt_dir = write_isolated_sequence_attempt(
            args.candidate_dir / "generic_sequence_solver_attempts",
            prepared.preparation.problem,
            result,
        )
        if isinstance(prepared, CapabilityObjectProblemPreparation):
            template_rows = list(prepared.template_rows)
        else:
            with (profile.result_dir / "object_pose_init.csv").open(newline="") as handle:
                import csv

                template_rows = list(csv.DictReader(handle))
        gate, hard_metrics = evaluate_object_publication_gate(prepared.preparation.problem, result)
        if gate.passed and not args.allow_accepted_write:
            gate = ObjectPublicationGate(
                passed=False,
                gate_ids=(*gate.gate_ids, "explicit_promotion_authorized"),
                blocking_reasons=("promotion_not_requested",),
            )
        arbitration_record = (
            factor_arbitration_ledger_record(prepared.factor_arbitration)
            if isinstance(prepared, CapabilityObjectProblemPreparation)
            else None
        )
        if arbitration_record is not None and bool(arbitration_record.get("blocking", False)):
            gate = ObjectPublicationGate(
                passed=False,
                gate_ids=(*gate.gate_ids, "vlm_factor_arbitration_clear"),
                blocking_reasons=(*gate.blocking_reasons, "vlm_factor_arbitration_unclear"),
            )
        update_isolated_attempt_evidence(
            attempt_dir,
            hard_metrics=hard_metrics,
            vlm_gates=arbitration_record,
        )
        publication = AcceptedObjectOutputPublisher().publish(
            result=result,
            state_spec=prepared.state_adaptation.state_spec,
            template_rows=template_rows,
            candidate_dir=args.candidate_dir,
            accepted_result_dir=profile.result_dir,
            gate=gate,
        )
        publication_record = object_publication_record(publication, gate)
        publication_record["attempt_dir"] = str(attempt_dir)
        _write_atomic(args.candidate_dir / "generic_object_publication.json", publication_record)
    print(args.output)


if __name__ == "__main__":
    main()
