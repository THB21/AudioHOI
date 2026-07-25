#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.state import build_canonical_state_parity_reports  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.state.golden import CANONICAL_CASE_DIRECTORIES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify five-case StateSpec parity reports.")
    parser.add_argument("--result-name", default="benchmark_vlm_qwen")
    parser.add_argument("--strict-warnings", action="store_true", help="Treat warnings as failures.")
    args = parser.parse_args()

    reports = build_canonical_state_parity_reports(result_name=args.result_name)
    errors: list[str] = []
    warnings: list[str] = []
    for case_name in CANONICAL_CASE_DIRECTORIES:
        report = reports["cases"][case_name]
        summary = report["summary"]
        print(
            f"{case_name}: status={summary['status']} "
            f"passed={summary['passed']} warnings={summary['warnings']} failed={summary['failed']} "
            f"{summary['canonical_sha256']}"
        )
        for check in report["checks"]:
            if check["status"] == "fail":
                errors.append(f"{case_name}:{check['check_id']}: {check['detail']}")
            elif check["status"] == "warn":
                warnings.append(
                    f"{case_name}:{check['check_id']}: {check['violations']} violations, "
                    f"max_abs_error={check['max_abs_error']}"
                )

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if args.strict_warnings:
        errors.extend(warnings)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
