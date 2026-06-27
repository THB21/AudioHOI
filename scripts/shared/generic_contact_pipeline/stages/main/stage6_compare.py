from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[5]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.evaluation.compare import compare_case  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.evaluation.audit import build_audit, write_audit_csv  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.config import available_cases, load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import write_json  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.schema import stage_paths  # noqa: E402


def run(profile):
    report = compare_case(profile)
    report["migration_audit"] = build_audit(profile)
    report["migration_audit_csv"] = write_audit_csv(profile, report["migration_audit"])
    scan_pass = bool(report["migration_audit"].get("old_script_reference_scan", {}).get("pass"))  # type: ignore[union-attr]
    report["checks"]["old_script_reference_scan_pass"] = scan_pass  # type: ignore[index]
    report["checks"]["overall_pass"] = bool(report["checks"]["overall_pass"] and scan_pass)  # type: ignore[index]
    write_json(stage_paths(profile)["compare_report"], report)
    write_json(stage_paths(profile)["migration_audit"], report["migration_audit"])
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare generic pipeline outputs against solved baselines.")
    ap.add_argument("--case", default="all", help="Case name or all")
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    failed = False
    for case in cases:
        report = run(load_case_profile(case))
        ok = bool(report["checks"]["overall_pass"])  # type: ignore[index]
        print(f"{case}: {'PASS' if ok else 'FAIL'}")
        failed = failed or not ok
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
