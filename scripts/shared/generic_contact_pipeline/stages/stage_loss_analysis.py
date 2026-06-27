from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.config import available_cases, load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.loss_analysis import run_loss_analysis  # noqa: E402
from scripts.shared.generic_contact_pipeline.stages.stage_related_work_proxy import run_case as run_related_work_proxy  # noqa: E402


def run(profile):
    loss = run_loss_analysis(profile)
    related = run_related_work_proxy(profile.case_name, profile.result_name)
    return {
        "stage": "stage7_loss_and_related_work_proxy",
        "case_name": profile.case_name,
        "loss_analysis": loss,
        "related_work_proxy": related,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate non-invasive loss/residual analysis from generic pipeline outputs.")
    ap.add_argument("--case", default="all", help="Case name or all")
    args = ap.parse_args()
    cases = available_cases() if args.case == "all" else [args.case]
    failed = False
    for case in cases:
        try:
            summary = run(load_case_profile(case))
            print(f"{case}: {summary['loss_trace']}")
        except Exception as exc:
            failed = True
            print(f"{case}: FAILED: {exc}", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
