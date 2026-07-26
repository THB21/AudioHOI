#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import repo_path  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.solver.candidate import default_candidate_dir  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.solver.sphere_sequence import solve_sphere_sequence_candidate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a typed sphere sequence into an isolated candidate sandbox.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--contact-events-csv", type=Path, default=None)
    parser.add_argument("--human-sites-csv", type=Path, default=None)
    parser.add_argument("--support-geometry-json", type=Path, default=None)
    parser.add_argument("--candidate-dir", type=Path, default=None)
    args = parser.parse_args()

    profile = load_case_profile(args.case)
    result_dir = repo_path(args.result_dir)
    contact_events = repo_path(args.contact_events_csv) if args.contact_events_csv else result_dir / "contact_events.csv"
    if not contact_events.exists():
        contact_events = result_dir / "contact_candidates_internal/contact_candidates_labeled.csv"
    human_sites = repo_path(args.human_sites_csv) if args.human_sites_csv else result_dir / "human_sites.csv"
    support_geometry = repo_path(args.support_geometry_json) if args.support_geometry_json else result_dir / "support_geometry.json"
    candidate_dir = repo_path(args.candidate_dir) if args.candidate_dir else default_candidate_dir(result_dir, profile.case_name)
    attempt = solve_sphere_sequence_candidate(
        profile,
        result_dir,
        contact_events_csv=contact_events,
        human_sites_csv=human_sites,
        support_geometry_json=support_geometry,
        candidate_dir=candidate_dir,
    )
    print(json.dumps(attempt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
