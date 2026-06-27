#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
else:
    REPO = Path(__file__).resolve().parents[4]


@dataclass
class RelatedWork:
    method: str
    repo: str
    repo_status: str
    source_evidence: str
    local_clone_status: str
    local_clone_path: str
    run_attempt_status: str
    run_attempt_reason: str
    direct_audiohoi_baseline: str
    comparison_role: str


METHODS = [
    {
        "method": "InterCap",
        "repo": "https://github.com/YinghaoHuang91/InterCap",
        "repo_status": "found_public_repo",
        "source_evidence": "gh search/view: Source code for GCPR 2022 InterCap; default branch main; no license metadata returned",
        "direct_audiohoi_baseline": "partial",
        "comparison_role": "markerless human-object 3D tracking baseline if inputs can be adapted",
        "local_patterns": ["*InterCap*", "*intercap*"],
    },
    {
        "method": "InterDiff",
        "repo": "https://github.com/Sirui-Xu/InterDiff",
        "repo_status": "found_public_repo",
        "source_evidence": "local clone inspected: official ICCV 2023 PyTorch implementation; two-step interaction diffusion + interaction correction",
        "direct_audiohoi_baseline": "no",
        "comparison_role": "physics-informed HOI generation baseline, not source-video 6D recovery",
        "local_patterns": ["*InterDiff*", "*interdiff*"],
    },
    {
        "method": "HOI-PAGE",
        "repo": "https://github.com/craigleili/HOI-PAGE",
        "repo_status": "found_public_repo",
        "source_evidence": "local clone inspected: README-only public snapshot for ICML 2026 zero-shot HOI generation with part affordance guidance",
        "direct_audiohoi_baseline": "no",
        "comparison_role": "part-affordance/VLM-adjacent baseline, not metric object pose recovery",
        "local_patterns": ["*HOI-PAGE*", "*hoi-page*"],
    },
    {
        "method": "ZeroHSI",
        "repo": "https://github.com/awfuact/zerohsi",
        "repo_status": "found_public_repo",
        "source_evidence": "local clone inspected: project-page snapshot; code/data links marked coming soon",
        "direct_audiohoi_baseline": "no",
        "comparison_role": "zero-shot interaction reasoning/project-page style reference, not direct 6D/contact baseline",
        "local_patterns": ["*ZeroHSI*", "*zerohsi*"],
    },
    {
        "method": "MOVER",
        "repo": "https://github.com/yhw-yhw/MOVER",
        "repo_status": "found_public_repo",
        "source_evidence": "local clone inspected: official CVPR 2022 Human-Aware Object Placement repo; demo requires MOVER dataset, SMPL-X, POSA/PARE/PROX style inputs and CUDA-10 era env",
        "direct_audiohoi_baseline": "partial",
        "comparison_role": "physical human-scene interaction optimization reference: ordinal depth, SDF collision, contact, ground-plane support",
        "local_patterns": ["*MOVER*", "*mover*"],
    },
    {
        "method": "Open3DHOI / HOI-Gaussian",
        "repo": "https://github.com/wenboran2002/open-3dhoi",
        "repo_status": "found_public_repo",
        "source_evidence": "local clone inspected: open-vocabulary HOI reconstruction repo with HOIGaussian optimizer; no separate exact Gaussian-HOI repo found by gh search",
        "direct_audiohoi_baseline": "partial",
        "comparison_role": "single-image/open-vocabulary HOI Gaussian optimization reference: mask/RGB fitting followed by contact/depth/collision refinement",
        "local_patterns": ["*open-3dhoi*", "*Open3DHOI*", "*Gaussian-HOI*", "*Gaussian*HOI*"],
    },
]


def find_local_clone(patterns: list[str]) -> tuple[str, str]:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(p for p in REPO.glob(pattern) if p.is_dir())
        candidates.extend(p for p in (REPO / "external").glob(pattern) if p.is_dir() if (REPO / "external").exists())
        candidates.extend(p for p in (REPO / "external_baselines").glob(pattern) if p.is_dir() if (REPO / "external_baselines").exists())
        candidates.extend(p for p in (REPO / "third_party").glob(pattern) if p.is_dir() if (REPO / "third_party").exists())
    if not candidates:
        return "not_found_locally", ""
    candidates = sorted(set(candidates), key=lambda p: str(p))
    return "found_locally", str(candidates[0])


def has_runnable_entry(path: str) -> bool:
    if not path:
        return False
    root = Path(path)
    markers = ["requirements.txt", "environment.yml", "setup.py", "pyproject.toml"]
    if any((root / marker).exists() for marker in markers):
        return True
    if any(root.glob("*.py")) or any(root.glob("*.sh")):
        return True
    if any((root / name).exists() for name in ["scripts", "src", "tools", "demo", "install"]):
        return True
    return any(root.glob("**/requirements.txt")) or any(root.glob("**/environment.yml")) or any(root.glob("**/*.py"))


def audit() -> list[RelatedWork]:
    rows: list[RelatedWork] = []
    for item in METHODS:
        local_status, local_path = find_local_clone(item["local_patterns"])
        if local_status != "found_locally":
            run_status = "not_attempted"
            reason = "no local clone or installed runnable baseline found under repository workspace"
        elif not has_runnable_entry(local_path):
            run_status = "not_attempted_no_runnable_entry"
            reason = "local clone exists but contains no requirements/setup/scripts/src Python runnable entry"
        elif item["direct_audiohoi_baseline"] == "no":
            run_status = "not_attempted_adjacent_method"
            reason = "method is not a direct source-video 6D/contact recovery baseline; code/method inspected for constraints and positioning"
        else:
            run_status = "not_run_requires_adapter_or_data"
            reason = "local source exists, but fair execution requires method-specific data/models/env and an AudioHOI input/output adapter"
        rows.append(
            RelatedWork(
                method=item["method"],
                repo=item["repo"],
                repo_status=item["repo_status"],
                source_evidence=item["source_evidence"],
                local_clone_status=local_status,
                local_clone_path=local_path,
                run_attempt_status=run_status,
                run_attempt_reason=reason,
                direct_audiohoi_baseline=item["direct_audiohoi_baseline"],
                comparison_role=item["comparison_role"],
            )
        )
    return rows


def write_outputs(rows: list[RelatedWork]) -> dict[str, str]:
    out_dir = REPO / "docs" / "related_work_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "related_work_audit.csv"
    json_path = out_dir / "related_work_audit.json"
    md_path = out_dir / "related_work_attempt_log.md"
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Related Work Attempt Log",
        "",
        "This file is generated by `scripts/shared/generic_contact_pipeline/stages/stage_related_work_audit.py`.",
        "",
        "| Method | Repo status | Local clone | Run attempt | Direct AudioHOI baseline? | Role |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        repo = f"[repo]({row.repo})" if row.repo else "none"
        local = row.local_clone_status if not row.local_clone_path else f"{row.local_clone_status}: `{row.local_clone_path}`"
        lines.append(
            f"| {row.method} | {row.repo_status}; {repo} | {local} | "
            f"{row.run_attempt_status}: {row.run_attempt_reason} | {row.direct_audiohoi_baseline} | {row.comparison_role} |"
        )
    lines.extend(
        [
            "",
            "Conclusion:",
            "",
            "- InterCap, InterDiff, HOI-PAGE, ZeroHSI, MOVER, and Open3DHOI/HOI-Gaussian were source-visible checked locally where source is available.",
            "- No related-work baseline is claimed as reproduced locally. InterCap/HOI-PAGE/ZeroHSI public snapshots are not runnable baselines; InterDiff/MOVER/Open3DHOI have runnable-looking code but require their own data, models, and adapters.",
            "- InterDiff, MOVER, and Open3DHOI are useful for loss/constraint comparison: relative/contact-frame motion, ordinal depth, collision/SDF, Chamfer contact, and render-mask objectives.",
            "- These methods are adjacent references unless an AudioHOI adapter is implemented; they do not directly replace the solved-case object_pose.csv/contact recovery pipeline.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    return {"csv": str(csv_path), "json": str(json_path), "md": str(md_path)}


def main() -> None:
    rows = audit()
    outputs = write_outputs(rows)
    print(json.dumps({"rows": len(rows), "outputs": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
