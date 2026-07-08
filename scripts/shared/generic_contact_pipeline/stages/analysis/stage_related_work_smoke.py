#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[5]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
else:
    REPO = Path(__file__).resolve().parents[5]

from scripts.shared.generic_contact_pipeline.core.base.io import write_csv, write_json  # noqa: E402


OUT_DIR = REPO / "docs" / "related_work_audit"


@dataclass
class SmokeRow:
    target: str
    check_type: str
    environment: str
    command: str
    cwd: str
    status: str
    returncode: int | str
    blocker: str
    stdout_tail: str
    stderr_tail: str


PY_IMPORT_MODULES = [
    "numpy",
    "cv2",
    "torch",
    "pytorch3d",
    "smplx",
    "open3d",
    "lpips",
    "diff_gaussian_rasterization",
    "simple_knn",
    "sdf",
]

KNOWN_ENVS = [
    "audiohoi",
    "bodyrender",
    "gvhmr",
    "hamer",
    "sam3d-objects-inference",
]


def tail(text: str, limit: int = 900) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def run_cmd(
    target: str,
    check_type: str,
    cmd: list[str],
    cwd: Path | None = None,
    env: str = "",
    timeout: int = 35,
) -> SmokeRow:
    cwd = cwd or REPO
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        output = "\n".join([proc.stdout or "", proc.stderr or ""])
        blocker = infer_blocker(output, proc.returncode)
        status = "ok" if proc.returncode == 0 else "failed"
        return SmokeRow(
            target=target,
            check_type=check_type,
            environment=env,
            command=cmd_str,
            cwd=str(cwd),
            status=status,
            returncode=proc.returncode,
            blocker=blocker,
            stdout_tail=tail(proc.stdout),
            stderr_tail=tail(proc.stderr),
        )
    except subprocess.TimeoutExpired as exc:
        return SmokeRow(
            target=target,
            check_type=check_type,
            environment=env,
            command=cmd_str,
            cwd=str(cwd),
            status="timeout",
            returncode="timeout",
            blocker=f"timeout after {timeout}s",
            stdout_tail=tail(exc.stdout if isinstance(exc.stdout, str) else ""),
            stderr_tail=tail(exc.stderr if isinstance(exc.stderr, str) else ""),
        )
    except FileNotFoundError as exc:
        return SmokeRow(
            target=target,
            check_type=check_type,
            environment=env,
            command=cmd_str,
            cwd=str(cwd),
            status="failed",
            returncode="missing_command",
            blocker=str(exc),
            stdout_tail="",
            stderr_tail="",
        )


def infer_blocker(output: str, returncode: int) -> str:
    if returncode == 0:
        return ""
    lowered = output.lower()
    if "modulenotfounderror" in lowered or "no module named" in lowered:
        for line in output.splitlines():
            if "ModuleNotFoundError" in line or "No module named" in line:
                return line.strip()
        return "missing Python module"
    if "importerror" in lowered:
        for line in output.splitlines():
            if "ImportError" in line:
                return line.strip()
        return "Python import error"
    if "cuda" in lowered and ("not found" in lowered or "unavailable" in lowered):
        return "CUDA/runtime dependency unavailable"
    if "usage:" in lowered and returncode != 0:
        return "CLI argument/import path failed before clean help exit"
    return tail(output, 240)


def conda_py(env: str, code: str) -> list[str]:
    return ["conda", "run", "-n", env, "python", "-c", code]


def conda_script(env: str, script: str, *args: str) -> list[str]:
    return ["conda", "run", "-n", env, "python", script, *args]


def env_import_probe(envs: list[str]) -> list[SmokeRow]:
    rows: list[SmokeRow] = []
    for env in envs:
        for module in PY_IMPORT_MODULES:
            code = f"import {module}; print('{module} OK')"
            rows.append(run_cmd(module, "env_import", conda_py(env, code), env=env, timeout=20))
    return rows


def method_entry_probe() -> list[SmokeRow]:
    rows: list[SmokeRow] = []
    interdiff = REPO / "external_baselines" / "InterDiff" / "interdiff"
    if interdiff.exists():
        rows.append(
            run_cmd(
                "InterDiff eval_skeleton",
                "entry_help",
                conda_script("audiohoi", "eval_skeleton.py", "--help"),
                cwd=interdiff,
                env="audiohoi",
            )
        )
        rows.append(
            run_cmd(
                "InterDiff optimization",
                "entry_help",
                conda_script("audiohoi", "optimization.py", "--help"),
                cwd=interdiff,
                env="audiohoi",
            )
        )
    opengaus = REPO / "external_baselines" / "open-3dhoi" / "HOIGaussian"
    if opengaus.exists():
        rows.append(
            run_cmd(
                "Open3DHOI train.py",
                "entry_help",
                conda_script("audiohoi", "train.py", "--help"),
                cwd=opengaus,
                env="audiohoi",
            )
        )
        rows.append(run_cmd("Open3DHOI test.sh", "shell_syntax", ["bash", "-n", "test.sh"], cwd=opengaus))
    mover = REPO / "external_baselines" / "MOVER"
    if mover.exists():
        rows.append(run_cmd("MOVER demo/run.sh", "shell_syntax", ["bash", "-n", "demo/run.sh"], cwd=mover))
        rows.append(run_cmd("MOVER demo/scene_refinement.sh", "shell_syntax", ["bash", "-n", "demo/scene_refinement.sh"], cwd=mover))
        rows.append(
            SmokeRow(
                target="MOVER demo execution",
                check_type="guarded_skip",
                environment="",
                command="bash demo/run.sh",
                cwd=str(mover),
                status="skipped",
                returncode="skipped",
                blocker="demo hardcodes external MOVER dataset path, cluster Python path, and CUDA module; requires adapter before fair AudioHOI run",
                stdout_tail="",
                stderr_tail="",
            )
        )
    for name, rel, reason in [
        ("InterCap", "external_baselines/InterCap", "README-only public snapshot; no runnable entry found"),
        ("HOI-PAGE", "external_baselines/HOI-PAGE", "README-only public snapshot; no runnable entry found"),
        ("ZeroHSI", "external_baselines/zerohsi", "project-page snapshot; code/data marked coming soon"),
    ]:
        path = REPO / rel
        if path.exists():
            rows.append(
                SmokeRow(
                    target=name,
                    check_type="source_availability",
                    environment="",
                    command="find runnable entry",
                    cwd=str(path),
                    status="not_runnable_snapshot",
                    returncode="n/a",
                    blocker=reason,
                    stdout_tail="",
                    stderr_tail="",
                )
            )
    return rows


def summarize(rows: list[SmokeRow]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    blockers: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
        if row.blocker:
            blockers[row.blocker] = blockers.get(row.blocker, 0) + 1
    return {"row_count": len(rows), "status_counts": counts, "blocker_counts": blockers}


def write_markdown(rows: list[SmokeRow], summary: dict[str, Any]) -> Path:
    path = OUT_DIR / "related_work_smoke_tests.md"
    lines = [
        "# Related Work Smoke Tests",
        "",
        "Generated by `scripts/shared/generic_contact_pipeline/stages/stage_related_work_smoke.py`.",
        "These checks are lightweight local probes, not full method reproductions.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Entry / Runnable Checks",
        "",
        "| target | check | env | status | blocker | command |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.check_type == "env_import":
            continue
        lines.append(
            f"| {row.target} | {row.check_type} | {row.environment} | {row.status} | "
            f"{row.blocker.replace('|', '/')} | `{row.command}` |"
        )
    lines.extend(["", "## Environment Import Matrix", ""])
    env_rows = [row for row in rows if row.check_type == "env_import"]
    modules = sorted({row.target for row in env_rows})
    envs = sorted({row.environment for row in env_rows})
    lines.append("| env | " + " | ".join(modules) + " |")
    lines.append("|---" + "|---" * len(modules) + "|")
    by_key = {(row.environment, row.target): row for row in env_rows}
    for env in envs:
        cells = []
        for module in modules:
            row = by_key.get((env, module))
            if row is None:
                cells.append("")
            elif row.status == "ok":
                cells.append("OK")
            else:
                cells.append(row.blocker.replace("|", "/") or row.status)
        lines.append("| " + env + " | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `failed` on `--help` still means the source entry was reached but imports failed before argument parsing.",
            "- MOVER execution is intentionally guarded because its public demo hardcodes a non-local dataset/env; this needs an AudioHOI adapter rather than blind execution.",
            "- Open3DHOI remains the most promising external full-run target because it ships `HOIGaussian/test_data`, but it needs missing Python modules and compiled Gaussian submodules.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def run(envs: list[str], skip_imports: bool = False) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[SmokeRow] = []
    if not skip_imports:
        rows.extend(env_import_probe(envs))
    rows.extend(method_entry_probe())
    dict_rows = [asdict(row) for row in rows]
    write_csv(OUT_DIR / "related_work_smoke_tests.csv", dict_rows)
    summary = summarize(rows)
    write_json(OUT_DIR / "related_work_smoke_tests.json", {"summary": summary, "rows": dict_rows})
    md = write_markdown(rows, summary)
    return {
        "rows": len(rows),
        "summary": summary,
        "csv": str(OUT_DIR / "related_work_smoke_tests.csv"),
        "json": str(OUT_DIR / "related_work_smoke_tests.json"),
        "md": str(md),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run lightweight smoke checks for inspected related-work repos.")
    ap.add_argument("--env", action="append", help="Conda env to import-probe. Repeatable. Defaults to known AudioHOI envs.")
    ap.add_argument("--skip-imports", action="store_true", help="Only run method entry/source checks.")
    args = ap.parse_args()
    envs = args.env or KNOWN_ENVS
    print(json.dumps(run(envs, skip_imports=args.skip_imports), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
