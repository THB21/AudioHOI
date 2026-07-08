#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else []


def key(row: dict[str, object]) -> tuple[str, str]:
    return str(row.get("frame", "")), str(row.get("query_type", ""))


def export(result_dir: Path) -> dict[str, object]:
    vlm_dir = result_dir / "vlm"
    rows: list[dict[str, object]] = []
    for stage_dir in sorted(p for p in vlm_dir.glob("stage*") if p.is_dir()):
        stage = stage_dir.name
        queries = read_csv(stage_dir / "vlm_queries.csv")
        raw_by_key = {key(row): row for row in read_json_rows(stage_dir / "qwen_raw_results.json")}
        gates_by_key = {key(row): row for row in read_csv(stage_dir / "vlm_gates.csv")}
        for query in queries:
            row_key = key(query)
            raw = raw_by_key.get(row_key, {})
            gate = gates_by_key.get(key(query), {})
            answer_status = "real_qwen_answer" if row_key in raw_by_key else "missing_real_qwen_answer"
            rows.append(
                {
                    "stage": stage,
                    "frame": query.get("frame", ""),
                    "query_type": query.get("query_type", ""),
                    "question": query.get("question", ""),
                    "choices": query.get("choices", ""),
                    "input_image_path": query.get("input_image_path", ""),
                    "input_render_path": query.get("input_render_path", ""),
                    "answer_status": answer_status,
                    "answer_label": raw.get("label", ""),
                    "confidence": raw.get("confidence", ""),
                    "short_reason": raw.get("short_reason", ""),
                    "raw_text": raw.get("raw_text", ""),
                    "pass_gate": gate.get("pass_gate", ""),
                    "repair_action": gate.get("repair_action", ""),
                    "allow_anchor_update": gate.get("allow_anchor_update", ""),
                    "allow_contact_residual": gate.get("allow_contact_residual", ""),
                    "allow_pose_refine": gate.get("allow_pose_refine", ""),
                    "report_only": gate.get("report_only", ""),
                }
            )

    out_json = vlm_dir / "vlm_qa_report.json"
    out_csv = vlm_dir / "vlm_qa_report.csv"
    out_md = vlm_dir / "vlm_qa_report.md"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    write_csv(out_csv, rows)

    lines = ["# VLM QA Report", ""]
    current_stage = ""
    for row in rows:
        stage = str(row["stage"])
        if stage != current_stage:
            current_stage = stage
            lines.extend([f"## {stage}", ""])
        lines.extend(
            [
                f"- Frame `{row['frame']}` `{row['query_type']}`",
                f"  - Question: {row['question']}",
                f"  - Choices: `{row['choices']}`",
                f"  - Evidence: `{row['input_render_path'] or row['input_image_path']}`",
                f"  - Answer: `{row['answer_label']}` confidence `{row['confidence']}`",
                f"  - Reason: {row['short_reason']}",
                f"  - Gate: `{row['pass_gate']}` action `{row['repair_action']}` anchor `{row['allow_anchor_update']}` contact `{row['allow_contact_residual']}` refine `{row['allow_pose_refine']}` report_only `{row['report_only']}`",
                "",
            ]
        )
    out_md.write_text("\n".join(lines))
    return {"rows": len(rows), "json": str(out_json), "csv": str(out_csv), "markdown": str(out_md)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a readable VLM question/answer report for one generic pipeline result.")
    ap.add_argument("--result-dir", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(export(args.result_dir), indent=2))


if __name__ == "__main__":
    main()
