#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[5]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
else:
    REPO = Path(__file__).resolve().parents[5]

from scripts.shared.generic_contact_pipeline.core.base.io import write_csv, write_json  # noqa: E402


OUT_DIR = REPO / "docs" / "related_work_audit"
PAPER_DIR = Path("/tmp/audiohoi_related_papers")


@dataclass
class PaperEvidenceSpec:
    method: str
    paper_txt: str
    evidence_topic: str
    start_line: int
    end_line: int
    expected_keywords: str
    method_takeaway: str
    audiohoi_implication: str
    audiohoi_probe: str


SPECS = [
    PaperEvidenceSpec(
        method="InterCap",
        paper_txt="InterCap.txt",
        evidence_topic="multi-view known-mesh object tracking from rendered mask/depth",
        start_line=306,
        end_line=325,
        expected_keywords="Object Tracking,render,mask,depth",
        method_takeaway="Track known object mesh by matching rendered segmentation/depth to observed RGB-D evidence.",
        audiohoi_implication="AudioHOI 2D-to-6D pose should keep K_fullimg projection and mask/depth residuals as primary evidence before contact refinement.",
        audiohoi_probe="observation_probe; render_depth_probe",
    ),
    PaperEvidenceSpec(
        method="InterCap",
        paper_txt="InterCap.txt",
        evidence_topic="joint human-object refinement with contact and temporal smoothing",
        start_line=384,
        end_line=405,
        expected_keywords="Joint Human-Object Tracking,contact,temporal smoothness,interpenetration",
        method_takeaway="Use contact labels and temporal smoothing to refine human/object over the sequence and reduce jitter/interpenetration.",
        audiohoi_implication="AudioHOI anchor update/freeze should happen after 2D alignment, with contact-gated refinement and jitter checks.",
        audiohoi_probe="method_loss_probe; semantic_contact_geometry_probe; sdf_penetration_probe",
    ),
    PaperEvidenceSpec(
        method="InterDiff",
        paper_txt="InterDiff.txt",
        evidence_topic="physics-informed diffusion correction from contact/penetration state",
        start_line=231,
        end_line=287,
        expected_keywords="Correction Schedule,contact,penetration,reference system",
        method_takeaway="A scheduler detects contact/penetration states and corrects generated object motion in a contact reference frame.",
        audiohoi_implication="AudioHOI should not blindly smooth all frames; it should use contact gates, anchor frames, and local reference stability.",
        audiohoi_probe="method_loss_probe; semantic_contact_geometry_probe; sdf_penetration_probe",
    ),
    PaperEvidenceSpec(
        method="InterDiff",
        paper_txt="InterDiff.txt",
        evidence_topic="evaluation reports penetration/contact-floating artifacts",
        start_line=476,
        end_line=489,
        expected_keywords="contact floating,penetration,correction",
        method_takeaway="Ablations show that removing correction/relative/contact terms causes floating contact and penetration.",
        audiohoi_implication="Use floating-contact/contact-gap and penetration probes as reportable failure labels.",
        audiohoi_probe="geometry_contact_probe; dense_mesh_probe; sdf_penetration_probe",
    ),
    PaperEvidenceSpec(
        method="HOI-PAGE",
        paper_txt="HOI-PAGE.txt",
        evidence_topic="part affordance graph from LLM reasoning",
        start_line=222,
        end_line=263,
        expected_keywords="Part Affordance Graph,LLM,contact,constraints",
        method_takeaway="Use LLM reasoning to infer object-part/body-part interaction edges and ground them to 3D object geometry.",
        audiohoi_implication="AudioHOI VLM/LLM should be a conservative semantic verifier for part/contact candidates, not a continuous pose optimizer.",
        audiohoi_probe="source/paper audit; VLM gate design; semantic_contact_geometry_probe",
    ),
    PaperEvidenceSpec(
        method="HOI-PAGE",
        paper_txt="HOI-PAGE.txt",
        evidence_topic="part affordance guided 4D optimization",
        start_line=301,
        end_line=329,
        expected_keywords="optimization,Chamfer,Penetration,2D",
        method_takeaway="Optimize object/body motion with part-level 3D/2D fitting plus penetration and motion-state constraints.",
        audiohoi_implication="Chair/mug should use semantic part anchors and contact part checks instead of whole-object center only.",
        audiohoi_probe="observation_probe; semantic_contact_geometry_probe; sdf_penetration_probe",
    ),
    PaperEvidenceSpec(
        method="HOI-PAGE",
        paper_txt="HOI-PAGE.txt",
        evidence_topic="part-level contact metrics",
        start_line=909,
        end_line=919,
        expected_keywords="Part-level Contact Metrics,Contact Accuracy,Contact Distance,minimum distance",
        method_takeaway="Evaluate realism with part-level minimum distances and thresholded contact accuracy.",
        audiohoi_implication="AudioHOI contact evaluation should report semantic part-to-part gap, not only global object trajectory error.",
        audiohoi_probe="semantic_contact_geometry_probe; dense_mesh_probe",
    ),
    PaperEvidenceSpec(
        method="ZeroHSI",
        paper_txt="ZeroHSI.txt",
        evidence_topic="video generation distilled through differentiable rendering",
        start_line=167,
        end_line=169,
        expected_keywords="video generation,differentiable,object 6D,rendered",
        method_takeaway="Generate an interaction video from a scene/text condition, then recover 4D motion through differentiable rendering.",
        audiohoi_implication="For AudioHOI, generated/video observation should remain an evidence source; 3D pose must be validated through rendering/depth consistency.",
        audiohoi_probe="observation_probe; render_depth_probe",
    ),
    PaperEvidenceSpec(
        method="ZeroHSI",
        paper_txt="ZeroHSI.txt",
        evidence_topic="Gaussian scene/object and 6D object pose optimized by rendering loss",
        start_line=110,
        end_line=131,
        expected_keywords="3D Gaussians,object’s 6D,rendering loss",
        method_takeaway="Represent scene/object as Gaussians and optimize camera, human, and 6D object pose with rendering/image losses.",
        audiohoi_implication="AudioHOI's 2D overlay/render-depth checks are the comparable audit path for Gaussian/render optimization methods.",
        audiohoi_probe="observation_probe; render_depth_probe; method_loss_probe",
    ),
    PaperEvidenceSpec(
        method="ZeroHSI",
        paper_txt="ZeroHSI.txt",
        evidence_topic="photometric, center, depth, and physics losses",
        start_line=261,
        end_line=286,
        expected_keywords="photometric,center point,depth regularization,physics",
        method_takeaway="Use photometric loss plus center/depth regularization and physics terms during per-frame optimization.",
        audiohoi_implication="AudioHOI report should separate visual projection residual, depth/order residual, and contact/physics residual.",
        audiohoi_probe="observation_probe; depth_probe; render_depth_probe; geometry_contact_probe",
    ),
    PaperEvidenceSpec(
        method="ZeroHSI",
        paper_txt="ZeroHSI.txt",
        evidence_topic="contact and penetration evaluation metrics",
        start_line=341,
        end_line=392,
        expected_keywords="Cont.,penetration,SDF,object penetration",
        method_takeaway="Dynamic-object evaluation uses contact ratio and object penetration depth from SDFs.",
        audiohoi_implication="AudioHOI SDF penetration and semantic contact probes are the right comparison metrics for zero-shot HSI methods.",
        audiohoi_probe="semantic_contact_geometry_probe; sdf_penetration_probe",
    ),
]


def normalize(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def verify(spec: PaperEvidenceSpec) -> dict[str, object]:
    path = PAPER_DIR / spec.paper_txt
    row: dict[str, object] = asdict(spec)
    row["source_path"] = str(path)
    row["exists"] = path.exists()
    if not path.exists():
        row.update({"valid_line_range": False, "keyword_hits": 0, "missing_keywords": spec.expected_keywords, "evidence_sha256": "", "snippet": ""})
        return row
    lines = path.read_text(errors="replace").splitlines()
    valid = 1 <= spec.start_line <= spec.end_line <= len(lines)
    block = lines[spec.start_line - 1 : spec.end_line] if valid else []
    norm = normalize(block)
    keywords = [k.strip() for k in spec.expected_keywords.split(",") if k.strip()]
    norm_lower = norm.lower()
    missing = [k for k in keywords if k.lower() not in norm_lower]
    row.update(
        {
            "valid_line_range": valid,
            "file_line_count": len(lines),
            "keyword_hits": len(keywords) - len(missing),
            "keyword_count": len(keywords),
            "missing_keywords": ",".join(missing),
            "evidence_sha256": hashlib.sha256(norm.encode("utf-8")).hexdigest() if norm else "",
            "snippet": norm[:320],
        }
    )
    return row


def write_markdown(rows: list[dict[str, object]]) -> Path:
    path = OUT_DIR / "related_work_paper_evidence_verified.md"
    lines = [
        "# Related Work Paper Evidence Verification",
        "",
        "Generated by `stage_related_work_paper_evidence.py`.",
        "This verifies paper-text evidence for methods whose public code is unavailable, incomplete, or not directly runnable as an AudioHOI baseline.",
        "",
        "| Method | Topic | Paper lines | Keywords | Evidence hash | AudioHOI implication | Probe |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        source = f"`{row['paper_txt']}:{row['start_line']}-{row['end_line']}`"
        kw = f"{row['keyword_hits']}/{row['keyword_count']}"
        sha = str(row.get("evidence_sha256", ""))[:12]
        lines.append(
            f"| {row['method']} | {row['evidence_topic']} | {source} | {kw} | `{sha}` | "
            f"{row['audiohoi_implication']} | {row['audiohoi_probe']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Paper evidence is used only when source code is README-only, unavailable, or not a direct AudioHOI baseline.",
            "- Snippets are stored in the CSV/JSON as short audit anchors; the report should cite the paper and summarize the method, not paste long copyrighted text.",
            "- `evidence_sha256` hashes the normalized evidence window so the extracted text can be checked for drift.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def run() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [verify(spec) for spec in SPECS]
    csv_path = write_csv(OUT_DIR / "related_work_paper_evidence_verified.csv", rows)
    json_path = write_json(OUT_DIR / "related_work_paper_evidence_verified.json", {"paper_dir": str(PAPER_DIR), "rows": rows})
    md_path = write_markdown(rows)
    failures = [r for r in rows if not r.get("exists") or not r.get("valid_line_range") or r.get("missing_keywords")]
    return {"rows": len(rows), "failures": len(failures), "csv": str(csv_path), "json": str(json_path), "md": str(md_path)}


def main() -> None:
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
