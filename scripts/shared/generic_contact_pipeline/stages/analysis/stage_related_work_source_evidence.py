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


@dataclass
class EvidenceSpec:
    method: str
    term: str
    source_path: str
    start_line: int
    end_line: int
    expected_keywords: str
    original_role: str
    audiohoi_probe: str
    audiohoi_metric: str
    full_reproduction_gap: str


SPECS = [
    EvidenceSpec(
        method="InterDiff",
        term="signed human-object collision / penetration",
        source_path="external_baselines/InterDiff/interdiff/optimization.py",
        start_line=62,
        end_line=76,
        expected_keywords="point2point_signed,o2h_signed,w_dist_neg,loss_dist_o",
        original_role="interaction correction penalizes penetration while keeping plausible contact",
        audiohoi_probe="related_work_sdf_penetration_probe + related_work_geometry_contact_probe",
        audiohoi_metric="sdf_max_penetration; mover_style_human_contact_gap",
        full_reproduction_gap="InterDiff correction network/data are not adapted to AudioHOI sequences",
    ),
    EvidenceSpec(
        method="InterDiff",
        term="object/body/hand temporal smoothness",
        source_path="external_baselines/InterDiff/interdiff/optimization.py",
        start_line=95,
        end_line=107,
        expected_keywords="loss_transl_v_reg,loss_obj_v_reg,loss_body_v_reg",
        original_role="second-difference and velocity penalties suppress jitter",
        audiohoi_probe="method_loss_probe",
        audiohoi_metric="method_interdiff_p95_translation_accel; method_interdiff_p95_rotation_accel",
        full_reproduction_gap="AudioHOI currently probes object pose smoothness; hand/body acceleration can be added from GVHMR",
    ),
    EvidenceSpec(
        method="InterDiff",
        term="contact-conditioned reference motion",
        source_path="external_baselines/InterDiff/interdiff/model/correction_smpl.py",
        start_line=84,
        end_line=136,
        expected_keywords="obj_trans_relative,contact_happen,hand_marker,torch.gather",
        original_role="predicts object motion relative to contact markers and prioritizes hand contacts",
        audiohoi_probe="method_loss_probe + semantic_contact_geometry_probe",
        audiohoi_metric="contact_frame_anchor_velocity_p95; mover_style_human_contact_gap",
        full_reproduction_gap="AudioHOI has sparse semantic anchors rather than dense InterDiff contact labels",
    ),
    EvidenceSpec(
        method="InterDiff",
        term="dense contact label generation",
        source_path="external_baselines/InterDiff/interdiff/data/prepare_behave.py",
        start_line=32,
        end_line=52,
        expected_keywords="signed_distance,contact_object_label,contact_human_label",
        original_role="supervises contact labels from human/object surface distance",
        audiohoi_probe="dense_mesh_probe + sdf_penetration_probe",
        audiohoi_metric="dense_mesh_contact_min_gap; sdf_contact_max_penetration",
        full_reproduction_gap="AudioHOI needs dense object/human contact labels if this is used as a training target",
    ),
    EvidenceSpec(
        method="MOVER",
        term="ground-plane support/contact",
        source_path="external_baselines/MOVER/mover/scene_reconstruction/_forward_hsi.py",
        start_line=42,
        end_line=86,
        expected_keywords="loss_gp_contact,loss_gp_support,ground_plane",
        original_role="keeps contacted feet on the ground plane and penalizes unsupported height",
        audiohoi_probe="method_loss_probe",
        audiohoi_metric="method_mover_tail_static_drift_mean_m",
        full_reproduction_gap="AudioHOI does not yet fit a shared ground plane for all cases",
    ),
    EvidenceSpec(
        method="MOVER",
        term="ordinal depth from human/object overlap",
        source_path="external_baselines/MOVER/mover/scene_reconstruction/_forward_hsi.py",
        start_line=103,
        end_line=123,
        expected_keywords="accumulate_ordinal_depth_from_human,compute_relative_depth_loss_range,loss_depth",
        original_role="uses rendered/masked depth ordering to constrain object placement",
        audiohoi_probe="related_work_depth_probe + related_work_render_depth_probe",
        audiohoi_metric="mover_contact_center_depth_gap_m_median; render_depth_contact_abs_gap",
        full_reproduction_gap="AudioHOI uses DA3/point-splat depth proxies, not MOVER's exact renderer",
    ),
    EvidenceSpec(
        method="MOVER",
        term="human body SDF penetration",
        source_path="external_baselines/MOVER/mover/scene_reconstruction/_accumulate_hsi_sdf.py",
        start_line=224,
        end_line=259,
        expected_keywords="compute_sdf_loss,grid_sample,body_sdf,sdf_penetration_loss",
        original_role="penalizes object vertices sampled inside the fused body SDF volume",
        audiohoi_probe="related_work_sdf_penetration_probe",
        audiohoi_metric="sdf_max_penetration; sdf_inside_vertex_ratio",
        full_reproduction_gap="AudioHOI queries per-frame SMPL-X mesh SDF via trimesh, not fused multi-frame MOVER SDF volume",
    ),
    EvidenceSpec(
        method="MOVER",
        term="coarse contact Chamfer with normal filtering",
        source_path="external_baselines/MOVER/mover/scene_reconstruction/_accumulate_hsi_contact_loss.py",
        start_line=26,
        end_line=98,
        expected_keywords="chamferDist,contact_body_vertices,contact_loss_y,contact_loss_z",
        original_role="matches accumulated body contact vertices to object vertices with normal-axis splits",
        audiohoi_probe="semantic_contact_geometry_probe + dense_mesh_probe",
        audiohoi_metric="mover_style_human_contact_gap; dense_mesh_contact_min_gap",
        full_reproduction_gap="AudioHOI uses semantic anchors and sampled mesh proximity, not MOVER's accumulated contact normals",
    ),
    EvidenceSpec(
        method="Open3DHOI / HOI-Gaussian",
        term="Gaussian RGB/mask/center image fit",
        source_path="external_baselines/open-3dhoi/HOIGaussian/train.py",
        start_line=198,
        end_line=250,
        expected_keywords="Ll1,mask_loss,ssim_loss,lpips_loss,centre_loss,size_loss",
        original_role="fits visible RGB/mask evidence before HOI physical refinement",
        audiohoi_probe="observation_probe + method_loss_probe",
        audiohoi_metric="opengaus_mask_center_loss_px_median; method_opengaus_median_visual_proxy_px",
        full_reproduction_gap="AudioHOI uses 2D observation residuals, not Gaussian raster RGB optimization",
    ),
    EvidenceSpec(
        method="Open3DHOI / HOI-Gaussian",
        term="contact Chamfer",
        source_path="external_baselines/open-3dhoi/HOIGaussian/hoi_optimizer/loss.py",
        start_line=24,
        end_line=32,
        expected_keywords="compute_contact_loss,chamfer_distance,loss_contact",
        original_role="pulls selected human/object contact vertex sets together",
        audiohoi_probe="semantic_contact_geometry_probe + dense_mesh_probe",
        audiohoi_metric="mover_style_human_contact_gap; dense_mesh_contact_min_gap",
        full_reproduction_gap="AudioHOI has contact anchors/mesh samples, not Open3DHOI selected contact vertex sets",
    ),
    EvidenceSpec(
        method="Open3DHOI / HOI-Gaussian",
        term="rendered ordinal depth",
        source_path="external_baselines/open-3dhoi/HOIGaussian/hoi_optimizer/loss.py",
        start_line=35,
        end_line=111,
        expected_keywords="compute_ordinal_depth_loss,MeshRasterizer,depth_h,depth_o,front_o_pred",
        original_role="penalizes wrong human/object front-back order under segmentation masks",
        audiohoi_probe="render_depth_probe",
        audiohoi_metric="render_depth_contact_abs_gap; object_in_front_ratio",
        full_reproduction_gap="AudioHOI point-splat z-buffer is not PyTorch3D differentiable rasterization",
    ),
    EvidenceSpec(
        method="Open3DHOI / HOI-Gaussian",
        term="bidirectional human-object collision",
        source_path="external_baselines/open-3dhoi/HOIGaussian/hoi_optimizer/loss.py",
        start_line=239,
        end_line=293,
        expected_keywords="compute_collision_loss2,HOCollisionLoss,h_in_o_collision_loss,o_in_h_collision_loss",
        original_role="penalizes human-inside-object and object-inside-human interpenetration",
        audiohoi_probe="sdf_penetration_probe + dense_mesh_probe",
        audiohoi_metric="sdf_max_penetration; dense_mesh_min_gap",
        full_reproduction_gap="AudioHOI does not run Open3DHOI HOCollisionLoss dependency stack",
    ),
]


def normalize_code(lines: list[str]) -> str:
    cleaned: list[str] = []
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        cleaned.append(re.sub(r"\s+", " ", text))
    return "\n".join(cleaned)


def enclosing_symbol(all_lines: list[str], start_line: int) -> str:
    for idx in range(start_line - 1, -1, -1):
        stripped = all_lines[idx].lstrip()
        match = re.match(r"(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
        if match:
            return f"{match.group(1)} {match.group(2)}"
    return ""


def verify(spec: EvidenceSpec) -> dict[str, object]:
    path = REPO / spec.source_path
    row: dict[str, object] = asdict(spec)
    row["exists"] = path.exists()
    if not path.exists():
        row.update({"valid_line_range": False, "keyword_hits": 0, "missing_keywords": spec.expected_keywords, "code_sha256": "", "enclosing_symbol": ""})
        return row
    all_lines = path.read_text(errors="replace").splitlines()
    valid = 1 <= spec.start_line <= spec.end_line <= len(all_lines)
    snippet = all_lines[spec.start_line - 1 : spec.end_line] if valid else []
    normalized = normalize_code(snippet)
    keywords = [k.strip() for k in spec.expected_keywords.split(",") if k.strip()]
    missing = [k for k in keywords if k not in normalized]
    row.update(
        {
            "valid_line_range": valid,
            "file_line_count": len(all_lines),
            "keyword_hits": len(keywords) - len(missing),
            "keyword_count": len(keywords),
            "missing_keywords": ",".join(missing),
            "code_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else "",
            "normalized_code_lines": len(normalized.splitlines()) if normalized else 0,
            "enclosing_symbol": enclosing_symbol(all_lines, spec.start_line),
        }
    )
    return row


def write_markdown(rows: list[dict[str, object]]) -> Path:
    path = OUT_DIR / "related_work_source_evidence_verified.md"
    lines = [
        "# Related Work Source Evidence Verification",
        "",
        "Generated by `stage_related_work_source_evidence.py`.",
        "The table verifies that the cited external source files and line ranges exist, records enclosing symbols, keyword hits, and a hash of the normalized code block.",
        "",
        "| Method | Term | Source | Symbol | Keywords | Hash | AudioHOI probe | Gap |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        source = f"`{row['source_path']}:{row['start_line']}-{row['end_line']}`"
        ok = f"{row['keyword_hits']}/{row['keyword_count']}"
        sha = str(row.get("code_sha256", ""))[:12]
        lines.append(
            f"| {row['method']} | {row['term']} | {source} | {row.get('enclosing_symbol', '')} | "
            f"{ok} | `{sha}` | {row['audiohoi_probe']} / {row['audiohoi_metric']} | {row['full_reproduction_gap']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is source evidence, not a claim that the external method is fully reproduced.",
            "- `code_sha256` hashes a whitespace-normalized version of the cited block, so future source edits are detectable.",
            "- `keyword_hits` is a conservative sanity check that the cited block still contains the expected loss/constraint symbols.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def run() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [verify(spec) for spec in SPECS]
    csv_path = write_csv(OUT_DIR / "related_work_source_evidence_verified.csv", rows)
    json_path = write_json(OUT_DIR / "related_work_source_evidence_verified.json", {"rows": rows})
    md_path = write_markdown(rows)
    failures = [r for r in rows if not r.get("exists") or not r.get("valid_line_range") or r.get("missing_keywords")]
    return {
        "rows": len(rows),
        "failures": len(failures),
        "csv": str(csv_path),
        "json": str(json_path),
        "md": str(md_path),
    }


def main() -> None:
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
