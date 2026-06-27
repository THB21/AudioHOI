#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[5]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
else:
    REPO = Path(__file__).resolve().parents[5]

from scripts.shared.generic_contact_pipeline.core.base.io import write_csv, write_json  # noqa: E402


OUT_DIR = REPO / "docs" / "related_work_audit"


LOSS_ROWS: list[dict[str, str]] = [
    {
        "method": "InterDiff",
        "term": "signed human-object collision / penetration",
        "source_path": "external_baselines/InterDiff/interdiff/optimization.py",
        "source_lines": "72-76,109-116",
        "source_evidence": "builds contact_human_label from object-human distances and accumulates loss_dist_o into final loss/collision log",
        "role_in_original": "interaction correction penalizes object-human penetration while preserving plausible contact",
        "audiohoi_proxy": "related_work_sdf_penetration_probe; related_work_geometry_contact_probe",
        "implemented_test": "mesh_sdf_proxy",
        "missing_for_full_test": "full InterDiff correction network/data adapter, not the collision term itself",
    },
    {
        "method": "InterDiff",
        "term": "object pose regularization",
        "source_path": "external_baselines/InterDiff/interdiff/optimization.py",
        "source_lines": "89-90,109-117",
        "source_evidence": "loss_obj_transl_reg and loss_obj_rot_reg keep optimized object near initial/generated trajectory",
        "role_in_original": "prevents correction from drifting away from generated object motion",
        "audiohoi_proxy": "generic/baseline pose delta and acceleration comparison",
        "implemented_test": "csv_proxy",
        "missing_for_full_test": "explicit candidate seed-vs-refined trajectory retained for every case",
    },
    {
        "method": "InterDiff",
        "term": "motion smoothness / acceleration",
        "source_path": "external_baselines/InterDiff/interdiff/optimization.py",
        "source_lines": "95-107",
        "source_evidence": "second-difference terms on translation, global rotation, hand pose, object, and body",
        "role_in_original": "suppresses jitter after interaction correction",
        "audiohoi_proxy": "method_interdiff_p95_translation_accel; method_interdiff_p95_rotation_accel",
        "implemented_test": "csv_proxy",
        "missing_for_full_test": "none for object pose CSV; hand/body pose acceleration can be added from GVHMR PKL",
    },
    {
        "method": "InterDiff",
        "term": "contact/reference-frame object motion",
        "source_path": "external_baselines/InterDiff/interdiff/model/correction_smpl.py",
        "source_lines": "74-79,126-136",
        "source_evidence": "selects human contact vertices and samples object motion relative to contact-conditioned candidates",
        "role_in_original": "makes interaction motion easier by anchoring it to contact regions",
        "audiohoi_proxy": "method_interdiff_contact_frame_anchor_velocity_p95; contact local anchor stability",
        "implemented_test": "csv_proxy_when_contact_local_points_exist",
        "missing_for_full_test": "dense contact vertex labels for all cases, not only local anchor points",
    },
    {
        "method": "InterDiff",
        "term": "contact label generation",
        "source_path": "external_baselines/InterDiff/interdiff/data/prepare_behave.py",
        "source_lines": "32-52,96-117",
        "source_evidence": "labels object and human contact vertices by nearest distance threshold",
        "role_in_original": "supervises contact-conditioned diffusion/correction",
        "audiohoi_proxy": "object_contact_points.csv nearest local point and contact gap columns",
        "implemented_test": "partial_csv_proxy",
        "missing_for_full_test": "dense object/human surface samples per AudioHOI frame",
    },
    {
        "method": "MOVER",
        "term": "ground-plane contact/support",
        "source_path": "external_baselines/MOVER/mover/scene_reconstruction/_forward_hsi.py",
        "source_lines": "42-52,71-86",
        "source_evidence": "pulls contacted foot vertices to ground plane and penalizes unsupported feet above plane",
        "role_in_original": "estimates/supports physically plausible camera/scene placement",
        "audiohoi_proxy": "method_mover_tail_static_drift_mean_m; static/freeze drift after place",
        "implemented_test": "csv_proxy",
        "missing_for_full_test": "explicit ground plane and foot contact vertices per frame",
    },
    {
        "method": "MOVER",
        "term": "ordinal depth",
        "source_path": "external_baselines/MOVER/mover/scene_reconstruction/_forward_hsi.py; external_baselines/MOVER/mover/scene_reconstruction/_accumulate_hsi_depth.py",
        "source_lines": "_forward_hsi.py:101-123; _accumulate_hsi_depth.py:25-70",
        "source_evidence": "accumulates human depth template, then computes relative depth/overlap/free-space losses for object meshes",
        "role_in_original": "uses 2D mask overlap to decide front/back ordering without metric depth labels",
        "audiohoi_proxy": "method_mover_median_depth_order_proxy_m from contact/depth gaps",
        "implemented_test": "partial_csv_proxy",
        "missing_for_full_test": "rendered human/object depth maps and overlap masks",
    },
    {
        "method": "MOVER",
        "term": "SDF collision",
        "source_path": "external_baselines/MOVER/mover/scene_reconstruction/_forward_hsi.py; external_baselines/MOVER/mover/scene_reconstruction/_accumulate_hsi_sdf.py",
        "source_lines": "_forward_hsi.py:126-154; _accumulate_hsi_sdf.py:224-259",
        "source_evidence": "fuses body SDF volume and penalizes object vertices inside negative SDF",
        "role_in_original": "prevents object placement from penetrating accumulated human body volume",
        "audiohoi_proxy": "related_work_sdf_penetration_probe",
        "implemented_test": "mesh_sdf_proxy",
        "missing_for_full_test": "full fused multi-frame MOVER body SDF volume; current test queries per-frame SMPL-X body mesh SDF",
    },
    {
        "method": "MOVER",
        "term": "coarse contact Chamfer with normal/axis filtering",
        "source_path": "external_baselines/MOVER/mover/scene_reconstruction/_forward_hsi.py; external_baselines/MOVER/mover/scene_reconstruction/_accumulate_hsi_contact_loss.py",
        "source_lines": "_forward_hsi.py:156-218; _accumulate_hsi_contact_loss.py:26-98",
        "source_evidence": "matches accumulated body contact vertices to object contact vertices with Chamfer-style distances split by normal axes",
        "role_in_original": "turns noisy body contact evidence into a region-level object placement constraint",
        "audiohoi_proxy": "method_mover_direct_contact_gap_median_m for chair; generic contact gap for other cases",
        "implemented_test": "partial_csv_proxy",
        "missing_for_full_test": "dense contact vertices/normals instead of single anchors",
    },
    {
        "method": "Open3DHOI / HOI-Gaussian",
        "term": "Gaussian RGB/mask image fit",
        "source_path": "external_baselines/open-3dhoi/HOIGaussian/train.py",
        "source_lines": "198-250",
        "source_evidence": "combines L1 RGB, alpha mask L2, SSIM, LPIPS, object center and size losses before HOI refinement",
        "role_in_original": "first fits visible image evidence before physical HOI losses",
        "audiohoi_proxy": "method_opengaus_median_visual_proxy_px; mask center velocity; mask area/width CV",
        "implemented_test": "csv_proxy",
        "missing_for_full_test": "full differentiable render/RGB loss over AudioHOI frames",
    },
    {
        "method": "Open3DHOI / HOI-Gaussian",
        "term": "contact Chamfer",
        "source_path": "external_baselines/open-3dhoi/HOIGaussian/hoi_optimizer/loss.py; external_baselines/open-3dhoi/HOIGaussian/hoi_optimizer/__init__.py",
        "source_lines": "loss.py:24-32; __init__.py:73-77",
        "source_evidence": "computes distance between selected human-contact and object-contact vertices",
        "role_in_original": "pulls Gaussian object and human surfaces together at contact labels",
        "audiohoi_proxy": "contact gap and direct chair pairprop contact metrics",
        "implemented_test": "partial_csv_proxy",
        "missing_for_full_test": "contact vertex sets from rendered/reconstructed human/object surfaces",
    },
    {
        "method": "Open3DHOI / HOI-Gaussian",
        "term": "ordinal depth render loss",
        "source_path": "external_baselines/open-3dhoi/HOIGaussian/hoi_optimizer/loss.py; external_baselines/open-3dhoi/HOIGaussian/hoi_optimizer/__init__.py",
        "source_lines": "loss.py:35-111; __init__.py:80-83",
        "source_evidence": "renders human/object depth maps and penalizes wrong front/back ordering under mask evidence",
        "role_in_original": "resolves object-human depth ordering in image space",
        "audiohoi_proxy": "method_mover_median_depth_order_proxy_m",
        "implemented_test": "partial_csv_proxy",
        "missing_for_full_test": "human/object depth render buffers under K_fullimg",
    },
    {
        "method": "Open3DHOI / HOI-Gaussian",
        "term": "collision loss",
        "source_path": "external_baselines/open-3dhoi/HOIGaussian/hoi_optimizer/loss.py; external_baselines/open-3dhoi/HOIGaussian/hoi_optimizer/__init__.py",
        "source_lines": "loss.py:239-293; __init__.py:90-93",
        "source_evidence": "computes bidirectional HO collision loss and suppresses small loss values under threshold",
        "role_in_original": "prevents interpenetration during HOI refinement",
        "audiohoi_proxy": "related_work_sdf_penetration_probe; related_work_dense_mesh_probe",
        "implemented_test": "mesh_sdf_proxy",
        "missing_for_full_test": "Open3DHOI HOCollisionLoss dependency stack and differentiable optimizer loop",
    },
]


def write_markdown(rows: list[dict[str, str]]) -> Path:
    path = OUT_DIR / "related_work_loss_catalog.md"
    lines = [
        "# Related Work Loss Catalog",
        "",
        "Generated by `scripts/shared/generic_contact_pipeline/stages/stage_related_work_loss_catalog.py`.",
        "This catalog records concrete source-level loss terms and how they are or are not tested on AudioHOI outputs.",
        "",
        "| method | term | source | original role | AudioHOI proxy | status | missing for full test |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        source = f"`{row['source_path']}:{row['source_lines']}`"
        lines.append(
            f"| {row['method']} | {row['term']} | {source} | {row['role_in_original']} | "
            f"{row['audiohoi_proxy']} | {row['implemented_test']} | {row['missing_for_full_test']} |"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def run() -> dict[str, str | int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(OUT_DIR / "related_work_loss_catalog.csv", LOSS_ROWS)
    json_path = write_json(OUT_DIR / "related_work_loss_catalog.json", {"rows": LOSS_ROWS})
    md_path = write_markdown(LOSS_ROWS)
    return {"rows": len(LOSS_ROWS), "csv": str(csv_path), "json": str(json_path), "md": str(md_path)}


def main() -> None:
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
