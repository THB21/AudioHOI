from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[5]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import available_cases, load_case_profile, with_runtime_overrides  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import write_csv, write_json  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.schema import stage_paths  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.gates.vlm import RESULT_FIELDS, STAGE_QUERY_TYPES, fuse_stage_decision, write_aggregate, write_stage_verification  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.gates.vlm_provider import load_vlm_provider  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.gates.vlm_gates import GATE_FIELDS, gate_row_from_result, write_stage_gates  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.gates.amodal_mask_completion import QUERY_TYPE as AMODAL_MASK_QUERY_TYPE, materialize_selected_masks  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.gates.semantic_relations import materialize_semantic_relation_artifacts  # noqa: E402


PROMPT_TEMPLATE = """You are a conservative visual verifier for an AudioHOI pipeline.

Rules:
- Answer exactly one forced-choice label.
- Choose "unclear" if the image is insufficient, ambiguous, or the requested highlight is not visible.
- Do not propose pose corrections, loss weights, or 3D coordinates.
- Ignore debug text unless it directly names the highlighted point/part.
- Estimate confidence from the visible evidence. Use 0 only for missing/invalid input; do not copy the example value.

Question:
{question}

Allowed labels:
{choices}

Return ONLY compact JSON:
{{"label":"one_allowed_label","confidence":0.73,"short_reason":"brief visual reason"}}
"""

AMODAL_PROMPT_TEMPLATE = """You are completing an amodal binary object mask from visual evidence.

Rules:
- The left CURRENT panel is the target coordinate system.
- Preserve all pixels already present in the CURRENT partial mask.
- Infer only the same rigid object behind the human; never include the person.
- Use the two reference panels to preserve rigid body size, orientation, handle attachment, and floor support.
- Approve completion only when the partial pieces and references support one rigid suitcase behind the person.
- Pixel filling is deterministic and limited to the existing body-region hull; you do not draw pixels or coordinates.
- Do not output pose, depth, quaternion, or loss weights.

Question:
{question}

Allowed labels:
{choices}

Return ONLY compact JSON:
{{"label":"one_allowed_label","confidence":0.73,"short_reason":"brief visual reason"}}
"""


PASS_LABELS = {
    "target_mask_check": {"yes", "partially"},
    "keypart_identity_check": "non_background_unclear",
    "keypart_visibility_check": {"visible", "partially_visible"},
    "track_stability_check": {"stable"},
    "contact_relation_check": "non_none_unclear",
    "overlay_alignment_check": {"aligned"},
    "anchor_update_check": {"improvement", "no_change"},
    "temporal_motion_check": {"consistent"},
    "constraint_reliability_check": {
        "visual_observation_reliable",
        "contact_relation_reliable",
        "both_consistent",
    },
    "interval_candidate_selection_check": {
        "candidate_a",
        "candidate_b",
    },
    AMODAL_MASK_QUERY_TYPE: {
        "completed_from_partial_and_references",
    },
    "post_render_sanity_check": {"pass"},
    "baseline_regression_check": {"no_regression"},
    "loss_diagnostic_check": {"normal"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_model(model_id: str, local_dir: str | None, device_map: str, load_4bit: bool):
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as ModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as ModelCls

    model_path = local_dir or model_id
    kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": device_map}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
    else:
        kwargs["torch_dtype"] = "auto"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = ModelCls.from_pretrained(model_path, **kwargs)
    return model, processor


def extract_json(text: str, choices: list[str]) -> dict[str, Any]:
    raw = text.strip()
    data: dict[str, Any] = {}
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
        except Exception:
            data = {}
    label = str(data.get("label", "")).strip()
    if label not in choices:
        lowered = {choice.lower(): choice for choice in choices}
        label = lowered.get(label.lower(), "")
    if not label:
        for choice in choices:
            if re.search(rf"\b{re.escape(choice)}\b", raw, re.I):
                label = choice
                break
    if label not in choices:
        label = "unclear" if "unclear" in choices else choices[-1]
    try:
        conf = float(data.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    polygon = data.get("amodal_polygon_norm")
    if not (
        isinstance(polygon, list)
        and len(polygon) == 4
        and all(isinstance(point, list) and len(point) == 2 for point in polygon)
    ):
        polygon = []
    return {
        "label": label,
        "confidence": max(0.0, min(1.0, conf)),
        "short_reason": str(data.get("short_reason", ""))[:240],
        "raw_text": raw,
        "amodal_polygon_norm": polygon,
    }


def ask_qwen(model, processor, image: Any, question: str, choices: list[str], max_new_tokens: int, *, query_type: str = "") -> str:
    template = AMODAL_PROMPT_TEMPLATE if query_type == AMODAL_MASK_QUERY_TYPE else PROMPT_TEMPLATE
    prompt = template.format(question=question, choices=" / ".join(choices))
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    except Exception:
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs.input_ids.shape[-1] :]
    return processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def query_image_path(query: dict[str, str]) -> Path | None:
    for key in ("input_render_path", "input_image_path"):
        value = query.get(key, "")
        if value:
            path = Path(value)
            if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                return path
    return None


def load_query_image(path: Path, resize_max: int) -> Any:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if resize_max > 0:
        w, h = image.size
        scale = min(1.0, float(resize_max) / float(max(w, h)))
        if scale < 1.0:
            image = image.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))))
    return image


def pass_gate(query_type: str, label: str) -> str:
    rule = PASS_LABELS.get(query_type)
    if rule == "non_background_unclear":
        return "pass" if label not in {"background", "unclear"} else "unclear" if label == "unclear" else "reject"
    if rule == "non_none_unclear":
        return "pass" if label not in {"nothing_nearby", "none", "no_contact", "unclear"} else "unclear" if label == "unclear" else "reject"
    if isinstance(rule, set):
        if label in rule:
            return "pass"
        if label == "unclear":
            return "unclear"
        return "reject"
    return "unclear" if label == "unclear" else "pass"


def repair_action(query_type: str, gate: str, label: str) -> str:
    if query_type == AMODAL_MASK_QUERY_TYPE:
        return "use_mask_fallback" if gate == "pass" else "unclear_no_update"
    if query_type == "constraint_reliability_check":
        if label == "both_consistent":
            return "accept_candidate"
        if label in {"visual_observation_reliable", "contact_relation_reliable"}:
            return "downweight_correspondence"
        return "unclear_no_update"
    if query_type == "interval_candidate_selection_check":
        if label == "candidate_b":
            return "select_occlusion_challenger"
        if label == "candidate_a":
            return "keep_stable_candidate"
        if label == "reject_both":
            return "reject_candidate_pair"
        return "unclear_no_update"
    if gate == "pass":
        return "accept_candidate"
    if gate == "unclear":
        return "unclear_no_update"
    if query_type == "contact_relation_check":
        return "disable_contact_frame"
    if query_type in {"keypart_visibility_check", "track_stability_check"}:
        return "keep_previous_anchor"
    if query_type == "anchor_update_check":
        return "keep_previous_anchor" if label in {"worse_overlay", "wrong_contact"} else "unclear_no_update"
    if query_type == "overlay_alignment_check":
        return "allow_small_se3_refine" if label in {"lateral_shift", "depth_shift", "rotation_error"} else "mark_for_report_only"
    if query_type == "temporal_motion_check":
        return "freeze_before_after_contact"
    if query_type == "loss_diagnostic_check":
        return "mark_for_report_only"
    if query_type == "target_mask_check":
        return "use_mask_fallback"
    return "mark_for_report_only"


def result_row(query: dict[str, str], label: str, gate: str, action: str) -> dict[str, object]:
    return {
        "case_name": query.get("case_name", ""),
        "stage": query.get("stage", ""),
        "frame": query.get("frame", ""),
        "query_type": query.get("query_type", ""),
        "normalized_label": label,
        "pass_gate": gate,
        "repair_action": action,
        "used_for_anchor_update": "1"
        if gate == "pass"
        and query.get("query_type") in {"keypart_identity_check", "keypart_visibility_check", "track_stability_check", "anchor_update_check"}
        else "0",
        "used_for_contact_residual": "1" if gate == "pass" and query.get("query_type") == "contact_relation_check" else "0",
        "used_for_report": "1",
    }


def evaluate_stage(profile, stage: str, args: argparse.Namespace) -> dict[str, object]:
    paths = stage_paths(profile)
    if args.refresh_queries:
        write_stage_verification(profile, stage, args.query_type or None)
    qpath = paths["vlm_dir"] / stage / "vlm_queries.csv"
    if not qpath.exists():
        raise FileNotFoundError(f"Missing VLM queries: {qpath}. Run stage_vlm_verify first.")
    queries = read_csv(qpath)
    if args.query_type:
        queries = [row for row in queries if row.get("query_type") == args.query_type]
        if not queries:
            raise ValueError(f"No {args.query_type} queries were generated for {stage}")
    debug_limited = args.limit > 0
    if args.limit > 0:
        queries = queries[: args.limit]
    model, processor = load_model(args.model_id, args.local_dir, args.device_map, args.load_4bit)
    results: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for idx, query in enumerate(queries, start=1):
        choices = [c for c in query.get("choices", "").split("|") if c]
        image_path = query_image_path(query)
        if not choices or image_path is None:
            label = "unclear" if "unclear" in choices else "missing_input"
            gate = "unclear"
            action = "unclear_no_update"
            raw = {"label": label, "confidence": 0.0, "short_reason": "missing image or choices", "raw_text": ""}
        else:
            print(f"[qwen-generic] {profile.case_name} {stage} {idx}/{len(queries)} frame={query.get('frame')} type={query.get('query_type')}", file=sys.stderr, flush=True)
            image = load_query_image(image_path, args.resize_max)
            text = ask_qwen(
                model,
                processor,
                image,
                query["question"],
                choices,
                max(args.max_new_tokens, 220) if query.get("query_type") == AMODAL_MASK_QUERY_TYPE else args.max_new_tokens,
                query_type=query.get("query_type", ""),
            )
            raw = extract_json(text, choices)
            label = str(raw["label"])
            gate = pass_gate(query.get("query_type", ""), label)
            action = repair_action(query.get("query_type", ""), gate, label)
        results.append(result_row(query, label, gate, action))
        raw_rows.append(
            {
                **query,
                **raw,
                "pass_gate": gate,
                "repair_action": action,
                "provider": args.provider,
                "model": args.model_id,
            }
        )

    out_dir = paths["vlm_dir"] / stage
    decision = fuse_stage_decision(profile.case_name, stage, results, mode="qwen_vl")
    if debug_limited:
        write_csv(out_dir / "vlm_results_qwen_debug.csv", results, RESULT_FIELDS)
        gates = [gate_row_from_result({key: str(value) for key, value in row.items()}) for row in results]
        write_csv(out_dir / "vlm_gates_qwen_debug.csv", gates, GATE_FIELDS)
        write_json(
            out_dir / "vlm_gate_decision_qwen_debug.json",
            {
                "case_name": profile.case_name,
                "stage": stage,
                "gate_rows": len(gates),
                "effective_rows": sum(1 for row in gates if row["is_effective"] == "1"),
                "gate_csv": str(out_dir / "vlm_gates_qwen_debug.csv"),
                "debug_limit": args.limit,
                "note": "Debug Qwen gates are audit evidence only and do not overwrite full-stage vlm_gates.csv.",
            },
        )
        write_json(out_dir / "qwen_raw_results_debug.json", raw_rows)
        write_json(out_dir / "stage_decision_qwen_debug.json", decision)
    else:
        if args.query_type:
            existing_results = [
                row
                for row in read_csv(out_dir / "vlm_results.csv")
                if row.get("query_type") != args.query_type
            ] if (out_dir / "vlm_results.csv").exists() else []
            existing_raw: list[dict[str, object]] = []
            raw_path = out_dir / "qwen_raw_results.json"
            if raw_path.exists():
                payload = json.loads(raw_path.read_text())
                if isinstance(payload, list):
                    existing_raw = [
                        row
                        for row in payload
                        if isinstance(row, dict) and row.get("query_type") != args.query_type
                    ]
            results = [*existing_results, *results]
            raw_rows = [*existing_raw, *raw_rows]
            decision = fuse_stage_decision(profile.case_name, stage, results, mode="qwen_vl")
        write_csv(out_dir / "vlm_results.csv", results, RESULT_FIELDS)
        write_json(out_dir / "qwen_raw_results.json", raw_rows)
        if stage == "stage4":
            materialize_selected_masks(profile, raw_rows)
            materialize_semantic_relation_artifacts(out_dir, raw_rows, status="qwen_evaluated")
        write_json(out_dir / "stage_decision.json", decision)
        write_aggregate(profile)
        write_stage_gates(profile, stage)
    del model
    del processor
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    return decision


def main() -> None:
    provider = load_vlm_provider()
    provider.apply_env_defaults()
    ap = argparse.ArgumentParser(description="Run local Qwen-VL over generated generic VLM verification queries.")
    ap.add_argument("--case", default="all", help="Case name or all")
    ap.add_argument("--stage", default="stage2", help="Pipeline stage or all")
    ap.add_argument("--result-name", default="", help="Override case profile result_name.")
    ap.add_argument("--provider", default=provider.name, help="Provider name from configs/vlm_provider.yaml")
    ap.add_argument("--model-id", default=provider.model_id)
    ap.add_argument("--local-dir", default=str(provider.local_dir) if provider.local_dir else "")
    ap.add_argument("--device-map", default=provider.device_map)
    ap.add_argument("--load-4bit", action="store_true", default=provider.load_4bit)
    ap.add_argument("--no-load-4bit", dest="load_4bit", action="store_false")
    ap.add_argument("--max-new-tokens", type=int, default=provider.max_new_tokens)
    ap.add_argument("--resize-max", type=int, default=provider.resize_max, help="Resize the longest image side before inference. 0 disables resizing.")
    ap.add_argument("--limit", type=int, default=0, help="Debug limit per stage. 0 means all queries.")
    ap.add_argument("--query-type", default="", help="Evaluate one registered query type while preserving other stage results.")
    ap.add_argument("--no-refresh-queries", dest="refresh_queries", action="store_false", help="Use existing vlm_queries.csv without regenerating evidence images.")
    ap.add_argument("--doctor", action="store_true", help="Print provider/environment status and exit.")
    ap.set_defaults(refresh_queries=True)
    args = ap.parse_args()

    if args.provider != provider.name:
        provider = load_vlm_provider(args.provider)
        provider.apply_env_defaults()
        if args.model_id == ap.get_default("model_id"):
            args.model_id = provider.model_id
        if args.local_dir == ap.get_default("local_dir"):
            args.local_dir = str(provider.local_dir) if provider.local_dir else ""
        if args.device_map == ap.get_default("device_map"):
            args.device_map = provider.device_map
        if args.max_new_tokens == ap.get_default("max_new_tokens"):
            args.max_new_tokens = provider.max_new_tokens
        if args.resize_max == ap.get_default("resize_max"):
            args.resize_max = provider.resize_max

    if args.doctor:
        print(json.dumps(provider.doctor(), indent=2, ensure_ascii=False))
        return

    cases = available_cases() if args.case == "all" else [args.case]
    stages = list(STAGE_QUERY_TYPES) if args.stage == "all" else [args.stage]
    failed = False
    for case in cases:
        profile = with_runtime_overrides(load_case_profile(case), result_name=args.result_name or None)
        for stage in stages:
            try:
                decision = evaluate_stage(profile, stage, args)
                print(f"{case} {stage}: {decision['decision']} {decision.get('gate_counts', {})}")
            except Exception as exc:
                failed = True
                print(f"{case} {stage}: FAILED: {exc}", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
