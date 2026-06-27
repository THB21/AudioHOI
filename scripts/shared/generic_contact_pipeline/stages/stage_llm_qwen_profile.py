from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.config import load_case_profile, with_runtime_overrides  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.hoi_profile import collect_prompt_context, default_profile, seed_profile_path  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.io import write_json  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.schema import stage_paths  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.vlm_provider import load_vlm_provider  # noqa: E402


PROMPT = """You generate a conservative semantic HOI profile for AudioHOI.

Rules:
- Return JSON only.
- Do not output pose, coordinates, loss weights, or continuous corrections.
- Include only discrete semantic parts, body parts, interaction edges, support priors, motion priors, and VLM query policy.
- Use the seed/default profile as the schema guide.

Case config:
{case_config}

Original generation / model prompt context:
{prompt_context}

Seed/default profile:
{seed_profile}

Return one JSON object with keys:
case_name, object_category, object_family, object_parts, human_parts,
interaction_edges, support_priors, motion_priors, vlm_query_policy.
"""


def extract_json(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


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


def ask_text(model, processor, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs.input_ids.shape[-1] :]
    return processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main() -> None:
    provider = load_vlm_provider()
    provider.apply_env_defaults()
    ap = argparse.ArgumentParser(description="Generate a schema-guarded HOI profile candidate with local Qwen.")
    ap.add_argument("--case", required=True)
    ap.add_argument("--result-name", default="")
    ap.add_argument("--provider", default=provider.name)
    ap.add_argument("--model-id", default=provider.model_id)
    ap.add_argument("--local-dir", default=str(provider.local_dir) if provider.local_dir else "")
    ap.add_argument("--device-map", default=provider.device_map)
    ap.add_argument("--load-4bit", action="store_true", default=provider.load_4bit)
    ap.add_argument("--no-load-4bit", dest="load_4bit", action="store_false")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    if args.provider != provider.name:
        provider = load_vlm_provider(args.provider)
        provider.apply_env_defaults()

    profile = with_runtime_overrides(load_case_profile(args.case), result_name=args.result_name or None)
    seed_path = seed_profile_path(profile)
    if seed_path.exists():
        seed = json.loads(seed_path.read_text())
    else:
        seed = default_profile(profile)
    prompt_context = collect_prompt_context(profile)
    prompt = PROMPT.format(
        case_config=json.dumps(profile.data, ensure_ascii=False, indent=2),
        prompt_context=json.dumps(prompt_context, ensure_ascii=False, indent=2),
        seed_profile=json.dumps(seed, ensure_ascii=False, indent=2),
    )
    model, processor = load_model(args.model_id, args.local_dir, args.device_map, args.load_4bit)
    text = ask_text(model, processor, prompt, args.max_new_tokens)
    parsed = extract_json(text)
    paths = stage_paths(profile)
    out = paths["llm_prior_trace"].parent / "llm_prior_qwen_raw.json"
    write_json(
        out,
        {
            "case_name": profile.case_name,
            "provider": args.provider,
            "model_id": args.model_id,
            "local_dir": args.local_dir,
            "raw_text": text,
            "parsed_json": parsed,
            "parse_ok": parsed is not None,
            "prompt_context": prompt_context,
        },
    )
    print(str(out))
    if parsed is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
