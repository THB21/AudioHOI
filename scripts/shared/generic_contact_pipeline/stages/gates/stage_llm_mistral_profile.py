from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[5]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile, with_runtime_overrides  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.semantics.hoi_profile import collect_prompt_context, default_profile, seed_profile_path  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.io import write_json  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.gates.llm_provider import load_llm_provider  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.base.schema import stage_paths  # noqa: E402


SYSTEM_PROMPT = """You are the text-only semantic prior generator for AudioHOI.

You must return JSON only. Do not output prose.
You must not output pose, coordinates, loss weights, SE(3) updates, or continuous corrections.
You may only output discrete semantic parts, human parts, interaction edges, support priors,
motion priors, and VLM query policy.
"""


USER_PROMPT = """Build a conservative HOI semantic profile for this case.

Use the original video/model prompt context as evidence, but keep the output schema compatible
with the seed/default profile.

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


def ask_mistral(provider, prompt: str) -> tuple[str, dict[str, Any]]:
    if not provider.api_key:
        raise RuntimeError(f"Missing API key in environment variable {provider.api_key_env}")
    url = provider.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": provider.model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": provider.temperature,
        "max_tokens": provider.max_new_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=provider.timeout_seconds)
    if resp.status_code >= 400:
        raise RuntimeError(f"Mistral API error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    text = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
    return text, data


def main() -> None:
    default_provider = load_llm_provider()
    ap = argparse.ArgumentParser(description="Generate a schema-guarded HOI profile candidate with Mistral.")
    ap.add_argument("--case", required=True)
    ap.add_argument("--result-name", default="")
    ap.add_argument("--provider", default=default_provider.name)
    ap.add_argument("--max-new-tokens", type=int, default=0)
    args = ap.parse_args()

    provider = load_llm_provider(args.provider)
    if args.max_new_tokens:
        provider = type(provider)(provider.name, {**provider.data, "max_new_tokens": args.max_new_tokens})
    profile = with_runtime_overrides(load_case_profile(args.case), result_name=args.result_name or None)
    seed_path = seed_profile_path(profile)
    if seed_path.exists():
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
    else:
        seed = default_profile(profile)
    prompt_context = collect_prompt_context(profile)
    prompt = USER_PROMPT.format(
        case_config=json.dumps(profile.data, ensure_ascii=False, indent=2),
        prompt_context=json.dumps(prompt_context, ensure_ascii=False, indent=2),
        seed_profile=json.dumps(seed, ensure_ascii=False, indent=2),
    )
    text, response = ask_mistral(provider, prompt)
    parsed = extract_json(text)
    paths = stage_paths(profile)
    out = paths["llm_prior_trace"].parent / "llm_prior_mistral_raw.json"
    write_json(
        out,
        {
            "case_name": profile.case_name,
            "provider": provider.name,
            "provider_doctor": provider.doctor(),
            "raw_text": text,
            "parsed_json": parsed,
            "parse_ok": parsed is not None,
            "prompt_context": prompt_context,
            "usage": response.get("usage", {}),
            "response_id": response.get("id", ""),
        },
    )
    print(str(out))
    if parsed is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
