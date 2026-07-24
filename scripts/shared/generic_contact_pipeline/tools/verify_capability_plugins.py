#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO = Path(__file__).resolve().parents[4]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.core.base.config import load_case_profile  # noqa: E402
from scripts.shared.generic_contact_pipeline.core.plugins.registry import (  # noqa: E402
    REGISTRY,
    resolve_pipeline_plugins,
)


CASES = ("basketball", "football", "mug", "chair", "stick")


def main() -> None:
    for spec in REGISTRY.all():
        spec.load()
        print(f"loaded {spec.plugin_id} -> {spec.module}:{spec.entrypoint}")
    for case_name in CASES:
        resolved = resolve_pipeline_plugins(load_case_profile(case_name))
        active = [
            resolved.observation.plugin_id,
            resolved.contact.plugin_id,
            resolved.pose.plugin_id,
            *[spec.plugin_id for spec in resolved.for_stage("stage4")],
        ]
        print(f"resolved {case_name}: {','.join(active)}")


if __name__ == "__main__":
    main()
