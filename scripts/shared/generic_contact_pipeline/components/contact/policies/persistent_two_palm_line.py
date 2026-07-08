from __future__ import annotations

import json
import subprocess

from ....core.base.config import CaseProfile
from ....core.base.io import REPO
from ....core.base.runtime import runtime_python
from ....core.base.schema import stage_paths


def build(profile: CaseProfile) -> dict[str, object]:
    """Stable two-palm anchors for a hand-held line object.

    Combines the mug stable-grasp pattern (persistent object-local hand anchors)
    with the chair line/segment pattern (object as semantic line coordinates).
    """
    paths = stage_paths(profile)
    script = REPO / "scripts/shared/generic_contact_pipeline/components/contact/solvers/persistent_two_palm_line_builder.py"
    cmd = [
        runtime_python("audiohoi"),
        str(script),
        "--sample-dir",
        str(profile.sample_dir),
        "--object-observations",
        str(paths["object_observations"]),
        "--contact-candidates",
        str(paths["contact_candidates"]),
        "--contact-state",
        str(paths["contact_state"]),
        "--metrics",
        str(paths["stage2_metrics"]),
        "--line-correspondence",
        str(paths["line_correspondence"]),
    ]
    result = subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Persistent two-palm line contact builder failed\n"
            f"cmd: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(paths["stage2_metrics"].read_text())
