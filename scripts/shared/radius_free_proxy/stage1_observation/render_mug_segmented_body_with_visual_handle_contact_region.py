"""Minimal stub for a module the teammate referenced but never committed.

`render_mug_articraft_rigid_mesh_vlm` imports this at module load (`as m9`) but only calls
its functions inside its own VLM/overlay render path. We only need the geometry helpers
(`load_articraft_meshes*`, `rot_y`, `transform`) from that module, so an importable stub is
enough to unblock those. The functions below raise if actually invoked, so we never silently
produce wrong overlays.
"""
from __future__ import annotations


def _missing(*_a, **_k):  # pragma: no cover - only hit if the overlay path is used
    raise NotImplementedError(
        "render_mug_segmented_body_with_visual_handle_contact_region was never committed "
        "by the teammate; only the geometry helpers of the rigid module are usable.")


def load_visibility_policy(*a, **k):
    return _missing(*a, **k)


def build_phase_track(*a, **k):
    return _missing(*a, **k)


def visibility_kind(*a, **k):
    return _missing(*a, **k)


def draw_contact_region(*a, **k):
    return _missing(*a, **k)
