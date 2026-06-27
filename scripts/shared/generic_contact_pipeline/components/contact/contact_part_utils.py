from __future__ import annotations

from typing import Iterable

import numpy as np


LEFT_FOOT_ID = 10
RIGHT_FOOT_ID = 11


def build_palm_centers(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_ids = [20, 25, 28, 31, 34]
    right_ids = [21, 40, 43, 46, 49]
    left_palm = joints[:, left_ids, :].mean(axis=1)
    right_palm = joints[:, right_ids, :].mean(axis=1)
    return left_palm, right_palm


def build_contact_part_centers(joints: np.ndarray) -> dict[str, np.ndarray]:
    left_hand, right_hand = build_palm_centers(joints)
    return {
        "left_hand": left_hand,
        "right_hand": right_hand,
        "left_foot": joints[:, LEFT_FOOT_ID, :],
        "right_foot": joints[:, RIGHT_FOOT_ID, :],
    }


def normalize_contact_label(
    row: dict[str, object],
    default_part: str,
    fallback_side: str = "right",
) -> str:
    label = str(row.get("contact_label", "") or "").strip()
    if label:
        return label
    target = str(row.get("target", "") or "").strip()
    if target:
        return target
    active_hand = str(row.get("active_hand", "") or "").strip()
    if active_hand:
        return f"{active_hand}_hand"
    active_foot = str(row.get("active_foot", "") or "").strip()
    if active_foot:
        return f"{active_foot}_foot"
    side = str(row.get("contact_side", row.get("active_anchor_side", fallback_side)) or fallback_side).strip()
    if str(row.get("contact_part", "") or "").strip() == "floor" or str(row.get("contact_type", "") or "") == "floor_contact_event":
        return "floor"
    return f"{side}_{default_part}"


def choose_active_contact_relation(
    joints: np.ndarray,
    labels: list[str],
    fallback_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = build_contact_part_centers(joints)
    ys = np.zeros(len(labels), dtype=np.float64)
    zs = np.zeros(len(labels), dtype=np.float64)
    names = np.empty(len(labels), dtype=object)
    for i, raw_label in enumerate(labels):
        label = raw_label if raw_label in centers else fallback_label
        part = centers[label][i]
        ys[i] = part[1]
        zs[i] = part[2]
        names[i] = label
    return ys, zs, names


def split_contact_label(label: str) -> tuple[str, str, str]:
    clean = str(label or "").strip()
    if not clean or clean == "floor":
        return "floor", "", "floor"
    if "_" not in clean:
        return "", "", clean
    side, part = clean.split("_", 1)
    return part, side, clean


def build_contact_identity(
    active_label: str,
    event_on: bool,
    floor_event_on: bool,
    default_part: str,
) -> tuple[str, str, str]:
    if floor_event_on and not event_on:
        return "floor", "", "floor"
    part, side, label = split_contact_label(active_label)
    if part:
        return part, side, label
    fallback = default_part
    fallback_side = "right"
    return fallback, fallback_side, f"{fallback_side}_{fallback}"


def event_frames_by_type(
    event_rows: Iterable[dict[str, str]],
    accepted_contact_types: set[str],
) -> set[int]:
    return {
        int(row["frame"])
        for row in event_rows
        if row.get("contact_type", "") in accepted_contact_types
    }


def infer_default_part(rows: Iterable[dict[str, object]], fallback: str = "hand") -> str:
    for row in rows:
        label = normalize_contact_label(dict(row), default_part=fallback, fallback_side="right")
        part, _, _ = split_contact_label(label)
        if part and part != "floor":
            return part
    return fallback


def is_floor_contact_row(row: dict[str, object]) -> bool:
    return (
        str(row.get("contact_type", "") or "") == "floor_contact_event"
        or str(row.get("contact_part", "") or "") == "floor"
        or str(row.get("contact_label", "") or "") == "floor"
    )


def human_event_frames_generic(event_rows: Iterable[dict[str, str]]) -> set[int]:
    frames: set[int] = set()
    for row in event_rows:
        if is_floor_contact_row(row):
            continue
        ctype = str(row.get("contact_type", "") or "")
        if ctype.endswith("_contact_event"):
            frames.add(int(row["frame"]))
    return frames


def resolve_human_state_key(state_row: dict[str, str]) -> str | None:
    for key in ("human_contact_state", "anchor_contact_state", "hand_contact_state"):
        if key in state_row:
            return key
    return None
