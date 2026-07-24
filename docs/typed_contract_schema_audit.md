# Five-Case Stage Contract Audit

This audit describes the existing `benchmark_vlm_qwen` artifacts at the
artifact-store checkpoint. It does not claim that similarly named columns have
identical solver semantics.

## Stage 1

| Artifact | Cases | Adapter | Observed semantics |
| --- | --- | --- | --- |
| `object_observations.csv` | basketball, football, stick | `point_reference_observation_v1` | reference/support/contact pixels plus depth/confidence |
| `object_observations.csv` | mug | `mug_rigid_parts_observation_v1` | body/handle geometry, visibility, phase and contact-side observations |
| `object_observations.csv` | chair | `chair_semantic_graph_observation_v1` | rails, seat, feet, leg-line families and tracked semantic landmarks |
| `object_correspondence.csv` | basketball, football, mug, chair | `tracked_point_correspondence_v1` | frame/time point correspondence envelope |
| `object_correspondence.csv` | stick | `line_object_correspondence_v1` | frame-indexed compatibility envelope; historical `time` values are empty and line endpoints live in `line_observations.csv` |
| `line_observations.csv` | stick only | `line_observation_v1` | visible/physical endpoints, line trust and occlusion |
| `object_semantic_points.csv` | chair | `semantic_segment_v1` | static local semantic segments |
| `object_semantic_points.csv` | other four | `semantic_point_v1` | static local semantic points; currently empty in canonical runs |

The mug per-frame surface/contact table and chair static local geometry are not
treated as one shared point-cloud schema.

## Stage 2

`contact_candidates.csv` has a common identity/contact envelope but three
different local-coordinate capabilities:

- ball cases: no stable local coordinate in the generic row;
- mug/chair: local XYZ extensions;
- stick: scalar coordinate `s` on a line object.

The adapter therefore returns `PointLocalCoordinate`, `LineLocalCoordinate`, or
`None`. It never converts line `s` into invented XYZ values.

`anchor_state.csv` is an explicit tagged union:

| Cases | Adapter | Identity/local semantics |
| --- | --- | --- |
| basketball, football, mug, chair | `point_anchor_state_v1` | contact id, human/object parts, stable/observed XYZ and optional scalar fields |
| stick | `line_anchor_state_v1` | human side, observed/stable line `s`, anchor pixel/depth, trust and occlusion |

An empty `human_side` remains `None`; this occurs legitimately on non-human
support contacts and is not replaced by a guessed side.

## Stage 3 and Stage 4

All five cases expose the common SE(3) envelope
`frame,time,tx,ty,tz,qw,qx,qy,qz`. Additional Euler angles, articulation,
camera, projection, contact-lock, or line-fit columns remain case extensions.
The typed adapter checks finite translation/quaternion values and preserves the
original CSV unchanged.

Stage 4 motion-regime, physical-smooth residual, and optimizer-decision tables
share stable core columns. VLM optimizer fields are extensions rather than
requirements because current historical manifests do not consistently record
VLM execution even when VLM-named result directories exist.

## Audit Result

The canonical contract command passes all 20 case/stage combinations:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/verify_stage_contracts.py
```
