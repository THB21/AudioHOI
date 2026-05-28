## Contact Candidates

This folder is for generic 2D contact-candidate detection before 3D refinement.

We distinguish two levels:

1. **Contact events**: short or center-like moments, such as a bounce touch-down or a hand-contact keyframe.
2. **Contact states**: frame-wise or interval-wise contact that can last for multiple frames and can coexist with other contact states.

Planned pipeline:

1. Build contact candidates from stable 2D observations.
2. Split candidates by target/type, such as `hand_contact_event` and `floor_contact_event`.
3. Build frame-wise multi-label contact states, such as `hand_contact_state` and `floor_contact_state`.
4. Merge frame-wise states into contact intervals.
5. Optionally re-rank or label candidate windows with a VLM.
6. Feed typed candidates into downstream 3D optimization.

Current outputs:

- `hand_contact_candidates.csv`
- `floor_contact_candidates.csv`
- `contact_candidates_labeled.csv`
- `contact_state_frames.csv`
- `contact_intervals.csv`

The first entry points are:

- `run_contact_candidate_detection.py`
- `render_contact_candidates.py`
