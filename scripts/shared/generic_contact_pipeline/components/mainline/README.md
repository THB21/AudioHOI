# Generic Mainline Components

This directory contains the fixed pipeline components used by `run_pipeline.py`.

- `observation.py`: normalizes object observations and correspondence artifacts.
- `contact_anchor.py`: normalizes contact candidates and anchor state.
- `pose_init.py`: promotes every case into the shared SE3 pose schema.
- `sequence_refine.py`: runs the generic sequence SE3 refinement, including smoothness, anchor, gate, and static-tail decisions.

Object-specific code should not be added here unless it is exposed through a generic geometry/contact adapter and keeps the shared artifact schema.
