# Legacy Refinement Policy Adapters

These files are retained for compatibility and migration only. They are not intended to be selected as independent pipeline branches.

The fixed Stage4 path is:

`stages/main/stage4_contact_refine.py -> components/mainline/sequence_refine.py`

Current policy files may still be called by `sequence_refine.py` as seed builders, geometry adapters, or residual builders while behavior is migrated into the generic SE3 mainline. New object behavior should be implemented in the mainline artifact contracts rather than by adding another object-specific runner.
