# Five-Case Capability Plugin Matrix

The registry preserves the existing YAML selectors and implementation modules.
Capabilities describe why a selected module is compatible; they do not imply
that the underlying algorithms are generic.

| Case | Observation plugin | Contact plugin | Pose plugin | Ordered refinement plugins |
| --- | --- | --- | --- | --- |
| basketball | `mask_track_center` | `hand_floor` | `translation3` | `anchor_depth`, `backproject_xy` |
| football | `mask_track_center` | `foot_floor` | `translation3` | `anchor_depth`, `backproject_xy` |
| mug | `rigid_body_plus_parts` | `palm_handle_rim_body` | `rigid6_plus_phase` | `stable_grasp_anchor`, `anchor_depth`, `table_freeze` |
| chair | `semantic_graph_tracks` | `two_hand_toprail_endpoint` | `semantic_graph_6d` | `small_se3`, `anchor_propagate_freeze`, `sequence_se3_optimizer` |
| stick | `mask_track_center` | `persistent_two_palm_line` | `translation3` | `line_contact_lock`, `backproject_xy` |

## Capability distinctions

- `observation.point_reference`, `observation.rigid_parts`, and
  `observation.semantic_graph` remain separate observation capabilities.
- Point-anchor plugins provide `anchor.local_xyz`; the stick line plugin
  provides `anchor.local_s` and additionally requires `geometry.line_object`.
- All active pose plugins provide `pose.se3`; mug additionally provides
  `articulation.phase` and chair provides `articulation.support`.
- `sequence_se3_optimizer` is declared as a
  `mainline_marker_and_implementation`, matching current behavior: it is not
  invoked as a compatibility seed builder because the mainline sequence
  optimizer is executed explicitly later in Stage 4. It is implicitly active
  for all five cases even when absent from a case's YAML refinement list.
- `generic_line_physical_smooth` is a `mainline_implementation`, also excluded
  from compatibility-seed invocation to avoid double execution. It is
  implicitly active for line-object profiles such as stick.

The registry resolves the complete chain before `run_case` starts a stage. An
unknown selector or missing required capability therefore fails before any
stage artifact is overwritten.
