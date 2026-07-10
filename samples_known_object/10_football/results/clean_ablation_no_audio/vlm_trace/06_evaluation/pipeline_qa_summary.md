# Pipeline VLM/LLM QA Summary

This report aggregates pipeline-stage VLM/LLM questions, answers, gates, and affected constraints.

| source | stage | frame | question | parsed | gate | affected constraint | changed optimizer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vlm | stage0 | 1 | Does the highlighted mask correspond to the target football? | yes | pass | visual_overlay_residual | 1 |
| vlm | stage1 | 1 | Which football part is highlighted? | ball_center | pass | audit_report | 1 |
| vlm | stage1 | 122 | Which football part is highlighted? | ball_center | pass | audit_report | 1 |
| vlm | stage1 | 242 | Which football part is highlighted? | ball_center | pass | audit_report | 1 |
| vlm | stage1 | 1 | Is the highlighted reference point or part stable and visually trackable across neighboring frames? | stable | pass | visual_overlay_residual | 1 |
| vlm | stage1 | 122 | Is the highlighted reference point or part stable and visually trackable across neighboring frames? | stable | pass | visual_overlay_residual | 1 |
| vlm | stage1 | 242 | Is the highlighted reference point or part stable and visually trackable across neighboring frames? | stable | pass | visual_overlay_residual | 1 |
| vlm | stage2 | 2 | What is the highlighted object contact region closest to? | foot_kick_touch_ball_boundary | pass | contact_anchor_residual | 1 |
| vlm | stage2 | 44 | What is the highlighted object contact region closest to? | foot_kick_touch_ball_boundary | pass | contact_anchor_residual | 1 |
| vlm | stage2 | 242 | What is the highlighted object contact region closest to? | foot_kick_touch_ball_boundary | pass | contact_anchor_residual | 1 |
| vlm | stage2 | 1 | What is the highlighted object contact region closest to? | foot_kick_touch_ball_boundary | pass | contact_anchor_residual | 1 |
| vlm | stage2 | 122 | What is the highlighted object contact region closest to? | foot_kick_touch_ball_boundary | pass | contact_anchor_residual | 1 |
| vlm | stage3 | 1 | Does the rendered object overlay sit on top of the same visible physical object in the video? Choose aligned only if the rendered object follows the real visible object; choose lateral_shift, wrong_scale, or rotation_error if it is offset, too long/short, or at a different angle. | unclear | unclear | visual_overlay_residual | 1 |
| vlm | stage3 | 122 | Does the rendered object overlay sit on top of the same visible physical object in the video? Choose aligned only if the rendered object follows the real visible object; choose lateral_shift, wrong_scale, or rotation_error if it is offset, too long/short, or at a different angle. | unclear | unclear | visual_overlay_residual | 1 |
| vlm | stage3 | 242 | Does the rendered object overlay sit on top of the same visible physical object in the video? Choose aligned only if the rendered object follows the real visible object; choose lateral_shift, wrong_scale, or rotation_error if it is offset, too long/short, or at a different angle. | unclear | unclear | visual_overlay_residual | 1 |
| vlm | stage4 | 2 | Using the visual guide in the image: brown render is the predicted stick, green line is the tracked visible stick, yellow circles are active palm contacts, and gray circles are inactive palms. Does the predicted stick preserve alignment with the green visible stick and the active yellow palm contacts without creating a wrong contact? | unclear | unclear | audit_report | 1 |
| vlm | stage4 | 44 | Using the visual guide in the image: brown render is the predicted stick, green line is the tracked visible stick, yellow circles are active palm contacts, and gray circles are inactive palms. Does the predicted stick preserve alignment with the green visible stick and the active yellow palm contacts without creating a wrong contact? | unclear | unclear | audit_report | 1 |
| vlm | stage4 | 242 | Using the visual guide in the image: brown render is the predicted stick, green line is the tracked visible stick, yellow circles are active palm contacts, and gray circles are inactive palms. Does the predicted stick preserve alignment with the green visible stick and the active yellow palm contacts without creating a wrong contact? | unclear | unclear | audit_report | 1 |
| vlm | stage4 | 1 | Using the visual guide in the image: brown render is the predicted stick, green line is the tracked visible stick, yellow circles are active palm contacts, and gray circles are inactive palms. Does the predicted stick preserve alignment with the green visible stick and the active yellow palm contacts without creating a wrong contact? | unclear | unclear | audit_report | 1 |
| vlm | stage4 | 122 | Using the visual guide in the image: brown render is the predicted stick, green line is the tracked visible stick, yellow circles are active palm contacts, and gray circles are inactive palms. Does the predicted stick preserve alignment with the green visible stick and the active yellow palm contacts without creating a wrong contact? | unclear | unclear | audit_report | 1 |
| vlm | stage4 | 2 | In the three-panel neighboring-frame view, do the brown predicted stick and green tracked stick move consistently without sudden jumps, wrong-object tracking, or physically implausible motion? | unclear | unclear | sequence_optimizer | 1 |
| vlm | stage4 | 44 | In the three-panel neighboring-frame view, do the brown predicted stick and green tracked stick move consistently without sudden jumps, wrong-object tracking, or physically implausible motion? | unclear | unclear | sequence_optimizer | 1 |
| vlm | stage4 | 242 | In the three-panel neighboring-frame view, do the brown predicted stick and green tracked stick move consistently without sudden jumps, wrong-object tracking, or physically implausible motion? | unclear | unclear | sequence_optimizer | 1 |
| vlm | stage4 | 1 | In the three-panel neighboring-frame view, do the brown predicted stick and green tracked stick move consistently without sudden jumps, wrong-object tracking, or physically implausible motion? | unclear | unclear | sequence_optimizer | 1 |
| vlm | stage4 | 122 | In the three-panel neighboring-frame view, do the brown predicted stick and green tracked stick move consistently without sudden jumps, wrong-object tracking, or physically implausible motion? | unclear | unclear | sequence_optimizer | 1 |
| vlm | stage5 | 1 | Inspect the dark rendered object over the video. Does it accurately cover the real visible object without floating away, using the wrong angle/scale, or incorrectly covering the human? Choose pass only if alignment is visually good. | pass | pass | render_acceptance_gate | 1 |
| vlm | stage5 | 122 | Inspect the dark rendered object over the video. Does it accurately cover the real visible object without floating away, using the wrong angle/scale, or incorrectly covering the human? Choose pass only if alignment is visually good. | pass | pass | render_acceptance_gate | 1 |
| vlm | stage5 | 242 | Inspect the dark rendered object over the video. Does it accurately cover the real visible object without floating away, using the wrong angle/scale, or incorrectly covering the human? Choose pass only if alignment is visually good. | pass | pass | render_acceptance_gate | 1 |
| vlm | stage6 | 1 | Compared with the solved baseline, does the current render show a regression? | unclear | unclear | audit_report | 1 |
| vlm | stage6 | 122 | Compared with the solved baseline, does the current render show a regression? | unclear | unclear | audit_report | 1 |
| vlm | stage6 | 242 | Compared with the solved baseline, does the current render show a regression? | unclear | unclear | audit_report | 1 |
| vlm | stage7 | 129 | Which label best describes the highlighted loss diagnostic plots? | unclear | unclear | audit_report | 1 |
| vlm | stage7 | 127 | Which label best describes the highlighted loss diagnostic plots? | unclear | unclear | audit_report | 1 |
| vlm | stage7 | 125 | Which label best describes the highlighted loss diagnostic plots? | unclear | unclear | audit_report | 1 |
| vlm | stage7 | 1 | Which label best describes the highlighted loss diagnostic plots? | unclear | unclear | audit_report | 1 |
| vlm | stage7 | 122 | Which label best describes the highlighted loss diagnostic plots? | normal | pass | audit_report | 1 |
| llm | stage-1 |  | llm_csv_audit | schema_missing |  | audit_report | 0 |
| llm | stage0 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage1 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage2 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage3 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage4 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage4 |  | llm_csv_audit | unclear |  | audit_report | 0 |
| llm | stage5 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage5 |  | llm_csv_audit | unclear |  | audit_report | 0 |
| llm | stage6 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage6.5 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage7 |  | llm_csv_audit | pass |  | audit_report | 0 |
