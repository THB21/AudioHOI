# Pipeline VLM/LLM QA Summary

This report aggregates pipeline-stage VLM/LLM questions, answers, gates, and affected constraints.

| source | stage | frame | question | parsed | gate | affected constraint | changed optimizer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vlm | stage0 | 1 | Does the highlighted mask correspond to the target folding_chair? | yes | pass | visual_overlay_residual | 1 |
| vlm | stage1 | 1 | Which folding_chair part is highlighted? | seat | pass | audit_report | 1 |
| vlm | stage1 | 97 | Which folding_chair part is highlighted? | front_leg | pass | audit_report | 1 |
| vlm | stage1 | 192 | Which folding_chair part is highlighted? | left_top_rail_endpoint | pass | audit_report | 1 |
| vlm | stage1 | 1 | Is the highlighted reference point or part stable and visually trackable across neighboring frames? | stable | pass | visual_overlay_residual | 1 |
| vlm | stage1 | 97 | Is the highlighted reference point or part stable and visually trackable across neighboring frames? | stable | pass | visual_overlay_residual | 1 |
| vlm | stage1 | 192 | Is the highlighted reference point or part stable and visually trackable across neighboring frames? | stable | pass | visual_overlay_residual | 1 |
| vlm | stage2 | 20 | What is the highlighted object contact region closest to? | right_palm_grasp_seat_edge | pass | contact_anchor_residual | 1 |
| vlm | stage2 | 82 | What is the highlighted object contact region closest to? | right_palm_grasp_seat_edge | pass | contact_anchor_residual | 1 |
| vlm | stage2 | 144 | What is the highlighted object contact region closest to? | left_palm_on_top_rail | pass | contact_anchor_residual | 1 |
| vlm | stage2 | 1 | What is the highlighted object contact region closest to? | right_palm_on_top_rail | pass | contact_anchor_residual | 1 |
| vlm | stage2 | 89 | What is the highlighted object contact region closest to? | right_palm_on_top_rail | pass | contact_anchor_residual | 1 |
| vlm | stage3 | 1 | Does the rendered object overlay sit on top of the same visible physical object in the video? Choose aligned only if the rendered object follows the real visible object; choose lateral_shift, wrong_scale, or rotation_error if it is offset, too long/short, or at a different angle. | aligned | pass | visual_overlay_residual | 1 |
| vlm | stage3 | 97 | Does the rendered object overlay sit on top of the same visible physical object in the video? Choose aligned only if the rendered object follows the real visible object; choose lateral_shift, wrong_scale, or rotation_error if it is offset, too long/short, or at a different angle. | aligned | pass | visual_overlay_residual | 1 |
| vlm | stage3 | 192 | Does the rendered object overlay sit on top of the same visible physical object in the video? Choose aligned only if the rendered object follows the real visible object; choose lateral_shift, wrong_scale, or rotation_error if it is offset, too long/short, or at a different angle. | aligned | pass | visual_overlay_residual | 1 |
| vlm | stage4 | 20 | Using the visual guide in the image: brown render is the predicted stick, green line is the tracked visible stick, yellow circles are active palm contacts, and gray circles are inactive palms. Does the predicted stick preserve alignment with the green visible stick and the active yellow palm contacts without creating a wrong contact? | no_change | pass | audit_report | 1 |
| vlm | stage4 | 82 | Using the visual guide in the image: brown render is the predicted stick, green line is the tracked visible stick, yellow circles are active palm contacts, and gray circles are inactive palms. Does the predicted stick preserve alignment with the green visible stick and the active yellow palm contacts without creating a wrong contact? | no_change | pass | audit_report | 1 |
| vlm | stage4 | 144 | Using the visual guide in the image: brown render is the predicted stick, green line is the tracked visible stick, yellow circles are active palm contacts, and gray circles are inactive palms. Does the predicted stick preserve alignment with the green visible stick and the active yellow palm contacts without creating a wrong contact? | unclear | unclear | audit_report | 1 |
| vlm | stage4 | 1 | Using the visual guide in the image: brown render is the predicted stick, green line is the tracked visible stick, yellow circles are active palm contacts, and gray circles are inactive palms. Does the predicted stick preserve alignment with the green visible stick and the active yellow palm contacts without creating a wrong contact? | no_change | pass | audit_report | 1 |
| vlm | stage4 | 89 | Using the visual guide in the image: brown render is the predicted stick, green line is the tracked visible stick, yellow circles are active palm contacts, and gray circles are inactive palms. Does the predicted stick preserve alignment with the green visible stick and the active yellow palm contacts without creating a wrong contact? | no_change | pass | audit_report | 1 |
| vlm | stage4 | 20 | In the three-panel neighboring-frame view, do the brown predicted stick and green tracked stick move consistently without sudden jumps, wrong-object tracking, or physically implausible motion? | unclear | unclear | sequence_optimizer | 1 |
| vlm | stage4 | 82 | In the three-panel neighboring-frame view, do the brown predicted stick and green tracked stick move consistently without sudden jumps, wrong-object tracking, or physically implausible motion? | unclear | unclear | sequence_optimizer | 1 |
| vlm | stage4 | 144 | In the three-panel neighboring-frame view, do the brown predicted stick and green tracked stick move consistently without sudden jumps, wrong-object tracking, or physically implausible motion? | unclear | unclear | sequence_optimizer | 1 |
| vlm | stage4 | 1 | In the three-panel neighboring-frame view, do the brown predicted stick and green tracked stick move consistently without sudden jumps, wrong-object tracking, or physically implausible motion? | unclear | unclear | sequence_optimizer | 1 |
| vlm | stage4 | 89 | In the three-panel neighboring-frame view, do the brown predicted stick and green tracked stick move consistently without sudden jumps, wrong-object tracking, or physically implausible motion? | unclear | unclear | sequence_optimizer | 1 |
| vlm | stage5 | 1 | Inspect the dark rendered object over the video. Does it accurately cover the real visible object without floating away, using the wrong angle/scale, or incorrectly covering the human? Choose pass only if alignment is visually good. | pass | pass | render_acceptance_gate | 1 |
| vlm | stage5 | 97 | Inspect the dark rendered object over the video. Does it accurately cover the real visible object without floating away, using the wrong angle/scale, or incorrectly covering the human? Choose pass only if alignment is visually good. | pass | pass | render_acceptance_gate | 1 |
| vlm | stage5 | 192 | Inspect the dark rendered object over the video. Does it accurately cover the real visible object without floating away, using the wrong angle/scale, or incorrectly covering the human? Choose pass only if alignment is visually good. | pass | pass | render_acceptance_gate | 1 |
| vlm | stage6 | 1 | Compared with the solved baseline, does the current render show a regression? | unclear | unclear | audit_report | 1 |
| vlm | stage6 | 97 | Compared with the solved baseline, does the current render show a regression? | unclear | unclear | audit_report | 1 |
| vlm | stage6 | 192 | Compared with the solved baseline, does the current render show a regression? | no_regression | pass | audit_report | 1 |
| vlm | stage7 | 45 | Which label best describes the highlighted loss diagnostic plots? | visual_spike | reject | audit_report | 1 |
| vlm | stage7 | 121 | Which label best describes the highlighted loss diagnostic plots? | visual_spike | reject | audit_report | 1 |
| vlm | stage7 | 46 | Which label best describes the highlighted loss diagnostic plots? | visual_spike | reject | audit_report | 1 |
| vlm | stage7 | 1 | Which label best describes the highlighted loss diagnostic plots? | visual_spike | reject | audit_report | 1 |
| vlm | stage7 | 97 | Which label best describes the highlighted loss diagnostic plots? | visual_spike | reject | audit_report | 1 |
| llm | stage-1 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage0 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage1 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage2 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage3 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage4 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage4 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage5 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage5 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage6 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage6.5 |  | llm_csv_audit | pass |  | audit_report | 0 |
| llm | stage7 |  | llm_csv_audit | pass |  | audit_report | 0 |
