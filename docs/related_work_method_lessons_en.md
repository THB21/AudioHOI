# Related-Work Method Lessons for AudioHOI

This note complements the `generic_pipeline_v2_llm_vlm_gate` mainline design. The mainline document explains our system; this note explains what several related HOI/HSI papers do, what their central methodological ideas are, and how AudioHOI borrows those ideas.

## 1. Summary: The Three-Layer Abstraction We Borrow

AudioHOI v2 does not try to reproduce one specific paper. Instead, it abstracts several HOI/HSI methods into a three-layer design:

```text
LLM semantic prior
  -> object parts / human parts / interaction edges / affordance graph

VLM visual gate
  -> forced-choice verification for masks, keyparts, contacts, overlays, anchor updates, final renders

optimizer
  -> continuous 2D-to-6D, depth, contact, temporal, static/freeze, small SE(3) solving
```

The main lessons are:

- InterCap and MOVER show that contact, occlusion, support surfaces, and penetration/floor constraints should be treated as geometry, not as comments.
- HOI-PAGE and InteractAnything show that LLMs are useful for part-level affordance and interaction graphs, but should not output continuous pose.
- GenZI and ZeroHSI show that VLM/video priors are useful for candidate generation and visual reasoning, but they still need geometric optimization.
- InterDiff and CoopDiff show that dynamic HOI must be contact-consistent over time, not fitted frame by frame.
- Gaussian-HOI/Open3DHOI/WildHOI-style work shows that explicit contact regions and structured 3D HOI outputs are important, even when the representation differs.

## 2. Method Comparison

| Method | Core idea | How it works | What AudioHOI borrows |
|---|---|---|---|
| InterCap | Humans and objects must be estimated jointly; contact improves both body pose and object pose | Multi-view RGB-D + SMPL-X + known object mesh, jointly optimized with contact, ground, and object pose constraints | Treat contact as a real geometric residual: mug palm-handle, chair two-hand endpoint, ball hand/foot/floor contact |
| MOVER | Human motion constrains object placement and scene geometry | Optimizes camera, ground, object scale and placement with occlusion, depth ordering, free-space, and contact surface constraints | `E_depth_order`, `E_penetration_or_floor_violation`, `E_contact`, floor/table static priors |
| InterDiff / CoopDiff | Dynamic HOI needs human/object motion consistency and contact consistency | Uses diffusion models with physics/contact-aware mechanisms for human-object motion | We do not use diffusion as the solver, but keep contact intervals, temporal smoothness, rotation-jump penalties, static/freeze |
| HOI-PAGE | LLMs can reason about part-level affordance graphs | Generates part affordance graphs from text prompt and object parts to guide 4D HOI generation | Stage -1 writes `hoi_profile.json`: object parts, human parts, interaction edges, and VLM query policy |
| InteractAnything | Open-set object interaction needs relationship reasoning and affordance parsing | Uses LLM feedback to generate/refine interaction poses and object affordance | Mistral/Qwen are used only for discrete semantic priors and checks, not coordinates or loss weights |
| GenZI | VLMs can imagine plausible humans interacting with a scene | Uses VLM inpainting over scene views, then optimizes a 3D human-scene interaction | VLM gates mask/keypart/contact/render evidence; the optimizer handles continuous 3D/6D solving |
| ZeroHSI | Video generation provides a strong zero-shot motion prior | Uses video generation and differentiable rendering to reconstruct 4D human-scene interaction | Our input videos are generated, so we use SAM2/CoTracker/VLM/audio as evidence and rely on optimization for alignment |
| Open3DHOI / WildHOI | In-the-wild RGB can be converted into structured 3D HOI annotations/reconstruction | Combines 2D images, human estimation, object reconstruction/6D pose, and semantic labels | Unified CSV outputs: `object_pose.csv`, `object_contact_points.csv`, and part-level local points |
| Gaussian-HOI / HOIGS | HOI can be represented with explicit human/object dynamic fields and contact regions | Uses Gaussian representations and HOI-aware/contact-aware optimization for reconstruction | We do not do photorealistic neural rendering, but we explicitly output contact regions/points |

## 3. InterCap: Contact Is Geometry, Not Annotation

### Core idea

InterCap’s key observation is that humans and objects in interaction cannot be estimated independently. Hand-cup contact, foot-floor support, mouth-rim contact, and object-ground contact are strong 3D constraints.

### How it works

InterCap uses multi-view RGB-D, SMPL-X whole-body modeling, and known object meshes. It jointly optimizes human pose, object pose, contact, and scene geometry to produce stable human-object reconstruction.

### How AudioHOI borrows it

AudioHOI does not have multi-view RGB-D, so we cannot directly reproduce InterCap. We keep the core idea:

```text
contact is an optimization constraint
```

Examples:

- Mug: `palm_handle_rim_body` and `stable_grasp_anchor`; when the handle is hidden, we do not follow an incorrect visible point, but use a stable grasp prior.
- Chair: `two_hand_toprail_endpoint` and `small_se3`; left/right palms constrain the two top-rail endpoints.
- Basketball / football: hand/foot/floor contact becomes a depth anchor, not just a 2D center cue.

## 4. MOVER: Human Motion Constrains Depth, Occlusion, and Support

### Core idea

MOVER focuses on monocular scene reconstruction. Human motion and object layout constrain each other: the person creates depth ordering, free space, support surfaces, and contact evidence.

### How it works

MOVER optimizes camera, ground plane, object scale, and object position using:

- occlusion / depth ordering
- human free-space constraints
- human-object contact surface consistency
- scene/object placement priors

### How AudioHOI borrows it

AudioHOI turns these ideas into explicit energy terms:

```text
E_total =
  w_2d      * E_2d_projection
+ w_depth   * E_depth_order_or_metric
+ w_contact * E_contact
+ w_smooth  * E_temporal_smooth
+ w_static  * E_static_freeze
+ w_pen     * E_penetration_or_floor_violation
+ w_prior   * E_pose_prior
```

Case examples:

- Ball: floor support, hand/foot contact depth anchors, contact-window smoothing.
- Mug: static/freeze after table release; handle-phase continuity during drinking.
- Chair: lift/place is inferred from audio + floor support + GVHMR proximity; trusted middle frames propagate to both sides.

## 5. HOI-PAGE / InteractAnything: LLMs Provide Semantic Priors, Not Continuous Pose

### Core idea

HOI-PAGE uses LLMs to infer an affordance graph between object parts and human parts. InteractAnything similarly emphasizes relationship reasoning, affordance parsing, and detailed action semantics for open-set objects.

### How they work

These methods typically start from a text prompt and object mesh, then use an LLM to infer:

- which object parts are interactable
- which human parts are likely to contact them
- what contact relations should exist
- what the action phases or goals are

A motion/pose/rendering model then generates or optimizes the actual 3D interaction.

### How AudioHOI borrows it

AudioHOI Stage -1 uses the same semantic role, but conservatively:

```json
{
  "object_parts": ["top_rail", "seat", "front_leg", "rear_leg"],
  "human_parts": ["left_palm", "right_palm"],
  "interaction_edges": [
    {"human_part": "left_palm", "object_part": "right_top_rail_endpoint", "relation": "hold"}
  ]
}
```

The LLM may:

- produce object/human part lists
- produce interaction edges
- produce support and motion priors
- decide which VLM query types should be used at each stage

The LLM must not:

- output 2D/3D coordinates
- output SE(3) corrections
- output continuous loss weights
- directly overwrite optimizer pose

This is the role of `hoi_profile.json`.

## 6. GenZI / ZeroHSI: VLM and Video Priors Help Verification, But Do Not Replace Optimization

### Core idea

GenZI uses VLMs to imagine plausible human-scene interactions over scene views, then optimizes a 3D human-scene interaction. ZeroHSI uses video generation as a zero-shot motion prior, then reconstructs 4D interaction with differentiable rendering.

### How they work

These methods show two things:

1. Large visual/video models contain rich human-interaction priors.
2. Generated or imagined evidence still needs geometry, rendering consistency, or optimization constraints.

### How AudioHOI borrows it

Our input videos are generated videos, so we must not blindly trust the generator. AudioHOI uses VLM as a conservative verification layer:

```text
Stage0 target_mask_check
Stage1 keypart_identity_check / track_stability_check
Stage2 contact_relation_check
Stage3 overlay_alignment_check
Stage4 anchor_update_check
Stage5 post_render_sanity_check
```

Every VLM query is forced-choice, for example:

```text
Which chair part is highlighted?
labels: top_rail / front_leg / rear_leg / seat / stretcher / hole / background / unclear
```

The VLM only outputs:

```text
pass / reject / unclear / failure label
```

It does not modify pose directly. If VLM rejects a contact candidate, that frame does not activate `E_contact`; if the VLM says unclear, the anchor is not updated.

## 7. InterDiff / CoopDiff: Dynamic Consistency Matters More Than Single-Frame Fits

### Core idea

InterDiff and CoopDiff focus on dynamic HOI. Their key concern is not just a good single frame, but consistency between human motion, object motion, and contacts over time.

### How they work

InterDiff uses physics-informed diffusion to generate human-object motion. CoopDiff decouples human dynamics and object dynamics and links them through contact consistency.

### How AudioHOI borrows it

AudioHOI does not use diffusion to generate trajectories. Instead, dynamic consistency is encoded as refinement rules and residuals:

- `E_temporal_smooth`: prevents frame-to-frame jitter.
- `E_static_freeze`: freezes objects after placement.
- `rotation_jump_count`: detects mug handle or chair orientation jumps.
- `anchor_propagate_freeze`: propagates trusted contact intervals to the left and right.

This explains why chair fitting cannot be pure per-frame 2D overlay: if middle contact frames are more reliable in 3D, the system should allow small SE(3) refinement and propagate it temporally.

## 8. Open3DHOI / Gaussian-HOI: Explicit Contact Regions and Structured 3D HOI Outputs

### Core idea

Open-vocabulary 3D HOI and Gaussian-HOI-style methods emphasize that HOI is not only a human trajectory or an object trajectory; it is the joint structure of human, object, contact region, and semantic relation.

### How they work

These methods often combine:

- human pose / body mesh
- object reconstruction / object 6D pose
- semantic object category or open-vocabulary labels
- contact or interaction regions
- rendering or reconstruction consistency

Gaussian-HOI/HOIGS-style methods further represent humans and objects with Gaussian/dynamic fields and optimize spatial relations using contact-aware or HOI-aware losses.

### How AudioHOI borrows it

AudioHOI currently does not perform photorealistic Gaussian reconstruction, but it keeps the structured outputs:

```text
object_pose.csv
object_contact_points.csv
object_phase.csv
object_local_points.csv
object_local_segments.csv
vlm_gates.csv
stage7_loss_residuals.csv
```

In particular, `object_contact_points.csv` is not just debug visualization. It is a core artifact for later precise human PKL fitting, hand reconstruction, and teacher review.

## 9. Why We Use Method Proxies Instead of Directly Running All Papers

The assumptions of these papers differ from AudioHOI:

- InterCap requires calibrated multi-view RGB-D.
- MOVER mainly optimizes static scene layout and does not directly output our object-contact CSV.
- InterDiff / CoopDiff are motion-generation methods, not generated-video-to-object-6D/contact recovery methods.
- HOI-PAGE / InteractAnything generate HOI and affordance, but do not solve our video alignment problem.
- GenZI / ZeroHSI focus on generation/reconstruction and do not directly provide per-frame object 6D/contact recovery.
- Gaussian-HOI-style methods target neural reconstruction and use different outputs from our CSV/diagnostic-video format.

Therefore, v2 uses method proxy / ablation comparison:

```text
video_only_tracking
mesh_only_alignment
human_only_contact
no_audio_event
no_vlm_gate
no_llm_prior
no_contact_refine
ours_full
```

The goal is not to claim that “other code fails.” The goal is to identify what breaks when an information source is removed:

- Without contact: depth and release/static become unstable.
- Without object semantic parts: mug handle and chair top rail/legs fail.
- Without audio: lift/place/static timing is less stable.
- Without VLM gate: wrong anchors/contacts enter the optimizer.
- Without LLM prior: VLM questions and contact relations become too coarse for zero-shot objects.

## 10. Direct Impact on the Mainline Design

These papers motivate the following AudioHOI v2 decisions:

1. The LLM only generates a discrete HOI profile.
2. The VLM performs forced-choice gates at every stage.
3. The optimizer is the only continuous solver.
4. Contact candidates must be verified by visual/geometric gates before activating residuals.
5. Object-specific logic is implemented as reusable components, not one runner per object.
6. The output must include pose, contact points, phase, loss residuals, and six render videos for fair proxy comparison.

## References

- InterCap: Joint Markerless 3D Tracking of Humans and Objects in Interaction. Project page: https://intercap.is.tue.mpg.de/
- MOVER: Human-Aware Object Placement for Visual Environment Reconstruction. Project page: https://vlg.inf.ethz.ch/publications/Mover-Human-Aware-Object-Placement-for.html
- InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion. Project page: https://sirui-xu.github.io/InterDiff/
- CoopDiff: Contact-consistent decoupled Diffusion for anticipating 3D HOI. arXiv page: https://arxiv.org/html/2508.07162v1
- HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Graphs. Project page: https://craigleili.github.io/projects/hoipage/
- InteractAnything: Zero-shot Human Object Interaction Synthesis via LLM Feedback and Object Affordance Parsing. Project page: https://jinluzhang.site/projects/interactanything/
- GenZI: Zero-Shot 3D Human-Scene Interaction Generation. CVPR 2024 paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_GenZI_Zero-Shot_3D_Human-Scene_Interaction_Generation_CVPR_2024_paper.pdf
- ZeroHSI: Zero-Shot 4D Human-Scene Interaction by Video Generation. Project page: https://awfuact.github.io/zerohsi/
- Open-vocabulary / Gaussian-HOI style 3D HOI annotation and reconstruction. Project page: https://wenboran2002.github.io/3dhoi/
