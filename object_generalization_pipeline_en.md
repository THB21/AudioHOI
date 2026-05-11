# From Basketball Baseline to a General Object Pipeline

This note explains how the current basketball-specific baseline can grow into a more general pipeline that also works for irregular objects.

## 1. Why basketball is a special case

The basketball case is unusually convenient because it gives us several strong priors:

- the object is close to a sphere
- the object size is approximately fixed
- apparent radius is easy to measure
- bounce/contact events are clear
- motion is relatively simple and often follows piecewise parabolic segments

Because of these assumptions, the current baseline can use:

```text
u, v, r -> X, Y, Z
```

where:

- `u, v` are image coordinates
- `r` is the apparent ball radius
- `X, Y, Z` are lifted 3D coordinates

This works well for a ball, but it does not transfer directly to irregular objects.

---

## 2. Why irregular objects are harder

For non-spherical objects, several basketball assumptions break:

- there is no stable radius
- the object center is not necessarily the contact point
- the projected shape changes with viewpoint and rotation
- contact events are not always bounce-like
- object motion may involve sliding, placing, rotating, scraping, or closing

So the future pipeline cannot rely only on:

```text
center + radius + bounce
```

It needs a more general object representation and a more general event model.

---

## 3. The main shift: from object-specific to event-centric

The key idea is to move from:

```text
ball-specific geometry
```

to:

```text
object-agnostic state estimation + event-centric constraints
```

In practice, this means:

- do not hard-code logic for each object category whenever possible
- define reusable object states
- define reusable event types
- use audio and temporal structure as shared constraints

---

## 4. Proposed generalized pipeline

The long-term pipeline can be organized into four layers.

### Layer A: Object initialization

Find the target object to track.

Current options already available in the project:

- manual initialization
- known-category initialization with:

```text
object name -> detector (e.g. Grounding DINO) -> SAM2
```

This part is already close to a reusable pipeline component.

### Layer B: Object state representation

For irregular objects, a single center point is not enough.

A more general object state per frame should include:

- `mask_t`
- `bbox_t` or oriented bounding box
- sparse points / extremal points
- support/contact candidate points
- pose proxy

A practical representation could include:

1. **mask**
   - object silhouette

2. **bbox or oriented bbox**
   - coarse extent and orientation

3. **canonical sparse points**
   - left / right / top / bottom
   - lowest visible point
   - farthest support-facing point
   - edge / corner / handle candidates when relevant

4. **pose proxy**
   - principal axis direction
   - aspect ratio
   - orientation change
   - coarse rotation signal

This becomes the general replacement for the basketball-specific `(u, v, r)` state.

### Layer C: Event abstraction

Instead of writing separate logic for:

```text
basketball
mug
chair
drawer
hammer
```

the pipeline should reason in terms of event type.

Useful event classes include:

1. **impact**
   - basketball bounce
   - hammer hit
   - box impact

2. **placement**
   - mug placed on table
   - chair placed on floor

3. **closure**
   - drawer close
   - door close

4. **slide / scrape**
   - broom motion
   - dragging contact

5. **sustained contact**
   - resting support
   - brushing
   - leaning

This is important because the reconstruction logic should depend more on:

```text
what kind of interaction is happening
```

than on:

```text
what the object category is called
```

### Layer D: Reconstruction / fitting

For general objects, 3D recovery should be driven by multiple constraints rather than a single radius-to-depth rule.

Candidate constraints include:

- mask reprojection consistency
- sparse point reprojection consistency
- support-plane constraints
- contact constraints
- temporal smoothness
- category size / shape priors
- audio timing constraints

So the generalized version becomes:

```text
2D object state + event constraints + temporal fitting -> 3D object trajectory / pose
```

---

## 5. What replaces “radius” for irregular objects

For the basketball baseline, apparent radius is the main depth cue.

For irregular objects, several alternatives are possible.

### Option A: Category size priors

If the object category is known, we can assign rough size priors:

- mug height range
- chair seat height range
- hammer handle/head length range
- broom length range

Then depth can be estimated from:

```text
observed 2D extent + category size prior
```

instead of a single radius.

### Option B: Simplified shape templates

Instead of full mesh reconstruction, use a simple shape model per object family:

- mug -> cylinder-like proxy
- chair -> seat/back/legs proxy
- hammer -> handle + head proxy
- drawer -> cuboid front proxy
- broom -> elongated shaft + brush head proxy

Then fit the projection of that simplified shape over time.

### Option C: Learned or foundation-model 3D priors

This is a stronger option, but probably not the best next step for the current project stage.

Examples:

- category-conditioned 3D priors
- single-image 3D object estimators
- object pose models

These can be added later if needed.

---

## 6. Contact geometry matters more than object center

For basketball, tracking the center is often enough.

For irregular objects, the more important entities are often:

- mug bottom rim
- chair leg tip
- hammer head tip
- drawer front edge
- broom head

So the generalized pipeline should explicitly estimate:

- object center
- contact candidate points
- support-facing boundary
- principal axis

This is much more useful than assuming the center is also the interaction location.

---

## 7. How audio helps generalization

Audio becomes even more valuable once the geometry is less regular.

It should not only be treated as a final alignment check, but as a structural cue.

### 7.1 Event proposal

Audio can indicate:

```text
an interaction event happens around this time
```

### 7.2 Event type hint

Different sounds often imply different interaction types:

- sharp transient -> impact
- dull thud -> placement
- short closing sound -> closure
- continuous noisy band -> scrape or brushing

### 7.3 Temporal anchor

Even if the object geometry is messy, audio can still narrow the search window in time.

So the generalized pipeline can use:

```text
audio -> event window / event type prior
video -> object state / contact candidates
joint reasoning -> 3D interaction estimate
```

---

## 8. A realistic path from the current baseline to a general pipeline

The most practical way forward is incremental.

### Phase 1: Move from ball-specific to event-specific logic

Keep the known-category setup, but generalize the reconstruction logic by event type:

- impact pipeline
- placement pipeline
- closure pipeline

This is still manageable and suitable for a baseline paper or prototype.

### Phase 2: Standardize the object state

Make all objects output a common state representation:

- mask
- bbox / oriented bbox
- principal axis
- contact candidate points
- support-relative motion

This is the point where the pipeline becomes structurally reusable.

### Phase 3: Joint temporal reconstruction

Later, combine all signals into a single fitting objective, for example:

```text
E = E_mask_reproj
  + E_keypoint_reproj
  + E_contact
  + E_support
  + E_audio
  + E_temporal
  + E_shape_prior
```

That is much closer to a general multi-object, multi-event reconstruction framework.

---

## 9. Recommended near-term categories

A good next step is to choose object categories that span event types:

- **basketball** -> impact
- **hammer** -> impact
- **mug** -> placement
- **drawer** -> closure

These are especially useful because they let the method generalize across interaction types before it has to solve the hardest geometry cases.

More difficult categories can come later:

- **chair**
- **broom**

These require stronger modeling of contact regions and sustained-contact behavior.

---

## 10. Final summary

To make the current basketball work grow into a real pipeline for irregular objects, the core transition is:

from:

```text
center + radius + bounce
```

to:

```text
object state + contact geometry + event type + temporal constraints
```

That is the real generalization path.

The basketball case is still useful because it lets the project prototype:

- object tracking
- event timing
- audio-visual alignment
- 3D lifting structure

in a setting where the geometry is simple enough to get the pipeline skeleton right first.
