# Contact Constraint IR Branch Plan

Branch: `refactor/contact-constraint-ir`

Base: Measurement IR commit `66072ce4`

## Scope

Add typed `ContactConstraint`, `HumanSite`, local-coordinate union and read-only
adapters for five canonical contact CSVs. Keep Stage 2 outputs and Stage 3-4
consumers unchanged in this branch.

## Steps

| Step | Status | Evidence |
| --- | --- | --- |
| Create isolated worktree and hydrate inputs | done | 48 copied; 89 verified; 0 errors |
| Inventory contact schemas, states and ablation consumers | done | four coordinate schemas; dead flag remains rejected by registry |
| Add ContactConstraint types and validation | done | HumanSite, interval, state/mode, LocalXYZ/LineS/SurfaceUV and gate tests |
| Add schema-driven five-case read-only adapters | done | no case-name branch; source CSV byte-stable |
| Add deterministic shadow manifest and gate/state coverage | done | per-case hashes/counts below |
| Run full regressions and commit | done | 72 passed, 2 skipped; 20 contracts; plugins; encoded/decoded golden; 89-input dry-run |

## Inventory

| Schema cue | Cases | Object coordinate | Active rows |
| --- | --- | --- | ---: |
| base point/contact columns | basketball | feature id only | 63/192 |
| base point/contact columns | football | feature id only | 67/242 |
| `stable_local_x/y/z` | mug | `LocalXYZ` | 240/240 |
| `object_local_x/y/z` | chair | `LocalXYZ` | 250/317 |
| `object_local_s` | stick | `LineS` | 463/480 |

`no_contact_anchor` is still a rejected legacy label with no runtime consumer.
The supported mechanism flag is `disable_anchor_propagation`; the IR must report
gate provenance but must not revive the dead flag.

## Invariants

- Point, line and surface coordinates remain distinct tagged variants.
- Inactive and occluded-hold rows remain explicit states, not deleted rows.
- VLM/gate provenance may change state/confidence only; it cannot create a
  continuous coordinate.
- Adapter selection uses schema fields, not case name.
- Shadow output is never a solver input in this branch.

## Shadow evidence

| Case | Constraint states | Coordinate representation | Canonical record SHA-256 |
| --- | --- | --- | --- |
| basketball | active 63; inactive 129 | none/feature id 192 | `299bef2e66e19d5ec3ebde2ecaa9c7b2fb91eafa0d48aa02f29f3c6b0484bc90` |
| football | active 67; inactive 175 | none/feature id 242 | `a80de068d1f2e7c321cf71cd8172733422672d4e4bab8ec78ee7649e0e3b38eb` |
| mug | active 175; occluded_hold 65 | LocalXYZ 240 | `e9695ae19bfae44bb0415cbc0cb11f9d974536f26d80e10156277114e342bb1e` |
| chair | active 250; inactive 67 | LocalXYZ 250; none 67 | `8ea1843f6fd770e80b98d687a3828fc03e3a563713a59806fc809e006776fcc1` |
| stick | active 463; inactive 17 | LineS 480 | `57998f36ba12e11d35e23958288a0bcc933f6ae6366905a553f422965877a5fd` |

Inactive rows with no declared human/object feature use `mode=unknown`; they are
not mislabeled as grasp/release. Mug occlusion uses the explicit visibility
field (65 frames), not the broader `keep_previous` implementation detail (88).
