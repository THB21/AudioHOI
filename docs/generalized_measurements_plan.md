# Generalized Measurements Branch Plan

Branch: `refactor/generalized-measurements`

Base: chair solved-seed removal commit `29bae8ae`

## Scope

Introduce a typed, read-only Measurement IR and adapters for the five canonical
cases. This branch must not alter Stage 1 CSVs, solver inputs, loss values,
thresholds or selected outputs.

## Steps

| Step | Status | Evidence |
| --- | --- | --- |
| Create isolated worktree and hydrate inputs | done | 89 verified, 0 pending, 0 errors |
| Inventory five-case observation schemas and coordinate semantics | done | three legacy schemas; mapping and explicit unmapped coverage below |
| Add Measurement IR types and validation | done | tagged point/line/mask/depth/track/visibility types with unit/frame checks |
| Add read-only legacy CSV adapters | done | five canonical files adapt without byte changes or zero fill |
| Add shadow manifest/export and provenance hashes | done | opt-in CLI; source and canonical-record SHA-256; never solver-consumed |
| Run contracts, plugins, pytest and five-case golden | done | 63 passed, 2 skipped; 20 contracts; plugins; encoded/decoded golden; 89-input dry-run |

## Invariants

- Unknown values remain absent; adapters never replace them with numeric zero.
- Every measurement declares coordinate frame, unit, feature reference,
  confidence and source provenance.
- `FeatureRef` contains semantic role and geometry feature id, never case name.
- Adapters may branch on legacy schema shape, but IR consumers may not branch on
  case name.
- Shadow export is opt-in and cannot be consumed by Stage 2-4 in this branch.

## Schema inventory

| Legacy schema | Canonical cases | Primary mapped measurements |
| --- | --- | --- |
| `proxy_center_depth_v1` | basketball, football, stick | raw/smoothed center, support point, metric center depth |
| `rigid_body_parts_v1` | mug | object/body masks, body center, lowest-visible point, optional handle point, handle visibility |
| `semantic_graph_v1` | chair | mask bbox, top/seat edges and four leg lines |

Adapters select these schemas by required column sets, not by sample or case
name. The shadow manifest separately reports every non-empty legacy field not
yet represented in the IR. Derived phase, jitter, parallelism diagnostics,
source labels and redundant bbox summaries remain explicit in that list; they
are not silently discarded or converted into fake measurements.

## Findings

- The ball and stick Stage-1 CSVs share one schema even though their downstream
  geometry differs. This supports separating measurement type from StateSpec.
- Mug missing handle centers are genuinely absent in 81/240 frames; the adapter
  emits 159 points and 240 visibility states rather than filling `(0, 0)`.
- Chair's primary semantics are line measurements. Treating every endpoint as
  an unrelated point would lose segment identity and is therefore prohibited.
- Confidence/source/diagnostic columns are not interchangeable. This branch
  maps confidence used by a measurement and reports remaining provenance fields
  as coverage work instead of guessing a common meaning.

## Shadow counts

| Case | Measurements | Kind counts | Canonical record SHA-256 |
| --- | ---: | --- | --- |
| basketball | 768 | point2d 576; depth 192 | `470a171b0c7c9a73835058925d681a5c1ee4dc8cf820e88c4a4d7141ab36767d` |
| football | 968 | point2d 726; depth 242 | `584bb002c0c004b9738a43fc1e4a413a04045fb86256aaf53da8db193315788d` |
| mug | 1359 | point2d 639; mask 480; visibility 240 | `5f2e1e2ee6b1d22bce5d4f18d23cafced9dead4646ae23c414882f736f2788b2` |
| chair | 1014 | line2d 822; mask 192 | `8a6415fb7042660493afc7a8aa7bfa5b1dee36e7050ec2845ee527fef0a32816` |
| stick | 960 | point2d 720; depth 240 | `33d49d798b793ef52c2dcd78a8a5968ead441d0cb87a2c8261b5bca7269650f5` |
