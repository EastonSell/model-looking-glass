# Occlusion v1 annotation-only preparation

The fixed [protocol](../../docs/VISION_COCO_OCCLUSION_V1.md) and selector were
public at commit `ce111d175cb89bfa44f07d89011b079b6dd368ff` before this
annotation-only preparation. The local execution checkout was at
`403c3545b7f490843f9936895b45e4e2f5330f03`; the five new source blobs and
relevant inherited selector/manifest blobs were independently matched to the
public source. This is a relevant-file comparison, not a claim that the two
entire checkout trees are identical.

The [fixed manifest](../../experiments/vision_coco_occlusion_v1.json) has
SHA-256 `aa61b294710f5a6cd9682dc783085c7079f1addfbad8257fdbd893d143ce4cef`.
Of 32 candidates per species, 17 cats and 15 dogs have a qualifying equal-
radius object/background geometry; all 64 remain in the clean-screen inventory.
The [receipt](receipt.json) and [logs](logs.zip) record a clean exit and zero
model forwards under cooperative observation. The observer is retained as the
original local-layout source, not as a portable execution attestation.

An [independent input-audit bundle](independent-input-audits.zip) contains
two standalone no-model readers and their original reports. The full-ranking
reader reconstructs the first 32 candidates of each species; the selected-row
reader independently checks the recorded polygons, box-disjoint geometry,
JPEG and processed-pixel hashes. They also verify all 235 prior exclusions.
The bundle preserves report SHA-256 values
`2da176954a9a8fd349f555a12b24708c3b77a1c245db070cd0e061865d7b4ba5`
and `3094c46381896cca7de94ea6b886123534df1ee463aa54559ab87e2c8bc8d311`.
These checks do not establish origin, execution chronology or pretraining
independence.
