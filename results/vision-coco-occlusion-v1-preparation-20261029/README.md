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
