# Synthetic ViT known-injection localization on CPU

On four fixed synthetic held-out inputs, the supervised directional selector
identified the deliberately injected patch coordinate in **4/4** cases. The
largest-change selector also identified it in **4/4**; the one fixed random
sequence identified **1/4**. At the selected downstream site, clean-state
replacement reduced the full-logit distance from the clean run more than both
eligible matched controls in **4/4** cases. These are small, predeclared
mechanics results, with no statistical-significance or natural-vision claim.

The study used the cached `WinKawaks/vit-tiny-patch16-224` checkpoint, revision
`77d1485af66b34d4ed0fe95dbb0c60c7496f950b`, on CPU with float32 and eager
attention. Four procedural tensors were used for supervised discovery and four
different procedural tensors for evaluation. An explicit +4.0 addition to
coordinate 7 of one patch token at `vit.embeddings` was followed by captures
of four block-0 patch states and CLS. The held-out selector received the
fitted discovery direction and candidate deltas, but not the injected token
label. The experiment and all labels were public, so this is procedural
label-blinding rather than a fully blinded or population-sampled study.

| Held-out seed | Injected / selected token | Magnitude / random token | Injected distance | Selected patch | Reflected control | Off-target control |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | 103 / 103 | 103 / 166 | 0.17184 | 0.01998 | 1.68068 | 0.40963 |
| 22 | 166 / 166 | 166 / 166 | 0.21047 | 0.02922 | 0.97358 | 0.29498 |
| 23 | 16 / 16 | 16 / 52 | 0.26373 | 0.02313 | 1.55725 | 0.34093 |
| 24 | 52 / 52 | 52 / 166 | 0.13153 | 0.02285 | 0.81782 | 0.46636 |

Distances are L2 over the complete classifier-logit vector from the clean
output. All four target doses and both controls' realized doses were nonzero
and within the predeclared 5% matching tolerance. The reflected control moves
opposite the clean-donor direction at the selected site; the off-target control
scales its own clean-donor direction to the selected dose, and can extrapolate
beyond the clean state. Neither is a null or a functionally matched control.
The off-target token is chosen with knowledge of the injected label solely for
this evaluation. The selected patch is a later block output, not the actual
embedding injection boundary. In particular, the large residual contribution
at block 0 assists coordinate recovery, and attention can mix information
across tokens. These findings do not establish a unique causal route, semantic
pixel correspondence, generalization to natural images, or pretraining novelty.

The [fixed protocol](../../docs/VIT_SYNTHETIC_INJECTION_V1.md) and source were
published at commit `42a7ea04411f07210aaa9efc2e47956b16f4590c` before the
operator launched the attempt. The fixed bootstrap checks source bytes before
loading the model. The separately retained [source readback](public-source-readback.json)
was recorded **during** the run; it is not an independent pre-forward timestamp.
The [direct-child observation](direct-child-observation.json) reports exit 0,
reaping and no incomplete marker. Those observations and the saved data do not
independently authenticate every instruction or the claimed execution timing.

An independent standard-library [saved-byte reader](audit_vit_synthetic_saved.py)
reconstructed all 196 retained tensor slices, the discovery direction,
selections, complete same-forward landing and untouched-state checks, self and
clean-repeat controls, realized doses and all reported distances. Its
[report](saved-audit.json) is a consistency check on saved bytes, not a model
replay or proof of a unique mechanism. The attempt contains 35 files totaling
15,630,263 bytes; the original terminal SHA-256 is
`07b2bddb044457d69a156d3bb93131d42bc2a995462a1c6f0a09fe42eedd541e`.

The original attempt is preserved in 15 indexed archive parts. Run
`python verify-archive.py` to verify them without extraction, or
`python verify-archive.py --extract <new-directory>` to reconstruct the exact
`attempt/` files. After extraction, run
`python audit_vit_synthetic_saved.py <new-directory>/attempt --source-observation public-source-readback.json`.
The part and member indexes contain lengths and hashes; the extractor refuses
unsafe member names and oversized declared data. The `torch_dtype` deprecation
notice in the saved stderr did not indicate a failed load or forward.
