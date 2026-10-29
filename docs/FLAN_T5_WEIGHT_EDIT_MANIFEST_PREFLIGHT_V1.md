# Manifest-only preflight correction

The prospective executor source at `a78d42417c797b880cd7f9c74ccdc78c2b1421c3`
was followed by manifest `05ef8082edc25b5d07711e8289858a32be524638`
(SHA-256 `e25cda825aa728048e93c7b3af8a1091ae2849e84eaa280ca49045eb3875ee38`).
An operator's exact-source no-model rehearsal failed at manifest validation:
the JSON serializer represented two fixed grid values as integers, whereas
the source's typed contract requires the corresponding floating-point values.
The preflight stopped before creating a child or fixed phase claim; no model
was loaded and no study case was forwarded. The small rehearsal inputs are
retained at `E:\Codex\model-looking-glass\work\flan-t5-weight-edit-v1\source-rehearsal-05ef8082`;
the failure output is retained in the operator's tool transcript.

This commit preserves the exact executor bytes and records that failed
preflight. Its manifest-only successor uses Python's JSON serializer to retain
the fixed float types. The model, inputs, grid values, thresholds and other
scientific rules are unchanged. The successor and a fresh no-model rehearsal
require independent readback before any study forward.

Correction: the serializer changed four entries' JSON types: `1.0`, `2.0`,
`3.0` and `4.0` became integer `1`, `2`, `3` and `4`. The earlier reference
to two entries was inaccurate. The numerical grid itself did not change.
