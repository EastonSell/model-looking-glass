# Development-only constant output-slope reference

This fixed 50,257-component binary64 vector is a descriptive hypothetical
output-space reference for the prospective subject-dependence study. It was
constructed during protocol drafting from eight already-exposed context
fragments in the completed public agreement-direction experiment. It is not a
physical intervention, held-out result, calibrated null, or unique mechanism.
Those fragments retained number-marked attractors and their original doses
varied; scaling their average finite-dose response to a new q assumes a
first-order approximation.

For each vocabulary index j, the builder evaluates the fixed-order formula
`math.fsum((z_plus_i[j]-z_minus_i[j])/(2.0*q_i) for i in case order)/8.0`
in binary64, using saved little-endian f32 logits decoded to Python floats.
The 16 source tensor hashes and q values are in [manifest.json](manifest.json).
An independent integer-significand decoder verified every source hash and
reproduced all output bytes. All 804,112 source elements were finite normal
f32 values, so no subnormal handling depends on ambient flush-to-zero in this
particular input.

* Output vector: [output-slope.f64le](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/distilgpt2-agreement-subject-output-reference-v1/output-slope.f64le), 402,056 bytes,
  SHA-256 `bdb31d5b74b6cff17ed2a683415beab22e36b101220257d774909be6f9f804bd`.
* Manifest: SHA-256 `69c7bdadb365c83bd4a591da405cc5e1dccd686c204a68f9b03264c604e82ed5`.
* Source journal: SHA-256 `7c8be25d480d986b087052685720ba7d2813c53d27509dcc81e67f9e2049eca6`.
* Prior science report: SHA-256 `8b816905a3009e212ab138c62cbb24f1dd06d53079e228c283bb5d8f6ae1842f`.
* Builder: `tools/build_agreement_subject_output_reference_v1.py`, SHA-256
  `9573cb814854f5fedc159e3936b97724563a7d801b47fe9d24c7f6742cb5bcb1`.

Reassemble and verify the earlier 13-part evidence archive as described in
the [earlier results](../distilgpt2-agreement-direction-study-v1/RESULTS.md).
Then run from the repository root with new output directory paths:

```text
python -B tools/build_agreement_subject_output_reference_v1.py \
  --attempt PATH_TO_EXTRACTED_RUN_ATTEMPT \
  --report results/distilgpt2-agreement-direction-study-v1/science-replay.json \
  --output NEW_OUTPUT_DIRECTORY
```

The builder verifies the pinned journal/report, ordered context IDs, doses,
and each full-logit payload hash before writing the vector and manifest. A
written result is complete only after both files match the stated hashes.
