# Agreement-direction training from public discovery captures

These vectors were reconstructed from the 16 **already exposed** clean
DistilGPT-2 discovery activations, after the prospective study protocol was
published in [PR #404](../../docs/AGREEMENT_DIRECTION_STUDY_V1.md). They do not
use or reveal the sealed natural-agreement split. No new evaluation inputs or
model forwards were generated in this construction.

The positive u direction points from plural toward singular in the mean of
eight within-lemma pairs at the selected layer-3 attention output. v1 contrasts
the singular–plural differences between the two lemmas in each public family;
v2 is a fixed pseudorandom direction orthogonalized against both. Both are
descriptive controls, not a sampling distribution. This construction and the
original site were informed by the same exposed examples, and neither the
observed geometry nor the proposed interventions proves a grammatical-number
circuit or transfer.

The saved little-endian binary64 vector SHA-256s are:

| Vector | SHA-256 |
| --- | --- |
| u | `d1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0` |
| v1 | `4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb` |
| v2 | `4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02` |

[The manifest](manifest.json) fixes the 16 capture hashes, source archive,
pair order, seed, intermediate norms, individual training projections,
source hash and vector hashes. Its SHA-256 is
`91dc4170eff6ac60416e5b3f884f2a28f41f9a4c1ad7339b409e32baeadf3a0e`.
The source file has SHA-256
`45ad94dd92848cb87fe0ae06573c276ffb05e6a6073ce9224a4f2a04c529b63e`.
Two independent reviewers reconstructed all three vector hashes; one also
reviewed the bounded snapshot reader, source identity and output readback.
Seven fabricated, model-free tests passed, including archive-path replacement,
oversized input, corrupt saved-vector readback, degenerate direction and
binary32 edge cases.

The recorded construction used CPython 3.11.15, MSC v.1944 AMD64 on Windows,
with binary64 radix 2, 53-bit mantissa and `sys.float_info.rounds == 1`
reported at startup. This is not a continuous live floating-point-mode probe.
The exact operation
order and the runtime `sqrt` limitation are fixed in the protocol; a different
runtime should be checked against these hashes instead of assumed identical.
The historical model run did not record `MKL_CBWR` or `ATEN_CPU_CAPABILITY`;
this artifact uses its retained capture bytes rather than reproducing that
historical runtime.

From a checkout of the published source, with the original evidence archive in
place, run the model-free builder into a new, nonexistent directory:

```text
python -B -m looking_glass.agreement_direction_training_v1 --archive results/distilgpt2-natural-agreement-v1-20261028/discovery-evidence.zip --output-dir <new-output-directory>
python -B -m unittest discover -s tests -p test_agreement_direction_training_v1.py
```

The derivation checks the archive hash and parses the same bounded in-memory
snapshot, verifies each source capture and reads back every written vector
before producing the manifest. A saved-byte replay checks consistency; it does
not independently authenticate the original model execution or its custody.
