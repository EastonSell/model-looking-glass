# Agreement-study stored-float comparisons

`looking_glass.agreement_numeric` decodes bounded, immutable little-endian
float32 bytes by IEEE sign, exponent and fraction fields directly into
binary64. This avoids a float32 load or cast that can flush a stored
subnormal when CPU flush-to-zero is active. It preserves signed zero and
rejects nonfinite values, empty or malformed payloads, and payloads larger
than 1 MiB.

`compare_f32_le` reports exact stored-bit equality separately from the fixed
study tolerance: `abs(a-b) <= 1e-5 + 1e-5 * max(abs(a),abs(b))` for every
decoded binary64 value. The reported maximum absolute difference uses that
binary64 arithmetic. The caller must check tensor shapes and identities; equal
byte lengths alone do not establish either. The byte cap bounds each supplied
payload, not transient host RAM, CUDA memory, or aggregate experiment storage.

The implementation does not capture tensors or verify that an intervention
landed. Tests use synthetic bit patterns, including signed zeros, subnormals,
normal extrema and nonfinite encodings; the denormal-mode subprocess is
skipped when the host cannot set both modes.
