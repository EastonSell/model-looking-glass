# Known-injection number-study primitives

The current `looking_glass.pretrained_number_injection` module supplies the
fixed 768-component direction, integer-only IEEE binary32 addition, an exact
pre/post injection-bit check, and a CPU rounding/denormal sentinel for the
known-injection track of [the number-study protocol](PRETRAINED_NUMBER_PATCH_V1.md).
It does not yet execute the study, commit a hidden site, create a held-out
split, enforce experiment custody, or authorize discovery or GPU execution.

The reference addition rounds finite binary32 operands to nearest, ties to
even, with explicit signed-zero, subnormal and overflow handling. It does not
use a host float32 cast to decode subnormal operands. The selected runtime
sentinels check subnormal inputs and outputs and halfway rounding; they are
not a proof of all arithmetic behavior. The caller must run them on the actual
CPU execution thread and reject mismatches before an injection. Injection
verification consumes immutable little-endian snapshots of the whole selected
vector, rejects nonfinite or incorrect results and requires a changed element.
The caller is responsible for preserving those snapshots privately and
checking site, invocation, action order and cleanup.

CPU unit tests compare 4,096 finite operand pairs with the local PyTorch
runtime, cover fixed edge cases, and check that the sentinel rejects an enabled
flush-to-zero mode when the runtime supports changing it. An independent
reviewer also checked 274,480 operand pairs on this machine; that operator
observation is not a cross-platform certification.
