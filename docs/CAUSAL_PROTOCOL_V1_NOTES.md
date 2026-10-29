# Causal protocol v1: pre-run clarification

This note records a development event after the original protocol was committed
and before the recorded v1 run. It does not replace or rewrite the frozen
[protocol](CAUSAL_PROTOCOL_V1.md).

While developing a regression test for accidental reuse of reserved inputs, an
intentionally failing test passed one public v1 held-out vector into a
development run before the guard was implemented. That development run used
model seed 17 and random-site seed 3, different from v1's 2301 and 918. The
test ran at least twice, including once by the independent reviewer. Its output
was not deliberately inspected and the temporary run
directory was deleted. No result from the v1 model seed was evaluated in this
step, but the event means the suite must not be described as an untouched blind
holdout.

The recorded v1 result is therefore **exploratory**, using a public fixed
suite. The suite is still split by input identity for the stated experiment,
and predictions must still be published before the v1 held-out forwards. The
pre-run split and ordering are useful engineering controls, not proof of prior
blindness. A later confirmatory claim would require a newly frozen, genuinely
unexposed evaluation suite and protocol.

An independent review also found that changing the sign of zero could bypass
the original hash-only reserved-input guard. The guard now compares float32
input tuples, treating positive and negative zero as equal. Its regression test
blocks model forwards while verifying the rejection, so the test cannot expose
the reserved input to the model if the guard fails.
