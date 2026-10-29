# Projected-edit derivative and mode checks

This pre-execution supplement clarifies the interpretation of the published
[projected-edit protocol](TARGETED_EDIT_PROJECTED_CPU_V1.md), whose original
SHA-256 is `4bf403abba087c52d21e6f0cc0cd01b7f36a9713eb7da446dfea113ba363a6d8`.
The original protocol remains unchanged. The public evaluation cases had
already been assembled when these checks were added; no projected-study
checkpoint or case had been forwarded. The checks were motivated by
fabricated models, not scores on the public cases.

The full gradient `G` is computed in the grad-enabled discovery execution.
Its dot product with each actual rounded weight difference is the first-order
prediction for that execution mode. The study compares it with the separately
observed no-grad margin change; their agreement is a measurement, not an
assumed property of the model. Identical forward values alone cannot establish
that their downstream Jacobians are equal. This qualification applies to the
planned, intended and realized gradient dots and the ideal rank-one ratio.

Before constructing any edit, capture and save the complete discovery-site
input sequence, complete site-output sequence and final logit row in each
mode. Require corresponding float32 bytes to match. For every edited discovery
forward, also require its complete site input to match the original no-grad
input and save its input, output and final logit row. A mismatch makes the
attempt invalid/incomplete; retain its partial artifacts and do not retry the
condition. These checks detect the reproduced activation mismatch, but do not
prove equal downstream derivatives or establish general equivalence of the
two execution modes.

The future full study manifest and pre-evaluation record must bind this
supplement as well as the original protocol, source, checkpoint and public
case identities. The existing target/locality thresholds, controls, fixed
dose and negative-result rules remain as originally declared. The private
development helper does not itself authorize a model forward or publish a
study result.
