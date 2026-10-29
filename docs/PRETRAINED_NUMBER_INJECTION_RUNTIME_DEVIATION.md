# Prospective injection held-out runtime deviation

This document supplements `PRETRAINED_NUMBER_INJECTION_HELDOUT_SUPPLEMENT.md`
and is to be independently reviewed and published before the injection
held-out split is allocated. The frozen prediction, site commitment,
discovery evidence and original protocol remain unchanged. The earlier
supplement separately discloses a departure from the same-full-source rule.

## Recorded runtime difference

Protocol v1 also required the same bound runtime for discovery and held-out.
The published discovery runtime has SHA-256
`b6860ec21fe6f0ab626164b86b494d783f3c18c92a697744b118be4cdb33754e`
and remains the runtime identity of the discovery and its frozen prediction.
Its `system_environment_sha256` field is
`464d57c8112df2880d0261907da9764d08d1f1dab156c542bb6b8c15ca2ed9b1`.
The retained record does not contain the underlying original system-variable
values. That digest cannot recover them, and does not identify which values
changed. It is an integrity fingerprint, not guaranteed anonymization of
guessable local paths; underlying values remain private.

A separate supervised no-model diagnostic of the frozen source reported
that only the recorded `system_environment_sha256` field differed from the
discovery runtime. Its child exited nonzero, was reaped, and its Job had zero
active members before verified drain and close. No model was loaded, no model
forward ran and no held-out phase was claimed. The failed diagnostic and its
logs remain retained and charged to the physical phase allowance. This
diagnostic did not retain a new complete runtime artifact and does not
establish equality of the underlying environments.

The proposed evaluation therefore makes a **second, explicit departure** from
the original same-runtime rule. Before any held-out allocation, observe and
publish a separate canonical held-out runtime manifest using the same frozen
runtime producer. Its complete decoded record must match the discovery record
in every field except `system_environment_sha256`, which must be a valid
distinct SHA-256 digest of the actually observed curated environment. Bind
both manifests with unambiguous discovery and held-out labels, and bind this
document and the source and result of the fixed public-case check below in the
pre-draw operational record. Recompute the actual manifest before model work
and after execution; observe the current numerical and environment settings
after execution without resetting them first, so persistent drift at that
observation boundary is not masked by reinitialization.
Reject a mismatch or an unverified observation rather
than silently substitute a runtime. Preserve the original discovery runtime
hash in the prediction and journal bindings; that hash alone must not be
presented as the held-out process's actual runtime.

The checked record includes the original source and checkpoint, assets,
dependency versions, CPU numerical configuration, deterministic settings and
the explicit numerical backend environment. These checks do not bind every
native library or every feature of the machine. In particular, an unknown
system-path difference can affect executable or library resolution; the two
environments must not be described as equivalent. Keep raw system paths and
dependency-origin details privately for provenance where available. Every
result must identify this runtime departure, both runtime hashes and the
limited compatibility check; no result is unqualified protocol-v1 completion.

## Fixed public-case compatibility check

Before the held-out split draw, run one separately recorded compatibility
check on **already exposed discovery case index 0**, using the same private
committed injection site and frozen discovery executor. The immutable
references are the 16 `observation-00-<role>.json` files under
`results/pretrained-number-injection-discovery-v1-20261028/public/` at commit
`6b0dd7e1c7e30288d4f563175d589540ee8480fd`. The role order is `clean`,
`repeat`, `injected`, `forward_1` through `forward_4`, `reverse_1` through
`reverse_4`, `null`, `self`, `upstream`, `downstream`, and `verify_clean`:
exactly 16 model forwards. This is an exposed development check, not a
held-out observation, a new site draw, a selection step or a change to the
frozen prediction.

Publish and independently review the exact replay source, reference hashes,
actual runtime binding, one-use schedule, result schema and resource
reservation before its first forward. Reverify the original site commitment
inside trusted custody; do not log or publish the site, its nonce or private
action witnesses. Check each role's request, input, donor, action and trace
identity and audit the retained private landing and untouched-coordinate
witnesses. Compare all six attention and six MLP captures, final-block capture
and all 50,257 last-token logits for each role to the published references,
elementwise using the protocol's binary64-evaluated
`abs(a-b) <= 1e-5 + 1e-5 * max(abs(a), abs(b))` rule. Reject nonfinite values,
missing or wrongly ordered records, or any tolerance or control failure.
Retain all available bounded observations and private partials. Until the
authorized site reveal, keep per-boundary diagnostics, per-role comparisons,
private witnesses and failed-role details private. A successful site-blind
public projection may contain one overall pass statement, with the fixed
reference and execution identities; it must not encode the secret site
through metadata or failure messages.

Perform the check once, without outcome-driven retries, replacing cases or
adjusting settings. A failure, unknown cleanup or missing/ambiguous terminal
leaves the compatibility check and the affected prospective evaluation
incomplete. Require an independently observed clean process exit and separate
saved-evidence audit. Reserve and account all completed, failed and partial
pilot and rehearsal files within the reviewed study ledger before writing;
do not consume the held-out share twice. A pass supports reproducibility only
for these fixed public cases and operations. It does not prove identical
runtimes, general equivalence, or correctness on private held-out inputs.
