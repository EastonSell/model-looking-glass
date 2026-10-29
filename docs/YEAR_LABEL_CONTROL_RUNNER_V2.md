# Year/label continuation runner v2

`scripts/year_label_control_v2.py` implements the separately frozen exposed
three-family protocol. It is not a retry that completes v1. It selects rows
12–47 of the exact v1 case file, uses their original family indices and
rotations, and records 288 forwards if all checks pass. V1's four-family
decision remains incomplete. The 36 v2 inputs and all 48 v1 clean/no-op
outputs have already been exposed.

The runner checks the exact protocol/case bytes, the historical v26 public
reservation and every selected own row, and the SHA-pinned v1 direct-process
receipt plus all 51 retained run files. It extracts the prior clean logits and
passive block-output bytes for cross-run equality reporting. The v2 clean
logits, no-op captures, common doses and plans are freshly computed; a
cross-run difference is recorded, not an eligibility decision. V1's stderr
is not required by the child; the separately published projection does not
authenticate its private original. Launch records must separately identify
the latest official exposure snapshot and the actual observed v1 status.

Each non-self plan saves its binary64 displacement, exact before/after float32
bits and rational squared D2/R2/E2 certificate, including limits. The hook
rechecks the certificate from same-forward actual bits, checks changed bits,
requires exact site landing and untouched coordinates, and preserves the
unmodified binary32 full-block and full-vocabulary evidence. Self replacement
requires bit identity. The fixed input fidelity limit is 1/5000 of q in norm
and vector error, with the independent 1/10**12 ideal-norm guard. Any failed
numeric, landing, state, source or evidence check stops the attempt; results
are retained as incomplete and there is no automatic retry or exclusion.
Exact squared fractions are authoritative. A human-readable vector-error
float is `null` when its square underflows binary64; a decimal square-root
summary is retained, and the runner rejects flushed binary64 subnormals.

Raw payload is bounded by 100,363,392 bytes across the complete 32-token
schedule, and each forward by 397,636 bytes (201,028 logit bytes plus two
32-by-768 float32 captures). For the fixed 15-token prompts the completed
payload is 77,802,624 bytes. Journal metadata is capped at 16 MiB, each line
at 65,536 bytes, and each JSON document at 16 MiB. Each partial payload is
bounded by the same per-forward maximum and the aggregate raw cap. These
bounds do not bound resident memory or external package caches.

Before any pretrained forward, publish the protocol, exact cases/source and
pre-run record; independently review the numeric planner, both control signs,
full synthetic schedule, source and incomplete-return retention. Use the
pinned cached checkpoint, one CPU intra-op thread, offline settings, `-B`, and
a fresh empty absolute `-X pycache_prefix` checked before project or Torch
imports. The guard assumes a trusted interpreter and a prefix not concurrently
modified. Run in a new output directory outside the source checkout, with an
absolute prior-attempt directory and pinned prior receipt. Observe the direct
child exit and write a separate `year_label_receipt_v2.py` receipt, including
post-run cache-prefix match and emptiness. Retain stdout, stderr and all
bounded raw evidence; a public redaction must remain a separately hashed
derivative of any path-bearing original. Source hashes, saved records and
operator observations do not authenticate every execution or outside exposure.
