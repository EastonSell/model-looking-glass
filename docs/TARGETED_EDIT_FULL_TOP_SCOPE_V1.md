# Full-top study: pre-forward scope clarification

This supplements [the prospective protocol](TARGETED_EDIT_FULL_TOP_CPU_V1.md),
whose public protocol-only commit is `5a0f0ef151a3a17e263ee902730bb2bf0d00d7e4`
and whose SHA-256 is
`36f1a709a6951866d6835802be71c8a0b3ea1a8bbcbfab4c5c3a97be5392299f`.
The tokenizer-only proposals already existed when this clarification was
written; no proposed input had been forwarded through the model and no private
split-comparison outcome had been requested or received. The finalized case
bytes, executable source and manifest still must be published and reviewed
before the first model forward. Preserve the original protocol and this
clarification in the study's public ancestry.

## Original natural split: public support instead of private query

The protocol says that a custodian should make a private comparison where
protected rows exist. The original natural split custodian reports that a new
48-row private query falls outside its reviewed disclosure scope. This
clarification prospectively permits a different, conservative check for that
specific original split: independently reconstruct and verify **every** input
row in its public sampling frame, verify the published binding of that frame to
the preparation which created the still-current sealed split, and compare all
48 finalized proposed input rows to the complete frame in both full-row-prefix
directions. The custodian must confirm the original split has not been replaced
and that its rows were drawn only from that frame. Record the exact source,
tokenizer, frame and case identities, comparison and custodian scope.

If the entire public frame is clear, the fixed proposal is disjoint from that
split under the recorded sampling and tokenizer assumptions. This is a public
superset argument, **not** a direct private comparison; no protected rows or
nonce are accessed and no claim is made about unknown splits or history. If
the frame is incomplete, its binding is unverified, the custodian reports a
different split, the comparison collides, or a relevant new protected scope is
unknown, stop before forwarding until the actual scope is resolved. Existing
checks of the current public inventory, other relevant protected splits and
the separately published 256-row natural successor pool still apply. This is
an explicit deviation from the original direct-private-comparison wording,
made before any proposed model input was forwarded.

## Locality variants and correlated templates

The original wording permits spelling or capitalization variants. The fixed
locality accepted set may also include a predeclared single-token numeral that
is semantically equivalent to the expected number word: for example ` 8` as
well as ` eight` for an octagon's sides, if each verified tokenizer append
boundary holds. This is an explicit prospective expansion of that wording;
the rule still uses accepted-set NLL and separately reports canonical-token
NLL and exact top IDs. Accepted sets are small prespecified lists and are not
claimed exhaustive. They cannot be expanded from model outputs.

Some target candidates use a Spain–Madrid example in the prompt. The
discovery, efficacy-calibration and evaluation strings have distinct roles and
are held out as exact phrasings, but shared in-context material can still make
them correlated. They are not independent facts or independent samples. The
study measures a selected fact's public, baseline-conditioned first-token
behavior, not generalization to unseen facts or a completed generated answer.
