# Known DistilGPT2 development exposures: inventory v4

This immutable successor preserves the complete v3 snapshot, canonical
SHA-256 `76baf0c8996bf2872a3a1b8fd0fe2734e15525ce0f7014e54811145cb98ce912`,
and appends 13 public targeted-edit follow-on proposals. The v4 canonical
payload SHA-256 is
`6c065926d08cf08aa98d2c2725299d6bdc25e919246851c34bd9c7d63e8a4af1`.
It contains 70 records and 246 distinct known or reserved nonempty prefixes;
all 204 v3 prefixes remain, and 42 prefixes are new. Its categories are 40
saved observations, three operator-reported observations, 15 planned rows,
four reference-only rows and eight synthetic-only rows.

The new rows comprise four proposed target and nine locality inputs supplied by
the follow-on study owner after a pinned-tokenizer-only alignment check. The
proposals are public and non-blind. At this snapshot there was no asserted
model forward on them and their final protocol/source had not yet been
published, so they are labeled `planned`, with no invented execution or
source claim. The owner must freeze its actual protocol, source and cases
before a forward; subsequent execution or new evidence belongs in a successor.
The v3 observed targeted-edit study remains a separate negative result.

`looking_glass/pretrained_exposures_v4.py` verifies the predecessor digest
and checks its planned rows for duplicates and full-row prefix relationships
in either direction against prior records before constructing the payload.
Its strict 8+8/16+16 and public
discovery-reuse 16+16 checks retain the v3 semantics. All nonempty prefixes of
all categories are excluded where the applicable held-out rule requires;
common shorter prefixes alone are permitted. A longer candidate that extends
an unrelated known row is not automatically rejected. The check covers only
known and reserved token rows, not unknown history, semantic overlap or
pretraining data. It does not authorize a seal or a model run.

The trusted caller must bind the actual checkpoint/tokenizer and reviewed
snapshot before use and keep candidate lists quiescent during checking.
Exception messages are generic, but their Python tracebacks can retain private
input locals; keep them inside the custodian and publish only sanitized
statuses. A newer snapshot, even one preserving all v3 prefixes, does not
certify that the known history is complete. No hidden case or model forward
was created for this reservation.
