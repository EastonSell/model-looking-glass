# Known DistilGPT2 development exposures: inventory v5

This immutable successor preserves all v4 records and its canonical SHA-256
`6c065926d08cf08aa98d2c2725299d6bdc25e919246851c34bd9c7d63e8a4af1`.
The v5 canonical payload SHA-256 is
`1c597a028952a40ac48614c0e263a87a034d4f6a3df2b4cbffcdb6047adf6586`.
It has 74 records and 258 distinct known or reserved nonempty prefixes,
preserving all 246 v4 prefixes and adding 12. Its categories are 40 saved
observations, three operator-reported observations, 19 planned rows, four
reference-only rows and eight synthetic-only rows.

The four additions are the public synthetic token rows `[50256, 107, 207,
307]` through `[50256, 110, 210, 310]` from
`docs/GROUPED_GPU_DEV_PLAN.md` at public commit
`332c1c6511a34f9658a1af5e0f2dd83a93f2eeaa`. The plan proposes them as
broker-routing checks held out from its policy construction. At this snapshot,
the owner reports no model forwards for these four rows; they are reserved as
`planned`, not as observed results. The previously recorded EOS rows 103–106
remain development exposures. Public and synthetic inputs are not private
study cases or evidence of pretraining independence.

The checker verifies the pinned predecessor, retains its full prefix set, and
rejects full-row prefix relations in either direction between the new public
proposals and prior complete records. Its strict 8+8/16+16 and exposed
discovery 16+16 policies retain v4's scope: each held-out full row is rejected
when it equals a known or reserved nonempty prefix; a longer row extending an
unrelated complete known row is not automatically rejected. Unknown historical
use, semantic overlap and pretraining remain unverified. Checking these known
rows is not permission to seal a study or run a model.

The trusted custodian must bind the actual checkpoint/tokenizer and current
reviewed snapshot before use, preserve the immutable bytes for reveal, and
keep candidate inputs quiescent during copying. Generic exception text can
still have a Python traceback retaining private inputs; keep exceptions and
tracebacks private and publish only sanitized statuses. No held-out case or
model forward was generated for this snapshot.
