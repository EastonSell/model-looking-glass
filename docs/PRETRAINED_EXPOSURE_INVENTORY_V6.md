# Known DistilGPT2 development exposures: inventory v6

This immutable successor preserves all v5 records and the canonical v5 SHA-256
`1c597a028952a40ac48614c0e263a87a034d4f6a3df2b4cbffcdb6047adf6586`.
Its canonical v6 payload SHA-256 is
`76803775e6929b4a20463a50bcae839a727b1a875a0d433f7c27c9d1d69f0681`.
It contains 106 records and 610 distinct known or reserved nonempty prefixes,
preserving all 258 v5 prefixes and adding 352. The records comprise 40 saved
observations, three operator-reported observations, 51 planned rows, four
reference-only rows and eight synthetic-only rows.

The 32 additions are the public, non-blind copying-screen inputs specified in
`docs/DISTILGPT2_COPYING_DEV_V1.md` and `scripts/copying_dev_v1.py` at public
commit `55d11a53eb34a13eb6b14e19a5bd0c47172edda5`. Each of eight six-token
groups has one repeated-sequence input and three matched permutations, all
with the same five-token suffix. At this snapshot the owner reported no model
forward for these additions; they are `planned`, not observed. The runner's
33 scheduled forwards include a repeated first row, so the reservation covers
32 distinct inputs. A later run must be recorded as a successor rather than
changing this snapshot.

The checker verifies the pinned predecessor, retains its full prefix set, and
rejects full-row prefix relations in either direction between the new public
proposals and prior complete records. Strict 8+8/16+16 and exposed-discovery
16+16 policies retain v5's scope: a held-out full row is rejected when it equals
a known or reserved nonempty prefix; a longer row extending a shorter known
row is not automatically rejected. Unknown historical use, semantic overlap
and pretraining remain unverified. Checking known rows is not permission to
seal a study or run a model.

The trusted custodian must bind the actual checkpoint/tokenizer and current
reviewed snapshot before use, retain its immutable bytes for reveal, and keep
candidate inputs quiescent during copying. Generic exception text can still
have a Python traceback retaining private inputs; keep exceptions and
tracebacks private and publish only sanitized statuses. No held-out case or
model forward was generated for this snapshot.
