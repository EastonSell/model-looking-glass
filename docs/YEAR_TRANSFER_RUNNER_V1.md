# Year-transfer CPU runner

`scripts/year_transfer_v1.py` implements the fixed public protocol in
[DISTILGPT2_YEAR_TRANSFER_V1.md](DISTILGPT2_YEAR_TRANSFER_V1.md). It binds the
reviewed v16 reservation for the 48 published rows. Before launch the operator
must publish the exact source and pre-run record and independently check the
current reviewed public exposure record. Outside and pretraining exposure are
unknown; a reservation alone does not establish that a model was executed.

The runner makes 48 unhooked clean and 48 passive no-op calls, retaining a
full final-position float32 vocabulary vector for each and two full block
snapshots for each no-op. It checks all 24 same-wording position-zero pairs
and evaluates the clean gates. If both gates fail, it saves a complete
96-forward result and performs no intervention. If either gate passes, it
performs the four rotated actions and an unhooked clean-after for each row,
for 336 total calls. It reports both gates and every direction separately;
the exploratory gate does not authorize a comparator interpretation.

The family-wide common dose and the three non-self controls use the protocol's
binary64 arithmetic and software-defined float32 rounding. Every hooked action
must start from the saved full no-op bits and land its exact planned slice,
with all other output coordinates unchanged. Its reported achieved dose is
recomputed from that invocation's saved before/after bits. Full-logit clean,
no-op, self and clean-after equality is required. Scoring decodes the saved
float32 bit patterns to binary64, including subnormals, before applying the
stable three-token band formula. Each action record also retains the expected
before/after site bytes as hexadecimal under the separate metadata limits;
the rest of the expected tensor is the unchanged saved no-op snapshot.

The process uses the pinned cached checkpoint, CPU float32/eager/no-cache
execution and the prior CPU guard and runtime probes. It verifies the clean
committed source set and records source, dependency and model identities,
environment, CPU RNG state, model state and nonpersistent buffers. The raw
payload cap is 114,731,328 bytes, with separate per-forward and metadata
limits; these are retained-file bounds, not a total process-memory guarantee.

Each returned bounded payload is written and journaled before its numeric,
control and runtime checks. If a helper or post-return preparation fails before
a payload is recorded, available bounded bytes are preserved in a separate
partial file when the physical raw limit permits it; any incomplete original
write is left intact. The failure record inventories both file kinds and
identifies a failed partial save. A false `model_call_return_observed` or
`payload_returned` flag means that return was not observed, not that the
underlying call definitely did not return. Interrupted
or failed attempts are not complete results and are never retried silently.
Keep the external process exit/failure observation and independently replay
the saved logits, gates, witnesses and action ordering before reporting.
