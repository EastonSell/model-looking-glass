# Known DistilGPT2 development exposures: inventory v9

This immutable successor preserves the v8 predecessor snapshot with canonical
SHA-256 `22b1392ed43568fcac574b55ae6cf0212766771d8439ac39e599f7aa112b298f`.
The v9 canonical payload SHA-256 is
`e0c394786b316ea61b9f3697063eb86c7cf290d898c3ed84549880df8af91c4d`.
It retains all 138 v8 records and their 962 known or reserved nonempty
prefixes, then adds 32 planned public copying follow-up rows. The resulting
170 records contain 1,314 known or reserved prefixes: 104 saved observations,
three operator-reported observations, 51 planned, four reference-only and
eight synthetic-only. Existing observations and planned rows are unchanged.

The [follow-up protocol](DISTILGPT2_COPY_HEAD_FOLLOWUP_V2.md) was published
before case assembly at commit `a9f29a5421ba40404708d48913c7d599be69a429`;
its SHA-256 is
`70616bb01aa7ccb8440da9dc503c274a3d5cb7b24dd3965fd985de6c645ca68c`.
The separately published [case file](../experiments/distilgpt2_copy_head_followup_v2.json)
is present at commit `939d59540b642c83bc999bcf45220ff2155f41b2`, with SHA-256
`3831bfd1a48aaa3fcb67fbdb3cdc10f4422149b1fed90fd5267d3c73525255a8`.
It binds the reviewed v8 snapshot, eight groups of six token IDs, four fixed
permutations per group and the corresponding targets. All 48 group token IDs
are unique and disjoint from both earlier public copying datasets. Each new
full row was checked against every v8 full row and every other new full row;
equality and complete-row prefix relationships in either direction are
rejected. The case file and this reservation assert **no model forward** on
the new rows. These public examples are exploratory, not a blind holdout.

The module `looking_glass.pretrained_exposures_v9` exports `FOLLOWUP_GROUPS`,
`FOLLOWUP_ROWS`, canonical inventory bytes and digest, and bounded known-split
checks. Those checks reject a candidate full row equal to a known or reserved
nonempty prefix; they do not automatically reject a longer candidate extending
a shorter known row. The stronger bidirectional check above applies to the
32 fixed public follow-up rows at reservation time. Neither check rules out
unknown history, semantic overlap or pretraining, certifies a complete
exposure history, or authorizes a study seal or a model forward. A trusted
custodian must bind the actual checkpoint and tokenizer, the current reviewed
snapshot and immutable candidate bytes. Keep private inputs, exceptions and
tracebacks private. No private or hidden held-out case was generated, and no
model forward was run, while preparing this snapshot.
