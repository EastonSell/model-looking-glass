# DistilGPT2 development exposure inventory v20

This successor reserves 40 public agreement-direction prompts before a
pretrained forward on those cases. It preserves all 459 v19 records and adds
32 target and eight context prompts as `planned`. The known input-prefix
union grows from 3,522 to 3,620.

The canonical module is `looking_glass.pretrained_exposures_v20`. Its v19
predecessor payload SHA-256 is
`e2eb185b32e74a4442ec68c43e53bd851573f60b7d7423b0d83195950f21b175`;
the v20 canonical payload SHA-256 is
`3df696303a06b2ec6ddf4113df1969897dbae89566375b7d3a9cecb85b3d59cd`.
The resulting 499 records include the 40 new prompts. The module exports
`AGREEMENT_CASES`, `AGREEMENT_ROWS` and `AGREEMENT_ANSWER_IDS` in their
published order. Each record ID is `planned-` followed by its exact public
case ID, and `case_role` preserves the case's target or context role.

All four possible continuation IDs `[318, 389, 373, 547]` are recorded as
`candidate_token_ids`; none is represented as a single target token. These
are candidate metadata, not an assertion that continuation rows were
forwarded, and only input rows contribute prefixes to the exclusion set.
The reservation checks both complete-row prefix directions for the prompts
and their input-plus-candidate extensions against prior input and recorded
target-appended rows, and checks the new prompts and extensions internally.

The protocol was fixed at `a39121c2f3cf2602281252a8856d48efadcb2a98`,
SHA-256 `f539ad55b347f88641de3b76565104dc7be43e0c5d15f6d8fefdb4173694bbea`.
The public case package merged at `3a5d5d2088471a702acea42c2f38d2cb2e0072bc`;
the case file SHA-256 is
`a68aa63ff06aa97c5c68d941fe27160c4c4698cdf070325b3cefa2b29ec58b13`,
and the ordered input-row SHA-256 is
`4b4faf815ae0d3d0943efec5fbc318fd251dc6fea80b6e32c770b470f01dae1d`.
The cases record v18 as their construction snapshot; this reservation checks
the later v19 inventory, including its 44 screened-edit prompts.

The public natural-split comparator input and digest-only receipt in
`results/distilgpt2-agreement-direction-cases-v1/` have SHA-256 values
`62d762605c3ca9fa78f53c6f186c6b1f2e94e08e1479d5f3f16058a805eb28fe` and
`2f288aa89494b2e597b64ba594166d9c8bc78d2a9bc5831377971d8116c4b8d6`.
The input file preserves all 40 case rows in order; the receipt reports no
bidirectional complete-row-prefix collision with 16 sealed natural rows. It
is a trusted-operator overlap observation, not execution attestation or
proof about other private or unrecorded inputs.

No pretrained forward is asserted for the 40 new prompts. Known-prefix
checks cover recorded and reported history only; they do not establish all
prior exposure or pretraining and do not authorize a study seal.
