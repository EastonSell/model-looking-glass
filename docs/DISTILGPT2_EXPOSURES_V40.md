# DistilGPT-2 public exposure inventory v40

This append-only successor reserves the 68 fixed, public invariant-noun
agreement inputs and their eight candidate continuations. It asserts no model
forward. The [case bundle](../experiments/distilgpt2_agreement_invariant_noun_cases_v1.json)
SHA-256 is `50ef3aea53a21a76ecf1bc4910882f78939a4d2fcca12528013ea56eaa586827`;
the [protocol](AGREEMENT_INVARIANT_NOUN_V1.md) and case publication preceded
any new forwarding.

Canonical v39 SHA-256 `5e593d1b5a0ac35aba2791cd5eafe7114f7e93ba4bb1c90706e385d2fab0cadc`
contains 1,183 records. V40 preserves all of them and adds 68 planned records,
for 1,251 total, with input-prefix union 6,807 to 6,947. Each new record binds
the exact case ID, role, text, token row and all eight answer IDs. Its
canonical payload SHA-256 is
`685391b672c356dd627923681c4a93fc9a310c03a40fdd22e0b29af2ae9f7be3`.
The builder rejects complete-row prefix relations in either direction with
known input, attention-mask-active and scored-prefix sequences and the
recorded answer extensions, as well as cross-case relations.

Prior records and their labels are carried forward unchanged, so planned
labels may be historical after separately published work. This is a scoped
public inventory, not proof of complete previous exposure or private split
membership. It does not authorize a held-out seal or a model forward on its
own; future work must perform its applicable source, currentness and custody
checks.
