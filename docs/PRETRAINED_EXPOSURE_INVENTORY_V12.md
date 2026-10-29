# DistilGPT2 development exposure inventory v12

This snapshot reserves 32 public copying-head factorial cases before any
factorial model forward. It preserves every v11 record, including the observed
copying follow-up and the 13 planned projected-edit cases. This is a planned
reservation; it does not report an experiment result or authorize a forward.

The canonical inventory is `looking_glass.pretrained_exposures_v12`. Its
predecessor is v11, SHA-256
`84b1a95a41670798ff55c9f08272890c00a939a74a4cad7ad9b95fbf4bcdaa25`;
v12's canonical payload SHA-256 is
`af374aaceb708b3b38e06b3955ca975467b048162ee105cbfcc9452a8a600e49`.
It contains 215 records: 136 observed-saved, 3 observed-reported, 64 planned,
4 reference-only, and 8 synthetic-only. Its 1,709 known or reserved input
prefixes include all 1,357 from v11.

The factorial protocol was publicly fixed at commit
`9dfc2952d3b21c85f0f7afc6bc6d902ed210634c`, before the case file. The public
case-file head is `7a5a7d6eaea4c82f02f7f4e05b6e5d41c85ecb49`, merged via
`571389b491fc675c11a85654517c84d41b2ecca5`. Their bytes are:

| File | SHA-256 |
| --- | --- |
| `docs/DISTILGPT2_COPY_HEAD_FACTORIAL_V1.md` | `467b566fd149bc7e489c0be265c01b63c3f005804bf30c7873560df930aa9d92` |
| `experiments/distilgpt2_copy_head_factorial_v1.json` | `69f8463da5f919fb99aef05aa4401de439248100e99ca90899f8d9f98af90d24` |

The case file binds v11, the pinned DistilGPT2 model and tokenizer, eight
six-token groups, and four fixed permutations per group. The compact JSON
SHA-256 of the 32 ordered input-plus-target rows is
`8ecee93b9ec137be5523f20751e4f6d1db231fafa384d32270be9849cb09e632`.
The 224 forwards listed by the protocol are planned, not evidence of execution.
The cases and selection process are public and non-blind.

The successor checks each proposed input-plus-target row against predecessor
complete rows in both prefix directions, compares proposed complete rows with
one another, and rejects any proposed prefix already present in the known
prefix set. Its generic candidate checks also reject a known prefix or an
extension of a recorded complete input. The focused tests bind the exact
published case and protocol bytes, predecessor and ordered inputs/targets.

These checks cover known records only. They do not establish complete prior
exposure, pretraining history or private held-out disjointness, and do not
authorize a study seal. Future public inputs require a separately reviewed
reservation before forwarding.
