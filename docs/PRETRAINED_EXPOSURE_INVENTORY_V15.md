# DistilGPT2 development exposure inventory v15

This successor adds the 48 public, unpadded inputs evaluated by the completed
CPU greater-than screen and reserves 32 already-public copying pairs/dose
inputs before any model forward on those cases. It preserves all 279 v14
records, including the completed recipient screen, prior factorial evidence,
13 projected-edit reservations and the annotation of the eight reused
agreement inputs. Known input prefixes increase from 2,287 to 2,959.

The canonical inventory is `looking_glass.pretrained_exposures_v15`. Its v14
predecessor has payload SHA-256
`236e4ab86b006998cbf39096edf78222cc8506c3f2b81d5e7ae70b2a32b30d86`;
the v15 canonical payload SHA-256 is `6c6ed34755dd9daebdde9a2de07f65c0c5d20301656416d873f321fa2c146d34`. Of its 359 records, the 48
new greater-than rows are `observed-saved` and the 32 new pairs/dose rows are
`planned`. The latter do not assert a model forward. The module exports
`PAIRS_DOSE_GROUPS` and `PAIRS_DOSE_ROWS` in their frozen public order; each
planned record uses ID `planned-copy-pairs-dose-{group:02d}-{variant}` and
binds the group's sixth token as its target.

The greater-than case inventory was frozen at source commit
`47d42b8137203fa8e5ffdc60942a1adece6c0feb`, file SHA-256
`399994789ee4d3eff27dd6c9cce23ae27ab7761e4bc33caf58de4528194b01d2`.
Its completed evidence was merged at `b211bcbfc85f59167daee2d4c5ad5d822e32e834` under
`results/distilgpt2-greater-than-screen-v1-20261028/`. The saved manifest,
journal, result and direct-process observation report 48 returned and
recorded actions, exit code 0 and no failure file. The exclusion rows are the
exact input IDs in that manifest, not target-token additions. Independent
tokenizer-only replay of the pinned local tokenizer matched all 48 texts to
their stored IDs with no special tokens or padding: 16 rows each have lengths
12, 13 and 15.

| Greater-than evidence | SHA-256 |
| --- | --- |
| Run manifest | `5590f02a6becd05daffb0915088bbe66e41b3cc8dce2310030fb87162d63ee1c` |
| Run result | `65e48fdd8bfdb2a69455037e5ad059731106793014c366b06896252bb35970ef` |
| Attempt journal | `f280ef79bdd6b0d6aea66d8c35e0868d4a18193bc5ebb7654f20b25f2b82ac3a` |
| Retained logits | `e2cd020bc219d1e042fea90ecd098d95c0ccbf653e3526cfab1d6c39cb9b4737` |
| Direct-process receipt | `469b61cf70916434d2e96779ba842ba1dd9e2c50be9fd511a76ac8d3894bcd48` |

The public pairs/dose protocol was fixed at
`8be210ce88a536dc9efcd769e40a8175b9beddfa`, SHA-256
`811c18baef640e08ccff7f6cade26d041672edda969ddbc7a6babdb60d70bddb`.
The 32-case public file merged at `66ac1d16da70cc7f8c39399181e983e4bc944e32`,
SHA-256 `3e5d40fff832a6976f5e4f8d3bda916ab203ed93936463a1a0726e88dcab1106`.
The case file honestly retains its v13 construction pin and a separate check
against the subsequently observed recipient rows. This reservation independently
checks each complete input-plus-target against the newer v14 history and the
48 greater-than inputs, in both full-row prefix directions and against all
known input prefixes. It does not attest that a private held-out split is
disjoint; that must be checked through the custodian before a later forward.

Focused no-model checks reconcile the greater-than case list, all 48 journal
triplets and contiguous logits slices, and the pairs/dose cases and targets
against their public source. These checks establish consistency of saved bytes,
not independent execution attestation. Known-prefix checks cover recorded
history only; they do not establish all prior exposure, pretraining or private
held-out disjointness, and they do not authorize a study seal.
