# DistilGPT-2 agreement-head fresh-context cases

> **Superseded before model forwarding.** Three subject lemmas in these
> cases appeared in earlier exposed head-development cases. The corrected
> [v4.1 protocol](../../docs/DISTILGPT2_AGREEMENT_HEADS_FRESH_CONTEXT_V4_1.md)
> fixes a replacement case set; the original manifest and receipts are kept
> here as historical evidence and were not reserved or forwarded.

These are the 16 public, nonblind cases fixed by the [v4 protocol](../../docs/DISTILGPT2_AGREEMENT_HEADS_FRESH_CONTEXT_V4.md). They cross four sentence frames with four subject lemmas in the protocol's fixed quartet order. Case preparation used only the pinned tokenizer; no model forward on these cases is asserted. The manifest is a case inventory, not an execution result or an exposure reservation.

The four answer continuations are included for public overlap checking and are read from final-position logits if the study runs. The proposed model inputs are the 16 base rows, not base-plus-answer forwards. L3H4 and L3H8 were selected on previously exposed development evidence; the data-informed fixed-dose design, singular/plural token differences, repeated frames and lemmas, and other limits are specified in the protocol.

## Reproducing the cases

The superseded v4 preparation scripts have been retired from the active tree. Their immutable source links below preserve historical replay; use the corrected v4.1 tools for the replacement case set.

Run [`scripts/prepare_agreement_head_v4_fresh_cases.py`](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/scripts/prepare_agreement_head_v4_fresh_cases.py) with `--snapshot` pointing to the offline pinned DistilGPT-2 tokenizer snapshot, `--protocol` pointing to the original, pre-notice v4 protocol bytes at commit [`8d97920ff7e1214c14c3b0e256390d3df6de05f3`](https://github.com/EastonSell/model-looking-glass/blob/8d97920ff7e1214c14c3b0e256390d3df6de05f3/docs/DISTILGPT2_AGREEMENT_HEADS_FRESH_CONTEXT_V4.md) (SHA-256 `38ca8bc913ed1763bec798e2aa4119af1e8d2b4fe1525485e4c805c522ec4a81`) and `--output` pointing to a new file. The script checks the protocol and four tokenizer asset SHA-256s, requires `tokenizers` 0.22.2, loads no model, verifies one varying subject-token position per quartet and the four one-token answer continuations, and writes a create-only manifest. Both normal and optimized Python reproduced SHA-256 `e90584985df00a7749687d49139a83aa7488bff47f65af16f79b24572bba33ee`.

## Public-only comparisons

The accompanying historical receipts describe checks at the stated snapshots, not a complete exposure certificate or a private held-out query. Refresh the official public inventory, pending case bundles and applicable custody observations before a new model run.

- `public-overlap-v41.json` compares 80 head base/continuation sequences against 5,991 distinct sequences reconstructed from the v41 canonical inventory (SHA-256 `110ebcf674168e794544c257d158b3c3242f98bcf108375cdc073bb145e2bfad`) and 612 sequences in the public invariant-noun successor (SHA-256 `188782606abd576cde029408bb34975e7fd7ac2113bf17299dbb7e1aa2ddd76b`). Both prefix directions and the cross-case check are zero. Reconstruct v41 with `looking_glass.pretrained_exposures_v41.canonical_inventory_bytes()` at the published v41 source, and use [`scripts/check_agreement_head_v4_public_overlap.py`](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/scripts/check_agreement_head_v4_public_overlap.py) with explicit paths to the manifest, canonical bytes, invariant successor and a new output file. Its `--inventory-sha256` option permits an explicitly pinned later inventory for a fresh check.
- `natural-public-support.json` was independently prepared and replayed with [`scripts/natural_public_cases_crosscheck_v2.py`](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/scripts/natural_public_cases_crosscheck_v2.py). It checks both the head and invariant cases against the public 128-row original natural support, public 256-row successor pool, six public short-service rows, the public timing input and each other; it retokenizes all declared continuations. Counts are zero. The original-split scope is conditional on its recorded sampler, custody and tokenizer preparation; no protected content or private membership was queried. Supply the explicit files named by the script's `--help`.
- `france-public-overlap.json` compares the same 80 head sequences with 73 public France proposal sequences, with no prefix relation either way. The separately owned France manifest was a pending tokenizer-only bundle at this check, so this receipt binds its SHA-256 rather than claiming it was already published or forwarded. The overlap script accepts that exact manifest via `--france` and a new `--france-output` path.

The checks do not rule out unknown prior exposure, model pretraining on related text, indirect intervention effects or future changes in other public proposals. The proposed fresh inputs remain unforwarded in this case-publication record.
