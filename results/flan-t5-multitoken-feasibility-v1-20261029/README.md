# FLAN-T5 multi-subtoken tokenizer feasibility pilot

A tokenizer-only development screen found that the fixed proposed list can
form up to **seven** disjoint four-adjective units with at least two lexical
subtokens, subject to later semantic and exposure review. The pilot operator
reports no model load or forward; the saved files do not independently attest
execution. These are combinatorial upper bounds, not behavior, successful
interventions, or evidence of a sentiment mechanism.

This pilot was designed after the earlier
[selection-incomplete study](../flan-t5-sentiment-intervention-v1-20261029/README.md)
reported one eligible unit out of twelve; its custodian reported no model
forward. That result is unchanged. The pilot's 160 locally authored,
provisional-polarity adjectives and four exact templates are [recorded in full](candidates.json); they and
their 640 tokenized rows are outcome-informed development exposures, not a
blind lexical pool. The subjects `story` and `concert` and both instruction
frames are fixed in that file. No independent semantic review of these new
proposals occurred, and label applicability may differ by subject.

## Tokenization and matching

All **640/640** saved rows passed this pilot's structural span checks. The
screen uses the pinned FLAN-T5 tokenizer, checks the complete adjective and
its immediately preceding space, and records every input ID, attention mask,
special-token mask, offset, final adjective-overlapping position and rejection
in the [full report](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/flan-t5-multitoken-feasibility-v1-20261029/report.json.gz). Some tokenizations include a separate
leading-space token; such a token belongs to the boundary span but is not an
adjective-character-overlapping lexical token.

Before counting matched buckets, the pilot excluded **43/160** proposals that
appear in either of two explicitly checked public sources: the released
12-unit sentiment pool or the earlier eight-word sentiment model screen. It
checked no other public or private corpus and does not claim unknown-exposure
or pretraining disjointness. Complete matches group the four templates by
lexical-subtoken count, boundary-inclusive token count, absolute final lexical
position, and prefix/suffix IDs and masks outside the boundary span. Within a
bucket, each notional unit requires two distinct proposed positives and two
negative words; no model outcome enters the count.

| Lexical subtokens per word | Maximum disjoint four-word units |
| --- | ---: |
| 1 | 6 |
| 2 | 3 |
| 3 | 3 |
| 4 | 1 |
| 5 | 0 |

These capacities total **13**, including **7** with at least two lexical
subtokens. They do not require the four final token IDs to be distinct or the
same; different adjectives can share a final token. A separate later design
must inspect full word sequences, actual case masks and final positions, and
must state what any shared final IDs mean for its controls. A standalone
space-only token does not turn a one-lexical-token word into a two-token word.

The report also lists lexical-count histograms for **all 160** candidates,
including the 43 scoped overlaps: proposed positives have counts 61/8/7/3/1
at one through five lexical subtokens, while proposed negatives have
22/23/19/13/3. Those histograms describe this convenience list only; they
are not the post-exclusion bucket counts or a population estimate.

## Independent check and reproducibility

The [independent tokenizer check](independent-tokenizer-audit.json) compared
all 640 stored rows through both `T5TokenizerFast` and the raw `tokenizers`
interface, with zero ID, mask, offset, span or signature mismatches. The two
interfaces share the same Rust backend; this verifies interface/configuration
agreement, not two independent tokenization algorithms. The separate methods
review reconstructed the two scoped exclusions and all nine buckets. Neither
check establishes semantic correctness or behavior.

| Artifact | SHA-256 |
| --- | --- |
| [Candidate list](candidates.json) | `2b32da31aab398d19ef80e013c377d2ed3d1a06912a431ee918bfeccb64107d1` |
| [Original pilot source](original-pilot.py) | `8da53bef610521104f4c302e831ad7dd95599053fab0b95aa47b2743a4dc7b83` |
| Uncompressed report bytes | `c1426f1ea4690f7c5185580167001d946425dcf0e3d08664a1f993fc7ac0a62f` |
| [Compressed report](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/flan-t5-multitoken-feasibility-v1-20261029/report.json.gz) | `53686f255ac854e8aa6ceed88ce83a14c53dec6e29ce285081c6ed41b34c8b1e` |
| [Independent check](independent-tokenizer-audit.json) | `b477be3a621d7b21a3ce7ddcb72e0d809a5aab720a1fc947f2e4eea9e919634e` |
| [Independent check source](independent-tokenizer-audit.py) | `59596cf1a0d38fad7b1cb02aa709e2503a9d8b6abc12d445916a672bb5baf443` |
| [Portable replay](replay.py) | `fe41652633c506bcf0b9d1f32dec6e47f5a4bda8014d9346745978c719ec7cf5` |

The observed runtime was Python 3.11.15 and `tokenizers` 0.22.2. The original
pilot verified the pinned asset-manifest SHA-256
`d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083`
and the tokenizer JSON SHA-256
`fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a`.
The operator reports that it did not read model weights. The pool and old
screen were read from Git
object `461a6d44f180e4cf924a0ad47e3f624a9d5191ad`; the report records their
individual hashes; the portable replay verifies both hashes and ignores
ambient Git overrides and replacement objects. It disables Git lazy fetching,
so the two source objects must already be available in the checkout.

To reproduce with a checkout containing that Git object and the pinned local
tokenizer assets, run the [portable replay](replay.py), providing a new output
path. It compares all scientific fields to the original compressed report,
excluding the source and runtime metadata that legitimately differ. The
replay requires `tokenizers` but does not load a model or access a GPU.

```text
python -I -B results/flan-t5-multitoken-feasibility-v1-20261029/replay.py \
  --repo CHECKOUT --asset-dir PINNED_ASSETS \
  --candidates results/flan-t5-multitoken-feasibility-v1-20261029/candidates.json \
  --compare results/flan-t5-multitoken-feasibility-v1-20261029/report.json.gz \
  --output NEW_REPORT.json
```

The replay matched the original scientific fields in normal and optimized
Python modes. It also rejected deliberately altered candidate bytes and an
altered expected scientific field before writing an output. The hardened replay
continued to match with an invalid ambient `GIT_DIR`, and rejected a
deliberately wrong expected source hash before writing an output. The independent
check source accepts the uncompressed original report, candidate file and
pinned assets; decompressing the report preserves the recorded uncompressed
hash. The original source is retained as executed; use the portable replay
for paths on another machine.
