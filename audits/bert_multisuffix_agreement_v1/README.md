# Multisuffix agreement input checks

These are tokenizer-only checks. The 160 fixed variants in
[`bert_multisuffix_agreement_cases_v1.json`](../../experiments/bert_multisuffix_agreement_cases_v1.json)
have SHA-256 `bc8d1af844c52f46bf068c04257280ed37e4302028885540ea498103d6ce125e`.
The study is a prospective follow-up to the published single-suffix result;
these checks do not load a model or establish absence from pretraining.

The [owner comparison](public-comparison-v2.json) covers 1,448 row references
in 31 named sources, including repeated exposure ledgers and one separately
fixed, not-yet-public 55-row antonym proposal. There were zero cross-bundle
full-row-prefix collisions in 231,680 comparisons, both with and without the
terminal `[SEP]`. The only within-proposal relations were the 128 intended
terminal-versus-continuation pairs. The 12 subject/attractor surface WordPiece
IDs and the two exact frame-token sequences were absent from these named
source rows. A new surface frame does not imply novel syntax.

An independently authored [public-only review](independent-exposure-review.json)
checks 28 immutable public sources and 1,319 row references (211,040
comparisons), with no cross-bundle collisions and the same 128 expected
relations. Its [original script](independent-exposure-review.py) is retained verbatim;
its [portable path-adapted script](independent-exposure-review-portable.py)
reproduces the same report with the [pinned predecessor receipt](independent-prior-receipt.json).
The independent [pending-antonym review](independent-antonym-review.json)
checks the additional 8,800 proposal/antonym row pairs with zero collisions.
It binds the local proposal SHA-256
`284fc9adc0cc3caa52ea5d122c1f08db50c8dda2d881297e2dcd1f2229cb6386`;
replay of this pending comparison requires separately retained matching bytes.
A later tokenizer-only donor-context proposal with SHA-256
`917bad800f72a9613e69d8feae3faf90ffbae5cc446bbeb3e2c8df1d03678dc9`
had no full/SEP-stripped prefix relations in 5,120 multisuffix comparisons
([receipt](pending-donor-context-comparison.json)); it was pending and may
change. None of these comparisons covers unknown, private or model-pretraining
data.

To rerun the owner check, supply a checkout containing the pinned Git commits,
the pinned `tokenizer.json`, and the SHA-matching pending antonym manifest:

```sh
python scripts/check_bert_multisuffix_public_v1.py --repo /absolute/repo \
  --cases experiments/bert_multisuffix_agreement_cases_v1.json \
  --tokenizer /absolute/tokenizer.json --pending-antonym /absolute/pending.json \
  --output /absolute/new-report.json
```

The tool uses the `tokenizers` Python package and creates a new output file.
Refresh public and pending inventories, and obtain independent currentness
reports, immediately before any model forward; these are point-in-time checks.

The independent public-only reviewer can be replayed separately:

```sh
python audits/bert_multisuffix_agreement_v1/independent-exposure-review-portable.py \
  --repo /absolute/repo --candidate experiments/bert_multisuffix_agreement_cases_v1.json \
  --prior audits/bert_multisuffix_agreement_v1/independent-prior-receipt.json \
  --tokenizer /absolute/tokenizer.json > /absolute/new-independent-report.json
```

The report reproduced byte-for-byte with SHA-256
`3ba317a4215b1593f6dfd2d27a710741bdbcbae7bae7974c27c0c404f851d936`.
