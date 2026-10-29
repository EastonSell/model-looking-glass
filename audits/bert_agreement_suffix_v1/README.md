# BERT suffix input checks

These checks precede any model forward on the 64 proposed input variants.
The [prospective protocol](../../docs/BERT_AGREEMENT_SUFFIX_V1.md) and
[fixed manifest](../../experiments/bert_agreement_suffix_cases_v1.json)
state the exact paired design and SHA-256. Tokenizer assets were used for case
construction and verification; no model weights or private rows were read.

The [owner comparison](public-comparison-v2.json) checks 64 variants against
1,191 row references from 25 named, byte-pinned public or pending sources,
including the independently published cross-model and color sets. There were
no cross-bundle complete or SEP-removed prefix relations. Of 2,016 within-set
pairs, exactly the 32 declared terminal/continued pairs had the expected
SEP-removed prefix relation. The exact subject/attractor noun forms and two
surface-frame phrases did not occur in those source bytes. This scope excludes
unknown or private exposures and pretraining; unique surface words do not
imply novel syntax.

Two independent reviews reached the same scoped input conclusion. The
[first report](independent-input-check.json) independently reconstructs the
manifest from pinned vocabulary bytes, checks donor/token alignment and 24 named
source inventories, and is reproduced by [its script](independent-input-check.py).
The [second report](independent-exposure-check.json) checks 25 sources and
tokenizer behavior. It also shows that, after dropping the new preamble and
masking noun identities, all 32 continued rows reproduce a previously public
agreement core and all 32 terminal rows are prefixes of that core. Its
[script](independent-exposure-check.py) only changes the reviewer's original
local file paths to arguments; all scientific fields reproduced exactly. The
reviewer's original script SHA-256 is
`01de7e602bd2548189cd6048cecc4b3fb1f161b4e7047eb086a36cf768ad1cdd`.

The then-pending context-transfer source is now public in PR #660 at commit
`40752bde00f11b69832066c2d16ce52c49f96810`, path
`experiments/bert_cue_context_transfer_cases_v1.json`, SHA-256
`4fb31c4b49c24e78f6a4b0b59d928ebb6204a55ecdc2c9c4cd55528700d592a7`.
The scripts take that exact file as a separate argument. Launch-time
currentness still needs to be recorded for any later attempt.
