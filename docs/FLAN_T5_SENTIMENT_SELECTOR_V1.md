# FLAN sentiment intervention: tokenizer-only selector

This module implements the private pool/selection commitments and the
tokenizer-only first-four rule in the [prospective protocol](FLAN_T5_SENTIMENT_INTERVENTION_V1.md).
It loads no model and performs no pretrained forward. The protocol, runner,
this selector and the full record schema must be public and independently
reviewed before a pool is drawn or screened. The 12-unit pool, two salts,
complete screen and selected rows remain private until the specified reveal.
An external custodian is responsible for semantic review, an independently
observed public pool card, currentness of the public exposure inventory and
any other protected-input comparison. These observations are not authenticated
execution or proof of unknown pretraining exposure.

## Record shape

The exact pool keys, values, word constraints, canonical serialization and
domain-separated commitments are fixed in the protocol. The selection has
exactly the top-level keys listed there. `unit_screens` is ordered `u00`–`u11`;
each unit contains exactly `unit_id`, `semantic_ok`, `lexical_exclusion_ok`
and two `instructions` in index order. Every instruction record has exactly
`instruction_index`, `cases`, `source_comparisons`, `eligible` and
`failure_reasons`. All flags are JSON booleans, all counts and indices are
JSON integers rather than booleans, and failure reasons are ordered strings.

Each `cases` array has four objects in `P1,N1,P2,N2` order. Every object has
exactly `side`, `text`, `input_ids`, `attention_mask`, `special_tokens_mask`,
`offset_mapping`, `adjective_index` and `subject_index`. IDs and masks are
integer arrays, offsets are arrays of two integer character positions, and a
failed alignment is represented by a null index. Every `source_comparisons`
entry has exactly `source_index`, `comparisons` and `conflict`; the count is
the number actually evaluated, and every candidate is compared with every
recorded row from that source even after a conflict. `selected_units` contains
the first at most four semantically accepted units whose two instructions are
eligible. `donor_mapping` has one object per selected unit, with exactly
`unit_id`, `opposite` and `same`; the latter two objects have exactly the four
side keys and the fixed reciprocal values from the protocol. A selection of
fewer than four is a no-forward outcome, not a case replacement opportunity.

The selector requires the pinned prior five FLAN public case/screen files,
the revealed held-out pool and screen, and the completed sentiment screen
journal, with their exact SHA-256s. Additional publicly known source files can
be included in the ordered inventory. For each Git-bound JSON/JSONL source,
it recursively collects every recorded `input_ids` row and all text and
source/target word strings, comparing complete normalized words and EOS-stripped
token prefixes. It does not assume those eight files include every possible
exposure; the custodian must refresh and report the scope before execution.
Every source path, revision and digest is committed in the private pool and
carried into the selection. No unsalted private row digest is published.

## Operation

Provide a trusted checkout containing the referenced Git objects and the
privately prepared pool, with output paths that do not already exist. Before
launch, an outer operator creates a fresh empty bytecode-prefix directory and
checks it before and after the direct process. The isolated launcher adds only
the selected package directory, not the checkout root, to Python's import
system. It compares its own and the four imported project-module bytes with
their checkout Git commit before and after execution; it does not authenticate
Python, the checkout, or an already-poisoned prefix. Git reads ignore
replacement objects. All source, pool-card and runner revisions must resolve to Git commit
objects; this local check does not establish that any commit was public at a
trusted time. The
first command validates pool structure and prior lexical exclusions, writes a
new private 32-byte salt and public pool card, and does not load the tokenizer:

```sh
python -I -B -X pycache_prefix=FRESH_EMPTY_PREFIX tools/run_flan_sentiment_selector_v1.py commit-pool \
  --repo CHECKOUT --pool PRIVATE_POOL.json --pool-salt PRIVATE_POOL_SALT.bin \
  --pool-card NEW_PUBLIC_POOL_CARD.json
```

Publish the card and obtain a separate observer's record with exactly
`schema` (`flan-sentiment-pool-observation-v1`), `path`, `revision` and
`sha256`. The source revision is a Git commit and the digest is over the raw
published card bytes. The second command re-reads that exact Git blob, verifies
the pool commitment and pinned assets, then uses the local fast tokenizer to
record all 12 units and produce a new private selection and salt plus a public
selection card:

```sh
python -I -B -X pycache_prefix=FRESH_EMPTY_PREFIX tools/run_flan_sentiment_selector_v1.py select \
  --repo CHECKOUT --pool PRIVATE_POOL.json --pool-salt PRIVATE_POOL_SALT.bin \
  --pool-card PUBLIC_POOL_CARD.json --pool-observation POOL_OBSERVATION.json \
  --asset-dir PINNED_ASSET_DIR --selection PRIVATE_SELECTION.json \
  --selection-salt PRIVATE_SELECTION_SALT.bin --selection-card NEW_SELECTION_CARD.json
```

Both commands use exclusive creation, read back and recompute their written
commitments before reporting success, and retain partial files on an error.
The selector bounds its private pool to 256 KiB and the written/observed
custody artifacts to 8 MiB with a 256 KiB allowance for external records;
the surrounding custodian still accounts for its own logs and receipts.
The selector's printed status precedes the launcher's final source and prefix
checks. The external custodian must retain and observe the direct process exit,
accept a stage only after exit 0 and reconciliation of its saved private and
public artifacts, and treat a nonzero/unknown exit as incomplete even if a card
or success line exists. Preserve partial records without retrying the fixed
attempt.
For each stage, retain the checked selector checkout revision and actual
Python, Transformers and tokenizers versions. An independent reader must
replay the complete saved 12-unit tokenizer screen with those recorded
dependencies and compare all records before the first model forward. The
asset digest and tokenizer class alone do not pin library behavior.
The surrounding custodian records stage outcomes and sanitized failures,
does not retry the fixed attempt, and applies the protocol's reveal rule if
selection is incomplete. An external independent observer reads the public
selection card before any model forward. The tool checks bytes and Git objects;
it cannot establish the observer's independence, that a Git object was public
at a trusted time, or that the source inventory covers unknown exposures.
The pool and private selection contain inputs and must not be sent to the
independent scientific reviewers before the attempt closes.

The selector's fabricated-only tests use a small fake tokenizer; an additional
no-model check retokenized four already-exposed development inputs with the
pinned local T5 tokenizer and compared their saved IDs, masks and offsets.
