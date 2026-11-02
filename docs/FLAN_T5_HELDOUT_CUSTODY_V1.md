# Locally held-out FLAN-T5 custody component

This model-free component implements the pool and selection custody described in
the prospective held-out protocol. Its tests use fabricated tokens and values.
No hidden pool should be assembled until the combined protocol and runner are
published and independently reviewed. A local commitment binds the recorded
bytes but does not prove a trusted timestamp, prevent another process on the
same account from accessing them, or establish independence from pretraining.

## Interfaces and private files

`load_exposure(checkout, source_commit, tokenizer)` checks the five committed
prior manifests against the checkout, then returns their normalized source and
target texts, all respective word and target token IDs, and their immutable
source identities. `validate_pool(pool, exposure)` checks the fixed 24 pairs,
two bounded templates, normalized text exclusions, public identity fields and
the exact five exposure identities. Human review is still required for noun
meaning, template novelty, structural distinction and semantic fit.

`create_pool_attempt(root, pool, exposure)` exclusively creates the custody
root, writes a started marker, draws one OS-random 32-byte salt and retains
`pool.json`, `pool-salt.bin` and `pool-commitment.json`. The two salt files are
raw 32-byte files, not hexadecimal text. `screen_attempt(root, pool_card,
pool_observation, tokenizer, exposure)` requires an independently observed
published card, writes its private `pool-observation.json`, screens all 24 pairs
in both templates, and retains `screen.json`. It returns a complete or
`selection-incomplete` screen, with no replacement of failed candidates.

`seal_cases_attempt(root, pool_card, screen, semantic_verdicts)` requires twelve
ordered verdicts, with a reason for every ineligible cell; it draws one distinct
case salt and retains `case-salt.bin`, `cases-payload.json` and
`case-commitment.json`. The payload embeds the exact screen and twelve selected
rows in pair/template order. An ineligible semantic verdict stays with that
cell and never causes replacement. `verify_bundle(root, pool_card, case_card,
tokenizer, exposure)` recomputes both commitments, the full tokenizer screen,
the first-six selection and the twelve verdicts before returning the private
case rows. The owner separately checks the source and public card observations
and the model asset identities before any model load.

The pool and case records, cards, screen and phase markers use canonical JSON:
UTF-8, sorted keys, compact separators, `ensure_ascii=True`, no nonfinite
values, no BOM and no trailing newline. The pool and case commitments use the
domain-separated constructions in the scientific protocol. Public cards have
only their salted commitment and public source/protocol/asset/tokenizer
identities; never publish unsalted hashes of private rows or payloads. Record
the separate source, pool-card, case-card and auditor publication observations
inside this root. The pool-card observation includes a normalized
repository-relative `card_path` and the hash of the public card bytes.

`inventory_custody(root)` returns a deterministic list of every relative file
path and apparent size, plus the count and total. It rejects reparse points,
special files, too many entries and more than 4 MiB. Keep its output private:
file names themselves may disclose custody details. This check assumes a
trusted, quiescent root and is not a filesystem quota. The process owner must
count all four observations, bounded private stdout/stderr, exit/failure
receipts and unexpected retained files. The root must be a protected sibling
of the separate model-evidence directory. Review the inventory before and
after the attempt, including owner files added after the child exits.

Each stage uses an exclusive started marker and never retries a failed or
ambiguous stage. A bounded failure receipt is attempted when a stage fails;
if the initial marker, storage, or failure receipt itself cannot be written,
the outer owner must retain an incomplete/uncertain outcome and must not retry
or redraw. Partial files are retained and charged. Unexpected exceptions and
tracebacks can contain private data, so the owner must capture stdout and stderr
privately and publish only sanitized status. After the frozen result and audit,
reveal the original private records and salts as prescribed by the protocol.
