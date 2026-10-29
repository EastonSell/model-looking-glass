# Public DistilBERT agreement cases and development inventory v1

The frozen public case manifest is
`experiments/bert_agreement_public_cases_v1.json`, SHA-256
`20a8ce96bab80c1f3b19ca86ac43e5d32f62c4b4a9bbd3c23e1a3fc9abfd91e5`.
These cases are public and non-blind. Evaluation cases are held out from this
team's development and candidate selection, but there is no certification
of complete past model access or of absence from pretraining.

The manifest names `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be` and hashes the expected config,
weights, tokenizer, tokenizer config and vocabulary. Those are expected asset
identities, not a claim that this case checker loaded the checkpoint or
verified a later runner's model. The calling experiment must independently
verify and bind its actual assets, runtime, source revision and the frozen
scientific protocol before any evaluation forward.

The four unique rows in the earlier public exploratory pilot and its same-host
replay are labeled `observed-pilot-and-replay`. The original result SHA-256 is
`2602232c7b0d5382f578b455b83496721ce6a68ff4a9fef398cdb033862387e0`
and the replay result SHA-256 is
`d0c4acfec26b80fd8609b3a005e3970c73eff76f6686ddc30823d1df1f89076a`.
The other 28 development input rows are conservatively
`reserved-development`, including next-family donor rows; their listing is
not evidence of a forward. The 64 evaluation rows are labeled
`public-nonblind-evaluation`, not observed outcomes. Later forwards or
newly discovered history require a successor inventory or separate evidence;
do not silently rewrite this manifest.

The four development and eight evaluation families follow the public study
design. Each has four A-target inputs and four B-donor-only inputs, with both
subject and distractor numbers crossed. Each target row explicitly maps to
`A_same`, `A_opposite`, `B_same` and `B_opposite` donor row IDs. B is
the next family's subject lemma in the same split, cyclically, while the
distractor and its number remain those of the target family. `A_same` refers
to the target row itself. Within each family, targets are ordered
singular/singular, singular/plural, plural/singular and plural/plural;
scientific code should still resolve row IDs rather than depend on order.

All 96 rows store exact text, ten input IDs and all-one attention masks. The
subject, distractor and mask positions are 2, 5 and 6; `is` and `are` are
single tokens 2003 and 2024. Development uses `near`; evaluation uses
`beside`. Subject and distractor lemmas are disjoint across those splits.
`verify_manifest_bytes` checks the immutable byte snapshot, all 32 reserved
or observed development rows, all 64 evaluation target/donor rows, and their
exact complete-token-row disjointness. It does not attest semantic
independence, an unseen training history, a model execution or authorization.
The exact-row policy is appropriate to these full bidirectional masked inputs;
it is not a proof about partial contexts or other tokenizers.

The module reconstructs its pinned JSON without loading a model. To check the
actual tokenizer separately, run from the repository root:

```text
python -m scripts.verify_bert_agreement_tokenizer --tokenizer-dir PATH_TO_PINNED_LOCAL_SNAPSHOT
```

The verifier checks the four local config/tokenizer asset hashes, loads the
fast tokenizer with `local_files_only=True`, and independently tokenizes all
96 full public texts and the two output words. It does not read or verify the
weight file, load a model, or perform a forward. The saved owner-produced
`results/bert-agreement-tokenizer-check-v1-20261028.json` has SHA-256
`2c7ad4548af28032d7b77b25f388c6855085bbbf1baca0698e6d4fe54c7dc958`;
it reports Transformers 4.57.3 verified all 96 rows and four local assets
with no model load. Its source hash and other fields are provenance, not
authentication of the execution. The tokenizer and model assets still need
independent checking in the eventual study runner.

The check is cooperative: the Python code and its frozen constants must be
bound to the pre-forward source revision, and callers should not treat a
returned report as cryptographic proof of execution or of a complete
exposure history. A future change to the protocol, tokenizer, cases, donor
mapping, or evidence must be versioned and reviewed before model execution.
