# DistilBERT agreement exposure status v2

This public status snapshot supplements, and does not rewrite, the frozen
96-row `bert_agreement_public_cases_v1.json` manifest, SHA-256
`20a8ce96bab80c1f3b19ca86ac43e5d32f62c4b4a9bbd3c23e1a3fc9abfd91e5`.
The canonical compact JSON in `experiments/bert_agreement_exposures_v2.json`
has SHA-256
`224d683291b5b4f3a617697c8f9811351cf63bef19958fff9870c943e88fcdbc`.
It carries every v1 row ID, exact token IDs, attention mask and original status,
then records which cited evidence used that row.

Of 32 public development rows, ten appear in the cited model-forward evidence
and 22 remain reserved in this snapshot. The original pilot and replay used
four rows; two CPU family-1 attempts used eight rows, including two of those
four. The first CPU controller outcome was **incomplete**, even though its
saved journal contains 140 started and 140 completed forwards. The separate
retry was **complete** and records the same uncompressed journal SHA-256
`ff14574bf1b35058b3cff51e4752180ec589982cf3a547baba318a9591af8b67`.
Their preserved records are under `results/bert-agreement-dev-v1-20261028/`
at commit `bf3e5f94c2a723806aaafdfdc609a6340c807dc5`. A separate GPU
calibration at commit `eef5522ebdb0319cc57882e1a1cafbc5881efe6f` reused the
two already exposed farmer/farmers rows. It is cited as a development
calibration, not a new row or held-out evaluation.

The 64 public, non-blind evaluation target/donor rows are not observed in
these cited records. Exact full-token rows for all 64 are disjoint from all
32 reserved or observed development rows. This is a known-row comparison,
not a claim about unknown historical use, pretraining, semantic independence,
or any unlisted run. Case hashes and saved records do not prove faithful
execution, completeness of exposure history, or an independently witnessed
publication time. See the preserved attempt READMEs for execution and evidence
limitations; the incomplete attempt is not reclassified as complete.

`looking_glass/bert_agreement_exposures_v2.py` reconstructs the overlay and
checks its pinned predecessor and exact bytes without loading a model or
tokenizer. `verify_snapshot_bytes` reports the cited status counts and exact
row overlap; it does not authorize execution. A future run must bind its
current reviewed status snapshot and perform its own model/tokenizer,
protocol, resource and custody checks.
