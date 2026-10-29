# Saved-evidence replay

Run these commands from the directory containing the compact files, replacing `FULL_JOURNAL` with the local complete journal's path. They do not load the model. The output is JSON; compare it to the bundled reports. The complete journal is not part of the public repository copy.

The frozen full-witness reader checks saved source/runtime identities, schedule, clean and self/repeat/final-cue controls, intervention vectors, landing, dose and the signed decisions:

```text
python -I -B audit_bert_cue_context_signed_full_saved.py manifest.json FULL_JOURNAL result.json --started started.json --expected-started-sha256 c69862836f26d8a23baf25fdd65dc3690368aff01106acc418ca37b1eebdef60 --expected-result-sha256 0a788dfb0655ac8da55d94b1154fa240860850bf9464928b391d8cfa3327f62a
```

The independently developed score reader recomputes all selected scores, full and excluded total variation, probability masses and signed outcomes from full mask logits. It checks source/start/result consistency but delegates geometry and landing to the full-witness reader:

```text
python -I -B review_bert_cue_context_signed_scores.py manifest.json FULL_JOURNAL result.json --started started.json --expected-source 91f9960e419cf20df9df156a6ebc1980d1b6a864 --expected-started-sha256 c69862836f26d8a23baf25fdd65dc3690368aff01106acc418ca37b1eebdef60 --expected-result-sha256 0a788dfb0655ac8da55d94b1154fa240860850bf9464928b391d8cfa3327f62a
```

The portable projection replay needs no complete journal. It recomputes selected-score decisions and summary arithmetic; it cannot independently reconstruct full TV or establish full-vector, dose, landing or execution claims:

```text
python -I -B review_bert_cue_context_signed_subset.py manifest.json bert-cue-context-signed-score-tv-subset-91f9960e.jsonl.gz result.json --expected-manifest-sha256 843649c3f105e26185d96f31d769b0585d219bb7a45ba445cf2c2548598927df --expected-subset-sha256 7ab40cf95e3689d8c331cd5cc7ba26173acf42d4c63541ce8612de356d92d5ac --expected-result-sha256 0a788dfb0655ac8da55d94b1154fa240860850bf9464928b391d8cfa3327f62a --expected-original-journal-sha256 a6c586dd9a8f125f60aa4003a8d8cbeac9cfae0f7862332412b5dde4a6417909 --expected-source 91f9960e419cf20df9df156a6ebc1980d1b6a864 --expected-started-sha256 c69862836f26d8a23baf25fdd65dc3690368aff01106acc418ca37b1eebdef60
```

With the complete journal, the operator-derived projection can be regenerated at a **new** output path:

```text
python -I -B derive_bert_cue_context_signed_subset.py manifest.json FULL_JOURNAL result.json NEW_SUBSET --expected-manifest-sha256 843649c3f105e26185d96f31d769b0585d219bb7a45ba445cf2c2548598927df --expected-journal-sha256 a6c586dd9a8f125f60aa4003a8d8cbeac9cfae0f7862332412b5dde4a6417909 --expected-result-sha256 0a788dfb0655ac8da55d94b1154fa240860850bf9464928b391d8cfa3327f62a --started started.json --expected-started-sha256 c69862836f26d8a23baf25fdd65dc3690368aff01106acc418ca37b1eebdef60 --expected-source 91f9960e419cf20df9df156a6ebc1980d1b6a864
```

All source/record identities and expected hashes establish saved-byte consistency within the stated scope; none of these commands independently proves execution or pretraining nonexposure.
