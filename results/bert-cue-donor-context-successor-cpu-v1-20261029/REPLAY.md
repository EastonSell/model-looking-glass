# Model-free replays

## Public score projection

The following needs only the included projection, manifest, result and the same-directory `review_bert_donor_context_scores.py` dependency. It recomputes selected-score decisions and checks the copied TV-summary arithmetic; it does not reconstruct full-vocabulary TV or authenticate original execution.

```sh
python -I -B review_bert_donor_context_subset.py manifest.json score-tv-subset.jsonl.gz result.json --expected-manifest-sha256 917bad800f72a9613e69d8feae3faf90ffbae5cc446bbeb3e2c8df1d03678dc9 --expected-subset-sha256 43088c7fb9763bf80df52a4459b1c1e42d37b5de120582b58b37f59eacea2467 --expected-result-sha256 3674c152b3c6e24383f191443ca50d2697baaf0c9334c22fa7766f759ace05c4 --expected-original-journal-sha256 cde003fb88e17fa4b9af3215e8e0ccbef23fb2d5df83e1ef859715b4bc5c17de --expected-started-sha256 b7b77af5eec57a5ef05e6ea6e7a4ec576844da82248a0b9d8a572050c90f5703 --expected-source b6667ad5bfcadac6e0f45d4ad6530cca1f4584ca > subset-replay.json
```

Compare the parsed output with `subset-audit.json`; shell line endings can differ. Both the subset derivation and this compact reader were written after outcome inspection.

The full journal is retained locally, not supplied for public download. Set `JOURNAL` to its closed `events.jsonl.gz` path. Set `VOCAB` to the pinned model's `vocab.txt` (SHA-256 `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`). Commands below run from this package directory. Use a non-optimized Python interpreter; no model is loaded.

```sh
python -I -B audit-full.py manifest.json "$JOURNAL" result.json --started started.json --expected-started-sha256 b7b77af5eec57a5ef05e6ea6e7a4ec576844da82248a0b9d8a572050c90f5703 --expected-result-sha256 3674c152b3c6e24383f191443ca50d2697baaf0c9334c22fa7766f759ace05c4 > full-replay.json
python -I -B review_bert_donor_context_scores.py manifest.json "$JOURNAL" result.json --started started.json --expected-source b6667ad5bfcadac6e0f45d4ad6530cca1f4584ca --expected-started-sha256 b7b77af5eec57a5ef05e6ea6e7a4ec576844da82248a0b9d8a572050c90f5703 --expected-result-sha256 3674c152b3c6e24383f191443ca50d2697baaf0c9334c22fa7766f759ace05c4 > score-replay.json
python -I -B geometry.py manifest.json "$JOURNAL" result.json geometry-replay.json --expected-manifest-sha256 917bad800f72a9613e69d8feae3faf90ffbae5cc446bbeb3e2c8df1d03678dc9 --expected-journal-sha256 cde003fb88e17fa4b9af3215e8e0ccbef23fb2d5df83e1ef859715b4bc5c17de --expected-result-sha256 3674c152b3c6e24383f191443ca50d2697baaf0c9334c22fa7766f759ace05c4 --expected-source b6667ad5bfcadac6e0f45d4ad6530cca1f4584ca
python -I -B audit-top-tokens.py manifest.json "$JOURNAL" result.json "$VOCAB" top-replay --expected-manifest-sha256 917bad800f72a9613e69d8feae3faf90ffbae5cc446bbeb3e2c8df1d03678dc9 --expected-journal-sha256 cde003fb88e17fa4b9af3215e8e0ccbef23fb2d5df83e1ef859715b4bc5c17de --expected-result-sha256 3674c152b3c6e24383f191443ca50d2697baaf0c9334c22fa7766f759ace05c4 --expected-vocab-sha256 07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3
python -I -B make-tables.py result.json geometry.json table-replay --expected-result-sha256 3674c152b3c6e24383f191443ca50d2697baaf0c9334c22fa7766f759ace05c4 --expected-geometry-sha256 3da420387a023a0977c8859bddacb3fb29f72c00c3216ed43d897a77cf8a3628
```

The geometry and table outputs must not already exist. Compare the first two parsed JSON outputs with `full-witness.json` and `scores-tv.json`; shell line endings can differ. The full-witness reader checks saved full-logit controls, same-forward landing, dose, baseline and decisions. The separate score reader recomputes full/excluded TV, masses and score decisions; it does not independently verify landing. Geometry records matched clean-vector distances/cosines and stored dose fields; it relies on the full witness for landing. The top-token scan and tables were written after outcome inspection and are descriptive. No replay attests actual execution or unknown pretraining exposure.
