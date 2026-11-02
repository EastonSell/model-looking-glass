# Replaying the saved explicit-year evidence

Both readers use only saved bytes; neither runs a model. Run from the root of
a checkout containing this result package. `SOURCE` below is a separate clean
checkout of execution source commit `4059fa97d78033b1f382c09b4b9b93d8fb4885bd`.
It must also have the three observed-main commits in its Git object history;
a shallow checkout may need additional history fetched. Do not silently use a
later result-package checkout as the execution source.

For a POSIX shell, set `SOURCE` to that checkout and `RESULT` to
`results/distilgpt2-explicit-year-direction-v1-20261029`. The primary reader
prints the reconstructed report to standard output:

```sh
python -B "$RESULT/audits/independent-saved-replay.py" \
  --run "$RESULT/run" --cases "$SOURCE/experiments/distilgpt2_explicit_year_direction_v1.json" \
  --receipt-sha256 a792c872f2a5e6776daea6bf780722b023e94387d305d75e4fe2e90e42f1a926 \
  --pre-run-sha256 676dd2ac8b2ceb2c2311d38b01f4d9e8a348c5b29430662926757677c9aded49 \
  --launch-observation-sha256 a2d7916ccb1e604fc392b1133b258aafb8fd1ff629e9235e2b4ba475736ab641 \
  --launch-main fc6496e89b9130805f484555378f29b74dd7ba76 \
  --launch-update-sha256 6f45668e730b33e04e21721a52c719643355717d155a1a45c74887af7c10717b \
  --launch-update-main 6f78a42ed66615d2ac8d5ad09bfeeae7d7ce68a3 \
  --launch-start-sha256 399dd79bf9467ec38a283e4380a8a1e4aa064ac4d7f67dcd4ad2eca57356b744 \
  --launch-start-main ce813d7ad41b87a6bac24db9e0f565877ba2733a \
  --source-revision 4059fa97d78033b1f382c09b4b9b93d8fb4885bd \
  --source-tree bff359603ba5ba3ae76b8abd7c1af273ec259b35 --source-root "$SOURCE" \
  --reservation-version distilgpt2-development-exposures-v21 \
  --reservation-sha256 a391c5f3a8ca33ad32e9378417b424338b12ea55257979977d2afd5a0c7f90ae \
  --current-exposure-version distilgpt2-development-exposures-v21 \
  --current-exposure-sha256 a391c5f3a8ca33ad32e9378417b424338b12ea55257979977d2afd5a0c7f90ae \
  > explicit-year-replay.json
```

The separately written control reader independently recomputes saved
scores, gates, doses and landing bits. It checks the manifest's declared
source identity, but does not independently verify the source checkout or
direct-process receipt:

```sh
python -B "$RESULT/audits/independent-control-audit.py" \
  --run "$RESULT/run" --cases "$SOURCE/experiments/distilgpt2_explicit_year_direction_v1.json" \
  --terminal-sha d6ae653ca2e96864e8332bc6ad9b6fbd0c5b546570971fd4b274172ac8089bc3 \
  --source-commit 4059fa97d78033b1f382c09b4b9b93d8fb4885bd \
  > explicit-year-control-audit.json
```

The published reports preserve the original readers' output. Replays check
saved-data consistency and stated identities; they do not independently prove
execution, full process custody or all training and prior-input exposure.
