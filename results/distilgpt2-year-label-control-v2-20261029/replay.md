# Model-free replay of the completed v2 attempt

All paths below are relative to the repository root. These readers do not load
a model. They require a compatible little-endian host with preserved binary
floating-point subnormals; exact floating calculations may depend on the host.
The direct-process receipt is an operator observation, not an execution or
descendant-process attestation.

For the broader receipt, source and prior-baseline replay, make a clean checkout
of source commit `b256a6846ad2386d0c0c306003026bbd9b2893a8` at `<source>`.
The public v1 directory must include its `supervisor.json` and all 51
receipt-bound files under `run/`. Its private original stderr is not required
and is not verified. The three files `year_label_v2_saved_replay.py`,
`explicit_year_saved_replay.py` and `year_label_v2_numeric_oracle.py` must be
in the same directory. The source file map is verified by content, not by Git
ancestry. From a repository checkout with both public result packages:

```sh
python -B results/distilgpt2-year-label-control-v2-20261029/year_label_v2_saved_replay.py \
  --attempt results/distilgpt2-year-label-control-v2-20261029 \
  --cases experiments/distilgpt2_year_label_control_v1.json \
  --source <source> \
  --prior-attempt results/distilgpt2-year-label-control-v1-20261029 \
  --pre-run-sha256 cf0e84371a7e516861006ab26aaab42c3d5c7ba8f8a514cfee609cb70e863abe \
  --receipt-sha256 8e41226e78793580e09985232e8dd68415ac54c5863c4157189e08835d387978
```

The second independent reader checks the closed run's records, scores,
certificates, landings and decision, using an external journal pin. It does
not independently attest the process, source or prior publication:

```sh
python -B results/distilgpt2-year-label-control-v2-20261029/year_label_v2_saved_independent_audit.py \
  --run results/distilgpt2-year-label-control-v2-20261029/run \
  --cases experiments/distilgpt2_year_label_control_v1.json \
  --journal-sha256 0c5132426cf1e855c93fa9f3fd4828f6c734d971522efa3a3f84ae354ff24f75
```

The separately saved cross-run reader compares v1 and v2 raw baseline segments
without making their equality a gate:

```sh
python -B results/distilgpt2-year-label-control-v2-20261029/year_label_v2_cross_run_independent.py \
  --v1-run results/distilgpt2-year-label-control-v1-20261029/run \
  --v2-run results/distilgpt2-year-label-control-v2-20261029/run \
  --cases experiments/distilgpt2_year_label_control_v1.json \
  --v2-journal-sha256 0c5132426cf1e855c93fa9f3fd4828f6c734d971522efa3a3f84ae354ff24f75 \
  --v2-result-sha256 a46dc2205169cbfdc313fa47096a70245cc074bec8dba7f19949f5ed0e29027c
```

The separate reports retain the original audit outputs. The readers are
separately written; the second adapts its author's earlier v1 replay. Their
reconstructions verify retained bytes and arithmetic, not a second model run.
