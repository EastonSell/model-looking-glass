# Independent FLAN-T5 dose-response synthetic review

These scripts exercise a frozen implementation using **fabricated CPU inputs and a fabricated model**. They do not load a pretrained checkpoint, run study examples through a real model, establish execution provenance, or replace the independent saved-byte audit.

`decision-fixtures.json` contains hand-labelled positive, weak, reversed, nonmonotone, matched-control, exact-boundary and unevaluable aggregate cases. `independent_runner_oracle.py` tests those cases, an additional binary64 rounding boundary, 100 independently calculated dose sets, exact float32 replacement bytes, and ineligible/unmatchable doses. Supply the exact implementation checkout, commit, runner and helper SHA-256 values with `--checkout`, `--commit`, `--source-sha256` and `--helper-sha256`.

`no_forward_preflight.py` separately checks the committed study source, pinned seven-file asset manifest, fixed case digest and actual tokenizer alignment without loading a model or running a forward. It takes `--checkout`, `--assets` and `--commit`.

`full_fake_oracle.py` runs the actual study loop with patched synthetic tokenizer/model loaders and fake source/publication/asset bindings. Supply a fresh output directory, the implementation checkout and its exact commit and runner hash. With `--mixed`, the fake produces baseline-unsupported, negative and positive cells and an inconclusive aggregate; without it, the fake produces all positive cells. These are engineered test cases, not research findings.

Use a fresh process with CUDA hidden, `-B`, an empty bytecode prefix and a tested CPU environment. The actual implementation checkout and the fake output directory must remain distinct and quiescent. An example invocation structure is:

```text
python -I -B -X pycache_prefix=<fresh-empty> independent_runner_oracle.py --checkout <source-checkout> --commit <commit> --source-sha256 <sha> --helper-sha256 <sha>
python -I -B -X pycache_prefix=<fresh-empty> full_fake_oracle.py --checkout <source-checkout> --commit <commit> --source-sha256 <sha> --output <new-output-directory> --mixed
```

The scripts check the relevant working source against committed Git bytes before execution and verify it did not change during the check. Results from the synthetic cases should be reported as component tests only.
