# Real-checkpoint CPU verifier challenge v2

This opt-in correctness check loads the pinned offline DistilGPT2 checkpoint
once on CPU and challenges the loaded-model verifier with controlled changes.
It uses `distilbert/distilgpt2` and tokenizer at immutable revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`. The public fixed input is
`input_ids=[[15496,995]]`, whose canonical bootstrap case hash is
`a137759c4f2a40c4da15d1a4625086e160edc75f161748db91b07d83dae751e4`.
This is an exposed development case, not a held-out behavioral evaluation.

The test creates a synthetic `SourcePreflight` containing the local pinned
asset paths solely to exercise `verify_loaded_model`. This fixture is not a
full source, package, hardware, or supervisor attestation and cannot be used
to authorize a GPU bootstrap. The loader checks all pinned assets offline;
the verifier rechecks the pinned config and weight file as part of its own
contract.

The original model must pass. Individual modifications must fail: one
attention parameter value, an attention mask value, its masked-bias buffer,
vocabulary configuration, training mode, attention scaling, equal-valued
noncontiguous float weight and bool mask, nonzero parameter storage offset,
equal-valued shared mask storage, an untied output embedding and a distinct
output-embedding Parameter sharing the original storage, and local and
process-global forward hooks. The test restores each change, rechecks a clean
model between groups, and requires the final logits to match the original
bitwise. It checks that the parameter, mask-value and attention-scaling
mutations change logits for this case; these observations do not establish a
general behavioral explanation.

A separate direct base-interpreter child makes one explicit loader call, runs
with CUDA hidden, and has selected
CUDA initialization, runtime-query and seeding entry points guarded after
PyTorch import; socket connections and DNS are guarded too. The test uses the
CPU default generator, one intra-op thread, eager evaluation and inference
mode. The selected guards do not cover PyTorch's initial import or every native
CUDA path. No GPU profile, performance measurement or general safety guarantee
is produced. No logits, prompts, credentials, child cache or scratch paths are
printed; the unittest runner may show the test-file path in its own traceback.

From a repository root containing the reviewed verifier, install the
`transformer` extra and set an existing scratch directory and the pinned cache:

```powershell
$env:LOOKING_GLASS_TEST_CACHE = (Resolve-Path 'work/hf-cache/hub').Path
$env:LOOKING_GLASS_TEST_SCRATCH = (Resolve-Path 'work').Path
$env:LOOKING_GLASS_VERIFIER_CPU_SUMMARY = '1'
python -m unittest discover -s tests -p test_pretrained_verifier_cpu.py -v
```

The test skips only when the cache variable is absent. After opt-in, missing
assets, an unexpected verifier result, a change to the verifier file during the
run, or a guarded call fail the check. The optional summary reports source
hash, case hash, named rejection outcomes and aggregate logit differences only.

For a saved aggregate record, run the versioned collector from a clean checkout
and choose a new output path outside it:

```powershell
python tools/pretrained_verifier_cpu_evidence.py --repository . `
  --cache $env:LOOKING_GLASS_TEST_CACHE --scratch $env:LOOKING_GLASS_TEST_SCRATCH `
  --output ../pretrained-verifier-cpu-result.json --purpose correctness
```

The collector verifies its own bytes and selected source files against their
Git blobs, isolates the unittest parent with a fresh bytecode-cache prefix, and
stores only hashes of raw process output. It checks the selected file bytes
and HEAD again after the run and records whether Git reports a clean status;
this is not an attestation of unlisted imported modules or of index flags that
may affect Git status. It accepts only the defined aggregate summary fields and
finite, positive differences. A standard-library regression proves
that this parent ignores an otherwise valid poisoned test bytecode cache on
CPython. A launched-test failure is recorded if the output can be written;
early setup, interruption or write failures can leave no complete record.
`controlled-missing-cache` requires a path absent before execution and exercises
child startup failure; it is not a loader-level missing-asset test.

The [saved CPU challenge records](PRETRAINED_VERIFIER_CPU_EVIDENCE.md) preserve
both the earlier checks and the later tied-parameter failure and fix.
