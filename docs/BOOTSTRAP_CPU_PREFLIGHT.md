# Bootstrap pretrained CPU preflight v1

This opt-in correctness test checks the pretrained checkpoint and the fixed
two-token case proposed for the GPU resource-profile bootstrap. It runs a
separate, directly invoked base Python interpreter with CUDA hidden; it does
not run a GPU owner, collect a GPU resource profile, measure throughput, or
explain a model behavior. See the bootstrap protocol for the distinct GPU
ownership and measurement requirements.

The test requires a clean Git checkout and a local cache containing all seven
files for `distilbert/distilgpt2` at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`. It independently checks each file's
size and SHA-256, and makes only offline Hub lookups. A disposable partial
cache checks that a missing asset and a same-size tampered config are rejected
before model deserialization. Network connection and DNS entry points are
guarded during these checks. The original cache is never modified by the
negative cases.

For the verified CPU model, the test checks the configuration, eager evaluation
mode, disabled cache, float32 CPU parameters and tied input/output embeddings.
The input is exactly `input_ids=[[15496,995]]`, shape `[1,2]`, int64 and 16
bytes; its canonical case hash is
`a137759c4f2a40c4da15d1a4625086e160edc75f161748db91b07d83dae751e4`.
The output must be finite float32 logits of shape `[1,2,50257]`, exactly 402,056
bytes and below the 512 KiB tensor limit. The test also exercises the production
IPC result and frame encoders with synthetic envelope metadata and an in-memory
sink, checks that the complete resulting frame fits the 4 MiB wire limit, and
verifies bit-exact float32 recovery of every encoded logit.
This frame check does not exercise a live network connection or attest to
the size of every possible service response.

The parent invokes the actual base interpreter with `-S`, disables user-site
loading and bytecode writes, supplies an initially empty private bytecode-cache
directory, and checks that the cache stays empty. The child verifies its
interpreter and the origins of imported project and selected package modules;
imported project bytes and the executed test file must match their Git blobs at
the unchanged clean source revision. The test does not hash every distribution,
standard-library or native dependency and is not a substitute for the
bootstrap's full source manifest or a trusted launcher.

After importing PyTorch, the test requires its CUDA context to be uninitialized
and guards selected CUDA initialization, runtime device-query, synchronization
and seed entry points during checkpoint loading and CPU inference. It uses the
CPU default generator directly. The initial PyTorch import is not instrumented,
and these selected Python-level guards do not prove that every possible native
CUDA entry point was intercepted. The test cannot authorize or certify a
subsequent GPU operation.

Install the `transformer` extra and set the following PowerShell variables to
run the test. It skips only when the cache variable is absent; an opted-in
missing or corrupt cache, unexpected import origin, guard call, or mismatch is
a failure. The scratch directory must already exist and should be private.

```powershell
$env:LOOKING_GLASS_TEST_CACHE = (Resolve-Path 'work/hf-cache/hub').Path
$env:LOOKING_GLASS_TEST_SCRATCH = (Resolve-Path 'work').Path
$env:LOOKING_GLASS_BOOTSTRAP_CPU_SUMMARY = '1'
python -m unittest discover -s tests -p test_bootstrap_pretrained_cpu.py -v
```

The optional summary contains only aggregate sizes, hashes, counts and the Git
revision. Failures report a bounded phase and exception type; full logits,
credentials, child cache or scratch paths, and child tracebacks are not emitted.
The unittest runner may show the local test-file path in its own traceback.
