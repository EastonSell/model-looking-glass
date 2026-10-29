# Disclosed-site DistilGPT2 CPU reference check

This opt-in development check compares the ordered intervention executor with
independent plain PyTorch hooks on the pinned `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`. It uses the already-exposed public
two-token row `[15496, 995]` and a disclosed injection at MLP layer 2. It is a
mechanics check, not a blind discovery, a held-out evaluation, an explanation
of model reasoning, or a GPU benchmark.

The test loads the seven hash-verified assets offline into one CPU model,
uses an all-one attention mask, eager float32 evaluation with no cache,
deterministic algorithms, one CPU thread, preserved denormals and fixed CPU
seed `20261028`. It checks the loaded model before and after using the
verifier's synthetic source fixture; that fixture is not a full installed
source/environment attestation. It hides CUDA at interpreter startup and
guards selected CUDA initialization/device/RNG and network entry points
after the dependency imports; a passing check is limited to those guards.

Plain hooks and the executor separately perform clean, injected, same-site
clean-to-injected patch and same-site reverse-patch forwards. The check compares
all 13 selected boundary vectors and the complete final-token vocabulary
logits under the fixed `1e-5` absolute/relative tolerance, validates each trace
coordinate, and compares private add/patch full-output witnesses byte for
byte. It checks same-site clean and reverse logit restoration and hook/model
cleanup. These are four reference and four executor forwards on one input.

Run from the repository with the pinned checkpoint already in an explicit
local Hugging Face hub cache:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:LOOKING_GLASS_TEST_CACHE = '<absolute workspace cache>/hub'
$env:LOOKING_GLASS_REFERENCE_SUMMARY = '1'
python -m unittest tests.test_number_injection_pretrained_reference -v
```

If the opt-in cache variable is absent, the default test skips; once supplied,
missing or corrupt assets fail. The optional summary contains only the pinned
revision, weight/input hashes, disclosed site, forward counts, selected guard
count and aggregate maximum discrepancies. Save a separate source/runtime
record for each attempt, including failures, before making a claim about a
specific run. Aggregate agreement does not imply that unrelated model
capabilities were preserved.
