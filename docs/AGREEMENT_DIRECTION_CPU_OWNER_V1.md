# Agreement-direction CPU owner

This entry implements the fixed public-input protocol in
`AGREEMENT_DIRECTION_STUDY_V1.md`. Its 784-forward worker emits only a
`complete_candidate`. A separately retained observed zero exit and an
independent saved-byte scientific audit are both needed to interpret a run.
The full sequence is checked for finiteness by the owner, but only final
logits and specified hook witnesses are retained on successful forwards.

The trusted operator must publish and independently read back the exact clean
source commit, prelaunch record, cases and latest reviewed exposure inventory
before any forward, refresh both scoped custodian observations, and ensure
the host runs only one attempt. The entry checks the fixed reservation and the
explicitly reviewed v23 module and reconstructs its canonical payload and
the v20-to-v23 predecessor chain; it cannot establish that v23 remains current.
The operator checks later reviewed exposure changes before launch. These checks
and the supervised exit are cooperative observations, not authenticated
execution, complete historical exposure or whole-process-tree cleanup. The
entry's exclusive claim prevents reuse of the same prelaunch hash under one
output parent; the operator is responsible for the wider one-shot rule and any
lost or uncertain process handle is incomplete.

The operator supplies a SHA-pinned prelaunch JSON and canonical latest
inventory, plus the original natural-comparison parent observation. The entry
checks its original hash and public comparison, retaining a sanitized
projection rather than copying the path- and PID-bearing original. The
prelaunch record separately states the refreshed natural and injection
custodian observations. A typical command is:

```text
BASE_PYTHON -I -S -B tools/agreement_direction_cpu_v1.py \
  --prelaunch ABSOLUTE_RECORD --prelaunch-sha256 SHA256 \
  --latest-exposure ABSOLUTE_CANONICAL_INVENTORY \
  --natural-parent-observation ABSOLUTE_ORIGINAL_OBSERVATION \
  --case-file ABSOLUTE_CASE_FILE --direction-dir ABSOLUTE_DIRECTION_DIRECTORY \
  --cache-dir ABSOLUTE_OFFLINE_HUB_CACHE --output-dir NEW_ABSOLUTE_DIRECTORY \
  --study-root ABSOLUTE_CPU_CUSTODY_ROOT --venv ABSOLUTE_VENV_ROOT \
  --child-custody NEW_ABSOLUTE_CHILD_RECORD
```

The reviewed CPU supervisor supplies an empty `CUDA_VISIBLE_DEVICES` mask
before interpreter startup. The child captures bounded local source bytes,
claims the supervised CPU lease before any Torch or Transformers import,
sets the mask to literal `-1`, and compares the captured sources to the clean
Git tree before model work. The parent also checks source independently before
the grant. Start with `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`; leave
`MKL_CBWR` and `ATEN_CPU_CAPABILITY` unset. The prelaunch record pins Python,
Torch, Transformers, tokenizers, NumPy and Hub versions. The model is loaded
only from seven verified offline checkpoint assets, and all 40 texts and four
continuation tokenizations are checked before the first forward. The owner
requires CPU float32, six blocks, width 768, vocabulary 50,257, eager
attention, disabled cache, one intra-op and one inter-op thread, deterministic
algorithms, and a calling-thread denormal probe. Runtime and model state are
checked around the forwards; this does not attest native library identity or
every worker thread's floating-point state.

`--validate-only` claims the supervised CPU session and performs the same
source, case, direction, inventory, custody-observation, environment and
free-disk checks without loading Torch, a tokenizer or a model and without
creating a study claim. Both modes write an exclusive child-custody record
with the run ID and PID for comparison with the reaped supervisor result. The
session and lease remain held through process exit; a clean attestation is
written only after successful validation or completed worker/model teardown.
The normal attempt saves
the prelaunch record, latest public inventory, natural public receipt and
input bundle, sanitized parent projection, model admission, journal and
candidate result. Errors retain available bounded partial evidence and do not
authorize a retry. A malformed oversized returned tensor can leave a labeled
bounded prefix rather than its entire content. The 512 MiB unique-tensor and
64 MiB journal limits, bounded metadata and retained public inventory leave
space within the 1 GiB physical-attempt cap. Sixteen MiB is reserved for the
claim, final metadata and the separately retained direct-operator receipts;
the operator must verify those external bytes and account publication copies
separately. The entry checks at least 2 GiB of free disk before claiming.
These are disk-accounting limits, not RAM or GPU-memory limits.
