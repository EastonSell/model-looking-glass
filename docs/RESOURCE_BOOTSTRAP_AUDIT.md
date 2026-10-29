# Auditing private GPU resource-bootstrap v1 artifacts

The bootstrap protocol is still proposed, and this repository has no measured
resource-bootstrap result. This CPU-only reader can check a future private run;
it never initializes CUDA, modifies artifacts, or establishes a safe memory
bound. The source protocol and its writer must be integrated and independently
reviewed before a run is made.

```sh
python -m looking_glass.resource_bootstrap_audit /path/to/private-run \
  --pins /separate/trusted/profile-pins.json \
  --repository /path/to/clean-recorded-checkout \
  --environment-root /path/to/recorded-environment \
  --site-packages /path/to/recorded-environment/site-packages \
  --python-executable /path/to/recorded-base-python \
  --asset-directory /path/to/pinned-checkpoint-snapshot
```

All arguments after the run directory are optional. Pins must be stored outside
the run directory as canonical JSON with exactly `load`, `cold`, and `warm`
keys and independently approved 64-character profile IDs. If pins are omitted,
the tool can recompute candidate IDs but reports `pins_verified: false`; this
is not approval. It exits 0 only for a structurally successful run with matching
pins, 2 for an incomplete, failed, cancelled, quarantined or unpinned run, and
1 for an invalid artifact. Its JSON summary omits physical GPU UUIDs, local
paths, and failure text. The CLI emits no traceback on validation failure; keep
API tracebacks and the original private evidence private.

For a success terminal the reader checks the exact file inventory, canonical
JSON and five ordered stage/seed and admission records; reconstructs the three
evidence documents, model/workload/profile IDs and arithmetic from each raw
sample; and checks the fixed case, limits, checkpoint hashes, runtime settings,
allocator readback, headroom arithmetic, output checks, source manifest and
profile margins. It binds child-owner identity to the reported supervisor result.
A missing terminal is incomplete. Failure or
cancellation terminals never become successful profiles; partial JSON-lines
are retained and reported as partial. The terminal is only an attestation,
and its inventory detects altered or missing bytes only while that terminal is
itself protected. Matching profile IDs are not authentication.

The parent can call `audit_candidate(directory, terminal_dict, ...)` after the
child has exited and before atomically publishing `terminal.json`. It validates
the prospective terminal and directory without writing either, and labels its
report `publication: candidate`; it rejects an existing terminal. The parent
must prevent files changing between this check and publication. The saved-run
reader labels an already published terminal separately.

The optional local checks provide additional, distinct evidence:

- `--repository` requires a clean checkout at the recorded commit/tree,
  enumerates every tracked `looking_glass/` file and the protocol file, and
  hashes each listed file's actual bytes.
- `--environment-root` and `--site-packages` are supplied together. The latter
  is an independently selected site directory that must appear in the recorded
  ordered import paths. The tool enumerates its installed distribution metadata,
  checks for unlisted or duplicate distributions, compares every declared file
  list with its installed `RECORD`, and hashes the listed bytes. The environment
  root is the common root used by the private manifest, not the site directory.
  Missing/outside files, unsupported metadata formats and links fail.
- `--python-executable` checks the recorded executable's size and bytes.
- `--asset-directory` hashes the seven fixed checkpoint assets. Snapshot links
  to content-addressed blobs are allowed; the bytes reached are checked.

The report labels each omitted local check `not_checked`. Matching a checkout,
packages, executable or asset directory does not prove the recorded process
actually used them. The reader cannot independently establish the runtime CUDA
flags, synchronization, origin of imported modules, device isolation, teardown,
or measurement timestamps. Allocator peaks and device-free endpoint changes
are incomplete memory observations; margins remain heuristics. Other native
dependencies and machine state are not comprehensively identified.
