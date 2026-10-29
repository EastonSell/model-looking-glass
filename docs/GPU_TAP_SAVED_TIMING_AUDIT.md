# Independent saved timing audit

`tools/audit_tap_repeated_saved.py` reads one closed provisional session and
recomputes its saved timing arithmetic on CPU. Its default remains the v1.2
reader. Pass `--protocol gpu-tap-latency-repeated-v1.3` explicitly to audit the
separate v1.3 session. Each invocation accepts one session and returns no
multi-session aggregate; it never pools v1.2 data with v1.3 data.

For v1.3 the auditor requires the frozen execution commit
`0c14778447ed6dcbc4890984641370659a09ffef`, the pinned v1.3 protocol source,
and consistent protocol identities in the candidate, public result,
preflight and captured traces. It replays the fixed schedule and the 36 paired
measurements, and checks the saved start, finish and cleanup barriers against
the recorded preflight interval. The v1.3 start grant is bounded to one second,
the gate to 250 ms, and the initial GPU and process observations to two-second
freshness. The older v1.2 calculations and default interface remain unchanged.

The arguments are `--run-dir`, `--public-result`, `--checkout`, `--git`, and
`--controller-exit`; the last is an externally supplied observation. A v1.3
call also supplies the explicit `--protocol` above. These files may contain
private process or device identifiers, so the caller must retain them under
the study's custody rules. The auditor emits a path-free summary and performs
no GPU work.

An incomplete attempt stays incomplete. If version or source identity cannot
be established from its partial records, the v1.3 output says so; a known
contradictory identity is rejected. The reader does not infer a timing result
from partial rows. Even for a completed record, a pass checks saved bytes and
arithmetic; it does not authenticate the source chronology, prove that the
controller or child exited cleanly, establish GPU isolation or verify the
truth of the activity screen. Those observations require independent review.

The v1.2 baseline was the independently reviewed local saved-evidence reader
with SHA-256 `01ec3f448032f7ac569c71eb856f65d82b7c21433345c9beadad263c525803f0`.
The v1.3 extension was developed while the session was underway and finalized
and reviewed after the controller's observed exit. It is a saved-record audit,
not a preregistered benchmark result.

The CPU fixture contains the exact protocol-module bytes from execution commit
`0c14778447ed6dcbc4890984641370659a09ffef` (SHA-256
`86b475a31d60a2158ee0de3ce0f4fe65068443335927bc0307d0daa8c1bf7503`).
Tests normalize checkout line endings and check this digest before using that
frozen module as the separate synthetic-summary oracle, so they do not depend
on the integration branch's current protocol revision or launch a model.
