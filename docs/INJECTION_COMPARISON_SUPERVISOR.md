# Injection comparison supervisor

`tools/injection_comparison_supervisor.py` provides direct observation for a
source-bound execution owner. It checks the published operator and its own
manifest identity, binds the path-only operator configuration, holds and
duplicates the same source and private-file protections, and launches the
direct-base operator through the reviewed outer observer. It returns only
after the operator is reaped and its Job is drained and closed, and the public
commitments and receipt agree.

The returned status is `clear` or `collision`; only `clear` may support the
separate shared-gate decision. The execution owner must load the supervisor
from its independently pinned manifest before use. If a separate process saves
the supervisor's result, its own successful exit must also be independently
observed before the saved result is used. No receipt or saved terminal alone
attests process exit or authorizes a model forward.
