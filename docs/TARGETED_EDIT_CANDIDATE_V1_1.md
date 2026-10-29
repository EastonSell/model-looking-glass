# Targeted-edit candidate maintenance entry point v1.1

`looking_glass.targeted_edit_candidate_v1_1.derive_candidate` is a separate,
maintenance-only entry point for deriving the two updates produced by the v1
candidate calculation. It is not a launcher, a study protocol, or a substitute
for the frozen v1 source and its published manifest and results. The existing
v1 and v2 entry points have not been changed to use it.

It accepts the same model, projection module, token IDs, target, competitor,
and optional expected shape as `targeted_edit_cpu_v1._candidate`. On a
successful synthetic CPU run, the candidate tensors and discovery details have
been compared with v1. The arithmetic and helper functions remain those of the
imported v1 module; this observation is not a claim about a pretrained model.

The successor records exceptions raised by its own body explicitly. It does not
use `sys.exception()` or treat an unrelated exception handled by its caller as
its own. It attempts to remove every recorded hook, clear the projection
gradient, and restore each recorded `requires_grad` flag. If an operation
raises, it retries that operation once and still reports a cleanup failure even
if the retry succeeds. A body error remains primary with the preferred cleanup
error chained, except that a fresh `KeyboardInterrupt` or `SystemExit` during
cleanup takes priority over an ordinary body error. An existing body
cancellation remains primary. When multiple cleanup operations fail, the
preferred error is reported; the helper does not promise a complete error log.

These are best-effort in-process operations, not automatic quarantine. A
trusted caller must own the model exclusively and discard the isolated model
and process after any exception or uncertain cleanup; the helper does not
prevent another caller from reusing them. An interruption or a registration
method that installs a hook but fails before returning its handle can leave an
unknown hook. Interruptions outside the protected action calls or failures
while setting up cleanup may also prevent later operations. Do not interpret a
failed call as proof of restoration. As in v1, the projection's existing
gradient is cleared rather than restored; callers must establish their own
appropriate initial state. Exception chains are private caller data.

The repository tests use fabricated CPU models. They cover output agreement,
hook and flag cleanup, ambient caller exceptions, multiple cleanup failures,
and cancellation precedence. A CPython 3.10 subprocess executes the new entry
point with dependency stubs and a synthetic body failure. With compatible
PyTorch installed, the new-helper tests can also run directly on Python 3.10;
the comparison that calls the known-incompatible historical v1 is skipped.

Before any future experiment uses this entry point, independently review and
freeze a new protocol and manifest that bind its exact source and every
imported execution dependency, verify actual import origins, and establish the
process isolation, source/runtime custody, failure reporting and rollback
contract. A v1 manifest does not bind this file, and hashes alone do not attest
execution. The separate v2 containment guard continues to govern the unchanged
v1 call until a future protocol explicitly adopts and binds this successor.
