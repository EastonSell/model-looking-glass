# Development-only phase-ordering exercise

`looking_glass.causal_v2_gate_dev` exercises ordering and failure handling with
fake identifiers and scalar scores. It does not generate cases or a site
schedule, execute a model, validate real v2 records, isolate secrets, or
authorize a synthetic-v2 study. Its callback is arbitrary Python. The object
cannot stop code outside the object from executing a model, reading private
data or modifying files, and is not a security boundary against other processes
using the same account.

The exercise creates a new bounded registry run with sorted `dev-cell-NN` and
`dev-case-NN` identifiers. A single owner process and thread publishes one
fake discovery receipt per declared cell, then freezes the complete receipt
set. It publishes the exact fake per-case prediction bytes and their salted
commitment before publishing a one-shot evaluation claim and calling the fake
callback. It verifies the same frozen bytes before and after the callback and
does not reopen an attempt. These fake predictions are public in this
exercise; a real v2 custodian must retain concrete held-out predictions
privately and publish only the commitment until the protocol allows reveal.

The owner must discard the object after **any** externally delivered
interruption, including an interruption delivered before the method's
failure handler starts. Reentrant calls and calls from a different process or
thread are rejected. The exercise assumes cooperative single-threaded use: a
concurrent foreign call is rejected, but an active publication may already
have passed its final violation check. Do not infer that every concurrent
violation invalidates an in-flight publication. A real runner must separately
enforce its lifecycle and fail-stop requirements.

The fake registry preserves published evidence when a normal callback or
publication error occurs. If a terminal publication or lock cleanup raises
after success, the gate checks the exact intended outcome and artifact hashes
before reporting the committed success with a cleanup error. A new
`KeyboardInterrupt` or `SystemExit` during cleanup is surfaced, with the
earlier error chained where applicable; an unverified terminal state remains
unknown. A process crash can leave an incomplete attempt. Registry publication
and local hashes do not establish a tamper-proof time, hostile-process
isolation, or durable survival of every operating-system or power failure.

The focused tests use only development identifiers and no model forwards:

```console
python -m unittest discover -s tests -p test_causal_v2_gate_dev.py
```

The production study still needs its own reviewed custodian, full record and
split validation, private storage, frozen prediction and commitment binding,
durable evidence accounting, actual intervention controls and execution
gates. This prototype does not substitute for any of those components.
