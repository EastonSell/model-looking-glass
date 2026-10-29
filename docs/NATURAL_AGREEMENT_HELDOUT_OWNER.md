# Natural agreement held-out owner

`tools/natural_agreement_heldout_owner.py` supervises one already committed
private phase. The separate custodian owns the study ledger, source and runtime
publication, split preparation, commitment publication and subsequent reveal.
No nonce, token row or private example is placed in the command line or the
path-only configuration. Output, logs, diagnostics and receipts remain private.

The trusted invoking process first checks this file's bytes against the
published operational binding and starts the direct base interpreter with
`-I -S -B`, a fresh nonexistent absolute `-X pycache_prefix`, CPU-only
environment, and `MLG_NATURAL_SITE` naming the pinned virtualenv's site-packages.
Both `MKL_CBWR` and `ATEN_CPU_CAPABILITY` must be absent before any Torch
import. A source-checking bootstrap may use the following code as the `-c`
argument, followed by the absolute owner path, its published SHA-256, and the
absolute private child-configuration path:

```python
import hashlib, os, sys
p = sys.argv.pop(1)
expected = sys.argv.pop(1)
data = open(p, 'rb').read(8388609)
if len(data) > 8388608 or hashlib.sha256(data).hexdigest() != expected:
    raise SystemExit('owner source does not match the published digest')
sys.path.append(os.environ['MLG_NATURAL_SITE'])
sys.argv[0] = p
exec(compile(data, p, 'exec', dont_inherit=True), {'__name__': '__main__', '__file__': p})
```

The parent independently authenticates the public operation and commitment,
checks the current inventory and preparation receipt, and holds source leases
over the frozen core and all operational sources. Restricted inheritable copies
of both leases are present in the gated owner from process creation until its
exit; the child also acquires its own leases. The existing Windows supervisor
assigns the child to a kill-on-close Job before granting its start token, bounds
logs, and verifies direct-child exit, reaping and Job drain. An uncertain spawn,
drain or handle close quarantines the process and protected objects; the owner
keeps that process alive, including after Ctrl+C or a broken diagnostic stream,
until it is externally stopped and reconciled. Normal Windows console interrupt
signals are ignored throughout this one-shot owner, including a healthy run;
an external stop must be followed by reconciliation. Forced termination closes OS
handles and is not proof that an uncertain child was reaped. Do not retry on
an unverified result. The protection is cooperative local process
custody, not a guarantee against native breakaway or a hostile account.

Only after confirmed cleanup does the structural callback hash the journal and
tensor inventory, compare the start/runtime and private committed row contract,
check the prescribed 608-forward event sequence and phase limits, and bind the
frozen prediction identities. It does not reconstruct causal effects, tensor
landing or numerical controls. The private `owner-terminal.json` binds the exact
supervisor receipt and records combined phase/side-file accounting after the
source leases have closed. Its `evidence_complete` status is structural and
requires the separate independent raw numerical/landing audit. If the terminal
is absent, incomplete or cleanup is unverified, preserve all files and do not
claim a successful evaluation. Observe the owner process's actual exit and
retain its receipt before publishing any sanitized projection.
