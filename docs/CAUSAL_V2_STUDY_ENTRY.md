# Synthetic v2 owner composition

`looking_glass.causal_v2_study_entry` composes the existing runtime preflight,
fixed exclusion inventories, one-shot custodian, study executor, and scorer.
It is not a standalone command or an authorization to prepare private cases.
The required guarded owner must first verify each supplied source against a
committed revision, keep the source and empty bytecode-cache protections for
the full process lifetime, and pass its actual source-only finder's inventory
callback. The entry rehashes the bounded source inventory and checks the fixed
pilot, development and public v1 exclusions before private preparation.
Immediately before preparation it exclusively creates and synchronizes a
bounded sibling `<output>.attempt.json` marker. A partial or complete marker
blocks reuse of that output path even if a crash occurs before the custodian
creates its evidence directory. The marker is charged to the public disk
reserve; it is a local operational record, not an independent timestamp.

The entry requires the custodian's phase-inventory callback before it prepares
any v2 inputs. It measures six CPU model-state hashes without a forward, checks
the CPU backend snapshot on both sides of that preflight, and gives the same
settings to the executor and pre-manifest. The custodian owns publication,
the discovery freeze and the commitment before any held-out forward. The
fixed scorer reads the saved evidence; the exposure result is `unknown` unless
an independently supported monitor is added. Local source checks, same-account
custody, and self-reported clocks do not establish secrecy, trusted timing,
or cryptographic attestation of interpreter and native-library bytes.

`scripts/run_synthetic_v2.py --rehearsal` uses the direct base interpreter,
guarded start frame, leased committed sources, private empty cache, Windows job,
and source-only bootstrap to check this owner with the real six CPU model
initializations. It generates no v2 inputs, performs zero model forwards and
does not create the requested evidence directory. `--study` is disabled in
both the parent and bootstrap pending independent review of the whole launch
path and phase-inventory integration. Supply a fresh absolute output path
outside the checkout, an existing separate work directory, the venv root and
its explicit dependency directory, and set `CUDA_VISIBLE_DEVICES=-1`.

The owner-composition tests use fabricated source trees, a stubbed custodian
and a stubbed preparation function. The Windows launcher test additionally
runs the real no-forward rehearsal from a temporary committed checkout whose
root model forward raises an error if called. None
of these tests generates synthetic v2 inputs or schedules, or runs a model
forward. The parent verifies the source file it is running on disk, but that
does not cryptographically attest already-executing parent bytes. A failed or
interrupted attempt must be retained as incomplete and never resumed.
