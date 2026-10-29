# Natural prediction publication handoff

This Windows-only, CPU-side handoff stages a natural discovery prediction and
checks its publication in two phases. It does not run a model, publish by
itself, create a held-out nonce, sample cases, or attest execution or secrecy.
The trusted custodian must establish independently that the discovery owner
used its already-published binding, completed and was reaped cleanly, and that
the reviewed independent auditor actually audited the final retained evidence.
Run the wrapper from a clean checkout at its separately recorded, publicly
available commit. The already-bound analysis checkout must also be clean at
the exact discovery source commit; the wrapper executes its transitive project
Python sources directly from hash-checked text while those files are protected
against ordinary concurrent writes. Native dependencies, OS access controls and
same-account adversaries remain outside this guarantee.

Before staging, retain the successful discovery's untouched `owner/attempt`
directory, `owner/supervisor.json`, launcher terminal, independent audit and
the exact public preflight. Preserve failed attempts as well. The wrapper
requires the original pre-discovery commit-pinned binding URL, source revision,
auditor and clarification hashes, and a new output directory **outside** the
wrapper checkout, analysis checkout and directories containing its inputs.
Output parents must already exist; the launcher normalizes parent traversal and
rejects reparse points and symlinks before checking these boundaries.
It checks the binding's public bytes, invokes the reviewed bound builder,
and independently streams every retained tensor file against the audited
inventory before and after writing. It writes an exclusive stage only after
the initial checks. Its last
`stage.json` records hashes, relative evidence names, source and wrapper
revisions and the discovery exposure-inventory digest; it does not expose
absolute local paths or raw supervisor logs. The original journal and tensor
files remain the authoritative private custody copy. Preserve them under
restricted, read-only custody, and make their exact bytes available with the
eventual result; the stage's hashes alone do not provide access to them.

Run in a dedicated, externally supervised process so its exit status and any
cleanup failure are retained separately. The following is a template; the
trusted custodian supplies exact existing paths and full digests. Set
`CUDA_VISIBLE_DEVICES=-1` in the process environment before staging:

```text
python -I -B tools/publish_natural_prediction_freeze.py stage \
  --wrapper-revision <wrapper-commit> --source-root <frozen-checkout> \
  --source-revision <discovery-source-commit> --binding <private-binding-copy> \
  --binding-url <original-commit-pinned-public-binding-url> \
  --owner <discovery-owner-directory> --launcher-terminal <private-launcher-terminal> \
  --audit <independent-audit-json> --preflight-manifest <frozen-preflight-json> \
  --auditor-sha256 <auditor-digest> --clarification-sha256 <clarification-digest> \
  --output <new-stage-directory>
```

Proceed only after the supervisor observes exit code zero and verifies the
returned stage digest against the exact `stage.json` bytes. A completed-looking
file after an interrupted write or an `fsync`/close error is not by itself a
successful process exit. Preserve partial output; do not overwrite it or retry
an experiment. This wrapper never retries model work.

Publish the exact staged `natural-prediction-v2.json` and `stage.json` as
`results/distilgpt2-natural-agreement-v1-20261028/natural-prediction-v2.json`
and `natural-prediction-provenance.json`, respectively, with the original
public binding unchanged. Use the authorized GitHub connector or a separately
authenticated non-force Git push; if the push outcome is uncertain, reread
official `main` and the commit-pinned bytes before any retry. **Do not publish
the aggregate yet.** Record the prediction commit and run:

```text
python -I -B tools/publish_natural_prediction_freeze.py track \
  --wrapper-revision <wrapper-commit> --stage <stage-directory> \
  --stage-sha256 <stage-digest> --prediction-commit <prediction-commit> \
  --receipt <new-track-receipt-path>
```

The receipt must live outside the protected stage and wrapper checkout. The
check requires the source commit to precede the original binding commit, and
the binding and wrapper commits to precede the prediction commit. It checks
exact official GitHub bytes at the prediction commit and observed `main` head,
reachability from that head, and absence of the aggregate at both locations.
After a successful supervised track observation, publish
the exact staged `natural-prediction-freeze.json` at its same-named result path
in a **distinct descendant of the observed head**. Then run:

```text
python -I -B tools/publish_natural_prediction_freeze.py aggregate \
  --wrapper-revision <wrapper-commit> --stage <stage-directory> \
  --stage-sha256 <stage-digest> --aggregate-commit <aggregate-commit> \
  --track-receipt <completed-track-receipt> --receipt <new-aggregate-receipt-path>
```

Put this receipt outside the protected stage, wrapper checkout and directory
containing the track receipt. The check repeats the track predicates, verifies
the later aggregate and unchanged prior bytes at both the aggregate commit and
the final observed official `main` head, and records
the observed head and digests. GitHub transport or rate-limit failure stops
the handoff; a 404 is accepted only for the explicit aggregate-absence checks.
The receipts are consistency and observation records, not cryptographic proof
of faithful execution, publication time, or the absence of private draws.

Only a separate trusted custodian may proceed after observing the completed
aggregate receipt and independently establishing that no held-out nonce or
case was generated earlier. The receipt identifies the discovery exposure
snapshot; before drawing, the custodian must verify the then-current reviewed
successor and apply its known-exposure checks. For shared held-out cases with
the injection track, independently validate and publish that track and a new
joint aggregate before any draw. Never alter or add a track after reveal.
