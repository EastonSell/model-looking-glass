# FLAN-T5 GPU smoke owner-record reader, v1

`tools/audit_flan_t5_gpu_smoke_saved.py` is a separate standard-library reader
for the saved owner portion of the fixed public-pair development smoke. It
loads no model or accelerator library. Its command-line entry point requires
an absolute owner directory, a separate saved supervisor record, and the exact
40-character source commit:

```text
python -I -S -B tools/audit_flan_t5_gpu_smoke_saved.py OWNER_DIR SUPERVISION_JSON --source-commit COMMIT
```

It independently checks the pinned staged manifest and all seven saved asset
hashes, six ordered inputs and forwards, raw finite float32 tensor bytes and
coordinates, repeat/self/null controls, the donor landing and logit-change
control, the 37 ordered memory-stage records, and the owner's final cleanup
counters. The owner describes these samples as synchronized, but the saved
bytes do not independently prove that CUDA synchronization occurred. The
command-line entry uses the fixed manifest and source
path identities. Test-only function arguments may replace those identities for
fabricated data; such calls do not validate the pinned checkpoint.
The output labels any explicit identity override `synthetic-overrides`, even
if its values equal the defaults; the command-line entry never uses overrides.

The output is `owner-records-consistent-not-approved` when these checks pass.
The owner directory and supervisor record must remain quiescent for the entire
audit: stable individual file reads do not establish a coherent snapshot if
another process changes the tree between reads. The later controller-envelope
audit must compare the exact protected evidence hash inventory and reject
unexpected files.
It does not verify the source checkout or dependency bytes, independently
witness the supervisor receipt or controller exit, observe device-wide
cleanup, or establish that the recorded model execution actually happened.
Model-state stability and settings not present in the records remain owner
observations. It does not replace the separate controller-envelope audit,
external controller-exit observation, or device admission and recovery. Run
and process identifiers in the output belong in private evidence until an
appropriate redacted export is reviewed.
