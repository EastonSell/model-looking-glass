# Independent saved-byte replays

The two readers were developed independently before inspecting the closed
attempt. They make no model calls. Use a compatible little-endian Python with
binary64/libm support and assertions enabled. Numeric replay can depend on the
host's binary64/libm behavior. Paths below are examples; the audited v1
case file and a clean checkout of source commit
`72ff52feca7c102e611023b54f031fbb8ff17bd4` are external inputs. That checkout
must contain the observed-main commit objects as well as the execution source.

The saved-bit replay and its separate rounding diagnostic do not read the
supervisor or stderr and therefore do not attest process exit, source
execution, stderr authenticity or publication chronology:

```text
python -B audits/year_label_partial_audit.py --run run --cases CASES_JSON --failure-sha256 b59687e3f63573ca4ecbe0b34ba98ead6cdcdf3e035e4e54247abd237ca6e6ef --journal-sha256 a2f5ba4f980bde155bf4eb5729f154f7422956eb9c0920421f13327038e057f5
python -B audits/year_label_failure_probe.py --run run --failure-sha256 b59687e3f63573ca4ecbe0b34ba98ead6cdcdf3e035e4e54247abd237ca6e6ef --journal-sha256 a2f5ba4f980bde155bf4eb5729f154f7422956eb9c0920421f13327038e057f5
python -B audits/year_label_rounding_analysis.py --run run --cases CASES_JSON
```

The other replay additionally checks the externally pinned source, process
receipt, run hashes and launch records. Its three Python files must remain
in the same directory. Public mode verifies the separately pinned redacted
stderr projection, but cannot independently verify the private original's
receipt hash or the redaction fidelity. Its output states that limitation;
the retained `year_label_partial_replay_report.json` is the original private
mode report, which was checked against the original stderr before publication.
`year_label_partial_public_mode_report.json` is the separate projection-mode
report and records the narrower stderr scope.
Neither mode independently authenticates actual execution or all exposures.

```text
python -B audits/year_label_partial_replay.py --run run --cases CASES_JSON --receipt-sha256 f3b418c7760ed23e2f3c3f90bd39b33538b2b9913ffad2dcf0dd7546742f55bf --pre-run-sha256 624590a8a547f32e90cc69ae6ab1583e51fa81306037e8874b90c5cf52f8a972 --launch-observation-sha256 97aad4e204848e4ff602181e0703c064a4b93f0a366eb9fb9b1ed8dedc737096 --launch-main ef20d6649adeff9c04dfb2a95bc6455adbf69290 --launch-start-sha256 3f374248dd01f955182e83c0f9007409351e7f3a67eb603c39c4395f38e529ab --start-main 4453c47a888362673432a88a5602080ee0e863fc --source-revision 72ff52feca7c102e611023b54f031fbb8ff17bd4 --source-tree 389d89769f8853d45376843b37547eaaffb93f97 --source-root CLEAN_SOURCE_CHECKOUT --current-exposure-version distilgpt2-development-exposures-v25 --current-exposure-sha256 32cc4bcbe4b482ded122715c5f96c6c5c765a76a1ef3532360cdf9890827f144 --stderr-projection stderr-redacted.txt --stderr-projection-sha256 0c9d3176ff1dfd47f33d77b166487de301810628f731f9863706fab735c2b0a4
```

`supervisor.json` remains byte-identical and binds the original stderr SHA;
`stderr-redaction.json` describes the separate public derivative. The raw
original remains private, and must be supplied instead of projection flags
for the private-original verification mode.
