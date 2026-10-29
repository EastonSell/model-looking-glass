# Prospective C: headroom amendment for the FLAN weight-change study

This note amends the C: operational threshold in the earlier
`flan_weight_edit_volume_plan.md`, SHA-256
`b61d09688242e322e525adc74b7e9336b33f471fef20a883a724200805a0815c`.
That plan remains part of the record. No FLAN weight-change model forward has
occurred when this amendment was prepared. The earlier direct-child no-model
preflight was refused when an operator observation found C: free space at
2,269,274,112 bytes, below its 2,281,701,376-byte threshold; the synthetic
test exited before creating a phase attempt. Its failure output and the C:
reading are retained in the operator's tool transcript; the synthetic pytest
temporary directory was subject to pytest retention and is no longer present.
This is an operator observation, not an independently authenticated time.

The prior C: floor was a conservative reserve for interpreter, source and
staging, not the allocation for the study's raw logits. The checkpoint and
trusted Python libraries remain on C: and are read in place. The two phase
directories, saved logits, process logs, temporary files and disabled bytecode
prefix are on E:. The 25 current small runtime/test source files total about
200 KiB; the additional 128 MiB staging allowance covers small source, manifest
and Git-object activity with substantial margin. Large model copies and large
evidence or archive staging on C: remain disallowed. Publication of large saved
outputs requires its own path and free-space check and must not silently use
the C: Git object store for large temporary copies.

For this fixed, not-yet-forwarded study, replace the C: admission floor with
**1.5 GiB plus 128 MiB**, exactly 1,744,830,464 bytes, checked immediately
before child grant and again immediately before the first model call in each
phase. Keep the E: requirement unchanged: at least 2 GiB free in addition to
the 256 MiB cumulative evidence budget and 1 MiB failure reserve. Do not launch
if either volume is below its applicable floor. Do not delete prior attempts or
evidence to meet the floor.

The C: reserve is a pragmatic heuristic, not a bound on operating-system
paging, native-library behavior, filesystem caches or concurrent work. The
observed free space is transient and not an OS reservation; all of those writes
may consume it during a phase. Directing known temporary paths to E: does not
prove that no other C: write occurs. The change affects admission headroom
only: model, input, scientific thresholds, call limits and all failed preflight
records remain unchanged. Independent review and publication of the exact
amendment precede any model forward.
