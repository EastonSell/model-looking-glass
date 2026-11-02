# Separate-volume evidence for a prospective v1.3 timing session

This disabled and unexecuted source successor adapts the storage proposal from
PR #699. It is not the historically observed source at
`0c14778447ed6dcbc4890984641370659a09ffef`, and does not alter that frozen
source or its earlier observations. Before any future run, the complete new
source and observer must be frozen and reviewed for currentness, followed by a
no-model rehearsal, separate enablement review and fresh coordinated admission.

The output's existing canonical parent is the parent for a fresh private run
directory. Its temporary files, child bytecode prefix, manifest, journal and
failure evidence remain inside that directory. The operator must bind fresh
output and observer directories and verify their resolved volume before any
future attempt. An incomplete attempt is retained and is not automatically
retried. Path checks are point-in-time checks; they do not lock the parent
against replacement or reserve storage. A trusted, stable local filesystem
remains an assumption.

The canonical checkout, offline model cache, workspace identity and physical
device lock/unsafe marker remain in the workspace. The proposed storage change
does not alter slot, seeds, paired schedule, timing definitions, activity
screen, status or availability rules. Available free space is a point
observation, not a reservation. The controller does not measure output-volume
free space or verify parent-directory access controls. Before enablement, the
operator must check capacity and provide private, trusted, stable storage; a
supported Windows rehearsal must verify volume, junction and access-control
behavior. Any new observation must retain its own source
identity and limitations and must not be pooled with or relabeled as a prior
isolated result.

`python3 -I tools/check_tap_evidence_parent_stdlib.py` checks extracted storage
helpers with the standard library alone. It does not run the controller or
establish Windows volume, junction, or device-custody behavior.
