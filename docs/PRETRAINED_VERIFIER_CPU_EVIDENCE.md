# Saved real-checkpoint CPU verifier challenges

These records follow the [CPU challenge protocol](PRETRAINED_VERIFIER_CPU.md).
They are correctness observations on a single exposed two-token input, using
the pinned offline DistilGPT2 checkpoint. They are not a GPU profile, a speed
measurement, a held-out behavior study, or a full execution-environment
attestation. The `SourcePreflight` used by the challenge is a synthetic test
fixture, not an authorization for GPU execution.

| Record | Executed source commit | Outcome |
| --- | --- | --- |
| [v1 correctness](../evidence/pretrained-verifier-cpu-v1.json) | `053f0191d8c4eeff13755777245fb82062c513e1` | Passed 13 prescribed mutation checks; this version did not test distinct Parameter objects sharing the embedding/head storage. |
| [v1 absent cache](../evidence/pretrained-verifier-cpu-absent-cache-v1.json) | `053f0191d8c4eeff13755777245fb82062c513e1` | Expected failure during startup with the designated cache path absent. |
| [v2 old-verifier red](../evidence/pretrained-verifier-cpu-alias-red-v2.json) | `366cbf837aa60513f15f741d87a1431150b564fe` | Failed at the new shared-storage/distinct-Parameter challenge against the old verifier. |
| [v2 fixed verifier](../evidence/pretrained-verifier-cpu-v2.json) | `f3c541deeb899a4b3b553ecaaea574f243e2dd29` | Passed all 14 prescribed mutation checks against the updated verifier. |
| [v2 absent cache](../evidence/pretrained-verifier-cpu-absent-cache-v2.json) | `f3c541deeb899a4b3b553ecaaea574f243e2dd29` | Expected failure during startup with the designated cache path absent. |

In each successful run, a single explicit loader call was made, the selected
CUDA/network guards reported no calls after PyTorch import, and the final
restored output matched the original float32 bits. For the v2 input, changing
the selected attention parameter, mask value and scaling flag changed maximum
absolute logits by approximately 0.0864, 4.2101 and 0.7630 respectively.
These are individual-case effects, not claims about a general behavior.

The absent-cache records verify that the nominated path did not exist before
the launched test and record its startup failure; they do not isolate a
loader-level missing-asset error. Failure output is retained only as hashes
and an allowlisted phase. Selected source files and Git-reported cleanliness
are checked, but unlisted imported modules and index flags are outside this
collector's attestation. Early setup, interruption or output-write failures
can prevent a complete record.

An independent reviewer replayed the v2 old-verifier source and observed the
same designated failure, then replayed the fixed source and observed all 14
rejections and bitwise restoration. The reviewer also checked the records'
selected source hashes and redaction. These independent replays are reviewer
observations; the JSON files record the original collector runs, not those
separate replays.
