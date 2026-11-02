# Bounded CPU owner for the masked-color study

This owner implements the [frozen protocol](DISTILBERT_COLOR_EDIT_TOP_V1.md)
and [planned cases](https://github.com/EastonSell/model-looking-glass/blob/f3d3e28980379640f7548541225ef40b017e5557/docs/DISTILBERT_COLOR_EDIT_CASES_V1.md). Its public source and
the exact cases require independent review before any pretrained forward.
The single masked-token output is not a generated answer. The four target
rephrasings use the same public object, and the 12 fixed locality rows are not
a broad capability benchmark.

The screen, development and evaluation are separate supervised processes in
one new run directory. The first stage forwards all nine fixed screen rows
once and reconstructs first-eligible selection from their saved full f32 logits.
If none qualifies, the result is baseline-ineligible and development stops.
An eligible screen's exact results, including input IDs and the selected
candidate, must be public before the development process can begin. The
development process saves the complete fixed dose grid, the final-FFN
gradient/site witnesses, opposite, random and wrong-target controls, original
decoder-bias bits and the discovery-only bias solve. It rebuilds its choices
from those records. Its exact public archive is required before evaluation.
When the bias solve is unavailable, or calibration fails, that condition and
the retained evidence are reported as specified in the protocol; no hidden
row is used to retune the edit.

Each started forward and completion is sequenced with its phase, case, arm,
input IDs, attention mask and position. Full mask-vocabulary logits are
f32 little-endian bytes with a length and SHA-256. A returned vector and any
available site/gradient witnesses are retained in an incomplete record if a
later validation or cleanup fails, when the returned CPU-f32 mask row is
indexable and no wider than the fixed vocabulary. An invalid returned shape
is recorded; an unindexable, empty, oversized or incompatible output has an
explicit omission reason. The engine compares the affected weight
and decoder bias against their expected values before restoration, then checks
the full registered state and hook cleanup. A mismatch, partial journal,
failed cleanup, timeout or absent receipt is invalid/incomplete. The run
directory permits one claim per stage; the coordinator must not bypass that
one-shot rule with a fresh directory after an uncertain attempt.

All original stage files, duplicated prelaunch-input bytes, the external
observer's retained command/start metadata and public archive indexes are
included in the shared 128 MiB retained-byte accounting, as are the exact
public run-record sidecars; later
compressed archive copies are not counted twice. All stage journals share the
256-start limit. A fixed terminal reserve leaves room for failure and exit
metadata. The caps do not bound model memory, cached checkpoint bytes or
temporary staging. The owner checks available space before spawning. The
supervisor's child PID, exit and file hashes are cooperative observations,
not independently authenticated execution or model provenance.
The supervisor writes `validated.json` only after its child is reaped and its
budget and saved-result checks pass. A separate observer must record the
supervisor's actual zero exit and the hash of that marker in
`supervisor-observation.json` before the stage can be promoted. The marker
alone does not establish the supervisor's exit, and neither cooperative
record attests model execution or the whole process tree. Git ancestry is
checked for the owner source, screen publication and development publication;
an independent observer must additionally confirm each commit was public
before the next stage starts. A lost spawn return or interrupted cleanup is
incomplete, and its one-shot attempt cannot be retried.

The checkpoint is staged offline and verified against all five pinned file
sizes and SHA-256s before and after loading; the staged copy is verified too.
The actual model, eager CPU f32 configuration, tokenizer, final FFN site,
tied input/output weight and decoder-bias bytes are checked before action.
The model process records Python 3.11.15, one intra-op thread, one inter-op
thread and enabled deterministic algorithms; CUDA visibility is disabled.
Each stage requires the exact saved original model-state digest and runtime
from both predecessors where present, as well as their exact source maps.
The supervisor and worker require the direct base Python interpreter,
`-E -s -B`, a newly created empty `-X pycache_prefix=<directory>` and an
explicit trusted site-packages directory. The observer and worker launch use
the fixed bootstrap string in the owner to insert the exact checkout and
site paths before running the module. This avoids the Windows virtualenv
redirector's different launcher and interpreter PIDs. The external observer
verifies a clean launcher environment. The owner requires an existing scratch
root disjoint from the checkout, cached checkpoint, dependency site, run and
observer directories. The worker's temporary files and staged checkpoint go
in a fresh child there; the resolved scratch-root path hash is checked across
stages. This is not a sandbox. No generation
or concurrent use of that model is allowed.
Source bytes and the case manifest are checked at stage start and again before
success. The prepared prelaunch observation checks the named public histories
and any explicitly supplied additional public manifests; its timestamp and
scope remain an operator report rather than a complete exposure history.

For each full row, ranks are **competition ranks**: one plus the number of
strictly larger logits, so tied tokens share a rank. A tied top has no unique
top. Accepted-set NLL is computed from the fixed accepted token IDs; those sets
are not exhaustive semantic answer lists. KL and TV use compensated binary64
calculations and a log-domain treatment of tiny probability terms. Extremely
small KL values can still suffer cancellation: for fabricated f32 rows
`(0, 0)` and `(0, 2**-149)`, the raw reduction can be about `7.0e-46` while a
high-precision calculation is about `2.5e-91`. Do not treat those metrics as
uniformly exact or infer a general error bound from this example. Retain the
raw KL, reject values below `-1e-12`, and clamp only the stated tiny negative
range for comparisons. A separate high-precision audit of the actual saved
rows is required before interpreting near-zero values or threshold decisions.
Signed f32/f64 subnormal handling is checked; no GPU is needed for the audit.

From a clean checkout of the independently reviewed public owner commit, first
prepare a new scoped input observation. Add each later relevant public input
manifest with `--additional <commit> <path> <case-rows|cue-variants|bert-cases>`; omit
that option only after checking the current custodians and public history.
The generator and supervisor compare the observed official main ref before
any stage claim. If main moves before the claim, prepare a fresh observation;
no model input was run by that failed preflight.

```text
python -m scripts.prepare_distilbert_color_prelaunch_v1 --repo . --source-commit <owner-commit> --phase screen --additional d5b23ff25134e6a22084bc0d9f1c8edab1863211 experiments/bert_cue_factorial_cases_v1.json cue-variants --additional b75e5bcc3a866a9af56ac1ac3ba54c6abe1691a4 experiments/bert_cue_dose_response_cases_v1.json cue-variants --additional a49cec14227ac1823b67f63e2793b513210fca7c experiments/cross_model_agreement_cases_v1.json bert-cases --scope <scoped-owner-observation> --output <new-prelaunch.json>
python -m scripts.observe_distilbert_color_supervisor_v1 --python <direct-base-python> --site <trusted-site> --repo <absolute-repo> --cache <hf-cache-hub> --scratch <disjoint-scratch-root> --run-root <new-run-directory> --observer-dir <new-observer-directory> --phase screen --source-commit <owner-commit> --prelaunch <new-prelaunch.json> --prelaunch-sha256 <sha256>
python -m looking_glass.distilbert_color_edit_publication_v1 --stage <run-directory>/screen --phase screen --output <new-public-bundle-directory>
```

The independently reviewed observer saves its exit observation in the stage
directory. Publish the bundle unchanged and have its saved records and process receipt
independently reviewed. For development, use a new phase-specific prelaunch
record and the same run directory, with `--phase development`,
`--screen-public-commit <commit>` and `--screen-public-prefix <repository-path>`.
For evaluation, also supply `--development-public-commit` and
`--development-public-prefix`. Each predecessor's compressed parts are read
from the immutable Git commit, hash-checked, unpacked under a size bound and
compared byte-for-byte with the local original stage files. Each indexed bundle
also publishes the exact parent `run.json` as a sidecar so a reader can check
the observer's run hash. The verifier does
not establish that a local Git object is public; the independent observer must
confirm the public commit and result before the next stage.
