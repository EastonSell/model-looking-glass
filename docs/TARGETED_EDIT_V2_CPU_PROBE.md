# Targeted-edit v2 CPU runtime rehearsal

`scripts/targeted_edit_v2_cpu_probe.py` is a no-model diagnostic for the
prospective v2 study. It loads no checkpoint, constructs no model and performs
no model forward. A matching report is not a study result, a source attestation,
or authorization to run either phase; both live entry points remain disabled.

On the recorded Windows host, the v1 `torch.__config__.show()` hash differs
between an empty CUDA visibility value and the literal `-1`, despite using the
same Torch installation. The original v1 owner used `-1`. This rehearsal
first obtains the cooperative CPU-study claim with the supervisor-required
empty value and no model/CUDA-capable dependency imported. Only then it sets
`CUDA_VISIBLE_DEVICES=-1`, checks the literal before the first model dependency
import, and checks it again after imports, before writing the report, and
before clean attestation. It rejects an initialized Torch CUDA context. The
visibility setting and checks rely on trusted cooperative code; they do not
prove that arbitrary Python or native code cannot access a GPU.

The direct child requires an isolated interpreter with site initialization and
bytecode writing disabled, a fresh empty bytecode prefix, a dedicated persistent
study root, and an explicitly provided trusted dependency site. A narrow
project-only finder resolves the probe's fixed set of `looking_glass` Python
sources without adding the checkout root to `sys.path`. This excludes checkout
root shadow modules from the probe's import search but does not authenticate
the project, interpreter, standard library or dependencies. The prospective
study still requires its independently reviewed frozen-source gate.

On Windows the probe replaces CPython's private `platform._syscmd_ver` with
values from `sys.getwindowsversion()` before importing Torch. It rejects an
already-cached platform identity and compares the resulting execution-relevant
v1 environment fields to the exact, bounded and digest-pinned published
precommit. The shim is tied to this
diagnostic and is not independent proof of operating-system identity. A
sentinel-verified Python audit hook rejects Python-visible process creation;
it cannot observe or prevent all native process creation, and this diagnostic
does not attest that no native descendants existed. A production owner needs
independently verified whole-process-tree accounting without weakening the
supervisor's cleanup rules.

The report is written before child teardown and clean attestation. Consumers
must also verify the exact supervisor run ID and PID, clean zero-exit reaping,
no timeout, and the expected custody root. A report alone can remain after a
failed or interrupted cleanup. An uncertain or failed attempt leaves the
supervisor's unsafe marker; do not overwrite or silently reuse its root.
