# Injection preparation startup probe

`tools/injection_preparation_child.py --validate-only CONFIG` is a gated,
CPU-only no-model startup probe. It checks the frozen discovery launcher and
the separately published preparation sources, runs the launcher's existing
isolated boot and runtime initialization, and compares the resulting canonical
runtime to the prospectively published held-out runtime. It then loads the
four verified preparation modules from their raw bytes and checks that the
approved site-packages path occurs once. A successful probe reports zero model
forwards; it does not claim a phase, access the concealed site, or draw cases.
The zero-forward result describes the reviewed no-model code path; it is not a
measurement from an independent runtime forward counter.

The probe is not an end-to-end preparation caller. A real caller still needs
the published replay plan and independent result, current exposure, protected
ledger and side-evidence checks, private site custody, tokenizer verification,
shared allocation gate, one-use launch and process-exit observations required
by `INJECTION_SPLIT_CUSTODY.md`. It must bind its final exact source and perform
another no-model rehearsal of its actual production command. The caller must
also independently verify published bytes, protected paths, source/cache
leases, approved interpreter and dependency environment. A result file or a
successful probe alone does not establish those duties.
