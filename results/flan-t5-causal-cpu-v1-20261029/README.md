# FLAN-T5 encoder-state intervention: CPU feasibility

This public exploratory study ran `google/flan-t5-small` on two fixed English
to German word pairs. It used the repository's activation sessions on the
512-dimensional final encoder state during complete model forwards. The
checkpoint, public inputs, target tokens, score, intervention and decision
rule were published before this run in [the frozen protocol](https://github.com/EastonSell/model-looking-glass/blob/5c8d3c1b577a904417ce0bff601afdd45b522581/docs/FLAN_T5_CAUSAL_FEASIBILITY_V1.md)
and [case manifest](https://github.com/EastonSell/model-looking-glass/blob/5c8d3c1b577a904417ce0bff601afdd45b522581/docs/FLAN_T5_PUBLIC_CASES_V1.json).
No pretrained checkpoint had been forwarded on these cases at that freeze.

The cat/dog pair did **not** satisfy the prespecified behavioral baseline:
the `Katze` minus `Hund` teacher-forced sequence-score margins were +4.4728
for cat and +3.4487 for dog, whereas the criterion required the second to be
negative. Its intervention forwards were skipped. This does not establish
what the model would generate when asked to translate.

The book/house pair satisfied the baseline. Its `Buch` minus `Haus` sequence
score margins were +5.3850 and −8.3697. Replacing the book-position final
encoder vector with the house vector changed the first margin to −8.2591;
the reverse replacement changed the second margin to +5.0881. The respective
movements, 13.6441 and 13.4579, met the fixed exploratory criterion. The same
cross-source replacement at instruction token position 0 moved the margins
by 0.2471 and 0.1613, but its vector L2 distance was 0.5327 compared with
2.0146 for the word-position replacement. This comparison is not dose
matched and does not establish unique word-local specificity.

Both targets include EOS and are scored as the **sum of teacher-forced token
log probabilities**. In the first pair `Katze` uses three tokens including
EOS and `Hund` uses two, so length and prior frequency can affect the raw
margin. `Buch` and `Haus` each use two tokens including EOS. Context-only
scores are saved alongside source-conditioned scores. Scores are not greedy
generated translations. These inputs are public and not blinded or known to
be absent from pretraining. The 20% and 5% thresholds are fixed exploratory
decision rules, not a calibrated null distribution or significance test.
One supported public pair does not establish generalization or a complete
mechanism for translation.

The full raw output is in `raw/`: each `forward-NNN.bin` concatenates
little-endian float32 logits and encoder states, with byte lengths, shapes,
token IDs and SHA-256 in its corresponding JSON file. The original
`completed.json` SHA-256 is
`57cbbc319fe4dcc84f3f3a4c50a5bceb633cf653886e8fe16d379cccd9f09afc`.
There were 44 root-model forwards and 12,445,178 bytes of original retained
evidence. A separate [saved-byte audit](audit/independent-saved-audit.py)
recomputed every score, landing, dose, identity/null/capture/full-swap
control and decision from these files; [its output](audit/independent-audit.json)
is a consistency check on retained bytes, not an independent execution of
the model.

The source commit was `5c8d3c1b577a904417ce0bff601afdd45b522581`; checkpoint
revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`; checkpoint license
Apache-2.0. The 307,867,048-byte safetensors file has SHA-256
`495fa51e204676f1a857a9fc13c4c89f3f5ba9f480b898cebca02add25e6d749`.
All seven pinned asset identities are in `asset-manifest.json`. The operator
observed CPU-only process exit 0, a clean source checkout and empty bytecode
prefix after the run; this is a retained [operator observation](operator-observation.json),
not authenticated execution or timing proof. The process reported a peak
working set of 945,954,816 bytes; this is an observed process peak, not a
resource bound or a GPU-memory measurement. Model parameter bytes were
307,844,608. The run was not a latency benchmark.

To replay the saved-byte audit, use the frozen source checkout, the saved
`raw/` directory and `docs/FLAN_T5_PUBLIC_CASES_V1.json` as the two arguments
to `audit/independent-saved-audit.py`, with the environment recorded in
`raw/environment.json`. The audit uses NumPy and PyTorch, does not execute a
model, and its exact numerical replay depends on the recorded runtime and
CPU floating-point settings.
