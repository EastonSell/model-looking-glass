# Compare a known corruption and test restoration

`compare_known_corruption` is a small CPU workflow built from the existing
GPT-2 trace, transformer comparison and ordered intervention APIs. It records
where **selected observations** differ after a known input-embedding edit,
then tests a clean-state patch at a chosen changed site. It also runs a
target-site self-patch and a clean-state patch at another site in the same
layer. An optional matched-dose pair, enabled by the example CLI, compares
partial clean-direction patches at those sites. These are controlled
demonstrations, not a search for a natural circuit.

## Run the example

From an installed checkout with PyTorch and the optional transformer extra:

```sh
python -m pip install -e ".[transformer]"
python examples/gpt2_localization_workflow.py --cache-dir /path/to/huggingface/hub --output gpt2-localization.json
```

The pinned DistilGPT2 loader uses the supplied writable cache offline by
default. Add `--allow-download` only to fetch missing assets. The output file
must be new. The [loader guide](PRETRAINED_MODEL.md) describes the asset
verification, model revision and staging space. The example uses the public
`Hello world` input (`[15496, 995]`) and writes a JSON report. A
[matched-dose sample report](../examples/gpt2_matched_control_workflow.sample.json)
from a cached CPU run is included; its numerical values can vary with the runtime.
The [earlier unmatched sample](../examples/gpt2_localization_workflow.sample.json)
remains available.

The library function can also be called with a loaded model and another short
input:

```python
from looking_glass.gpt2_trace_workflow import compare_known_corruption
from looking_glass.pretrained import load_distilgpt2

bundle = load_distilgpt2(cache_dir="/path/to/huggingface/hub")
inputs = bundle.encode("Hello world", max_tokens=2)
report = compare_known_corruption(
    bundle.model, inputs["input_ids"], coordinate=0, amount=5.0,
    readout_token=13, layers=(0, 1),
    target_site=(0, "block"), comparison_site=(0, "attn"),
    matched_control=True,
)
print(report["localization"]["first_observed_interval"])
print(report["candidate_sites"])
print(report["interventions"])
```

The caller chooses the perturbation, selected layers, two sites and the
vocabulary ID whose last-position logit is reported. The input is one unpadded
CPU int64 row of 2–16 tokens; the perturbation is a finite nonzero addition
to one coordinate of the **last** token's embedding. Sites are distinct
`(layer, "attn"|"mlp"|"block")` pairs in the same selected layer. The model
must be an unmodified evaluation-mode CPU float32 `GPT2LMHeadModel` with no
other local or global forward hooks. The caller must exclusively own the model
and prevent concurrent mutation. The workflow makes no download and does not
mutate model weights.

## Reading the result

The workflow checks that an ordinary input-ID forward and the clean embedding
forward have bit-identical logits, then separately traces the clean embeddings
and a copy with just the selected coordinate changed. Each trace has distinct
run identity, one explicit uncached forward, the same mask and tap policy,
and at most 18 events and 128 KiB of captured tensor values. The exact
expected execution order, source shapes, output selectors and token positions
are checked before supplying the pairing and order attestations to
`compare_transformer_traces`. Those attestations apply to these cooperative
runs; they are not independent proof of model identity or execution. The
report includes the comparator's metadata, status and every different sampled
site. A first observed difference is a boundary in the selected trace, not a
proven origin or exhaustive localization.

If the chosen target is not observed as different, or comparison is
inconclusive, interventions are marked `not_run`; clean/corrupt readouts and
the input-path control remain in the report. Otherwise each patch runs in its
own `ActivationPlan`, using the clean vector for the **same token and site**.
Before/after captures cover the complete module output. The workflow checks
that the receiver matches the corrupted trace, the selected output matches
its source, and all other token positions are bit-identical before and after.
The target-site self-patch must reproduce the corrupted logits, and clean and
corrupted repeats must reproduce their earlier logits. These are inference
controls; self-patching is not a gradient-preserving no-op. Sessions run
sequentially because they require exclusive ownership.

Each arm reports the chosen next-token logit and maximum absolute distance
between all its logits and the clean and corrupted logits. No recovery ratio
or success threshold is inferred. The other-site patch is a descriptive
comparison: module outputs can differ in scale, execution position and
downstream sensitivity. Its realized L2 dose is reported separately and need
not equal the target dose; it is not assumed to be ineffective. The JSON saves
metadata and summary measurements, not full logits or activation snapshots.

With `matched_control=True`, the workflow also measures each site's natural
clean-to-corrupted vector distance and chooses their smaller value `q` as the
common dose. It interpolates each corrupted vector toward its clean vector in
CPU float64, then rounds once to float32; neither direction is extrapolated.
Each partial patch has the same same-forward receiver, landing and untouched-
position checks as the full patches. The actual post-action L2 at each site
must be positive, within `min(max(1e-5, 1e-5*q), 0.01*q)` of `q`, and within
1% of the actual target-site dose. The report retains both natural distances,
the common dose, interpolation fractions, planned and actual doses, witnesses,
and the same logit measurements as the full patches. If either natural
distance is at most `1e-6`, or rounding cannot deliver the required matched
doses, the matched arm is labeled `dose_ineligible` or `dose_unmatchable`;
the full restoration and existing controls are still reported. Nonfinite
directions are errors, not a successful match. The library default remains
`matched_control=False`; opting in uses the `gpt2-known-corruption-workflow-v2`
report schema and up to two additional forwards.

The matched comparison controls raw vector L2 magnitude, not module scale,
direction, position in the computation, or downstream sensitivity. A larger
or smaller response at one boundary does not establish a unique causal site
or natural feature; the interpolated vectors can also be off-distribution.
In the saved CPU example, the common dose is about 10.833 at both sites; the
comparison-site natural distance is about 12.434, so its interpolation uses
about 87.1% of that direction. The matched target patch restores the clean
logits exactly, while the matched comparison patch leaves a maximum logit
difference of about 0.636 from clean. These are single-input, runtime-specific
observations; the target restoration is expected from the construction
described below.

In the supplied example the only changed input is the last token. Patching its
full layer-0 block output with the clean vector restores that entire block
output in a deterministic causal GPT-2 execution: earlier token states do
not depend on the changed later token. Full downstream restoration in this
setup is therefore expected by construction and tests the mechanics; it does
not show that the block uniquely contains a natural representation. An
attention-output patch need not restore the residual stream. More general
claims require separately designed inputs, controls and held-out evaluations.

The trace and plan limits cover their declared captured/source tensors, not
model weights, logits, metadata, intermediates, temporary copies or peak
memory. Model hidden width is limited to 1,024, vocabulary to 65,536 and the
trace to six layers. Failures and invalid or nondeterministic controls raise
instead of being reported as successful comparisons. See
[transformer comparison](TRANSFORMER_LOCALIZATION.md),
[GPT-2 tracing](TRANSFORMER_TRACE.md) and [ordered plans](INTERVENTION_PLANS.md)
for the full contracts.
