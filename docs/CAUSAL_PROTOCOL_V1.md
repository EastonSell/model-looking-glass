# Causal intervention study, protocol v1

Status: prespecified synthetic experiment. This protocol describes the test to
run, not a finding about a pretrained model. The implementation and this
protocol must be committed before the recorded held-out run. The run must
atomically publish the precommitment artifact, with file contents flushed,
before the first held-out model forward. This does not guarantee directory
entry survival after sudden power loss.

Run as a standalone process from a clean, committed repository checkout:

```bash
python -m looking_glass.causal_experiment --output results/causal-v1-run
```

The output directory must not exist. The writer retains an incomplete or
failed record and any already-completed held-out rows if a run fails. Calling
the Python API with other case values or seeds produces a development run and
rejects any float32 input already present in the reserved v1 suite. This guard
prevents accidental exposure through the runner, not prior knowledge of its
public inputs.

## Question

Can a first-observed activation discrepancy on discovery inputs nominate a
boundary whose intervention effects and repair of an injected change can be
predicted on separate inputs? Compare it with a magnitude-ranked boundary, one
random boundary and a zero-effect predictor. This is an instrumentation and
causal-testing sanity check: the model, injection and boundary set are known,
and no inference about a pretrained model or a unique mechanism is warranted.

## Model, inputs, and measurements

- CPU eager PyTorch float32, one thread, a `TinyResidual` with width 8, depth
  3 and route 0. Initialize once from model seed 2301; do not train the model.
- Use the fixed, versioned synthetic suite: 8 discovery and 16 held-out cases,
  each an independently identified 8-dimensional vector, disjoint by both ID
  and canonical contents. Record its exact digest and each case digest. These
  are distinct deterministic examples, not independent model seeds.
- For each input, run a clean pass and a corrupt pass with a +1.0 addition to
  element `[0, 0]` immediately before the `block.1.activation` tap. This is a
  known injected change, not an inferred fact. The common scalar task metric
  is output logit 1 minus output logit 0. The target correction is clean metric
  minus corrupt metric. This is an arbitrary scalar on an untrained network,
  not a language task or a capability measure.
- Directly capture every active `TapPoint`. Compare matched labels and
  occurrences in execution order with exact equality. All matched traces must
  have complete, agreeing execution metadata and shapes; abort if they do not.
  Each trace is capped at 12 events and 384 logical snapshot bytes; this is not
  a claim about process peak memory or capture temporaries.
  This study places a dense tap at the injected boundary, and therefore does
  not test localization when the change occurs between sparse observations.

## Freeze candidates and predictions before held-out evaluation

On discovery cases only:

1. For each paired trace, find the first differing observed boundary. Select
   the most frequent first boundary; resolve ties by execution order. This
   method must not receive the known injection label as selection input.
2. Independently rank sites by the mean Euclidean norm of the paired activation
   difference and select the largest; resolve ties by execution order.
3. Draw one active site uniformly using Python's `random.Random(918)` and the
   stable execution order. Sampling includes the selected sites. This single
   draw is descriptive and does not estimate a distribution over random sites.
   Under this architecture it selects `block.2.output`, the complete state
   immediately before the head. Patching this state from the clean run is
   expected to restore its output regardless of the earlier injection site;
   this fact is a useful non-uniqueness control, not a localization success.
4. For each site and discovery case, replace the *corrupt* output of that
   site, invocation zero, with the same-input *clean* output at that site, and
   measure the patched metric minus corrupt metric. Predict each held-out
   effect with the discovery mean; record the discovery minimum and maximum as
   a descriptive range, not a confidence or calibrated prediction interval.
   The zero-effect baseline predicts zero. Every selected site receives the
   same number of discovery patch experiments. Identical selected sites may
   produce identical effects and must not be reported as independent methods.

Before executing any held-out forward, atomically write `precommit.json`,
including input IDs and content digests, site choices and invocation, source
and base run definitions, all predicted effects and descriptive ranges,
operation and strength (+1.0), metric, and comparisons. Save hashes of the
protocol, exact model state, relevant source files and suite, along with the
source revision and runtime versions. These hashes and recorded operation
ordering support later checking, but do not prove execution order externally;
they do not prove that nobody previously viewed the public inputs.

## Held-out interventions and baselines

For every held-out input, record the paired clean and corrupt metrics and raw
effects for each frozen candidate. Measure the actual clean-to-corrupt patch
effect at its frozen site. Also run these controls:

- A null intervention and a corrupt-to-corrupt identity patch at the nominated
  boundary, each applied to a corrupt base run.
- A clean-to-corrupt patch at `block.0.output`, before the injected change.
- A corrupt-to-clean reverse patch at the nominated boundary.
- A clean-base null intervention with no injected change, and a separately
  recorded clean-versus-clean pair of dense traces with no differing boundary.

The null, identity, upstream and clean-null controls must reproduce the full
two-logit output exactly, not merely preserve their scalar difference. A
nonzero change invalidates the run and is recorded as a failure.

The reference source and base use the same held-out input; the source tensor is
captured at the same site and invocation, with the entire tensor replaced. The
precommitment freezes this rule; the specific source tensor is measured only
after held-out evaluation begins. The upstream location is an injection-aware
negative control, not a discovery method. Retain the raw case-level metrics
and intervention deltas: do not divide by a possibly near-zero clean–corrupt
gap. Evaluate the candidate predictions against their *own measured patch
effects* and against the common target correction. Separately measure each
patch's absolute residual from the target and compare with the zero correction.
Report MAE and sign accuracy with symmetric absolute zero tolerance `1e-7`;
exclude cases from sign accuracy only when the observed effect is at or below
that tolerance. A prediction at or below that tolerance is incorrect on a
nonzero observed effect. Report excluded counts. Report descriptive statistics
for these fixed cases without inferential bootstrap intervals: distinct IDs
alone do not establish a sampled population or independent model seeds. Save
failures and all predictions even if a
held-out evaluation fails; never overwrite a previous output directory.

## Interpretation and limits

Same-input clean-to-corrupt patching tests a conditional restoration under
this injection; reverse patching tests a complementary conditional failure.
Neither establishes global necessity or sufficiency. Later boundaries may
also restore the output, and off-distribution replacements and redundant or
interacting pathways can change the answer. Trace differences give observed
locations, not unique causes or the model's reasoning. A location is only a
discrete hit on the known injection when the reported tap label equals the
injected label; it is not a general localization accuracy estimate. This is
one model seed, one injection and one random draw. The public fixed split can
be inspected, and once repeatedly tuned against it it is no longer a blind
held-out set. Test and debugging runs must use separate case values and seeds;
if final cases influence changes, label the run exploratory and freeze a new
protocol/split before making confirmatory claims.
