"""Explicit, producer-supplied context for a narrow encoder comparison.

This is separate from the versioned event schema. It is a comparison contract,
not a proof that a producer actually used the recorded inputs or model.
"""

from dataclasses import dataclass, field

from .trace_schema import GenerationContext, RunIdentity, TraceEvent


DISTILBERT_CONTRACT = "distilbert-masked-lm-eager-v1"
_ROLES = ("attention", "ffn", "block")


def _selection(name, value):
    if (type(value) is not tuple or not value
            or any(type(x) is not int or x < 0 for x in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name} must be a nonempty tuple of distinct nonnegative integers")


@dataclass(frozen=True, slots=True)
class EncoderForwardContext:
    """Recorded inputs and selected tap policy for one supported encoder call.

    Values are immutable tuples, but the record is producer-supplied. A caller
    must separately attest model/runtime pairing and the producer's integrity.
    """

    contract: str
    run_id: str
    input_ids: tuple[tuple[int, ...], ...] = field(repr=False)
    attention_mask: tuple[tuple[int, ...], ...] = field(repr=False)
    num_layers: int
    hidden_size: int
    vocab_size: int
    max_positions: int
    layers: tuple[int, ...]
    modules: tuple[str, ...]
    samples: tuple[int, ...]
    tokens: tuple[int, ...]

    def __post_init__(self):
        if type(self.contract) is not str or not self.contract.strip():
            raise ValueError("contract must be a nonempty string")
        if type(self.run_id) is not str or not self.run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        for name in ("num_layers", "hidden_size", "vocab_size", "max_positions"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (type(self.input_ids) is not tuple or not self.input_ids
                or any(type(row) is not tuple or not row for row in self.input_ids)):
            raise ValueError("input_ids must be a nonempty tuple of nonempty rows")
        length = len(self.input_ids[0])
        if (length > self.max_positions
                or any(len(row) != length for row in self.input_ids)
                or any(type(x) is not int or x < 0 or x >= self.vocab_size
                       for row in self.input_ids for x in row)):
            raise ValueError("input IDs must be rectangular and within model bounds")
        if (type(self.attention_mask) is not tuple
                or len(self.attention_mask) != len(self.input_ids)
                or any(type(row) is not tuple or len(row) != length for row in self.attention_mask)
                or any(type(x) is not int or x not in (0, 1)
                       for row in self.attention_mask for x in row)):
            raise ValueError("attention_mask must be a matching binary tuple of rows")
        for name in ("layers", "samples", "tokens"):
            _selection(name, getattr(self, name))
        if self.layers != tuple(sorted(self.layers)) or any(x >= self.num_layers for x in self.layers):
            raise ValueError("layers must be sorted and within the model")
        if (type(self.modules) is not tuple or not self.modules
                or self.modules != tuple(x for x in _ROLES if x in self.modules)):
            raise ValueError("modules must be distinct roles in execution order")
        if any(x >= len(self.input_ids) for x in self.samples) or any(x >= length for x in self.tokens):
            raise ValueError("selected sample or token is out of range")


def _matches_distilbert_events(context, observations):
    """Check the complete selected event policy, not just shared coordinates."""
    if context.contract != DISTILBERT_CONTRACT:
        return False
    count = len(context.layers) * len(context.modules) * len(context.samples) * len(context.tokens)
    if len(observations) != count:
        return False
    batch, length = len(context.input_ids), len(context.input_ids[0])
    selected = ((layer, role, sample, token) for layer in context.layers
                for role in context.modules for sample in context.samples for token in context.tokens)
    for sequence, (observation, (layer, role, sample, token)) in enumerate(zip(observations, selected)):
        path = f"distilbert.transformer.layer.{layer}"
        if role != "block":
            path += f".{role}"
        event = getattr(observation, "event", None)
        if type(event) is not TraceEvent:
            return False
        if (event.identity != RunIdentity(context.run_id) or event.sequence != sequence
                or event.forward_index != 0 or event.invocation != 0
                or event.module_path != path or event.module_kind != role or event.layer_index != layer
                or event.output_path != (() if role == "ffn" else (0,))
                or event.shape != (batch, length, context.hidden_size)
                or event.dtype != "torch.float32" or event.device != "cpu"
                or event.capture_mode != "direct" or event.batch_axis != 0 or event.token_axis != 1
                or event.batch_positions != (sample,) or event.token_positions != (token,)
                or event.generation != GenerationContext()
                or event.saved_shape != (1, 1, context.hidden_size)
                or event.saved_dtype != "torch.float32" or event.saved_device != "cpu"
                or getattr(observation, "original_saved_shape", None) != (context.hidden_size,)
                or getattr(observation, "comparison_transform", None)
                != "restore_singleton_batch_and_token_axes"):
            return False
    return True


def check_encoder_pair(contexts, reference, candidate, expected_input_differences):
    """Validate the opt-in contract without reporting input IDs or mask values."""
    report = {"verified": False, "issues": [], "actual_input_differences": None}
    if (type(contexts) is not tuple or len(contexts) != 2
            or any(type(context) is not EncoderForwardContext for context in contexts)):
        report["issues"].append("two encoder-forward contexts are required")
        return report
    left, right = contexts
    if (type(expected_input_differences) is not tuple
            or any(type(p) is not tuple or len(p) != 2
                   or any(type(x) is not int or x < 0 for x in p)
                   for p in expected_input_differences)
            or len(set(expected_input_differences)) != len(expected_input_differences)):
        report["issues"].append("explicit distinct input-difference coordinates are required")
        return report
    if not _matches_distilbert_events(left, reference):
        report["issues"].append("reference events do not match the supported encoder policy")
    if not _matches_distilbert_events(right, candidate):
        report["issues"].append("candidate events do not match the supported encoder policy")
    if left.run_id == right.run_id:
        report["issues"].append("encoder run IDs are not distinct")
    if (left.contract, left.num_layers, left.hidden_size, left.vocab_size, left.max_positions,
            left.layers, left.modules, left.samples, left.tokens) != (
            right.contract, right.num_layers, right.hidden_size, right.vocab_size, right.max_positions,
            right.layers, right.modules, right.samples, right.tokens):
        report["issues"].append("encoder model dimensions or selected policies differ")
    if len(left.input_ids) != len(right.input_ids) or len(left.input_ids[0]) != len(right.input_ids[0]):
        report["issues"].append("input shapes differ")
    else:
        actual = tuple((row, token) for row in range(len(left.input_ids))
                       for token in range(len(left.input_ids[0]))
                       if left.input_ids[row][token] != right.input_ids[row][token])
        report["actual_input_differences"] = [list(position) for position in actual]
        if set(actual) != set(expected_input_differences):
            report["issues"].append("input differences do not match the declared coordinates")
        if left.attention_mask != right.attention_mask:
            report["issues"].append("attention masks differ")
    report["verified"] = not report["issues"]
    return report
