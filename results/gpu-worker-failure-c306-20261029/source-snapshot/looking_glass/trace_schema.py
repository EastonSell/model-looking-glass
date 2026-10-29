"""Versioned, immutable metadata for observations of model tensor outputs.

The schema does not hold tensor values and does not assert causal provenance.
It is independent of the existing ``probes.TraceEvent`` API.
"""
from dataclasses import dataclass, field
import json
from typing import Iterable


SCHEMA_VERSION = "1.0"


def _text(name, value, *, optional=False):
    if optional and value is None:
        return
    if type(value) is not str:
        raise TypeError(f"{name} must be a string" + (" or None" if optional else ""))
    if not value.strip():
        raise ValueError(f"{name} must not be empty or whitespace")


def _nonnegative(name, value, *, optional=False):
    if optional and value is None:
        return
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer" + (" or None" if optional else ""))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _indices(name, value, *, optional=False):
    if optional and value is None:
        return
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple" + (" or None" if optional else ""))
    for index, item in enumerate(value):
        _nonnegative(f"{name}[{index}]", item)


def _fields(name, value, expected):
    if type(value) is not dict:
        raise TypeError(f"{name} must be an object")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{name} field names must be strings")
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        raise ValueError(f"{name} has missing fields {sorted(missing)!r} "
                         f"and unexpected fields {sorted(extra, key=repr)!r}")


def _tuple_from_json(name, value, *, optional=False):
    if optional and value is None:
        return None
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array" + (" or null" if optional else ""))
    return tuple(value)


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Producer-assigned identifiers; None denotes an unknown identifier."""

    run_id: str
    job_id: str | None = None
    request_id: str | None = None
    sample_id: str | None = None

    def __post_init__(self):
        _text("run_id", self.run_id)
        for name in ("job_id", "request_id", "sample_id"):
            _text(name, getattr(self, name), optional=True)

    def to_dict(self):
        return {name: getattr(self, name)
                for name in ("run_id", "job_id", "request_id", "sample_id")}

    @classmethod
    def from_dict(cls, value):
        _fields("identity", value, {"run_id", "job_id", "request_id", "sample_id"})
        return cls(**value)


@dataclass(frozen=True, slots=True)
class GenerationContext:
    """Producer-supplied generation/cache context; None means unknown."""

    stage: str = "unknown"
    generation_step: int | None = None
    cache_position: int | None = None
    cache_length_before: int | None = None
    cache_length_after: int | None = None
    use_cache: bool | None = None

    def __post_init__(self):
        if type(self.stage) is not str:
            raise TypeError("stage must be a string")
        if self.stage not in ("unknown", "forward", "prefill", "decode"):
            raise ValueError("stage must be unknown, forward, prefill, or decode")
        for name in ("generation_step", "cache_position", "cache_length_before",
                     "cache_length_after"):
            _nonnegative(name, getattr(self, name), optional=True)
        if self.use_cache is not None and type(self.use_cache) is not bool:
            raise TypeError("use_cache must be a boolean or None")

    def to_dict(self):
        return {name: getattr(self, name) for name in
                ("stage", "generation_step", "cache_position", "cache_length_before",
                 "cache_length_after", "use_cache")}

    @classmethod
    def from_dict(cls, value):
        _fields("generation", value, {"stage", "generation_step", "cache_position",
                                     "cache_length_before", "cache_length_after", "use_cache"})
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One observation or selected tensor slice, with its original coordinates."""

    identity: RunIdentity
    sequence: int
    forward_index: int
    module_path: str
    invocation: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    capture_mode: str
    module_kind: str | None = None
    layer_index: int | None = None
    output_path: tuple[int | str, ...] | None = None
    batch_positions: tuple[int, ...] | None = None
    token_positions: tuple[int, ...] | None = None
    batch_axis: int | None = None
    token_axis: int | None = None
    generation: GenerationContext = field(default_factory=GenerationContext)
    saved_shape: tuple[int, ...] | None = None
    saved_dtype: str | None = None
    saved_device: str | None = None

    def __post_init__(self):
        if type(self.identity) is not RunIdentity:
            raise TypeError("identity must be a RunIdentity")
        if type(self.generation) is not GenerationContext:
            raise TypeError("generation must be a GenerationContext")
        for name in ("sequence", "forward_index", "invocation"):
            _nonnegative(name, getattr(self, name))
        for name in ("layer_index", "batch_axis", "token_axis"):
            _nonnegative(name, getattr(self, name), optional=True)
        for name in ("module_path", "dtype", "device", "capture_mode"):
            _text(name, getattr(self, name))
        for name in ("module_kind", "saved_dtype", "saved_device"):
            _text(name, getattr(self, name), optional=True)
        _indices("shape", self.shape)
        for name in ("batch_positions", "token_positions", "saved_shape"):
            _indices(name, getattr(self, name), optional=True)
        if self.output_path is not None:
            if type(self.output_path) is not tuple:
                raise TypeError("output_path must be a tuple or None")
            for index, component in enumerate(self.output_path):
                if type(component) is int:
                    _nonnegative(f"output_path[{index}]", component)
                elif type(component) is not str:
                    raise TypeError("output_path components must be integers or strings")
        for axis_name, positions_name in (("batch_axis", "batch_positions"),
                                          ("token_axis", "token_positions")):
            axis = getattr(self, axis_name)
            positions = getattr(self, positions_name)
            if axis is not None:
                if axis >= len(self.shape):
                    raise ValueError(f"{axis_name} is outside shape")
                if positions is not None and any(index >= self.shape[axis] for index in positions):
                    raise ValueError(f"{positions_name} contains an index outside shape")
        if self.batch_axis is not None and self.batch_axis == self.token_axis:
            raise ValueError("batch_axis and token_axis must be distinct")

    def to_dict(self):
        """Return an independent JSON-compatible object with explicit nulls."""
        result = {"schema_version": SCHEMA_VERSION, "identity": self.identity.to_dict(),
                  "generation": self.generation.to_dict()}
        for name in ("sequence", "forward_index", "module_path", "invocation", "dtype",
                     "device", "capture_mode", "module_kind", "layer_index", "batch_axis",
                     "token_axis", "saved_dtype", "saved_device"):
            result[name] = getattr(self, name)
        for name in ("shape", "output_path", "batch_positions", "token_positions", "saved_shape"):
            value = getattr(self, name)
            result[name] = None if value is None else list(value)
        return result

    @classmethod
    def from_dict(cls, value):
        expected = {"schema_version", "identity", "sequence", "forward_index", "module_path",
                    "invocation", "shape", "dtype", "device", "capture_mode", "module_kind",
                    "layer_index", "output_path", "batch_positions", "token_positions", "batch_axis",
                    "token_axis", "generation", "saved_shape", "saved_dtype", "saved_device"}
        _fields("trace event", value, expected)
        if type(value["schema_version"]) is not str or value["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema version: {value['schema_version']!r}")
        values = {key: item for key, item in value.items() if key != "schema_version"}
        values["identity"] = RunIdentity.from_dict(value["identity"])
        values["generation"] = GenerationContext.from_dict(value["generation"])
        values["shape"] = _tuple_from_json("shape", value["shape"])
        for name in ("output_path", "batch_positions", "token_positions", "saved_shape"):
            values[name] = _tuple_from_json(name, value[name], optional=True)
        return cls(**values)

    def to_json(self):
        """Return one deterministic, compact JSON object, excluding tensor payloads."""
        return json.dumps(self.to_dict(), ensure_ascii=True, allow_nan=False,
                          sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value):
        """Parse a single event, rejecting duplicate keys and nonstandard numbers."""
        if not isinstance(value, str):
            raise TypeError("JSON input must be a string")

        def unique_pairs(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key!r}")
                result[key] = item
            return result

        def reject_constant(constant):
            raise ValueError(f"nonstandard JSON constant: {constant}")

        return cls.from_dict(json.loads(value, object_pairs_hook=unique_pairs,
                                        parse_constant=reject_constant))


def validate_trace(events: Iterable[TraceEvent]):
    """Reject duplicate (run_id, sequence) identities in an event collection.

    Input order and sequence contiguity are intentionally not required, since
    partial traces may be filtered or gathered from independently stored slices.
    """
    seen = set()
    for event in events:
        if type(event) is not TraceEvent:
            raise TypeError("trace members must be TraceEvent instances")
        key = (event.identity.run_id, event.sequence)
        if key in seen:
            raise ValueError(f"duplicate trace event identity: {key!r}")
        seen.add(key)
