"""Behavioral tests for the versioned metadata-only trace contract."""
import dataclasses
import importlib.util
import json
import unittest


class TraceSchemaTests(unittest.TestCase):
    def setUp(self):
        # Keep the first test run a useful red failure before implementation.
        self.assertIsNotNone(importlib.util.find_spec("looking_glass.trace_schema"),
                             "versioned trace schema is missing")
        from looking_glass import trace_schema
        self.schema = trace_schema

    def make_event(self, **overrides):
        values = dict(
            identity=self.schema.RunIdentity("run-1", "job-1", "request-1", "sample-1"),
            sequence=12, forward_index=2, module_path="transformer.h.1.attn",
            invocation=1, shape=(4, 5, 6), dtype="torch.float32", device="cuda:0",
            capture_mode="direct", module_kind="attn", layer_index=1,
            output_path=(0,),
            batch_positions=(3,), token_positions=(1, 4), batch_axis=0, token_axis=1,
            generation=self.schema.GenerationContext(
                stage="decode", generation_step=2, cache_position=7,
                cache_length_before=7, cache_length_after=8, use_cache=True),
            saved_shape=(1, 2, 6), saved_dtype="torch.float32", saved_device="cpu",
        )
        values.update(overrides)
        return self.schema.TraceEvent(**values)

    def test_json_round_trip_retains_identity_coordinates_and_saved_placement(self):
        event = self.make_event()
        wire = json.loads(event.to_json())
        self.assertEqual(wire["schema_version"], "1.0")
        self.assertEqual(wire["identity"], {
            "run_id": "run-1", "job_id": "job-1", "request_id": "request-1",
            "sample_id": "sample-1"})
        self.assertEqual(wire["batch_positions"], [3])
        self.assertEqual(wire["token_positions"], [1, 4])
        self.assertEqual(wire["output_path"], [0])
        self.assertEqual(wire["shape"], [4, 5, 6])
        self.assertEqual(wire["saved_shape"], [1, 2, 6])
        self.assertEqual(wire["device"], "cuda:0")
        self.assertEqual(wire["saved_device"], "cpu")
        self.assertEqual(wire["generation"]["cache_position"], 7)
        self.assertEqual(self.schema.TraceEvent.from_json(event.to_json()), event)

    def test_unknowns_are_explicit_and_not_conflated_with_empty_selection(self):
        event = self.make_event(
            identity=self.schema.RunIdentity("run-1"), module_kind=None,
            layer_index=None, output_path=None, batch_positions=None, token_positions=(),
            batch_axis=None, token_axis=None, generation=self.schema.GenerationContext(),
            saved_shape=None, saved_dtype=None, saved_device=None)
        wire = event.to_dict()
        self.assertIsNone(wire["identity"]["job_id"])
        self.assertIsNone(wire["identity"]["request_id"])
        self.assertIsNone(wire["identity"]["sample_id"])
        self.assertIsNone(wire["batch_positions"])
        self.assertIsNone(wire["output_path"])
        self.assertEqual(wire["token_positions"], [])
        self.assertEqual(wire["generation"], {
            "stage": "unknown", "generation_step": None, "cache_position": None,
            "cache_length_before": None, "cache_length_after": None, "use_cache": None})
        self.assertEqual(self.schema.TraceEvent.from_dict(wire), event)

    def test_metadata_is_immutable_and_does_not_alias_mutable_lists(self):
        event = self.make_event()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            event.sequence = 99
        with self.assertRaises(dataclasses.FrozenInstanceError):
            event.identity.sample_id = "other"
        wire = event.to_dict()
        wire["shape"][0] = 100
        wire["token_positions"].append(2)
        self.assertEqual(event.shape, (4, 5, 6))
        self.assertEqual(event.token_positions, (1, 4))
        with self.assertRaises((TypeError, ValueError)):
            self.make_event(token_positions=[1, 4])

    def test_json_is_utf8_encodable_even_for_unpaired_surrogates_in_identifiers(self):
        event = self.make_event(module_path="module-\ud800")
        encoded = event.to_json().encode("utf-8")
        self.assertEqual(self.schema.TraceEvent.from_json(encoded.decode("utf-8")), event)

    def test_invalid_coordinates_types_and_axes_are_rejected(self):
        changes = [
            {"sequence": True}, {"sequence": -1}, {"forward_index": -1},
            {"invocation": -1}, {"layer_index": True}, {"layer_index": -1},
            {"shape": (4, -1, 6)}, {"shape": (4, True, 6)},
            {"output_path": [0]}, {"output_path": (-1,)}, {"output_path": (True,)},
            {"batch_positions": (-1,)}, {"token_positions": (True,)},
            {"batch_axis": 3}, {"token_axis": 0},
            {"batch_positions": (4,)}, {"token_positions": (5,)},
            {"module_path": " "}, {"module_kind": ""},
            {"dtype": ""}, {"device": ""}, {"capture_mode": ""},
        ]
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises((TypeError, ValueError)):
                    self.make_event(**change)

    def test_unknown_axis_does_not_invent_a_bounds_check(self):
        event = self.make_event(token_positions=(99,), token_axis=None)
        self.assertEqual(event.token_positions, (99,))
        self.assertIsNone(event.token_axis)

    def test_output_selector_distinguishes_root_unknown_and_nested_tensor(self):
        for path, wire_value in [(None, None), ((), []), ((0, "hidden"), [0, "hidden"])]:
            with self.subTest(path=path):
                event = self.make_event(output_path=path)
                self.assertEqual(event.to_dict()["output_path"], wire_value)
                self.assertEqual(self.schema.TraceEvent.from_json(event.to_json()).output_path, path)

    def test_invalid_generation_metadata_and_identity_are_rejected(self):
        for values in [dict(stage="invented"), dict(generation_step=True),
                       dict(cache_position=-1), dict(cache_length_before=-1),
                       dict(cache_length_after=False), dict(use_cache=1)]:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    self.schema.GenerationContext(**values)
        for args in [("",), ("   ",), ("run", ""), ("run", None, 7),
                     ("run", None, None, " ")]:
            with self.subTest(args=args):
                with self.assertRaises((TypeError, ValueError)):
                    self.schema.RunIdentity(*args)

    def test_json_deserialization_rejects_wrong_versions_and_incomplete_objects(self):
        wire = self.make_event().to_dict()
        for mutation in (lambda d: d.update(schema_version="2.0"),
                         lambda d: d.pop("layer_index"),
                         lambda d: d.update(unexpected="field"),
                         lambda d: d["identity"].pop("sample_id"),
                         lambda d: d["generation"].update(unexpected=True)):
            data = json.loads(json.dumps(wire))
            mutation(data)
            with self.subTest(data=data):
                with self.assertRaises((TypeError, ValueError)):
                    self.schema.TraceEvent.from_dict(data)
        for raw in ('[]', '{"schema_version":"1.0","schema_version":"2.0"}',
                    '{"nonfinite":NaN}'):
            with self.subTest(raw=raw):
                with self.assertRaises((TypeError, ValueError)):
                    self.schema.TraceEvent.from_json(raw)

    def test_sample_slices_can_share_one_module_invocation_without_aliasing_identity(self):
        first = self.make_event(sequence=3, batch_positions=(0,),
                                identity=self.schema.RunIdentity("run", "job", "req", "a"))
        second = self.make_event(sequence=4, batch_positions=(1,),
                                 identity=self.schema.RunIdentity("run", "job", "req", "b"))
        self.assertEqual((first.forward_index, first.module_path, first.invocation),
                         (second.forward_index, second.module_path, second.invocation))
        self.assertNotEqual((first.identity.run_id, first.sequence),
                            (second.identity.run_id, second.sequence))
        self.assertEqual(self.schema.TraceEvent.from_json(first.to_json()).identity.sample_id, "a")
        self.assertEqual(self.schema.TraceEvent.from_json(second.to_json()).identity.sample_id, "b")
        self.schema.validate_trace([first, second])
        with self.assertRaises(ValueError):
            self.schema.validate_trace([first, dataclasses.replace(second, sequence=3)])
        # Sequence numbers are scoped to their run and may recur in a different run.
        other_run = dataclasses.replace(second, sequence=3,
                                       identity=self.schema.RunIdentity("other-run"))
        self.schema.validate_trace([first, other_run])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_legacy_probe_event_retains_positional_construction_and_metadata(self):
        from looking_glass.probes import TraceEvent as LegacyTraceEvent
        legacy = LegacyTraceEvent("run", "node", 4, 2, (3, 5), "torch.float32", "cpu", None)
        self.assertEqual(legacy.metadata(), {
            "run_id": "run", "node_id": "node", "sequence": 4, "occurrence": 2,
            "shape": [3, 5], "dtype": "torch.float32", "device": "cpu", "tensor_bytes": 0})


if __name__ == "__main__":
    unittest.main()
