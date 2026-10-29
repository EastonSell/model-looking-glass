"""Request boundaries and failure semantics of batched trace capture."""
import asyncio
from concurrent.futures import CancelledError, Future
import gc
import threading
import unittest
from unittest import mock
import weakref

import torch
from torch import nn

from looking_glass.capture_dispatcher import (
    CaptureAbortedError,
    CaptureDispatcher,
    CaptureLimitError,
    CaptureRequest,
    SampleSelection,
)
from looking_glass.trace_schema import TraceEvent, validate_trace


class RepeatedTap(nn.Module):
    def __init__(self):
        super().__init__()
        self.tap = nn.Identity()

    def forward(self, x):
        first = self.tap(x)
        return self.tap(first + 100)


def request(request_id, samples, *, run_id="run", job_id=None,
            max_events=20, max_trace_bytes=1000, forward_index=0,
            generation_step=None):
    return CaptureRequest(
        run_id=run_id, job_id=job_id or f"job-{request_id}",
        request_id=request_id, samples=tuple(samples),
        forward_index=forward_index, generation_step=generation_step,
        max_events=max_events, max_trace_bytes=max_trace_bytes,
    )


def attach_tap(model, dispatcher):
    def capture(_module, _inputs, output):
        dispatcher.record("block.0.tap", layer_index=0,
                          module_kind="block", output=output, output_path=())
    return model.tap.register_forward_hook(capture)


def make_dispatcher(requests, **kwargs):
    kwargs.setdefault("sequence_starts", {r.run_id: 0 for r in requests})
    return CaptureDispatcher(requests, **kwargs)


class CaptureDispatcherTests(unittest.TestCase):
    def test_sequence_start_must_be_explicit_for_every_run(self):
        r = request("a", [SampleSelection(0, "s", (0,))])
        with self.assertRaises(ValueError):
            CaptureDispatcher((r,), batch_size=1, sequence_length=1,
                              max_events=2, max_trace_bytes=64)

    def test_splits_batch_and_repeated_invocations_with_owned_values(self):
        model = RepeatedTap()
        x = torch.arange(24.0).reshape(3, 4, 2).requires_grad_()
        a = request("a", [SampleSelection(0, "a0", (0, 3)),
                          SampleSelection(2, "a1", (1,))],
                    forward_index=4, generation_step=7)
        b = request("b", [SampleSelection(1, "b0", (2, 3))], forward_index=4)
        dispatcher = make_dispatcher((a, b), batch_size=3, sequence_length=4,
                                       max_events=6, max_trace_bytes=80)
        handle = attach_tap(model, dispatcher)
        try:
            output, events = dispatcher.run(lambda: model(x))
        finally:
            handle.remove()

        self.assertTrue(torch.equal(output, x + 100))
        self.assertEqual(len(events), 6)
        self.assertEqual([e.sequence for e in events], list(range(6)))
        self.assertEqual([e.invocation for e in events], [0, 0, 0, 1, 1, 1])
        self.assertEqual([e.request_id for e in events], ["a", "a", "b"] * 2)
        self.assertEqual([e.sample_id for e in events], ["a0", "a1", "b0"] * 2)
        self.assertEqual([e.batch_position for e in events], [0, 2, 1] * 2)
        self.assertEqual([e.token_positions for e in events], [(0, 3), (1,), (2, 3)] * 2)
        self.assertEqual(events[0].forward_index, 4)
        self.assertEqual(events[0].generation_step, 7)
        self.assertEqual(events[0].module_path, "block.0.tap")
        self.assertEqual(events[0].layer_index, 0)
        self.assertEqual(events[0].module_kind, "block")
        self.assertEqual(events[0].output_path, ())
        self.assertEqual(events[0].source_shape, (3, 4, 2))
        self.assertEqual(events[0].saved_shape, (2, 2))
        self.assertEqual(events[0].source_device, "cpu")
        self.assertEqual(events[0].saved_device, "cpu")
        self.assertTrue(torch.equal(events[0].value, torch.tensor([[0., 1.], [6., 7.]])))
        self.assertTrue(torch.equal(events[1].value, torch.tensor([[18., 19.]])))
        self.assertTrue(torch.equal(events[2].value, torch.tensor([[12., 13.], [14., 15.]])))
        self.assertTrue(torch.equal(events[3].value, torch.tensor([[100., 101.], [106., 107.]])))
        self.assertTrue(all(not event.value.requires_grad for event in events))
        self.assertEqual(sum(e.tensor_bytes for e in events), 80)
        metadata = [event.to_trace_event() for event in events]
        validate_trace(metadata)
        self.assertEqual([TraceEvent.from_json(event.to_json()) for event in metadata], metadata)
        self.assertEqual(metadata[0].identity.run_id, "run")
        self.assertEqual(metadata[0].identity.job_id, "job-a")
        self.assertEqual(metadata[0].identity.request_id, "a")
        self.assertEqual(metadata[0].identity.sample_id, "a0")
        self.assertEqual(metadata[0].batch_positions, (0,))
        self.assertEqual((metadata[0].batch_axis, metadata[0].token_axis), (0, 1))
        self.assertEqual(metadata[0].shape, (3, 4, 2))
        self.assertEqual(metadata[0].saved_shape, (2, 2))
        self.assertEqual(metadata[0].generation.stage, "unknown")
        self.assertEqual(metadata[0].generation.generation_step, 7)
        self.assertEqual(metadata[0].output_path, ())
        events[0].value[0, 0] = -99
        self.assertEqual(x[0, 0, 0].item(), 0)
        self.assertEqual(events[3].value[0, 0].item(), 100)
        self.assertEqual(dispatcher.retained_bytes, 0)

    def test_invalid_or_incomplete_ownership_is_rejected_before_forward(self):
        a = request("a", [SampleSelection(0, "a0", (0,))])
        cases = [
            (a,),  # row 1 has no owner
            (a, request("b", [SampleSelection(0, "b0", (0,))])),
            (a, request("b", [SampleSelection(2, "b0", (0,))])),
            (a, request("b", [SampleSelection(1, "b0", (4,))])),
            (a, request("b", [SampleSelection(True, "b0", (0,))])),
            (a, request("b", [SampleSelection(1, "b0", (-1,))])),
            (a, request("b", [SampleSelection(1, "b0", (True,))])),
            (a, request("b", [SampleSelection(1, "b0", (1, 1))])),
            (a, request("a", [SampleSelection(1, "b0", (0,))])),
            (a, request("b", [SampleSelection(1, "b0", (0,))], forward_index=1)),
            (request("a", [SampleSelection(0, "a0", (0,)),
                           SampleSelection(1, "a0", (0,))]),),
        ]
        for requests in cases:
            with self.subTest(requests=requests):
                with self.assertRaises(ValueError):
                    make_dispatcher(requests, batch_size=2, sequence_length=4,
                                      max_events=10, max_trace_bytes=100)

    def test_post_validation_mutation_cannot_change_batch_ownership_or_caps(self):
        sample_a = SampleSelection(0, "a0", (0,))
        sample_b = SampleSelection(1, "b0", (1,))
        first = request("a", [sample_a])
        second = request("b", [sample_b])
        starts = {"run": 4}
        d = make_dispatcher((first, second), batch_size=2, sequence_length=2,
                            max_events=2, max_trace_bytes=64,
                            sequence_starts=starts)
        object.__setattr__(sample_a, "batch_position", 1)
        object.__setattr__(sample_a, "sample_id", "forged-sample")
        object.__setattr__(sample_a, "token_positions", (1,))
        object.__setattr__(first, "request_id", "forged-request")
        object.__setattr__(first, "job_id", "forged-job")
        object.__setattr__(first, "run_id", "forged-run")
        object.__setattr__(first, "max_events", 0)
        object.__setattr__(first, "max_trace_bytes", 0)
        starts["run"] = 90
        source = torch.tensor([[[11.], [12.]], [[21.], [22.]]])
        _, events = d.run(lambda: d.record("tap", layer_index=0,
                                          module_kind="block", output=source))
        self.assertEqual([e.request_id for e in events], ["a", "b"])
        self.assertEqual([e.job_id for e in events], ["job-a", "job-b"])
        self.assertEqual([e.run_id for e in events], ["run", "run"])
        self.assertEqual([e.sample_id for e in events], ["a0", "b0"])
        self.assertEqual([e.batch_position for e in events], [0, 1])
        self.assertEqual([e.token_positions for e in events], [(0,), (1,)])
        self.assertEqual([e.sequence for e in events], [4, 5])
        self.assertEqual([e.value.item() for e in events], [11., 22.])

    def test_request_spec_subclasses_and_string_subclasses_are_rejected(self):
        class RequestSubclass(CaptureRequest):
            pass
        class SampleSubclass(SampleSelection):
            pass
        class IDSubclass(str):
            pass
        r = request("a", [SampleSelection(0, "s", (0,))])
        subclassed = RequestSubclass(r.run_id, r.job_id, r.request_id, r.samples,
                                     r.forward_index, r.generation_step,
                                     r.max_events, r.max_trace_bytes)
        cases = [subclassed,
                 request("a", [SampleSubclass(0, "s", (0,))]),
                 request(IDSubclass("a"), [SampleSelection(0, "s", (0,))])]
        for candidate in cases:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    make_dispatcher((candidate,), batch_size=1, sequence_length=1,
                                    max_events=2, max_trace_bytes=64)
        with self.assertRaises(ValueError):
            make_dispatcher((r,), batch_size=1, sequence_length=1,
                            max_events=2, max_trace_bytes=64,
                            sequence_starts={IDSubclass("run"): 0})

    def test_caps_are_checked_before_selecting_or_copying_any_slice(self):
        x = torch.ones(2, 4, 2)
        cases = [
            (1, 100, 2, 100),     # request event cap
            (2, 15, 2, 100),      # request byte cap: two tokens require 16
            (2, 100, 1, 100),     # batch event cap
            (2, 100, 2, 31),      # batch byte cap: two requests need 32
        ]
        for per_events, per_bytes, total_events, total_bytes in cases:
            with self.subTest(caps=(per_events, per_bytes, total_events, total_bytes)):
                if per_events == 1:
                    requests = (request("a", [SampleSelection(0, "a0", (0, 1)),
                                                SampleSelection(1, "a1", (0, 1))],
                                        max_events=per_events, max_trace_bytes=per_bytes),)
                else:
                    requests = (request("a", [SampleSelection(0, "a0", (0, 1))],
                                        max_events=per_events, max_trace_bytes=per_bytes),
                                request("b", [SampleSelection(1, "b0", (0, 1))],
                                        max_events=per_events, max_trace_bytes=100))
                dispatcher = make_dispatcher(requests, batch_size=2, sequence_length=4,
                                               max_events=total_events,
                                               max_trace_bytes=total_bytes)
                with mock.patch.object(torch.Tensor, "index_select",
                                       side_effect=AssertionError("allocation occurred")):
                    with self.assertRaises(CaptureLimitError):
                        dispatcher.run(lambda: dispatcher.record(
                            "tap", layer_index=0, module_kind="block", output=x))
                self.assertEqual(dispatcher.retained_bytes, 0)

    def test_caught_capture_failure_still_aborts_and_discards_prior_events(self):
        selection = request("a", [SampleSelection(0, "s", (0,))], max_events=1)
        dispatcher = make_dispatcher((selection,), batch_size=1, sequence_length=1,
                                       max_events=2, max_trace_bytes=100)
        x = torch.ones(1, 1, 2)

        def forward():
            dispatcher.record("tap", layer_index=0, module_kind="block", output=x)
            try:
                dispatcher.record("tap", layer_index=0, module_kind="block", output=x)
            except CaptureLimitError:
                pass
            return x

        with self.assertRaises(CaptureAbortedError):
            dispatcher.run(forward)
        self.assertEqual(dispatcher.retained_bytes, 0)
        with self.assertRaises(RuntimeError):
            dispatcher.run(lambda: x)

    def test_forward_failure_does_not_leak_into_later_request(self):
        model = RepeatedTap()
        a = request("a", [SampleSelection(0, "s", (0,))])
        first = make_dispatcher((a,), batch_size=1, sequence_length=1,
                                  max_events=4, max_trace_bytes=64)
        h = attach_tap(model, first)
        try:
            def fail():
                model(torch.tensor([[[11.]]]))
                raise RuntimeError("forward failed")
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                first.run(fail)
        finally:
            h.remove()
        self.assertEqual(first.retained_bytes, 0)
        second = make_dispatcher((request("b", [SampleSelection(0, "t", (0,))]),),
                                   batch_size=1, sequence_length=1,
                                   max_events=4, max_trace_bytes=64)
        h = attach_tap(model, second)
        try:
            _, events = second.run(lambda: model(torch.tensor([[[22.]]])))
        finally:
            h.remove()
        self.assertEqual([e.request_id for e in events], ["b", "b"])
        self.assertEqual([e.value.item() for e in events], [22., 122.])

    def test_cancellation_and_oom_release_owned_slices_and_execution_lock(self):
        """Simulated failures exercise cleanup; no CUDA allocation is made."""
        original_select = torch.Tensor.index_select
        for failure_type in (CancelledError, torch.OutOfMemoryError, KeyboardInterrupt):
            with self.subTest(failure_type=failure_type):
                refs = []
                def track_select(tensor, *args, **kwargs):
                    value = original_select(tensor, *args, **kwargs)
                    refs.append(weakref.ref(value))
                    return value
                r = request("a", [SampleSelection(0, "s", (0,))])
                d = make_dispatcher((r,), batch_size=1, sequence_length=1,
                                    max_events=2, max_trace_bytes=64)
                def forward():
                    d.record("tap", layer_index=0, module_kind="block",
                             output=torch.ones(1, 1, 1))
                    raise failure_type("simulated")
                with mock.patch.object(torch.Tensor, "index_select", track_select):
                    with self.assertRaises(failure_type):
                        d.run(forward)
                gc.collect()
                self.assertEqual(len(refs), 1)
                self.assertIsNone(refs[0]())
                self.assertEqual(d.retained_bytes, 0)
                # A failed or cancelled call must release the process-local guard.
                later = make_dispatcher((r,), batch_size=1, sequence_length=1,
                                        max_events=2, max_trace_bytes=64)
                output, events = later.run(lambda: 42)
                self.assertEqual((output, events), (42, ()))

    def test_caught_limit_failure_releases_previously_owned_slices(self):
        original_select = torch.Tensor.index_select
        refs = []
        def track_select(tensor, *args, **kwargs):
            value = original_select(tensor, *args, **kwargs)
            refs.append(weakref.ref(value))
            return value
        r = request("a", [SampleSelection(0, "s", (0,))], max_events=1)
        d = make_dispatcher((r,), batch_size=1, sequence_length=1,
                            max_events=2, max_trace_bytes=64)
        def forward():
            value = torch.ones(1, 1, 1)
            d.record("tap", layer_index=0, module_kind="block", output=value)
            try:
                d.record("tap", layer_index=0, module_kind="block", output=value)
            except CaptureLimitError:
                pass
        with mock.patch.object(torch.Tensor, "index_select", track_select):
            with self.assertRaises(CaptureAbortedError):
                d.run(forward)
        gc.collect()
        self.assertEqual(len(refs), 1)
        self.assertIsNone(refs[0]())
        self.assertEqual(d.retained_bytes, 0)

    def test_retained_error_does_not_keep_partial_event_snapshots(self):
        original_select = torch.Tensor.index_select
        refs = []
        calls = 0
        def fail_second_selection(tensor, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated selection failure")
            value = original_select(tensor, *args, **kwargs)
            refs.append(weakref.ref(value))
            return value
        r = request("a", [SampleSelection(0, "a0", (0,)),
                          SampleSelection(1, "a1", (0,))])
        d = make_dispatcher((r,), batch_size=2, sequence_length=1,
                            max_events=2, max_trace_bytes=64)
        source = torch.ones(2, 1, 1)
        retained_error = None
        with mock.patch.object(torch.Tensor, "index_select", fail_second_selection):
            try:
                d.run(lambda: d.record("tap", layer_index=0, module_kind="block",
                                       output=source))
            except RuntimeError as error:
                retained_error = error
        self.assertIsNotNone(retained_error)
        self.assertIsNotNone(retained_error.__traceback__)
        gc.collect()
        self.assertEqual(len(refs), 1)
        self.assertIsNone(refs[0]())
        self.assertEqual(d.retained_bytes, 0)

    def test_interrupted_publication_drops_result_tuple_from_retained_traceback(self):
        class InterruptOnce(list):
            def __init__(self):
                super().__init__()
                self.interrupt = True
            def clear(self):
                super().clear()
                if self.interrupt:
                    self.interrupt = False
                    raise KeyboardInterrupt("simulated publication interrupt")
        original_select = torch.Tensor.index_select
        refs = []
        def track_select(tensor, *args, **kwargs):
            value = original_select(tensor, *args, **kwargs)
            refs.append(weakref.ref(value))
            return value
        r = request("a", [SampleSelection(0, "s", (0,))])
        d = make_dispatcher((r,), batch_size=1, sequence_length=1,
                            max_events=2, max_trace_bytes=64)
        d._events = InterruptOnce()
        retained_error = None
        with mock.patch.object(torch.Tensor, "index_select", track_select):
            try:
                d.run(lambda: d.record("tap", layer_index=0, module_kind="block",
                                       output=torch.ones(1, 1, 1)))
            except KeyboardInterrupt as error:
                retained_error = error
        self.assertIsNotNone(retained_error.__traceback__)
        gc.collect()
        self.assertEqual(len(refs), 1)
        self.assertIsNone(refs[0]())
        self.assertEqual(d.retained_bytes, 0)

    def test_forward_and_input_gradients_match_uninstrumented_reference(self):
        model = RepeatedTap()
        reference = torch.tensor([[[2., -3.], [4., 5.]]], requires_grad=True)
        observed = reference.detach().clone().requires_grad_()
        expected = model(reference)
        expected.square().sum().backward()
        d = make_dispatcher((request("a", [SampleSelection(0, "s", (1,))]),),
                              batch_size=1, sequence_length=2,
                              max_events=2, max_trace_bytes=64, capture_device="source")
        h = attach_tap(model, d)
        try:
            actual, _ = d.run(lambda: model(observed))
        finally:
            h.remove()
        actual.square().sum().backward()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(observed.grad, reference.grad))

    def test_nested_dispatch_is_rejected_even_if_outer_forward_catches_it(self):
        r = request("a", [SampleSelection(0, "s", (0,))])
        outer = make_dispatcher((r,), batch_size=1, sequence_length=1,
                                  max_events=2, max_trace_bytes=64)
        inner = make_dispatcher((r,), batch_size=1, sequence_length=1,
                                  max_events=2, max_trace_bytes=64)
        def forward():
            with self.assertRaises(RuntimeError):
                inner.run(lambda: None)
            outer.record("tap", layer_index=None, module_kind="test",
                         output=torch.ones(1, 1, 1))
        _, events = outer.run(forward)
        self.assertEqual(len(events), 1)

    def test_cross_thread_record_aborts_even_if_the_error_is_caught(self):
        r = request("a", [SampleSelection(0, "s", (0,))])
        dispatcher = make_dispatcher((r,), batch_size=1, sequence_length=1,
                                       max_events=2, max_trace_bytes=64)
        failures = []
        def forward():
            def other_thread():
                try:
                    dispatcher.record("tap", layer_index=0, module_kind="block",
                                      output=torch.ones(1, 1, 1))
                except RuntimeError as exc:
                    failures.append(exc)
            thread = threading.Thread(target=other_thread)
            thread.start()
            thread.join()
        with self.assertRaises(CaptureAbortedError):
            dispatcher.run(forward)
        self.assertEqual(len(failures), 1)
        self.assertEqual(dispatcher.retained_bytes, 0)

    def test_reentrant_record_poisoning_cannot_publish_partial_results(self):
        r = request("a", [SampleSelection(0, "s", (0,))])
        dispatcher = make_dispatcher((r,), batch_size=1, sequence_length=1,
                                       max_events=2, max_trace_bytes=64)
        x = torch.ones(1, 1, 1)
        original = torch.Tensor.index_select
        def reenter(tensor, *args, **kwargs):
            try:
                dispatcher.record("nested", layer_index=None, module_kind="test", output=x)
            except RuntimeError:
                pass
            return original(tensor, *args, **kwargs)
        with mock.patch.object(torch.Tensor, "index_select", reenter):
            with self.assertRaises(CaptureAbortedError):
                dispatcher.run(lambda: dispatcher.record(
                    "tap", layer_index=0, module_kind="block", output=x))
        self.assertEqual(dispatcher.retained_bytes, 0)

    def test_bad_tensor_contract_is_rejected_and_poisoned_even_when_caught(self):
        class FakeTensorSubclass(torch.Tensor):
            pass
        bad_values = [torch.ones(1, 1), torch.ones(1, 2, 1),
                      torch.empty(1, 1, 0), torch.ones(1, 1, 1, dtype=torch.int64),
                      (torch.ones(1, 1, 1),), torch.ones(1, 1, 1, device="meta"),
                      torch.ones(1, 1, 1).as_subclass(FakeTensorSubclass)]
        for value in bad_values:
            with self.subTest(value=value):
                d = make_dispatcher((request("a", [SampleSelection(0, "s", (0,))]),),
                                      batch_size=1, sequence_length=1,
                                      max_events=2, max_trace_bytes=64)
                def forward():
                    try:
                        d.record("tap", layer_index=0, module_kind="block", output=value)
                    except (TypeError, ValueError):
                        pass
                with self.assertRaises(CaptureAbortedError):
                    d.run(forward)
                self.assertEqual(d.retained_bytes, 0)

    def test_nonfinite_values_are_preserved_as_raw_unvalidated_data(self):
        d = make_dispatcher((request("a", [SampleSelection(0, "s", (0,))]),),
                              batch_size=1, sequence_length=1,
                              max_events=2, max_trace_bytes=64)
        def forward():
            d.record("tap", layer_index=0, module_kind="block",
                     output=torch.tensor([[[float("nan"), float("inf")]]]))
        _, events = d.run(forward)
        self.assertTrue(torch.isnan(events[0].value[0, 0]))
        self.assertTrue(torch.isinf(events[0].value[0, 1]))

    def test_non_schema_output_path_is_rejected_before_any_result(self):
        class Selector(str):
            pass
        d = make_dispatcher((request("a", [SampleSelection(0, "s", (0,))]),),
                            batch_size=1, sequence_length=1,
                            max_events=2, max_trace_bytes=64)
        def forward():
            try:
                d.record("tap", layer_index=0, module_kind="block",
                         output=torch.ones(1, 1, 1), output_path=(Selector("hidden"),))
            except ValueError:
                pass
        with self.assertRaises(CaptureAbortedError):
            d.run(forward)
        self.assertEqual(d.retained_bytes, 0)

    def test_schema_conversion_rejects_changed_payload_shape(self):
        d = make_dispatcher((request("a", [SampleSelection(0, "s", (0,))]),),
                            batch_size=1, sequence_length=1,
                            max_events=2, max_trace_bytes=64)
        def forward():
            d.record("tap", layer_index=0, module_kind="block", output=torch.ones(1, 1, 1))
        _, events = d.run(forward)
        events[0].value.resize_(3)
        with self.assertRaises(ValueError):
            events[0].to_trace_event()

    def test_inference_mode_capture_returns_independently_mutable_owned_tensor(self):
        d = make_dispatcher((request("a", [SampleSelection(0, "s", (0,))]),),
                            batch_size=1, sequence_length=1,
                            max_events=2, max_trace_bytes=64)
        with torch.inference_mode():
            source = torch.tensor([[[3.]]])
            def forward():
                d.record("tap", layer_index=0, module_kind="block", output=source)
            _, events = d.run(forward)
        events[0].value.add_(2)
        self.assertEqual(events[0].value.item(), 5.)
        self.assertEqual(source.item(), 3.)

    def test_owner_supplied_sequence_start_spans_multiple_forwards(self):
        r = request("a", [SampleSelection(0, "s", (0,))], forward_index=3)
        first = make_dispatcher((r,), batch_size=1, sequence_length=1,
                                  max_events=2, max_trace_bytes=64,
                                  sequence_starts={"run": 12})
        def capture(d):
            d.record("tap", layer_index=0, module_kind="block", output=torch.ones(1, 1, 1))
        _, one = first.run(lambda: capture(first))
        next_request = request("a", [SampleSelection(0, "s", (0,))], forward_index=4)
        second = make_dispatcher((next_request,), batch_size=1, sequence_length=1,
                                   max_events=2, max_trace_bytes=64,
                                   sequence_starts={"run": 13})
        _, two = second.run(lambda: capture(second))
        self.assertEqual((one[0].sequence, two[0].sequence), (12, 13))
        self.assertEqual((one[0].invocation, two[0].invocation), (0, 0))
        self.assertEqual((one[0].forward_index, two[0].forward_index), (3, 4))
        validate_trace((one[0].to_trace_event(), two[0].to_trace_event()))

    def test_async_context_and_async_forward_are_rejected(self):
        r = request("a", [SampleSelection(0, "s", (0,))])
        async def async_forward():
            return 1
        d = make_dispatcher((r,), batch_size=1, sequence_length=1,
                              max_events=2, max_trace_bytes=64)
        with self.assertRaises(RuntimeError):
            d.run(async_forward)
        future_dispatcher = make_dispatcher((r,), batch_size=1, sequence_length=1,
                                              max_events=2, max_trace_bytes=64)
        with self.assertRaises(RuntimeError):
            future_dispatcher.run(lambda: Future())
        async def in_loop():
            other = make_dispatcher((r,), batch_size=1, sequence_length=1,
                                       max_events=2, max_trace_bytes=64)
            with self.assertRaises(RuntimeError):
                other.run(lambda: 1)
        asyncio.run(in_loop())


if __name__ == "__main__":
    unittest.main()
