import concurrent.futures
import gc
import inspect
import sys
import threading
import unittest
import weakref
from unittest.mock import patch

import torch

from looking_glass.trace_store import (
    HostTraceStore, TraceReservation, TraceStoreCancelled, TraceStoreFull, TraceStoreNotReady,
)
import looking_glass.trace_store as trace_store


class HostTraceStoreTests(unittest.TestCase):
    def test_exact_global_limits_and_ownership_transfer(self):
        store = HostTraceStore(max_tensor_bytes=16, max_items=2)
        first = store.request("first")
        second = store.request("second")
        original = torch.tensor([1.0, 2.0])
        a = first.put(0, original)
        b = second.put(0, torch.tensor([3.0, 4.0]))
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (16, 2))
        with self.assertRaises(TraceStoreFull):
            first.put(1, torch.empty(0))  # even zero-byte entries consume a slot

        original.fill_(9)
        inspected = a.read()
        inspected.fill_(8)
        transferred = a.take()
        self.assertTrue(torch.equal(transferred, torch.tensor([1.0, 2.0])))
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (8, 1))
        first.put(1, torch.tensor([5.0, 6.0]))
        transferred.fill_(7)
        self.assertTrue(torch.equal(b.read(), torch.tensor([3.0, 4.0])))
        with self.assertRaises(KeyError):
            a.take()

    def test_metadata_only_and_zero_sized_tensors_consume_item_slots(self):
        store = HostTraceStore(max_tensor_bytes=0, max_items=2)
        request = store.request("run")
        metadata = request.put(0, None)
        empty = request.put(1, torch.empty(0))
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 2))
        self.assertIsNone(metadata.take())
        self.assertEqual(empty.take().numel(), 0)
        self.assertEqual(store.usage().claimed_items, 0)

    def test_inference_mode_copies_remain_mutable_after_context(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        request = store.request("run")
        with torch.inference_mode():
            handle = request.put(0, torch.tensor([1.0]))
            preview = handle.read()
        taken = handle.take()
        preview.add_(1)
        taken.add_(2)
        self.assertTrue(torch.equal(preview, torch.tensor([2.0])))
        self.assertTrue(torch.equal(taken, torch.tensor([3.0])))

    def test_invalid_limits_and_unsupported_inputs_do_not_claim(self):
        for kwargs in ({"max_tensor_bytes": -1, "max_items": 1},
                       {"max_tensor_bytes": True, "max_items": 1},
                       {"max_tensor_bytes": 1, "max_items": -1},
                       {"max_tensor_bytes": 1, "max_items": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                HostTraceStore(**kwargs)
        store = HostTraceStore(max_tensor_bytes=64, max_items=3)
        request = store.request("run")
        sparse = torch.sparse_coo_tensor([[0]], [1.0], (1,), check_invariants=True)
        for value in (sparse, torch.empty(1, device="meta"), [1.0]):
            with self.subTest(value=type(value)), self.assertRaises((TypeError, ValueError)):
                request.put(0, value)
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))

    def test_text_identifiers_are_bounded(self):
        store = HostTraceStore(max_tensor_bytes=0, max_items=1)
        with self.assertRaises(ValueError):
            store.request("r" * 257)
        request = store.request("r" * 256)
        with self.assertRaises(ValueError):
            request.reserve("i" * 257, 0)
        request.reserve("i" * 256, 0).cancel()
        self.assertEqual(store.usage().claimed_items, 0)

    def test_claim_precedes_copy_and_duplicate_key_is_rejected(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=2)
        request = store.request("run")
        claim = request.reserve(0, 4)
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (4, 1))
        with self.assertRaises(ValueError):
            request.reserve(0, 0)
        with patch("looking_glass.trace_store._clone_tensor", side_effect=AssertionError("copied")):
            with self.assertRaises(TraceStoreFull):
                request.put(1, torch.ones(1))
        with self.assertRaises(TraceStoreNotReady):
            claim.take()
        claim.cancel()
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))

    def test_unregistered_handle_cannot_copy_without_a_claim(self):
        store = HostTraceStore(max_tensor_bytes=0, max_items=0)
        claim = TraceReservation(store.request("run"), 0, 4)
        calls = []
        with patch("looking_glass.trace_store._clone_tensor", side_effect=lambda value: calls.append(value)):
            with self.assertRaises(RuntimeError):
                claim.commit(torch.ones(1))
        self.assertEqual(calls, [])
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))

    def test_context_and_copy_failure_release_claims(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        request = store.request("run")
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with request.reserve(0, 4):
                raise RuntimeError("deliberate")
        self.assertEqual(store.usage().claimed_items, 0)

        claim = request.reserve(0, 4)
        with patch("looking_glass.trace_store._clone_tensor", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                claim.commit(torch.ones(1))
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))
        self.assertTrue(torch.equal(request.put(0, torch.ones(1)).take(), torch.ones(1)))

    def test_exception_after_commit_rolls_back_context_reservation(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        request = store.request("run")
        with self.assertRaisesRegex(RuntimeError, "after commit"):
            with request.reserve(0, 4) as claim:
                claim.commit(torch.ones(1))
                raise RuntimeError("after commit")
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))
        with self.assertRaises(KeyError):
            claim.read()
        self.assertTrue(torch.equal(request.put(0, torch.ones(1)).take(), torch.ones(1)))

    def test_source_size_change_fails_without_publishing(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        request = store.request("run")
        claim = request.reserve(0, 4)
        with self.assertRaises(ValueError):
            claim.commit(torch.ones(2))
        self.assertEqual(store.usage().claimed_items, 0)
        with self.assertRaises(KeyError):
            claim.take()

        claim = request.reserve(0, 4)
        with patch("looking_glass.trace_store._clone_tensor", return_value=torch.ones(2)):
            with self.assertRaises(ValueError):
                claim.commit(torch.ones(1))
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))

    def test_failed_postcopy_validation_does_not_retain_clone_in_traceback(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        claim = store.request("run").reserve(0, 4)
        references = []

        def oversized_copy(_value):
            copied = torch.ones(2)
            references.append(weakref.ref(copied))
            return copied

        with patch("looking_glass.trace_store._clone_tensor", side_effect=oversized_copy):
            try:
                claim.commit(torch.ones(1))
            except ValueError as error:
                retained_error = error
            else:
                self.fail("a mismatched copy was accepted")
        gc.collect()
        self.assertIsInstance(retained_error, ValueError)
        self.assertIsNone(references[0]())
        self.assertEqual(store.usage().claimed_bytes, 0)

    def test_failed_read_does_not_keep_dropped_store_tensor_in_traceback(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        handle = store.request("run").put(0, torch.ones(1))
        reference = weakref.ref(handle._value)
        with patch.object(torch.Tensor, "clone", side_effect=MemoryError("copy failed")):
            try:
                handle.read()
            except MemoryError as error:
                retained_error = error
            else:
                self.fail("read failure did not occur")
        handle.drop()
        gc.collect()
        self.assertIsInstance(retained_error, MemoryError)
        self.assertIsNone(reference())
        self.assertEqual(store.usage().claimed_bytes, 0)

    def test_cancelling_inflight_copy_keeps_claim_until_acknowledged(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        request = store.request("run")
        claim = request.reserve(0, 4)
        started, resume = threading.Event(), threading.Event()
        clone = trace_store._clone_tensor

        def slow_clone(value):
            started.set()
            if not resume.wait(5):
                raise TimeoutError("test did not release clone")
            return clone(value)

        with patch("looking_glass.trace_store._clone_tensor", side_effect=slow_clone):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(claim.commit, torch.ones(1))
                try:
                    self.assertTrue(started.wait(5))
                    request.cancel()
                    self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (4, 1))
                    with self.assertRaises(TraceStoreCancelled):
                        request.reserve(1, 0)
                    with self.assertRaises(TraceStoreFull):
                        store.request("other").reserve(0, 4)
                finally:
                    resume.set()
                with self.assertRaises(TraceStoreCancelled):
                    future.result(timeout=5)
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))
        self.assertTrue(torch.equal(store.request("other").put(0, torch.ones(1)).take(), torch.ones(1)))

    def test_cancellation_before_copy_starts_avoids_copy(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        request = store.request("run")
        claim = request.reserve(0, 4)
        started, resume = threading.Event(), threading.Event()
        inspect = trace_store._payload_info

        def slow_inspect(value):
            started.set()
            if not resume.wait(5):
                raise TimeoutError("test did not release preflight")
            return inspect(value)

        with patch("looking_glass.trace_store._payload_info", side_effect=slow_inspect):
            with patch("looking_glass.trace_store._clone_tensor", side_effect=AssertionError("copied")):
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(claim.commit, torch.ones(1))
                    try:
                        self.assertTrue(started.wait(5))
                        request.cancel()
                    finally:
                        resume.set()
                    with self.assertRaises(TraceStoreCancelled):
                        future.result(timeout=5)
        self.assertEqual(store.usage().claimed_items, 0)

    def test_two_committers_cannot_share_one_claim(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        claim = store.request("run").reserve(0, 4)
        started, resume = threading.Event(), threading.Event()
        clone = trace_store._clone_tensor

        def slow_clone(value):
            started.set()
            if not resume.wait(5):
                raise TimeoutError("test did not release clone")
            return clone(value)

        with patch("looking_glass.trace_store._clone_tensor", side_effect=slow_clone):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(claim.commit, torch.tensor([1.0]))
                try:
                    self.assertTrue(started.wait(5))
                    with self.assertRaises(RuntimeError):
                        claim.commit(torch.tensor([2.0]))
                finally:
                    resume.set()
                future.result(timeout=5)
        self.assertTrue(torch.equal(claim.take(), torch.tensor([1.0])))
        self.assertEqual(store.usage().claimed_items, 0)

    def test_interrupt_immediately_after_copy_transition_releases_claim(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        claim = store.request("run").reserve(0, 4)
        previous_trace = sys.gettrace()

        def interrupt(frame, event, arg):
            if (frame.f_code is trace_store.TraceReservation.commit.__code__ and
                    event == "line" and claim._state == "copying"):
                sys.settrace(None)
                raise KeyboardInterrupt()
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                claim.commit(torch.ones(1))
        finally:
            sys.settrace(previous_trace)
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))

    def test_interrupt_before_publication_releases_claim_and_copy(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        claim = store.request("run").reserve(0, 4)
        underlying = store._lock

        class InterruptingLock:
            def __init__(self):
                self.calls = 0

            def __enter__(self):
                self.calls += 1
                if self.calls == 3:
                    raise KeyboardInterrupt()
                return underlying.__enter__()

            def __exit__(self, *args):
                return underlying.__exit__(*args)

        store._lock = InterruptingLock()
        references = []
        clone = trace_store._clone_tensor

        def watch_clone(value):
            result = clone(value)
            references.append(weakref.ref(result))
            return result

        with patch("looking_glass.trace_store._clone_tensor", side_effect=watch_clone):
            try:
                claim.commit(torch.ones(1))
            except KeyboardInterrupt as error:
                retained_error = error
            else:
                self.fail("publication interruption did not occur")
        gc.collect()
        self.assertIsInstance(retained_error, KeyboardInterrupt)
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))
        self.assertIsNone(references[0]())

    def test_aliases_are_independently_copied_and_stale_handles_are_safe(self):
        store = HostTraceStore(max_tensor_bytes=16, max_items=2)
        request = store.request("run")
        source = torch.arange(1000, dtype=torch.float32)
        a = request.put(0, source[:2])
        b = request.put(1, source[:2])
        self.assertEqual(a.read().untyped_storage().nbytes(), 8)
        source[:2].fill_(9)
        taken = a.take()
        taken.fill_(7)
        self.assertTrue(torch.equal(b.read(), torch.tensor([0.0, 1.0])))
        replacement = request.put(0, torch.tensor([5.0, 6.0]))
        self.assertFalse(a.drop())
        self.assertTrue(torch.equal(replacement.take(), torch.tensor([5.0, 6.0])))
        request.cancel()
        with self.assertRaises(KeyError):
            b.read()
        self.assertEqual(store.usage().claimed_items, 0)

    def test_discarded_payload_is_freed_before_its_claim_is_released(self):
        for operation in ("drop", "cancel"):
            with self.subTest(operation=operation):
                store = HostTraceStore(max_tensor_bytes=4, max_items=1)
                request = store.request("run")
                observed = []
                references = []
                clone = trace_store._clone_tensor

                def watch_clone(value):
                    result = clone(value)
                    references.append(weakref.ref(
                        result, lambda _ref: observed.append(
                            (store.usage().claimed_bytes, store.usage().claimed_items))))
                    return result

                with patch("looking_glass.trace_store._clone_tensor", side_effect=watch_clone):
                    handle = request.put(0, torch.ones(1))
                if operation == "drop":
                    handle.drop()
                else:
                    request.cancel()
                self.assertEqual(observed, [(4, 1)])
                self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))

    def test_disposal_callback_cannot_erase_a_reentrant_reservation(self):
        store = HostTraceStore(max_tensor_bytes=8, max_items=2)
        request = store.request("run")
        handle = request.put(0, torch.ones(1))
        created = []
        reference = weakref.ref(handle._value, lambda _ref: created.append(request.reserve(1, 4)))
        handle.drop()
        self.assertIsNone(reference())
        self.assertEqual(len(created), 1)
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (4, 1))
        with self.assertRaises(TraceStoreFull):
            request.reserve(2, 8)
        created[0].cancel()
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))

    def test_interrupted_take_preserves_reentrant_disposal_claim(self):
        store = HostTraceStore(max_tensor_bytes=8, max_items=2)
        request = store.request("run")
        handle = request.put(0, torch.ones(1))
        created = []
        reference = weakref.ref(handle._value, lambda _ref: created.append(request.reserve(1, 4)))
        previous_trace = sys.gettrace()
        source, first_line = inspect.getsourcelines(trace_store.HostTraceStore._remove_locked)
        remove_line = first_line + next(i for i, line in enumerate(source)
                                        if line.strip() == "del self._entries[key]")

        def interrupt(frame, event, arg):
            if (frame.f_code is trace_store.HostTraceStore._remove_locked.__code__ and
                    event == "line" and frame.f_lineno == remove_line):
                sys.settrace(None)
                raise KeyboardInterrupt()
            return interrupt

        try:
            sys.settrace(interrupt)
            try:
                handle.take()
            except KeyboardInterrupt as error:
                retained_error = error
            else:
                self.fail("take interruption did not occur")
        finally:
            sys.settrace(previous_trace)
        self.assertIsInstance(retained_error, KeyboardInterrupt)
        self.assertIsNone(reference())
        self.assertEqual(len(created), 1)
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (4, 1))
        with self.assertRaises(TraceStoreFull):
            request.reserve(2, 8)
        created[0].cancel()

    def test_concurrent_reservations_obey_aggregate_caps(self):
        store = HostTraceStore(max_tensor_bytes=20, max_items=5)
        request = store.request("run")
        barrier = threading.Barrier(20)

        def reserve(index):
            barrier.wait(timeout=5)
            try:
                return request.reserve(index, 4)
            except TraceStoreFull:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            handles = list(pool.map(reserve, range(20)))
        self.assertEqual(sum(h is not None for h in handles), 5)
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (20, 5))
        request.cancel()
        self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))

    def test_interrupted_reservation_registration_is_rolled_back(self):
        for partial_items, partial_bytes in ((0, 0), (1, 0), (1, 4)):
            with self.subTest(partial_items=partial_items, partial_bytes=partial_bytes):
                store = HostTraceStore(max_tensor_bytes=4, max_items=1)
                request = store.request("run")
                previous_trace = sys.gettrace()

                def interrupt(frame, event, arg):
                    if (frame.f_code is trace_store.TraceRequest.reserve.__code__ and
                            event == "line" and len(store._entries) == 1 and
                            store._claimed_items == partial_items and store._claimed_bytes == partial_bytes):
                        sys.settrace(None)
                        raise KeyboardInterrupt()
                    return interrupt

                try:
                    sys.settrace(interrupt)
                    with self.assertRaises(KeyboardInterrupt):
                        request.reserve(0, 4)
                finally:
                    sys.settrace(previous_trace)
                self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))
                request.reserve(0, 4).cancel()

    def test_interrupted_disposal_completes_accounting(self):
        for entries, nbytes, items in ((1, 4, 1), (0, 4, 1), (0, 0, 1), (0, 0, 0)):
            with self.subTest(entries=entries, nbytes=nbytes, items=items):
                store = HostTraceStore(max_tensor_bytes=4, max_items=1)
                request = store.request("run")
                handle = request.put(0, torch.ones(1))
                previous_trace = sys.gettrace()

                def interrupt(frame, event, arg):
                    if (frame.f_code is trace_store.HostTraceStore._remove_locked.__code__ and
                            event == "line" and handle._state == "releasing" and
                            handle._value is None and len(store._entries) == entries and
                            store._claimed_bytes == nbytes and store._claimed_items == items):
                        sys.settrace(None)
                        raise KeyboardInterrupt()
                    return interrupt

                try:
                    sys.settrace(interrupt)
                    with self.assertRaises(KeyboardInterrupt):
                        handle.drop()
                finally:
                    sys.settrace(previous_trace)
                self.assertEqual((store.usage().claimed_bytes, store.usage().claimed_items), (0, 0))
                with self.assertRaises(KeyError):
                    handle.read()
                request.reserve(0, 4).cancel()


if __name__ == "__main__":
    unittest.main()
