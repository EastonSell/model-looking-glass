"""Trusted CPU-only sealed-batch handoff and delivery custody."""
import importlib
import importlib.util
import dis
import sys
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from looking_glass.cpu_trace_owner import CpuTraceOwner
from looking_glass.host_budget import HostBudget, HostBudgetExceeded
from looking_glass.resident_worker import ForwardRequest
from looking_glass.trace_producer import ClaimLimit, ProducerProfile
from looking_glass.trace_store import HostTraceStore
from looking_glass.traced_broker_worker import (
    BrokerJobIdentity, SampleTrace, SiteTrace, TraceLimits, TraceSpec, bind_trace_plan,
)

if __package__:
    from .opcode_support import run_opcode_trace
else:
    from opcode_support import run_opcode_trace


@unittest.skipUnless(importlib.util.find_spec("transformers") is not None,
                     "requires the optional transformers dependency")
class TraceBatchHandoffTests(unittest.TestCase):
    def make_sealed_job(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from looking_glass.traced_worker import TracedGPT2Worker
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(71)
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=32, n_positions=8, n_ctx=8, n_embd=8, n_layer=1,
                n_head=2, use_cache=False, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()
        worker = TracedGPT2Worker("tiny", model, device="cpu", max_batch_size=2,
                                  max_sequence_length=3, max_input_bytes=48,
                                  max_output_bytes=768)
        self.addCleanup(worker.close)
        budget = HostBudget(max_bytes=2_000_000, max_items=96, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=4096, max_items=8)
        owner = CpuTraceOwner(worker=worker, budget=budget, store=store,
                              broker_id="a" * 32, process_generation="b" * 32,
                              max_slots=4, max_event_bytes=64)
        profile = ProducerProfile(ClaimLimit(96, 8), ClaimLimit(1536, 2),
                                  ClaimLimit(128, 4), ClaimLimit(100_000, 1))
        identity = BrokerJobIdentity("a" * 32, 7, "team", "request")
        spec = TraceSpec("run", (SampleTrace("left", (0,)), SampleTrace("right", (2,))),
                         (SiteTrace(0, "mlp"), SiteTrace(0, "attn")), 0, 10, None, 4, 128)
        plan = bind_trace_plan(identity, spec, TraceLimits(2, 3, 1, 2, 4, 128, 3),
                               batch_size=2, sequence_length=3, layer_count=1)
        owner.register(plan, profile=profile, registration_id="c" * 32, generation=1,
                       batch_size=2, sequence_length=3)
        job = owner.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        caller = budget.open_request("caller")
        claim = caller.reserve("caller", "input", 48, items=1)
        ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.int64)
        request = ForwardRequest(plan.worker_request_id, {"input_ids": ids})
        job.execute(request, input_owner=caller, input_claim=claim)
        job.discard_result()
        job.confirm_producer_disposed()
        job.seal()
        request = ids = None
        claim.release()
        caller.close()
        return owner, job, budget

    def make_bridge(self):
        try:
            module = importlib.import_module("looking_glass.trace_batch_handoff")
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, "sealed trace-batch handoff is not implemented")
        owner, job, budget = self.make_sealed_job()
        registry = module.SealedBatchRegistry(
            budget=budget, broker_id="a" * 32, process_generation="b" * 32,
            max_jobs=1, max_events=4)
        registry.register(owner, job, transfer_id="d" * 32, recipient_id="reader")
        bridge = registry.lookup(7, registration_id="c" * 32, generation=1,
                                 transfer_id="d" * 32)
        return module, registry, bridge, budget

    def test_complete_manifest_and_one_shot_receipt_before_release(self):
        module, registry, bridge, budget = self.make_bridge()
        baseline = budget.snapshot().used_bytes
        bridge.prepare()
        self.assertGreater(budget.snapshot().used_bytes, baseline)
        bridge.offer()
        recipient = bridge.accept(recipient_id="reader", transfer_id="d" * 32)
        self.assertIs(recipient, bridge.recover_recipient(recipient_id="reader", transfer_id="d" * 32))
        with self.assertRaises(module.TraceHandoffStateError):
            recipient.ack_manifest()
        manifest = recipient.manifest()
        keys = recipient.events()
        with self.assertRaises(module.TraceHandoffStateError):
            recipient.send(keys[0], lambda payload: None)
        self.assertTrue(recipient.ack_manifest())
        self.assertFalse(recipient.ack_manifest())
        self.assertEqual(len(manifest.events), 4)
        with self.assertRaises(module.TraceHandoffStateError):
            recipient.complete()
        seen = []
        for index, key in enumerate(keys):
            recipient.send(key, lambda payload: seen.append(len(payload)))
            self.assertTrue(recipient.ack_received(key))
            self.assertFalse(recipient.ack_received(key))
            if index < len(keys) - 1:
                with self.assertRaises(module.TraceHandoffStateError):
                    recipient.complete()
        self.assertEqual(seen, [32] * 4)
        marker = recipient.complete()
        self.assertEqual(marker.event_count, 4)
        self.assertFalse(recipient.ack_manifest())
        for index, key in enumerate(keys):
            self.assertTrue(recipient.ack_release(key))
            self.assertFalse(recipient.ack_release(key))
            self.assertFalse(recipient.ack_received(key))
            if index == 0:
                self.assertIs(recipient.complete(), marker)
        # The manifest, keys, marker, callback frames and every alias covered by
        # the bridge claim must be disposed or separately charged before this.
        manifest = keys = marker = key = None
        recipient.confirm_disposed()
        self.assertTrue(bridge.dispose(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_cancel_before_transfer_drains_without_recipient(self):
        for prepared in (False, True):
            with self.subTest(prepared=prepared):
                module, registry, bridge, budget = self.make_bridge()
                if prepared:
                    bridge.prepare()
                    bridge.offer()
                bridge.disconnect()
                self.assertEqual(bridge.state, "cancelled_unowned")
                self.assertTrue(bridge.abort_before_transfer(timeout=5))
                registry.retire(bridge)
                self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_manifest_admission_denial_can_abort_without_copy(self):
        module, registry, bridge, budget = self.make_bridge()
        competitor = budget.open_request("competitor")
        available = budget.max_bytes - budget.snapshot().used_bytes
        occupied = competitor.reserve("competitor", "workspace", available - 1024, items=1)
        with self.assertRaises(HostBudgetExceeded):
            bridge.prepare()
        self.assertEqual(bridge.state, "admission_failed")
        occupied.release()
        competitor.close()
        self.assertTrue(bridge.abort_before_transfer(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_same_live_job_cannot_be_registered_in_a_second_registry(self):
        module, registry, bridge, budget = self.make_bridge()
        other = module.SealedBatchRegistry(
            budget=budget, broker_id="a" * 32, process_generation="b" * 32,
            max_jobs=1, max_events=4)
        with self.assertRaises(module.TraceHandoffStateError):
            other.register(bridge._owner, bridge._job, transfer_id="e" * 32,
                           recipient_id="another-reader")
        self.assertTrue(bridge.abort_before_transfer(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_interrupted_registration_before_insertion_can_retry_same_job(self):
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing tracing session")
        from looking_glass import trace_batch_handoff as module
        owner, job, budget = self.make_sealed_job()
        registry = module.SealedBatchRegistry(
            budget=budget, broker_id="a" * 32, process_generation="b" * 32,
            max_jobs=1, max_events=4)
        code = type(registry).register.__code__
        offsets = [instruction.offset for instruction in dis.get_instructions(code)
                   if instruction.opname == "LOAD_GLOBAL"
                   and instruction.argval == "_HandoffJob"]
        self.assertEqual(len(offsets), 1)

        def interrupt(frame, event, arg):
            if frame.f_code is code:
                if event == "call":
                    frame.f_trace_opcodes = True
                elif event == "opcode" and frame.f_lasti in offsets:
                    sys.settrace(None)
                    raise KeyboardInterrupt("before bridge insertion")
            return interrupt

        def register():
            registry.register(owner, job, transfer_id="d" * 32, recipient_id="reader")

        with self.assertRaises(KeyboardInterrupt):
            run_opcode_trace(interrupt, register)
        with self.assertRaises(module.TraceHandoffStateError):
            registry.lookup(7, registration_id="c" * 32, generation=1,
                            transfer_id="d" * 32)
        register()
        bridge = registry.lookup(7, registration_id="c" * 32, generation=1,
                                 transfer_id="d" * 32)
        self.assertTrue(bridge.abort_before_transfer(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_interrupted_registration_return_recovers_the_installed_job(self):
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing tracing session")
        from looking_glass import trace_batch_handoff as module
        owner, job, budget = self.make_sealed_job()
        registry = module.SealedBatchRegistry(
            budget=budget, broker_id="a" * 32, process_generation="b" * 32,
            max_jobs=1, max_events=4)
        code = type(registry).register.__code__
        offsets = [instruction.offset for instruction in dis.get_instructions(code)
                   if instruction.opname in {"RETURN_VALUE", "RETURN_CONST"}]
        self.assertTrue(offsets)

        def interrupt(frame, event, arg):
            if frame.f_code is code:
                if event == "call":
                    frame.f_trace_opcodes = True
                elif event == "opcode" and frame.f_lasti in offsets:
                    sys.settrace(None)
                    raise KeyboardInterrupt("registration return interrupted")
            return interrupt

        with self.assertRaises(KeyboardInterrupt):
            run_opcode_trace(
                interrupt, lambda: registry.register(
                    owner, job, transfer_id="d" * 32, recipient_id="reader"))
        bridge = registry.lookup(7, registration_id="c" * 32, generation=1,
                                 transfer_id="d" * 32)
        self.assertTrue(bridge.abort_before_transfer(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_pre_send_capacity_denial_is_retryable_without_sink_call(self):
        module, registry, bridge, budget = self.make_bridge()
        bridge.prepare()
        bridge.offer()
        recipient = bridge.accept(recipient_id="reader", transfer_id="d" * 32)
        manifest = recipient.manifest()
        recipient.ack_manifest()
        keys = recipient.events()
        competitor = budget.open_request("competitor")
        available = budget.max_bytes - budget.snapshot().used_bytes
        occupied = competitor.reserve("competitor", "workspace", available - 16, items=1)
        calls = []
        with self.assertRaises(HostBudgetExceeded):
            recipient.send(keys[0], lambda data: calls.append(len(data)))
        self.assertEqual(calls, [])
        self.assertEqual(bridge.state, "owned")
        occupied.release()
        competitor.close()
        for key in keys:
            recipient.send(key, lambda data: calls.append(len(data)))
            recipient.ack_received(key)
        self.assertEqual(calls, [32] * 4)
        marker = recipient.complete()
        for key in keys:
            recipient.ack_release(key)
        manifest = keys = marker = key = None
        recipient.confirm_disposed()
        self.assertTrue(bridge.dispose(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_sink_capacity_exception_is_ambiguous_and_not_retryable(self):
        from looking_glass.trace_transport import DeliveryError
        module, registry, bridge, budget = self.make_bridge()
        bridge.prepare()
        bridge.offer()
        recipient = bridge.accept(recipient_id="reader", transfer_id="d" * 32)
        recipient.manifest()
        recipient.ack_manifest()
        key = recipient.events()[0]
        calls = []

        def failing_sink(data):
            calls.append(len(data))
            raise HostBudgetExceeded("sink outcome is uncertain")

        with self.assertRaises(DeliveryError):
            recipient.send(key, failing_sink)
        self.assertEqual(calls, [32])
        self.assertEqual(bridge.state, "quarantined")
        with self.assertRaises(module.TraceHandoffStateError):
            recipient.send(key, lambda data: None)
        self.assertGreater(budget.snapshot().used_bytes, 0)

    def test_complete_handoff_does_not_enter_cuda(self):
        # Resolve optional dependency imports before guarding CUDA entry points.
        from transformers import GPT2Config, GPT2LMHeadModel  # noqa: F401
        from looking_glass.traced_worker import TracedGPT2Worker  # noqa: F401

        def reject_cuda(*args, **kwargs):
            raise AssertionError("CPU handoff entered CUDA")

        with patch("torch.cuda.device_count", side_effect=reject_cuda), \
                patch("torch.cuda.manual_seed_all", side_effect=reject_cuda), \
                patch("torch.cuda.get_rng_state", side_effect=reject_cuda), \
                patch("torch.cuda.set_rng_state", side_effect=reject_cuda), \
                patch("torch.cuda._lazy_init", side_effect=reject_cuda):
            module, registry, bridge, budget = self.make_bridge()
            bridge.prepare()
            bridge.offer()
            recipient = bridge.accept(recipient_id="reader", transfer_id="d" * 32)
            manifest = recipient.manifest()
            recipient.ack_manifest()
            keys = recipient.events()
            for key in keys:
                recipient.send(key, lambda payload: None)
                recipient.ack_received(key)
            marker = recipient.complete()
            for key in keys:
                recipient.ack_release(key)
            manifest = keys = marker = key = None
            recipient.confirm_disposed()
            self.assertTrue(bridge.dispose(timeout=5))
            registry.retire(bridge)
            self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_interrupted_committed_receipt_ack_is_idempotently_recoverable(self):
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing tracing session")
        module, registry, bridge, budget = self.make_bridge()
        bridge.prepare()
        bridge.offer()
        recipient = bridge.accept(recipient_id="reader", transfer_id="d" * 32)
        manifest = recipient.manifest()
        recipient.ack_manifest()
        keys = recipient.events()
        recipient.send(keys[0], lambda payload: None)
        code = type(recipient)._ack.__code__
        instructions = list(dis.get_instructions(code))
        offsets = [instructions[index].offset for index in range(1, len(instructions))
                   if instructions[index].opname == "RETURN_VALUE"
                   and instructions[index - 1].opname == "LOAD_FAST"
                   and instructions[index - 1].argval == "changed"]
        self.assertTrue(offsets)

        def interrupt(frame, event, arg):
            if frame.f_code is code:
                if event == "call":
                    frame.f_trace_opcodes = True
                elif event == "opcode" and frame.f_lasti in offsets:
                    sys.settrace(None)
                    raise KeyboardInterrupt("receipt return interrupted")
            return interrupt

        with self.assertRaises(KeyboardInterrupt):
            run_opcode_trace(interrupt, lambda: recipient.ack_received(keys[0]))
        self.assertFalse(recipient.ack_received(keys[0]))
        for key in keys[1:]:
            recipient.send(key, lambda payload: None)
            self.assertTrue(recipient.ack_received(key))
        marker = recipient.complete()
        for key in keys:
            recipient.ack_release(key)
        manifest = keys = marker = key = None
        recipient.confirm_disposed()
        self.assertTrue(bridge.dispose(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_interrupted_committed_send_is_recoverable_without_resend(self):
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing tracing session")
        module, registry, bridge, budget = self.make_bridge()
        bridge.prepare()
        bridge.offer()
        recipient = bridge.accept(recipient_id="reader", transfer_id="d" * 32)
        manifest = recipient.manifest()
        recipient.ack_manifest()
        keys = recipient.events()
        calls = []
        code = type(recipient).send.__code__
        offsets = [instruction.offset for instruction in dis.get_instructions(code)
                   if instruction.opname in {"RETURN_VALUE", "RETURN_CONST"}]
        self.assertTrue(offsets)

        def interrupt(frame, event, arg):
            if frame.f_code is code:
                if event == "call":
                    frame.f_trace_opcodes = True
                elif event == "opcode" and frame.f_lasti in offsets:
                    sys.settrace(None)
                    raise KeyboardInterrupt("send return interrupted")
            return interrupt

        with self.assertRaises(KeyboardInterrupt):
            run_opcode_trace(interrupt,
                             lambda: recipient.send(keys[0], lambda data: calls.append(len(data))))
        self.assertEqual(calls, [32])
        with self.assertRaises(module.TraceHandoffStateError):
            recipient.delivery_state(replace(keys[0], process_generation="f" * 32))
        self.assertEqual(recipient.delivery_state(keys[0]), "sent")
        with self.assertRaises(module.TraceHandoffStateError):
            recipient.complete()
        with self.assertRaises(module.TraceHandoffStateError):
            recipient.send(keys[0], lambda data: calls.append(len(data)))
        self.assertTrue(recipient.ack_received(keys[0]))
        for key in keys[1:]:
            recipient.send(key, lambda data: calls.append(len(data)))
            recipient.ack_received(key)
        marker = recipient.complete()
        for key in keys:
            recipient.ack_release(key)
        manifest = keys = marker = key = None
        recipient.confirm_disposed()
        self.assertTrue(bridge.dispose(timeout=5))
        registry.retire(bridge)
        self.assertEqual(calls, [32] * 4)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_interrupted_manifest_ack_return_does_not_hold_control_lock(self):
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing tracing session")
        module, registry, bridge, budget = self.make_bridge()
        bridge.prepare()
        bridge.offer()
        recipient = bridge.accept(recipient_id="reader", transfer_id="d" * 32)
        manifest = recipient.manifest()
        code = type(recipient).ack_manifest.__code__
        instructions = list(dis.get_instructions(code))
        offsets = [instructions[index].offset for index in range(1, len(instructions))
                   if instructions[index].opname == "RETURN_VALUE"
                   and instructions[index - 1].opname == "LOAD_FAST"
                   and instructions[index - 1].argval == "changed"]
        self.assertTrue(offsets)

        def interrupt(frame, event, arg):
            if frame.f_code is code:
                if event == "call":
                    frame.f_trace_opcodes = True
                elif event == "opcode" and frame.f_lasti in offsets:
                    sys.settrace(None)
                    raise KeyboardInterrupt("manifest acknowledgement return interrupted")
            return interrupt

        for _ in range(2):
            with self.assertRaises(KeyboardInterrupt):
                run_opcode_trace(interrupt, recipient.ack_manifest)
            self.assertFalse(recipient.ack_manifest())
            observed = []
            thread = threading.Thread(target=lambda: observed.append(bridge.state))
            thread.start()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(observed, ["owned"])
        keys = recipient.events()
        for key in keys:
            recipient.send(key, lambda payload: None)
            recipient.ack_received(key)
        marker = recipient.complete()
        for key in keys:
            recipient.ack_release(key)
        manifest = keys = marker = key = None
        recipient.confirm_disposed()
        self.assertTrue(bridge.dispose(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_foreign_identity_rejects_ack_without_affecting_current_slot(self):
        module, registry, bridge, budget = self.make_bridge()
        bridge.prepare()
        bridge.offer()
        recipient = bridge.accept(recipient_id="reader", transfer_id="d" * 32)
        recipient.manifest()
        recipient.ack_manifest()
        key = recipient.events()[0]
        forged = replace(key, process_generation="f" * 32)
        with self.assertRaises(module.TraceHandoffStateError):
            recipient.send(forged, lambda payload: None)
        recipient.send(key, lambda payload: None)
        self.assertTrue(recipient.ack_received(key))
        bridge.disconnect()
        self.assertEqual(bridge.state, "quarantined")
        self.assertGreater(budget.snapshot().used_bytes, 0)

    def test_disconnect_during_blocked_send_keeps_claims_and_no_marker(self):
        module, registry, bridge, budget = self.make_bridge()
        bridge.prepare()
        bridge.offer()
        recipient = bridge.accept(recipient_id="reader", transfer_id="d" * 32)
        recipient.manifest()
        recipient.ack_manifest()
        key = recipient.events()[0]
        entered, resume = threading.Event(), threading.Event()
        failures = []

        def blocked_sink(payload):
            entered.set()
            if not resume.wait(10):
                raise RuntimeError("sink timed out")

        def send():
            try:
                recipient.send(key, blocked_sink)
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=send)
        thread.start()
        try:
            self.assertTrue(entered.wait(10))
            bridge.disconnect()
            self.assertTrue(budget.snapshot().operations)
            with self.assertRaises(module.TraceHandoffStateError):
                recipient.complete()
        finally:
            resume.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertTrue(failures)
        self.assertEqual(bridge.state, "quarantined")
        self.assertGreater(budget.snapshot().used_bytes, 0)


if __name__ == "__main__":
    unittest.main()
