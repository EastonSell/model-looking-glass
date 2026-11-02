"""Supervised recovery of quarantined in-process CPU trace handoffs."""
import importlib
import importlib.util
import dis
import sys
import threading
import unittest
import weakref
from dataclasses import replace
from unittest.mock import patch

from looking_glass.host_budget import HostBudgetExceeded, HostBusy, HostClaim, HostClaimGuard, HostRequest
from looking_glass.trace_transport import TraceDelivery

if __package__:
    from . import test_trace_batch_handoff as handoff_fixtures
else:
    import test_trace_batch_handoff as handoff_fixtures


@unittest.skipUnless(importlib.util.find_spec("transformers") is not None,
                     "requires the optional transformers dependency")
class TraceHandoffRecoveryTests(unittest.TestCase):
    make_sealed_job = handoff_fixtures.TraceBatchHandoffTests.make_sealed_job
    make_bridge = handoff_fixtures.TraceBatchHandoffTests.make_bridge

    def make_quarantined(self):
        _, registry, bridge, budget = self.make_bridge()
        bridge.prepare()
        bridge.offer()
        recipient = bridge.accept(recipient_id="reader", transfer_id="d" * 32)
        manifest = recipient.manifest()
        recipient.ack_manifest()
        keys = recipient.events()
        bridge.disconnect()
        self.assertEqual(bridge.state, "quarantined")
        return registry, bridge, budget, recipient, manifest, keys

    def register_recovery(self, bridge):
        module = importlib.import_module("looking_glass.trace_handoff_recovery")
        module.register_recovery(
            bridge, broker_id="a" * 32, process_generation="b" * 32,
            job_id=7, registration_id="c" * 32, generation=1,
            transfer_id="d" * 32, recovery_id="e" * 32)
        recovery = module.lookup_recovery(bridge, recovery_id="e" * 32)
        return module, recovery

    def test_exact_identity_attestation_drains_quarantined_job(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        self.assertTrue(recovery.bind())
        batch_ref = weakref.ref(bridge._job._batch)
        scope = recovery.scope()
        self.assertEqual(scope.events, keys)
        with self.assertRaises(module.RecoveryStateError):
            recovery.ack_scope(replace(scope, process_generation="f" * 32))
        with self.assertRaises(module.RecoveryStateError):
            recovery.ack_scope(replace(scope, inventory_claim_id=True))
        wrong_slot = replace(scope.events[0], slot_generation=True)
        wrong_delivery = replace(
            scope.events[0], delivery=replace(scope.events[0].delivery, delivery_id="f" * 32))
        for wrong_event in (wrong_slot, wrong_delivery):
            with self.assertRaises(module.RecoveryStateError):
                recovery.ack_scope(replace(scope, events=(wrong_event,) + scope.events[1:]))
        wrong_slot = wrong_delivery = wrong_event = None
        self.assertEqual(recovery.state, "bound")
        with self.assertRaises(module.RecoveryStateError):
            recovery.confirm_disposed()
        self.assertTrue(recovery.ack_scope(scope))
        self.assertFalse(recovery.ack_scope(scope))
        recipient = manifest = keys = scope = None
        recovery.confirm_disposed()
        with self.assertRaises(module.RecoveryStateError):
            recovery.scope()
        self.assertTrue(recovery.drain(timeout=5))
        self.assertEqual(bridge.state, "disposed")
        self.assertIsNone(batch_ref())
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_operation_capacity_denial_releases_pre_copy_inventory_claim(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        competitor = budget.open_request("operation-competitor")
        claim = competitor.reserve("competitor", "input", 1)
        operations = [competitor.begin_operation(f"operation-{index}", (claim,))
                      for index in range(budget.max_operations)]
        baseline = budget.snapshot()
        self.assertFalse(recovery.bind())
        self.assertEqual(recovery.state, "unbound")
        self.assertEqual(budget.snapshot().used_bytes, baseline.used_bytes)
        self.assertEqual(budget.snapshot().used_items, baseline.used_items)
        self.assertFalse(any(info.stage == "inventory" for info in budget.snapshot().claims))
        for operation in operations:
            operation.finish()
        claim.release()
        competitor.close()
        self.assertTrue(recovery.bind())
        scope = recovery.scope()
        recovery.ack_scope(scope)
        recipient = manifest = keys = scope = None
        recovery.confirm_disposed()
        self.assertTrue(recovery.drain(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_lost_pin_return_keeps_claims_and_requires_supervision(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        original = HostRequest.begin_operation

        def lose_pin(request, label, claims):
            operation = original(request, label, claims)
            if label.startswith("handoff-recovery-"):
                raise KeyboardInterrupt("pin return lost")
            return operation

        with patch.object(HostRequest, "begin_operation", lose_pin):
            with self.assertRaises(KeyboardInterrupt):
                recovery.bind()
        self.assertEqual(recovery.state, "uncertain")
        self.assertTrue(budget.snapshot().operations)
        self.assertGreater(budget.snapshot().used_bytes, 0)
        with self.assertRaises(HostBusy):
            bridge._manifest_claim.release()
        with self.assertRaises(module.RecoveryStateError):
            recovery.bind()

    def test_lost_inventory_reservation_return_keeps_exact_guard(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        original = HostClaimGuard.reserve

        def lose_reservation(guard):
            result = original(guard)
            if guard._stage == "inventory":
                raise KeyboardInterrupt("inventory reservation return lost")
            return result

        with patch.object(HostClaimGuard, "reserve", lose_reservation):
            with self.assertRaises(KeyboardInterrupt):
                recovery.bind()
        self.assertEqual(recovery.state, "uncertain")
        self.assertIn(recovery._guard, budget.active_claim_guards())
        self.assertTrue(any(info.stage == "inventory" for info in budget.snapshot().claims))
        with self.assertRaises(module.RecoveryStateError):
            recovery.bind()

    def test_budget_error_after_inventory_commit_stays_uncertain(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        original = HostClaimGuard.reserve

        def lose_reservation(guard):
            result = original(guard)
            if guard._stage == "inventory":
                raise HostBudgetExceeded("reservation return was lost")
            return result

        with patch.object(HostClaimGuard, "reserve", lose_reservation):
            with self.assertRaises(HostBudgetExceeded):
                recovery.bind()
        self.assertEqual(recovery.state, "uncertain")
        self.assertIn(recovery._guard, budget.active_claim_guards())
        self.assertTrue(any(info.stage == "inventory" for info in budget.snapshot().claims))

    def test_budget_error_after_pin_commit_stays_uncertain(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        original = HostRequest.begin_operation

        def lose_pin(request, label, claims):
            operation = original(request, label, claims)
            if label.startswith("handoff-recovery-"):
                raise HostBudgetExceeded("pin return was lost")
            return operation

        with patch.object(HostRequest, "begin_operation", lose_pin):
            with self.assertRaises(HostBudgetExceeded):
                recovery.bind()
        self.assertEqual(recovery.state, "uncertain")
        self.assertTrue(budget.snapshot().operations)
        with self.assertRaises(HostBusy):
            bridge._manifest_claim.release()

    def test_budget_error_after_guard_adoption_stays_uncertain(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        original = HostClaimGuard.adopt

        def lose_adoption(guard):
            claim = original(guard)
            if guard._stage == "inventory":
                raise HostBudgetExceeded("claim adoption return was lost")
            return claim

        with patch.object(HostClaimGuard, "adopt", lose_adoption):
            with self.assertRaises(HostBudgetExceeded):
                recovery.bind()
        self.assertEqual(recovery.state, "uncertain")
        self.assertIn(recovery._guard, budget.active_claim_guards())
        self.assertTrue(any(info.stage == "inventory" for info in budget.snapshot().claims))

    def test_extremely_large_timeout_is_rejected_as_value_error(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        with self.assertRaises(ValueError):
            recovery.drain(timeout=10**500)

    def test_interrupted_scope_ack_and_attestation_returns_are_recoverable(self):
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing tracing session")
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        self.assertTrue(recovery.bind())
        scope = recovery.scope()

        def interrupt_return(method, call):
            code = method.__code__
            offsets = [item.offset for item in dis.get_instructions(code)
                       if item.opname in {"RETURN_VALUE", "RETURN_CONST"}]
            self.assertTrue(offsets)

            def interrupt(frame, event, arg):
                if frame.f_code is code:
                    if event == "call":
                        frame.f_trace_opcodes = True
                    elif event == "opcode" and frame.f_lasti in offsets:
                        sys.settrace(None)
                        raise KeyboardInterrupt("recovery return interrupted")
                return interrupt

            with self.assertRaises(KeyboardInterrupt):
                handoff_fixtures.run_opcode_trace(interrupt, call)

        interrupt_return(type(recovery).ack_scope, lambda: recovery.ack_scope(scope))
        self.assertFalse(recovery.ack_scope(scope))
        recipient = manifest = keys = scope = None
        interrupt_return(type(recovery).confirm_disposed, recovery.confirm_disposed)
        self.assertEqual(recovery.state, "attested")
        with self.assertRaises(module.RecoveryStateError):
            recovery.scope()
        self.assertTrue(recovery.drain(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_interrupted_successful_drain_return_keeps_disposed_outcome(self):
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing tracing session")
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        self.assertTrue(recovery.bind())
        scope = recovery.scope()
        recovery.ack_scope(scope)
        recipient = manifest = keys = scope = None
        recovery.confirm_disposed()
        code = type(recovery).drain.__code__
        instructions = list(dis.get_instructions(code))
        offsets = [instructions[index].offset for index in range(1, len(instructions))
                   if (instructions[index].opname == "RETURN_CONST"
                       and instructions[index].argval is True)
                   or (instructions[index].opname == "RETURN_VALUE"
                       and instructions[index - 1].opname == "LOAD_CONST"
                       and instructions[index - 1].argval is True)]
        self.assertTrue(offsets)

        def interrupt(frame, event, arg):
            if frame.f_code is code:
                if event == "call":
                    frame.f_trace_opcodes = True
                elif event == "opcode" and frame.f_lasti in offsets:
                    sys.settrace(None)
                    raise KeyboardInterrupt("successful drain return interrupted")
            return interrupt

        with self.assertRaises(KeyboardInterrupt):
            handoff_fixtures.run_opcode_trace(interrupt, lambda: recovery.drain(timeout=5))
        self.assertEqual(recovery.state, "disposed")
        self.assertEqual(bridge.state, "disposed")
        self.assertIs(module.lookup_recovery(bridge, recovery_id="e" * 32), recovery)
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_recovery_does_not_enter_cuda(self):
        # Load optional dependencies before guarding CUDA entry points.
        from transformers import GPT2Config, GPT2LMHeadModel  # noqa: F401
        from looking_glass.traced_worker import TracedGPT2Worker  # noqa: F401

        def reject_cuda(*args, **kwargs):
            raise AssertionError("CPU recovery entered CUDA")

        with patch("torch.cuda.device_count", side_effect=reject_cuda), \
                patch("torch.cuda.manual_seed_all", side_effect=reject_cuda), \
                patch("torch.cuda.get_rng_state", side_effect=reject_cuda), \
                patch("torch.cuda.set_rng_state", side_effect=reject_cuda), \
                patch("torch.cuda._lazy_init", side_effect=reject_cuda):
            registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
            module, recovery = self.register_recovery(bridge)
            self.assertTrue(recovery.bind())
            scope = recovery.scope()
            recovery.ack_scope(scope)
            recipient = manifest = keys = scope = None
            recovery.confirm_disposed()
            self.assertTrue(recovery.drain(timeout=5))
            registry.retire(bridge)
            self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_interrupted_pre_copy_rollback_is_not_retried_blindly(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        competitor = budget.open_request("rollback-competitor")
        claim = competitor.reserve("competitor", "input", 1)
        operations = [competitor.begin_operation(f"operation-{index}", (claim,))
                      for index in range(budget.max_operations)]
        original = HostClaim.release

        def lose_release(handle):
            result = original(handle)
            if recovery._guard is not None and recovery._guard._handle is handle:
                raise KeyboardInterrupt("inventory rollback return lost")
            return result

        try:
            with patch.object(HostClaim, "release", lose_release):
                with self.assertRaises(KeyboardInterrupt):
                    recovery.bind()
            self.assertEqual(recovery.state, "uncertain")
            self.assertFalse(any(info.stage == "inventory" for info in budget.snapshot().claims))
            with self.assertRaises(module.RecoveryStateError):
                recovery.bind()
        finally:
            for operation in operations:
                operation.finish()
            claim.release()
            competitor.close()

    def test_unproven_missing_child_fails_closed_with_claims_pinned(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        self.assertTrue(recovery.bind())
        scope = recovery.scope()
        recovery.ack_scope(scope)
        recipient = manifest = keys = scope = None
        recovery.confirm_disposed()
        slot_id, _, slot, _ = recovery._children[0]
        with recovery._deliveries._lock:
            del recovery._deliveries._slots[slot_id]
        self.assertNotEqual(slot._state, "retired")
        with self.assertRaises(module.RecoveryStateError):
            recovery.drain(timeout=0)
        self.assertEqual(recovery.state, "uncertain")
        self.assertTrue(budget.snapshot().operations)
        with self.assertRaises(HostBusy):
            bridge._manifest_claim.release()

    def test_changed_child_capability_fails_closed_with_claims_pinned(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        self.assertTrue(recovery.bind())
        scope = recovery.scope()
        recovery.ack_scope(scope)
        recipient = manifest = keys = scope = None
        recovery.confirm_disposed()
        first_slot = recovery._children[0][2]
        foreign_child = recovery._children[1][3]
        with recovery._deliveries._lock:
            first_slot._delivery = foreign_child
        with self.assertRaises(module.RecoveryStateError):
            recovery.drain(timeout=0)
        self.assertEqual(recovery.state, "uncertain")
        self.assertTrue(budget.snapshot().operations)
        with self.assertRaises(HostBusy):
            bridge._manifest_claim.release()

    def test_wrong_registration_key_does_not_disturb_existing_owner(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module = importlib.import_module("looking_glass.trace_handoff_recovery")
        with self.assertRaises(module.RecoveryStateError):
            module.register_recovery(
                bridge, broker_id="a" * 32, process_generation="f" * 32,
                job_id=7, registration_id="c" * 32, generation=1,
                transfer_id="d" * 32, recovery_id="e" * 32)
        module, recovery = self.register_recovery(bridge)
        with self.assertRaises(module.RecoveryStateError):
            module.lookup_recovery(bridge, recovery_id="f" * 32)
        self.assertIs(module.lookup_recovery(bridge, recovery_id="e" * 32), recovery)
        self.assertTrue(recovery.bind())
        scope = recovery.scope()
        recovery.ack_scope(scope)
        recipient = manifest = keys = scope = None
        recovery.confirm_disposed()
        self.assertTrue(recovery.drain(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_interrupted_registration_return_recovers_same_capability(self):
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing tracing session")
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module = importlib.import_module("looking_glass.trace_handoff_recovery")
        code = module.register_recovery.__code__
        offsets = [item.offset for item in dis.get_instructions(code)
                   if item.opname in {"RETURN_VALUE", "RETURN_CONST"}]
        self.assertTrue(offsets)

        def interrupt(frame, event, arg):
            if frame.f_code is code:
                if event == "call":
                    frame.f_trace_opcodes = True
                elif event == "opcode" and frame.f_lasti in offsets:
                    sys.settrace(None)
                    raise KeyboardInterrupt("recovery registration return interrupted")
            return interrupt

        def register():
            module.register_recovery(
                bridge, broker_id="a" * 32, process_generation="b" * 32,
                job_id=7, registration_id="c" * 32, generation=1,
                transfer_id="d" * 32, recovery_id="e" * 32)

        with self.assertRaises(KeyboardInterrupt):
            handoff_fixtures.run_opcode_trace(interrupt, register)
        recovery = module.lookup_recovery(bridge, recovery_id="e" * 32)
        self.assertIs(module.lookup_recovery(bridge, recovery_id="e" * 32), recovery)
        with self.assertRaises(module.RecoveryStateError):
            register()
        self.assertTrue(recovery.bind())
        scope = recovery.scope()
        recovery.ack_scope(scope)
        recipient = manifest = keys = scope = None
        recovery.confirm_disposed()
        self.assertTrue(recovery.drain(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_blocked_sink_cannot_be_recovered_before_send_unwinds(self):
        handoff, registry, bridge, budget = self.make_bridge()
        bridge.prepare()
        bridge.offer()
        recipient = bridge.accept(recipient_id="reader", transfer_id="d" * 32)
        manifest = recipient.manifest()
        recipient.ack_manifest()
        keys = recipient.events()
        entered, resume = threading.Event(), threading.Event()
        failures = []

        def sink(data):
            entered.set()
            if not resume.wait(10):
                raise RuntimeError("sink gate timed out")

        def send():
            try:
                recipient.send(keys[0], sink)
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=send)
        thread.start()
        try:
            self.assertTrue(entered.wait(10))
            bridge.disconnect()
            with self.assertRaises(importlib.import_module(
                    "looking_glass.trace_handoff_recovery").RecoveryStateError):
                self.register_recovery(bridge)
            self.assertTrue(budget.snapshot().operations)
            self.assertGreater(budget.snapshot().used_bytes, 0)
        finally:
            resume.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertTrue(failures)
        failures.clear()
        thread = recipient = manifest = keys = None
        module, recovery = self.register_recovery(bridge)
        self.assertTrue(recovery.bind())
        scope = recovery.scope()
        recovery.ack_scope(scope)
        scope = None
        recovery.confirm_disposed()
        self.assertTrue(recovery.drain(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)
        self.assertFalse(budget.snapshot().operations)

    def test_partial_real_child_drain_is_retryable_and_remains_pinned(self):
        registry, bridge, budget, recipient, manifest, keys = self.make_quarantined()
        module, recovery = self.register_recovery(bridge)
        self.assertTrue(recovery.bind())
        scope = recovery.scope()
        recovery.ack_scope(scope)
        recipient = manifest = keys = scope = None
        recovery.confirm_disposed()
        pinned = bridge._manifest_claim
        with self.assertRaises(HostBusy):
            pinned.release()

        first_slot, second_slot = recovery._children[0][2], recovery._children[1][2]
        second_child = recovery._children[1][3]
        entered, resume = threading.Event(), threading.Event()
        result, failures = [], []
        original = TraceDelivery._cleanup

        def gated_cleanup(child):
            if child is second_child:
                entered.set()
                if not resume.wait(10):
                    raise RuntimeError("cleanup gate timed out")
            return original(child)

        def drain():
            try:
                result.append(recovery.drain(timeout=2))
            except BaseException as error:
                failures.append(error)

        with patch.object(TraceDelivery, "_cleanup", gated_cleanup):
            thread = threading.Thread(target=drain)
            thread.start()
            try:
                self.assertTrue(entered.wait(10))
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertFalse(failures)
                self.assertEqual(result, [False])
                self.assertEqual(first_slot._state, "retired")
                self.assertNotEqual(second_slot._state, "retired")
                self.assertTrue(budget.snapshot().operations)
                self.assertGreater(budget.snapshot().used_bytes, 0)
                with self.assertRaises(HostBusy):
                    pinned.release()
            finally:
                resume.set()
                thread.join(10)
            self.assertTrue(recovery.drain(timeout=5))
        registry.retire(bridge)
        self.assertEqual(budget.snapshot().used_bytes, 0)


if __name__ == "__main__":
    unittest.main()
