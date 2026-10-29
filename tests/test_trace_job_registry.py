"""Recoverable control-plane receipts; no tensors or delivery implementation."""
import copy
import dis
import pickle
import sys
import threading
import unittest
from unittest.mock import patch

from looking_glass.trace_job_registry import (
    JobRegistryFull, JobStateError, StaleReceipt, TraceJobRegistry,
)

REG_A = "c" * 32
REG_B = "d" * 32


class JobRegistryTests(unittest.TestCase):
    def make_registry(self, maximum=2):
        return TraceJobRegistry(broker_id="a" * 32, process_generation="b" * 32,
                                max_jobs=maximum)

    def test_registration_is_bounded_and_recovers_exact_job(self):
        registry = self.make_registry(maximum=1)
        receipt = registry.register(7, registration_id=REG_A)
        snap = registry.inspect(receipt)
        self.assertEqual((snap.broker_id, snap.process_generation, snap.job_id,
                          snap.registration_id, snap.worker_request_id, snap.phase, snap.cancel_requested),
                         ("a" * 32, "b" * 32, 7, REG_A, "lg-" + "a" * 32 + "-7",
                          "preparing", False))
        recovered = registry.recover(7, registration_id=REG_A)
        self.assertEqual(registry.inspect(recovered), snap)
        with self.assertRaises(JobStateError):
            registry.register(7, registration_id=REG_A)
        with self.assertRaises(JobRegistryFull):
            registry.register(8, registration_id=REG_B)

    def test_lost_registration_return_is_recoverable(self):
        registry = self.make_registry()
        with patch("looking_glass.trace_job_registry._new_receipt", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                registry.register(7, registration_id=REG_A)
        receipt = registry.recover(7, registration_id=REG_A)
        self.assertEqual(registry.inspect(receipt).phase, "preparing")
        self.assertEqual(registry.active_jobs(), (7,))

    def test_lost_handoff_return_is_recoverable_and_ack_is_idempotent(self):
        registry = self.make_registry()
        receipt = registry.register(7, registration_id=REG_A)
        registry.seal(receipt)
        with patch("looking_glass.trace_job_registry._new_handoff", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                registry.begin_handoff(receipt, "delivery-owner")
        handoff = registry.recover_handoff(receipt, "delivery-owner")
        with self.assertRaises(JobStateError):
            registry.recover_handoff(receipt, "other-owner")
        self.assertTrue(registry.ack_handoff(handoff))
        self.assertFalse(registry.ack_handoff(handoff))
        self.assertEqual(registry.inspect(receipt).phase, "owned")

    def test_interrupted_ack_return_remains_recoverable_and_idempotent(self):
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing tracing session")
        registry = self.make_registry()
        receipt = registry.register(7, registration_id=REG_A)
        registry.seal(receipt)
        handoff = registry.begin_handoff(receipt, "delivery-owner")
        code = TraceJobRegistry.ack_handoff.__code__
        instructions = list(dis.get_instructions(code))
        matches = [instructions[i - 1].offset for i, instruction in enumerate(instructions)
                   if instruction.opname == "RETURN_VALUE" and i > 0
                   and instructions[i - 1].opname == "LOAD_FAST"
                   and instructions[i - 1].argval == "changed"]
        self.assertEqual(len(matches), 1)
        def interrupt(frame, event, arg):
            if event == "call":
                frame.f_trace_opcodes = True
            if frame.f_code is code and event == "opcode" and frame.f_lasti == matches[0]:
                sys.settrace(None)
                raise KeyboardInterrupt("injected after handoff commit")
            return interrupt
        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                registry.ack_handoff(handoff)
        finally:
            sys.settrace(None)
        self.assertEqual(registry.inspect(registry.recover(7, registration_id=REG_A)).phase, "owned")
        self.assertFalse(registry.ack_handoff(handoff))

    def test_interrupted_handoff_return_does_not_leave_registry_lock_held(self):
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing tracing session")
        registry = self.make_registry()
        receipt = registry.register(7, registration_id=REG_A)
        registry.seal(receipt)
        code = TraceJobRegistry.begin_handoff.__code__
        instructions = list(dis.get_instructions(code))
        matches = [instructions[i - 1].offset for i, instruction in enumerate(instructions)
                   if instruction.opname == "RETURN_VALUE" and i > 0
                   and instructions[i - 1].opname == "LOAD_FAST"
                   and instructions[i - 1].argval == "handoff"]
        self.assertEqual(len(matches), 1)
        def interrupt(frame, event, arg):
            if event == "call":
                frame.f_trace_opcodes = True
            if frame.f_code is code and event == "opcode" and frame.f_lasti == matches[0]:
                sys.settrace(None)
                raise KeyboardInterrupt("injected before handoff return")
            return interrupt
        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                registry.begin_handoff(receipt, "delivery-owner")
        finally:
            sys.settrace(None)
        observed = []
        thread = threading.Thread(target=lambda: observed.append(registry.active_jobs()), daemon=True)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(observed, [(7,)])
        handoff = registry.recover_handoff(receipt, "delivery-owner")
        self.assertTrue(registry.ack_handoff(handoff))

    def test_cancel_blocks_seal_or_unacknowledged_handoff_without_disposal(self):
        registry = self.make_registry()
        first = registry.register(1, registration_id=REG_A)
        registry.cancel(first)
        with self.assertRaises(JobStateError):
            registry.seal(first)
        self.assertTrue(registry.inspect(first).cancel_requested)
        self.assertIn(1, registry.active_jobs())
        second = registry.register(2, registration_id=REG_B)
        registry.seal(second)
        handoff = registry.begin_handoff(second, "delivery-owner")
        registry.cancel(second)
        with self.assertRaises(JobStateError):
            registry.ack_handoff(handoff)
        self.assertEqual(registry.inspect(second).phase, "handoff_pending")
        self.assertIn(2, registry.active_jobs())

    def test_cancel_and_seal_have_a_single_commit_order(self):
        for _ in range(50):
            registry = self.make_registry()
            receipt = registry.register(1, registration_id=REG_A)
            barrier = threading.Barrier(3)
            results = []
            def seal():
                barrier.wait()
                try:
                    registry.seal(receipt)
                    results.append("sealed")
                except JobStateError:
                    results.append("rejected")
            def cancel():
                barrier.wait()
                registry.cancel(receipt)
            threads = [threading.Thread(target=seal), threading.Thread(target=cancel)]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(5)
                self.assertFalse(thread.is_alive())
            self.assertIn(results[0], ("sealed", "rejected"))
            self.assertTrue(registry.inspect(receipt).cancel_requested)
            with self.assertRaises(JobStateError):
                registry.begin_handoff(receipt, "delivery-owner")

    def test_disposal_requires_explicit_attestation_and_stale_receipts_fail(self):
        registry = self.make_registry(maximum=1)
        receipt = registry.register(7, registration_id=REG_A)
        with self.assertRaises(JobStateError):
            registry.finish_disposal(receipt)
        registry.begin_disposal(receipt)
        registry.finish_disposal(receipt)
        replacement = registry.register(7, registration_id=REG_B)
        with self.assertRaises(StaleReceipt):
            registry.cancel(receipt)
        with self.assertRaises(StaleReceipt):
            registry.recover(7, registration_id=REG_A)
        self.assertFalse(registry.inspect(replacement).cancel_requested)

    def test_quarantine_retains_bounded_record_and_prevents_attestation(self):
        registry = self.make_registry(maximum=1)
        receipt = registry.register(7, registration_id=REG_A)
        registry.quarantine(receipt)
        self.assertEqual(registry.inspect(receipt).phase, "quarantined")
        with self.assertRaises(JobStateError):
            registry.begin_disposal(receipt)
        with self.assertRaises(JobRegistryFull):
            registry.register(8, registration_id=REG_B)

    def test_receipts_cannot_be_copied_or_pickled(self):
        registry = self.make_registry()
        receipt = registry.register(7, registration_id=REG_A)
        for clone in (copy.copy, copy.deepcopy, pickle.dumps):
            with self.assertRaises(TypeError):
                clone(receipt)
        registry.seal(receipt)
        handoff = registry.begin_handoff(receipt, "delivery-owner")
        for clone in (copy.copy, copy.deepcopy, pickle.dumps):
            with self.assertRaises(TypeError):
                clone(handoff)

    def test_invalid_identifiers_and_receiver_are_rejected(self):
        for broker, generation, size in (("x" * 32, "b" * 32, 1),
                                         ("a" * 32, "b", 1),
                                         ("a" * 32, "b" * 32, True)):
            with self.assertRaises(ValueError):
                TraceJobRegistry(broker_id=broker, process_generation=generation, max_jobs=size)
        registry = self.make_registry()
        for job in (True, 0, -1, 2**63):
            with self.assertRaises(ValueError):
                registry.register(job, registration_id=REG_A)
        with self.assertRaises(ValueError):
            registry.register(7, registration_id="invalid")
        receipt = registry.register(7, registration_id=REG_A)
        registry.seal(receipt)
        for owner in ("", "x" * 129):
            with self.assertRaises(ValueError):
                registry.begin_handoff(receipt, owner)


if __name__ == "__main__":
    unittest.main()
