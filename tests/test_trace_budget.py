import unittest
from unittest.mock import patch

import torch

from looking_glass.probes import TapPoint, TraceSession


class TraceBudgetTests(unittest.TestCase):
    def test_exact_limit_across_invocations_and_overflow_cleanup(self):
        for mode in ("direct", "gradient"):
            with self.subTest(mode=mode):
                tap = TapPoint("point")
                session = TraceSession(tap, mode=mode, max_tensor_bytes=16)
                with self.assertRaisesRegex(RuntimeError, "max_tensor_bytes"):
                    with session:
                        tap(torch.tensor([1.0, 2.0]))  # 8 bytes
                        tap(torch.tensor([3.0, 4.0]))  # exactly 16 bytes
                        self.assertEqual(session.tensor_bytes, 16)
                        tap(torch.tensor([5.0]))  # exceeds by 4 bytes

                self.assertEqual(session.tensor_bytes, 16)
                self.assertEqual(len(session.events), 2)
                self.assertEqual([e.occurrence for e in session.events], [0, 1])
                self.assertTrue(torch.equal(session.events[0].value, torch.tensor([1.0, 2.0])))
                self.assertTrue(torch.equal(session.events[1].value, torch.tensor([3.0, 4.0])))
                self.assertEqual(len(tap._forward_hooks), 0)
                tap(torch.ones(1))  # no leaked hook or further recording
                self.assertEqual(len(session.events), 2)

    def test_zero_limit_accepts_empty_and_metadata_only_events(self):
        for mode in ("direct", "gradient"):
            with self.subTest(mode=mode):
                tap = TapPoint("point")
                with TraceSession(tap, mode=mode, max_tensor_bytes=0) as session:
                    tap(torch.empty(0))
                self.assertEqual(session.tensor_bytes, 0)
                self.assertEqual(len(session.events), 1)
                self.assertEqual(session.events[0].value.numel(), 0)

        tap = TapPoint("id")
        with TraceSession(tap, mode="id", max_tensor_bytes=0) as session:
            tap(torch.tensor([float("nan"), float("inf")]))
            tap(torch.ones(10))
        self.assertEqual(len(session.events), 2)
        self.assertEqual(session.tensor_bytes, 0)
        self.assertTrue(all(event.value is None for event in session.events))

    def test_invalid_limits_are_rejected(self):
        tap = TapPoint("point")
        for limit in (-1, 1.5, True, float("nan"), float("inf"), "8"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                TraceSession(tap, max_tensor_bytes=limit)

    def test_limit_cannot_be_enabled_after_unlimited_captures(self):
        tap = TapPoint("point")
        with TraceSession(tap) as session:
            tap(torch.ones(1))
            with self.assertRaises(AttributeError):
                session.max_tensor_bytes = 4

    def test_budget_uses_tensor_dtype_size(self):
        for mode in ("direct", "gradient"):
            with self.subTest(mode=mode):
                tap = TapPoint("point")
                session = TraceSession(tap, mode=mode, max_tensor_bytes=8)
                with self.assertRaisesRegex(RuntimeError, "max_tensor_bytes"):
                    with session:
                        tap(torch.tensor([1.0, 2.0], dtype=torch.float64))  # 16 bytes
                self.assertEqual(session.events, [])

    def test_removing_events_does_not_restore_capture_allowance(self):
        tap = TapPoint("point")
        session = TraceSession(tap, max_tensor_bytes=4)
        with self.assertRaisesRegex(RuntimeError, "max_tensor_bytes"):
            with session:
                tap(torch.ones(1))
                session.events.clear()
                self.assertEqual(session.tensor_bytes, 0)
                tap(torch.ones(1))
        self.assertEqual(session.events, [])
        self.assertEqual(len(tap._forward_hooks), 0)

    def test_rejected_invocation_does_not_advance_sequence_or_allowance(self):
        tap = TapPoint("point")
        session = TraceSession(tap, max_tensor_bytes=8)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with session:
                tap(torch.tensor([1.0]))
                with self.assertRaisesRegex(RuntimeError, "max_tensor_bytes"):
                    tap(torch.tensor([2.0, 3.0]))
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    tap(torch.tensor([4.0]))
        self.assertTrue(session.failed)
        self.assertEqual(session.tensor_bytes, 4)
        self.assertEqual([event.sequence for event in session.events], [0])
        self.assertEqual([event.occurrence for event in session.events], [0])
        self.assertTrue(torch.equal(session.events[0].value, torch.tensor([1.0])))

    def test_sparse_value_capture_is_rejected_but_metadata_remains_allowed(self):
        sparse = torch.sparse_coo_tensor([[0]], [1.0], (1,), check_invariants=True)
        for mode in ("direct", "gradient"):
            with self.subTest(mode=mode):
                tap = TapPoint("point")
                session = TraceSession(tap, mode=mode, max_tensor_bytes=4)
                with self.assertRaisesRegex(ValueError, "strided"):
                    with session:
                        tap(sparse)
                self.assertEqual(session.events, [])
                self.assertEqual(len(tap._forward_hooks), 0)

        tap = TapPoint("id")
        with TraceSession(tap, mode="id", max_tensor_bytes=0) as session:
            tap(sparse)
        self.assertEqual(len(session.events), 1)
        self.assertIsNone(session.events[0].value)

    def test_nonfinite_values_inside_limit_abort_without_leaking_hooks(self):
        for mode in ("direct", "gradient"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(mode=mode, value=value):
                    tap = TapPoint("point")
                    session = TraceSession(tap, mode=mode, max_tensor_bytes=4)
                    with self.assertRaises(ValueError):
                        with session:
                            tap(torch.tensor([value]))
                    self.assertEqual(session.tensor_bytes, 0)
                    self.assertEqual(session.events, [])
                    self.assertEqual(len(tap._forward_hooks), 0)
                    self.assertTrue(torch.equal(tap(torch.ones(1)), torch.ones(1)))

    def test_overflow_is_checked_before_validation_or_gradient_readout(self):
        # These calls allocate temporary tensors; an oversized activation must
        # be rejected before either allocation path is entered.
        for mode, target in (("direct", "looking_glass.probes._validate_tensor"),
                             ("gradient", "looking_glass.probes.gradient_readout")):
            with self.subTest(mode=mode):
                tap = TapPoint("point")
                session = TraceSession(tap, mode=mode, max_tensor_bytes=3)
                with patch(target, side_effect=AssertionError("allocation path entered")):
                    with self.assertRaisesRegex(RuntimeError, "max_tensor_bytes"):
                        with session:
                            tap(torch.tensor([1.0]))  # 4 bytes
                self.assertEqual(session.events, [])
                self.assertEqual(len(tap._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
