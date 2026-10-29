"""CPU-only checks for the trusted continuous case-ownership seam."""

import threading

import unittest
import torch
from torch import nn

from looking_glass.intervention_plan import ActivationPlan
from looking_glass.model_ownership import ModelLease

class Model(nn.Module):
    def __init__(self, tap=None):
        super().__init__()
        self.tap = nn.Identity() if tap is None else tap

    def forward(self, value):
        return self.tap(value)

def make_plan(model, lease, *, value=None):
    step = {"name": "seen", "operation": "capture", "module_path": "tap"}
    if value is not None:
        step = {"name": "patched", "operation": "patch", "module_path": "tap", "value": value}
    return ActivationPlan(model, [step], _parent_lease=lease)

class _UncertainRegistration(nn.Identity):
    def register_forward_hook(self, *args, **kwargs):
        super().register_forward_hook(*args, **kwargs)
        raise MemoryError("installed without returning handle")

class _RemoveOnce(nn.Identity):
    def register_forward_hook(self, *args, **kwargs):
        handle = super().register_forward_hook(*args, **kwargs)
        if kwargs.get("always_call"):
            return handle

        class Wrapper:
            failed = False

            def remove(self):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("transient removal failure")
                handle.remove()

        return Wrapper()


class ParentLeaseTests(unittest.TestCase):
    def test_borrow_cannot_be_released_during_plan_entry_before_registration(self):
        model = Model()
        reached, resume = threading.Event(), threading.Event()
        shared, errors = {}, []

        def run_owner():
            try:
                lease = ModelLease(model).acquire()
                plan = make_plan(model, lease)
                shared.update(lease=lease, plan=plan)
                register = plan._register

                def gated_register(*args, **kwargs):
                    if not reached.is_set():
                        reached.set()
                        if not resume.wait(5):
                            raise RuntimeError("test gate timed out")
                    return register(*args, **kwargs)

                plan._register = gated_register
                with plan:
                    model(torch.tensor([1.]))
                lease.release()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=run_owner)
        thread.start()
        try:
            self.assertTrue(reached.wait(5))
            plan, lease = shared["plan"], shared["lease"]
            with self.assertRaisesRegex(RuntimeError, "hooks or forward"):
                plan._lease.release(plan._borrow_release_key)
            with self.assertRaisesRegex(RuntimeError, "active"):
                lease.release()
            with self.assertRaisesRegex(RuntimeError, "already leased"):
                ModelLease(model).acquire()
        finally:
            resume.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_parent_remains_leased_between_sequential_plans(self):
        model = Model()
        lease = ModelLease(model).acquire()
        try:
            for value in (torch.tensor([1.]), torch.tensor([2.])):
                plan = make_plan(model, lease)
                with plan:
                    with self.assertRaisesRegex(RuntimeError, "active"):
                        lease.release()
                    with self.assertRaisesRegex(RuntimeError, "already leased"):
                        ModelLease(model).acquire()
                    assert torch.equal(model(value), value)
                assert torch.equal(plan.captures["seen"], value)
                with self.assertRaisesRegex(RuntimeError, "already leased"):
                    ModelLease(model).acquire()
        finally:
            lease.release()
        ModelLease(model).acquire().release()


    def test_only_one_child_and_rejected_sibling_cannot_release_active_child(self):
        model = Model()
        lease = ModelLease(model).acquire()
        first, second = make_plan(model, lease), make_plan(model, lease)
        with first:
            with self.assertRaisesRegex(RuntimeError, "only plan cleanup"):
                first._lease.release()
            with self.assertRaisesRegex(RuntimeError, "hooks or forward"):
                first._lease.release(first._borrow_release_key)
            with self.assertRaisesRegex(RuntimeError, "another plan scope"):
                second.__enter__()
            with self.assertRaisesRegex(RuntimeError, "active"):
                lease.release()
            assert torch.equal(model(torch.tensor([3.])), torch.tensor([3.]))
        lease.release()


    def test_stale_child_release_cannot_remove_a_later_child(self):
        model = Model()
        lease = ModelLease(model).acquire()
        first, second = make_plan(model, lease), make_plan(model, lease)
        with first:
            model(torch.tensor([1.]))
        with second:
            with self.assertRaisesRegex(RuntimeError, "hooks or forward"):
                first._lease.release(first._borrow_release_key)
            with self.assertRaisesRegex(RuntimeError, "active"):
                lease.release()
            model(torch.tensor([2.]))
        lease.release()


    def test_parent_model_identity_and_held_state_are_required(self):
        one, two = Model(), Model()
        lease = ModelLease(one).acquire()
        with self.assertRaisesRegex(ValueError, "exact lease"):
            make_plan(two, lease)
        plan = make_plan(one, lease)
        lease.release()
        with self.assertRaisesRegex(RuntimeError, "parent model lease"):
            plan.__enter__()


    def test_foreign_thread_cannot_enter_borrowed_plan(self):
        model = Model()
        lease = ModelLease(model).acquire()
        plan = make_plan(model, lease)
        errors = []

        def try_enter():
            try:
                plan.__enter__()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=try_enter)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(errors) == 1 and "owner thread" in str(errors[0])
        with self.assertRaisesRegex(RuntimeError, "already leased"):
            ModelLease(model).acquire()
        lease.release()


    def test_tree_change_at_child_entry_quarantines_parent(self):
        model = Model()
        lease = ModelLease(model).acquire()
        plan = make_plan(model, lease)
        model.add_module("new_tap", nn.Identity())
        with self.assertRaisesRegex(RuntimeError, "tree changed"):
            plan.__enter__()
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            lease.release()
        with self.assertRaisesRegex(RuntimeError, "already leased"):
            ModelLease(model.tap).acquire()

    def test_failed_forward_with_tree_change_quarantines_and_keeps_original_error(self):
        class Mutating(Model):
            def forward(self, value):
                self.tap(value)
                self.new_tap = nn.Identity()
                raise ValueError("original forward error")

        model = Mutating()
        lease = ModelLease(model).acquire()
        plan = make_plan(model, lease)
        caught = None
        try:
            with plan:
                model(torch.tensor([1.]))
        except ValueError as error:
            caught = error
        self.assertIsNotNone(caught)
        self.assertEqual(str(caught), "original forward error")
        self.assertIsInstance(caught.__cause__, RuntimeError)
        self.assertIn("tree changed", str(caught.__cause__))
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            lease.release()
        with self.assertRaisesRegex(RuntimeError, "already leased"):
            ModelLease(model.tap).acquire()


    def test_indeterminate_hook_registration_cannot_release_parent(self):
        model = Model(_UncertainRegistration())
        lease = ModelLease(model).acquire()
        plan = make_plan(model, lease)
        with self.assertRaises(MemoryError):
            plan.__enter__()
        assert model.tap._forward_hooks
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            lease.release()
        with self.assertRaisesRegex(RuntimeError, "already leased"):
            ModelLease(model).acquire()
        with self.assertRaisesRegex(RuntimeError, "indeterminate"):
            plan.close()


    def test_failed_removal_keeps_parent_until_known_hook_retry(self):
        model = Model(_RemoveOnce())
        lease = ModelLease(model).acquire()
        plan = make_plan(model, lease)
        with self.assertRaisesRegex(RuntimeError, "transient removal"):
            with plan:
                model(torch.tensor([1.]))
        with self.assertRaisesRegex(RuntimeError, "active"):
            lease.release()
        plan.close()
        with self.assertRaisesRegex(RuntimeError, "already leased"):
            ModelLease(model).acquire()
        lease.release()


    def test_explicit_quarantine_permanently_retains_parent(self):
        model = Model()
        lease = ModelLease(model).acquire()
        lease._quarantine()
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            lease.release()
        with self.assertRaisesRegex(RuntimeError, "parent model lease"):
            make_plan(model, lease).__enter__()


    def test_borrowed_plan_matches_ordinary_plan_and_preserves_parent(self):
        model = Model()
        patched = torch.tensor([9.])
        with ActivationPlan(model, [{"name": "p", "operation": "patch", "module_path": "tap", "value": patched}]):
            ordinary = model(torch.tensor([1.]))
        lease = ModelLease(model).acquire()
        try:
            with make_plan(model, lease, value=patched):
                borrowed = model(torch.tensor([1.]))
            assert torch.equal(ordinary, borrowed)
        finally:
            lease.release()


