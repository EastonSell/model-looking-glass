"""Cancellation and recoverable one-commit publication of grouped jobs."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import inspect
import subprocess
import sys
import textwrap
import threading
import unittest
from unittest.mock import patch

import torch

from looking_glass.grouped_broker_state import (
    CohortMember, CohortState, CohortUnavailable, CohortRetired,
)
from looking_glass.microbatch import (BatchOutcome, BatchProfile, ForwardContract, Microbatch,
                                     MicrobatchRequest, RequestResult)


class CohortStateTests(unittest.TestCase):
    def setUp(self):
        self.members = (CohortMember('job-1', 'team-a', 'worker-1'),
                        CohortMember('job-2', 'team-b', 'worker-2'))
        contract = ForwardContract('model', 'checkpoint', 'adapter', 'cpu', torch.float32,
                                   runtime_id='cpu-test', seed=17, deterministic=True,
                                   batch_separable=True, mode='eval', rng_policy='none',
                                   cache_policy='disabled', mutable_state_policy='none',
                                   tracing=False, intervention=False)
        self.profile = BatchProfile('profile', 'cert', contract, (('input_ids', torch.int64),),
                                    (1, 1), torch.float32, 1, 0.0, 0.0, (2,), 2, 1,
                                    2, 2, 16, 8, 24)
        self.state = CohortState('batch-1', self.members, self.profile)

    def outputs(self):
        return BatchOutcome((RequestResult('job-1', 'team-a', 'worker-1', 'ok',
                                           torch.tensor([[[1.0]]])),
                             RequestResult('job-2', 'team-b', 'worker-2', 'ok',
                                           torch.tensor([[[2.0]]]))))

    def test_cancel_before_launch_blocks_group_without_retry(self):
        self.assertTrue(self.state.cancel('job-1'))
        self.assertFalse(self.state.try_commit_launch())
        publication = self.state.finish_cancelled_before_launch()
        self.assertFalse(publication.launch_committed)
        self.assertEqual([result.status for result in publication.members], ['cancelled', 'failed'])
        self.assertEqual([result.cancel_requested for result in publication.members], [True, False])
        self.assertEqual(publication.error_code, 'cohort_cancelled_before_launch')
        self.assertFalse(self.state.cancel('job-2'))
        self.assertIs(self.state.publication(), publication)

    def test_cancel_after_launch_suppresses_only_that_member(self):
        self.assertTrue(self.state.try_commit_launch())
        self.assertTrue(self.state.cancel('job-1'))
        outcome = self.outputs()
        publication = self.state.publish_success(outcome)
        self.assertEqual([r.status for r in publication.members], ['cancelled', 'ok'])
        self.assertIsNone(publication.members[0].logits)
        self.assertIs(publication.members[1].logits, outcome.results[1].logits)
        self.assertEqual(publication.members[1].member, self.members[1])
        self.assertFalse(self.state.cancel('job-2'))

    def test_real_microbatch_split_with_late_member_cancellation(self):
        jobs = tuple(MicrobatchRequest(member.job_id, member.owner, member.request_id,
                                       self.profile.contract,
                                       {'input_ids': torch.tensor([[index + 1]])})
                     for index, member in enumerate(self.members))
        plan = Microbatch(jobs, self.profile)
        self.assertTrue(self.state.try_commit_launch())
        launch = plan.launch()
        self.assertIs(plan.pending_launch(), launch)
        plan.claim_launch(launch)
        outcome = plan.finish(torch.tensor([[[1.0]], [[2.0]]]), launch_id=launch.launch_id)
        self.assertIs(plan.outcome(), outcome)
        self.assertTrue(self.state.cancel('job-1'))
        publication = self.state.publish_success(outcome)
        self.assertEqual([r.status for r in publication.members], ['cancelled', 'ok'])
        self.assertIsNone(publication.members[0].logits)
        self.assertTrue(torch.equal(publication.members[1].logits, torch.tensor([[[2.0]]])))

    def test_group_failure_dominates_cancellation_and_keeps_it_visible(self):
        self.assertTrue(self.state.try_commit_launch())
        self.assertTrue(self.state.cancel('job-1'))
        self.assertTrue(self.state.cancel('job-2'))
        publication = self.state.publish_failure('out_of_memory', quarantine_required=True)
        self.assertEqual([r.status for r in publication.members], ['failed', 'failed'])
        self.assertEqual([r.cancel_requested for r in publication.members], [True, True])
        self.assertEqual(publication.error_code, 'out_of_memory')
        self.assertTrue(publication.quarantine_required)
        with self.assertRaises(CohortUnavailable):
            self.state.retire()

    def test_wrong_result_identity_poisoned_without_partial_publication(self):
        self.assertTrue(self.state.try_commit_launch())
        result = BatchOutcome((RequestResult('job-1', 'team-b', 'worker-1', 'ok', torch.ones(1)),
                               RequestResult('job-2', 'team-b', 'worker-2', 'ok', torch.ones(1))))
        with self.assertRaises(ValueError):
            self.state.publish_success(result)
        with self.assertRaises(CohortUnavailable):
            self.state.publication()
        self.assertIsNone(self.state.cancellation_requested('job-1'))
        self.assertTrue(self.state.quarantine_required)

    def test_failure_during_outcome_allocation_exposes_no_partial_members(self):
        self.assertTrue(self.state.try_commit_launch())
        from looking_glass.grouped_broker_state import MemberCompletion
        calls = []
        def fail_second(*args):
            calls.append(None)
            if len(calls) == 2:
                raise MemoryError
            return MemberCompletion(*args)
        with patch('looking_glass.grouped_broker_state.MemberCompletion', side_effect=fail_second):
            with self.assertRaises(MemoryError):
                self.state.publish_success(self.outputs())
        self.assertEqual(len(calls), 2)
        with self.assertRaises(CohortUnavailable):
            self.state.publication()
        self.assertTrue(self.state.quarantine_required)

    def test_notification_failure_after_commit_is_recoverable(self):
        self.assertTrue(self.state.try_commit_launch())
        received, errors = [], []
        def wait():
            try:
                received.append(self.state.wait(timeout=2))
            except BaseException as exc:
                errors.append(exc)
        waiter = threading.Thread(target=wait, daemon=True)
        waiter.start()
        with patch.object(self.state._condition, 'notify_all', side_effect=MemoryError):
            with self.assertRaises(MemoryError):
                self.state.publish_success(self.outputs())
        waiter.join(2)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(errors, [])
        publication = self.state.publication()
        self.assertEqual([r.status for r in publication.members], ['ok', 'ok'])
        self.assertIs(self.state.wait(timeout=1), publication)
        self.assertEqual(received, [publication])
        self.assertFalse(self.state.quarantine_required)

    def test_overlapping_owner_outputs_fail_closed(self):
        self.state.try_commit_launch()
        shared = torch.ones((1, 1, 1))
        results = BatchOutcome((RequestResult('job-1', 'team-a', 'worker-1', 'ok', shared),
                                RequestResult('job-2', 'team-b', 'worker-2', 'ok', shared)))
        with self.assertRaises(ValueError):
            self.state.publish_success(results)
        with self.assertRaises(CohortUnavailable):
            self.state.publication()
        self.assertTrue(self.state.quarantine_required)

    def test_profile_shape_dtype_and_finite_values_are_mandatory(self):
        for bad in (torch.ones((2, 1, 1)), torch.ones((1, 1, 1), dtype=torch.float64),
                    torch.tensor([[[float('nan')]]]), torch.tensor([[[float('inf')]]])):
            with self.subTest(bad=bad):
                state = CohortState('batch', self.members, self.profile)
                state.try_commit_launch()
                outcome = self.outputs()
                results = (outcome.results[0], replace(outcome.results[1], logits=bad))
                with self.assertRaises(ValueError):
                    state.publish_success(BatchOutcome(results))
                self.assertTrue(state.quarantine_required)
                with self.assertRaises(CohortUnavailable):
                    state.publication()

    def test_line_interrupts_before_and_after_publication_commit(self):
        code = CohortState._publish.__code__
        lines, first = inspect.getsourcelines(CohortState._publish)
        targets = {line.strip(): first + i for i, line in enumerate(lines)
                   if line.strip() in ('self._state = publication', 'self._condition.notify_all()')}
        self.assertEqual(len(targets), 2)
        for fragment, target in targets.items():
            with self.subTest(fragment=fragment):
                state = CohortState('batch', self.members, self.profile)
                state.try_commit_launch()
                previous = sys.gettrace()
                def trace(frame, event, arg):
                    if event == 'line' and frame.f_code is code and frame.f_lineno == target:
                        raise KeyboardInterrupt
                    return trace
                try:
                    sys.settrace(trace)
                    with self.assertRaises(KeyboardInterrupt):
                        state.publish_success(self.outputs())
                finally:
                    sys.settrace(previous)
                states = []
                thread = threading.Thread(target=lambda: states.append(state.phase), daemon=True)
                thread.start()
                thread.join(1)
                self.assertFalse(thread.is_alive(), 'cohort lock was stranded')
                if fragment == 'self._state = publication':
                    self.assertEqual(states, ['unknown'])
                    self.assertTrue(state.quarantine_required)
                else:
                    self.assertEqual(states, ['terminal'])
                    self.assertEqual(len(state.publication().members), 2)

    def test_return_event_interrupt_recovers_launch_and_cancel_decisions(self):
        for method_name in ('try_commit_launch', 'cancel'):
            with self.subTest(method=method_name):
                state = CohortState('batch', self.members, self.profile)
                method = getattr(state, method_name)
                previous = sys.gettrace()
                def trace(frame, event, arg):
                    if event == 'return' and frame.f_code is method.__func__.__code__:
                        raise KeyboardInterrupt
                    return trace
                try:
                    sys.settrace(trace)
                    with self.assertRaises(KeyboardInterrupt):
                        method(*(('job-1',) if method_name == 'cancel' else ()))
                finally:
                    sys.settrace(previous)
                result = []
                thread = threading.Thread(target=lambda: result.append(
                    state.cancellation_requested('job-1') if method_name == 'cancel' else state.phase), daemon=True)
                thread.start()
                thread.join(1)
                self.assertFalse(thread.is_alive(), 'cohort lock was stranded')
                self.assertEqual(result, [True if method_name == 'cancel' else 'launch_committed'])

    def test_all_reached_publication_line_interrupts_release_lock_in_subprocess(self):
        # The subprocess bounds failures even if a CPython exception-table gap
        # strands the condition lock. Arbitrary opcode interrupts remain outside
        # this cooperative contract and require supervisor recovery.
        script = textwrap.dedent('''
            import inspect, sys, threading, torch
            from looking_glass.grouped_broker_state import CohortMember, CohortState
            from looking_glass.microbatch import BatchOutcome, BatchProfile, ForwardContract, RequestResult
            c=ForwardContract('m','c','a','cpu',torch.float32,'runtime',17,True,True,'eval','none','disabled','none',False,False)
            p=BatchProfile('p','v',c,(('input_ids',torch.int64),),(1,1),torch.float32,1,0.,0.,(2,),2,1,2,2,16,8,24)
            members=(CohortMember('j1','a','r1'),CohortMember('j2','b','r2'))
            result=BatchOutcome((RequestResult('j1','a','r1','ok',torch.ones((1,1,1))),
                                 RequestResult('j2','b','r2','ok',torch.ones((1,1,1)))))
            code=CohortState._publish.__code__
            lines=set()
            def collect(f,event,arg):
                if event=='line' and f.f_code is code: lines.add(f.f_lineno)
                return collect
            state=CohortState('g',members,p); state.try_commit_launch()
            sys.settrace(collect); state.publish_success(result); sys.settrace(None)
            for target in sorted(lines):
                state=CohortState('g',members,p); state.try_commit_launch(); hit=[]
                def trace(f,event,arg):
                    if event=='line' and f.f_code is code and f.f_lineno==target:
                        hit.append(target); sys.settrace(None); raise KeyboardInterrupt
                    return trace
                try:
                    sys.settrace(trace); state.publish_success(result)
                except KeyboardInterrupt: pass
                finally: sys.settrace(None)
                seen=[]
                t=threading.Thread(target=lambda: seen.append(state.phase),daemon=True); t.start(); t.join(.3)
                if t.is_alive(): raise AssertionError('stranded condition at line '+str(target))
                if not hit: raise AssertionError('did not reach line '+str(target))
            print('checked',len(lines),'reached lines')
        ''')
        result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('checked', result.stdout)

    def test_cancel_and_publication_race_linearizes_for_each_member(self):
        for _ in range(100):
            state = CohortState('batch', self.members, self.profile)
            self.assertTrue(state.try_commit_launch())
            barrier = threading.Barrier(2)
            def cancel():
                barrier.wait()
                return state.cancel('job-1')
            def finish():
                barrier.wait()
                return state.publish_success(self.outputs())
            with ThreadPoolExecutor(max_workers=2) as pool:
                a, b = pool.submit(cancel), pool.submit(finish)
                accepted, publication = a.result(3), b.result(3)
            member = publication.members[0]
            self.assertEqual(member.cancel_requested, accepted)
            self.assertEqual(member.status, 'cancelled' if accepted else 'ok')
            self.assertEqual(publication.members[1].status, 'ok')

    def test_cancel_and_launch_race_has_one_prelaunch_decision(self):
        for _ in range(100):
            state = CohortState('batch', self.members, self.profile)
            barrier = threading.Barrier(2)
            def cancel():
                barrier.wait()
                return state.cancel('job-1')
            def launch():
                barrier.wait()
                return state.try_commit_launch()
            with ThreadPoolExecutor(max_workers=2) as pool:
                c, l = pool.submit(cancel), pool.submit(launch)
                self.assertTrue(c.result(3))
                committed = l.result(3)
            if committed:
                publication = state.publish_success(self.outputs())
                self.assertTrue(publication.launch_committed)
            else:
                publication = state.finish_cancelled_before_launch()
                self.assertFalse(publication.launch_committed)
            self.assertEqual(publication.members[0].status, 'cancelled')

    def test_retirement_requires_a_terminal_result_and_drops_visibility(self):
        with self.assertRaises(RuntimeError):
            self.state.retire()
        self.state.try_commit_launch()
        self.state.publish_success(self.outputs())
        self.state.retire()
        with self.assertRaises(CohortRetired):
            self.state.publication()
        with self.assertRaises(CohortRetired):
            self.state.wait(timeout=0)


if __name__ == '__main__':
    unittest.main()
