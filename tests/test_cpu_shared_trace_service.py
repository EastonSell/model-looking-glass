"""CPU-only end-to-end checks for cooperative shared trace clients."""
import hashlib
import gc
import importlib
import importlib.util
import inspect
import sys
import threading
import time
import unittest
import weakref
from unittest import mock

import torch

from looking_glass.cpu_trace_owner import CpuTraceOwner
from looking_glass.host_budget import HostBudget
from looking_glass.producer_slot import ProducerSlotError
from looking_glass.trace_producer import ClaimLimit, ProducerProfile
from looking_glass.trace_store import HostTraceStore
from looking_glass.trace_transport import DeliveryError
from looking_glass.traced_broker_worker import SampleTrace, SiteTrace, TraceLimits
from looking_glass.traced_worker import TracedGPT2Worker

_QUARANTINE_ROOTS = []

@unittest.skipUnless(importlib.util.find_spec('transformers') is not None,
                     'requires the optional transformers dependency')
class SharedCpuTraceServiceTests(unittest.TestCase):
    def make_service(self, *, budget_bytes=1_000_000, trace_bytes=32, max_event_bytes=32, slots=1):
        try:
            service_module = importlib.import_module('looking_glass.cpu_shared_trace_service')
        except ModuleNotFoundError as error:
            if error.name != 'looking_glass.cpu_shared_trace_service':
                raise
            self.fail('shared CPU trace service is not implemented')
        from transformers import GPT2Config, GPT2LMHeadModel
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(71)
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=32, n_positions=8, n_ctx=8, n_embd=8, n_layer=1,
                n_head=2, use_cache=False, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()
        worker = TracedGPT2Worker('shared-tiny-gpt2', model, device='cpu', max_batch_size=1,
                                  max_sequence_length=3, max_input_bytes=24, max_output_bytes=384)
        budget = HostBudget(max_bytes=budget_bytes, max_items=64, max_requests=16, max_operations=16)
        owner = CpuTraceOwner(worker=worker, budget=budget,
                              store=HostTraceStore(max_tensor_bytes=1024, max_items=4),
                              broker_id='a' * 32, process_generation='b' * 32,
                              max_slots=slots, max_event_bytes=max_event_bytes)
        profile = ProducerProfile(ClaimLimit(48, 8), ClaimLimit(768, 2),
                                  ClaimLimit(trace_bytes, slots), ClaimLimit(100_000, 1))
        limits = TraceLimits(1, 3, 1, slots, slots, trace_bytes, 3)
        self.original_profile, self.original_limits = profile, limits
        service = service_module.CpuSharedTraceService(owner=owner, profile=profile,
                                                       limits=limits, team_ids=('alpha', 'beta'))
        def close_if_quiescent():
            if service._active is None and not service.quarantined:
                worker.close()
            else:
                # Do not dispose a worker whose attempt remains quarantined.
                _QUARANTINE_ROOTS.append((service, worker))
        self.addCleanup(close_if_quiescent)
        return service_module, service, budget

    def input_claim(self, budget, name, values):
        request = budget.open_request(name)
        claim = request.reserve(name, 'input', 24, items=1)
        tensor = torch.tensor([values], dtype=torch.int64)
        return request, claim, {'input_ids': tensor}

    def test_two_clients_receive_only_their_own_actual_trace_from_one_serial_owner(self):
        _, service, budget = self.make_service()
        alpha, beta = service.client('alpha'), service.client('beta')
        a_owner, a_claim, a_input = self.input_claim(budget, 'a-input', [1, 2, 3])
        b_owner, b_claim, b_input = self.input_claim(budget, 'b-input', [3, 2, 1])
        first_entered, allow_first, second_done = threading.Event(), threading.Event(), threading.Event()
        received, results, errors = [], {}, []

        def event(team, block):
            def sink(manifest, payload):
                identity = manifest.identity
                received.append((team, identity.team_id, identity.request_id, identity.sample_id,
                                 manifest.event.token_positions, len(payload), hashlib.sha256(payload).hexdigest()))
                if block:
                    first_entered.set()
                    if not allow_first.wait(5):
                        raise RuntimeError('test failed to release first callback')
            return sink

        def submit(client, team, inputs, owner, claim, position, block):
            try:
                results[team] = client.submit(
                    request_id='request-' + team, inputs=inputs, input_owner=owner, input_claim=claim,
                    samples=(SampleTrace('sample-' + team, (position,)),),
                    sites=(SiteTrace(0, 'attn'),), on_event=event(team, block))
            except BaseException as error:
                errors.append(error)
            finally:
                if team == 'beta':
                    second_done.set()

        first = threading.Thread(target=submit, args=(alpha, 'alpha', a_input, a_owner, a_claim, 0, True))
        second = threading.Thread(target=submit, args=(beta, 'beta', b_input, b_owner, b_claim, 2, False))
        first.start()
        try:
            self.assertTrue(first_entered.wait(5))
            second.start()
            self.assertFalse(second_done.wait(0.1), 'second client must wait during first delivery')
            self.assertGreater(budget.snapshot().used_bytes, 48)
        finally:
            allow_first.set()
            first.join(10)
            if second.ident is not None:
                second.join(10)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([item[:6] for item in received], [
            ('alpha', 'alpha', 'request-alpha', 'sample-alpha', (0,), 32),
            ('beta', 'beta', 'request-beta', 'sample-beta', (2,), 32)])
        self.assertNotEqual(received[0][6], received[1][6])
        self.assertEqual((results['alpha'].event_count, results['alpha'].trace_bytes), (1, 32))
        self.assertEqual((results['beta'].event_count, results['beta'].trace_bytes), (1, 32))
        self.assertNotEqual(results['alpha'].job_id, results['beta'].job_id)
        self.assertEqual(budget.snapshot().used_bytes, 48)
        a_input = b_input = None
        a_claim.release()
        b_claim.release()
        a_owner.close()
        b_owner.close()
        alpha.acknowledge(results['alpha'])
        beta.acknowledge(results['beta'])
        service.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_trace_cap_rejects_oversized_selection_before_forward_and_allows_retry(self):
        _, service, budget = self.make_service(trace_bytes=32, max_event_bytes=64)
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        seen = []
        with self.assertRaises(ValueError):
            client.submit(request_id='large', inputs=inputs, input_owner=owner, input_claim=claim,
                          samples=(SampleTrace('row', (0, 1)),), sites=(SiteTrace(0, 'attn'),),
                          on_event=lambda metadata, payload: seen.append(len(payload)))
        self.assertEqual(seen, [])
        self.assertEqual(budget.snapshot().used_bytes, 24)
        self.assertFalse(service.quarantined)
        receipt = client.submit(request_id='small', inputs=inputs, input_owner=owner, input_claim=claim,
                                samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                                on_event=lambda metadata, payload: seen.append(len(payload)))
        self.assertEqual(seen, [32])
        self.assertEqual(receipt.trace_bytes, 32)
        self.assertEqual(budget.snapshot().used_bytes, 24)
        inputs = None
        claim.release()
        owner.close()
        client.acknowledge(receipt)
        service.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_two_site_trace_is_complete_ordered_and_exactly_within_cap(self):
        _, service, budget = self.make_service(trace_bytes=64, slots=2)
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        seen = []

        def sink(metadata, payload):
            seen.append((metadata.identity.team_id, metadata.identity.sample_id,
                         metadata.event.module_kind, metadata.event.sequence, len(payload)))

        receipt = client.submit(request_id='two-sites', inputs=inputs, input_owner=owner, input_claim=claim,
                                samples=(SampleTrace('row', (0,)),),
                                sites=(SiteTrace(0, 'attn'), SiteTrace(0, 'mlp')), on_event=sink)
        self.assertEqual(seen, [('alpha', 'row', 'attn', 0, 32), ('alpha', 'row', 'mlp', 1, 32)])
        self.assertEqual((receipt.event_count, receipt.trace_bytes), (2, 64))
        self.assertEqual(budget.snapshot().used_bytes, 24)
        client.acknowledge(receipt)
        inputs = None
        claim.release()
        owner.close()
        service.close()

    def test_two_site_trace_over_aggregate_cap_is_rejected_before_forward(self):
        _, service, budget = self.make_service(trace_bytes=63, slots=2)
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        with mock.patch.object(service._owner._worker, 'execute_traced',
                               side_effect=AssertionError('forward must not run')) as forward:
            with self.assertRaises(ValueError):
                client.submit(request_id='too-large', inputs=inputs, input_owner=owner, input_claim=claim,
                              samples=(SampleTrace('row', (0,)),),
                              sites=(SiteTrace(0, 'attn'), SiteTrace(0, 'mlp')),
                              on_event=lambda metadata, payload: None)
            forward.assert_not_called()
        self.assertEqual(budget.snapshot().used_bytes, 24)
        self.assertFalse(service.quarantined)
        inputs = None
        claim.release()
        owner.close()
        service.close()

    def test_second_event_failure_never_publishes_complete_receipt(self):
        module, service, budget = self.make_service(trace_bytes=64, slots=2)
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        seen, held = [], []

        def sink(metadata, payload):
            seen.append(metadata.event.module_kind)
            if len(seen) == 2:
                held.append(payload)
                raise RuntimeError('second event failed')

        try:
            with self.assertRaises(DeliveryError):
                client.submit(request_id='partial', inputs=inputs, input_owner=owner, input_claim=claim,
                              samples=(SampleTrace('row', (0,)),),
                              sites=(SiteTrace(0, 'attn'), SiteTrace(0, 'mlp')), on_event=sink)
            self.assertEqual(seen, ['attn', 'mlp'])
            self.assertTrue(service.quarantined)
            self.assertEqual([c.nbytes for c in budget.snapshot().claims if c.stage == 'callback_exposure'], [32])
            with self.assertRaises(module.SharedTraceError):
                client.recover('partial')
            self.assertEqual(len(held[0]), 32)
        finally:
            _QUARANTINE_ROOTS.append((service, budget, owner, claim, inputs, held))

    def test_host_budget_denial_cleans_partial_admission_without_forward(self):
        _, service, budget = self.make_service(budget_bytes=50_000)
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        seen = []
        with self.assertRaises(ProducerSlotError):
            client.submit(request_id='small', inputs=inputs, input_owner=owner, input_claim=claim,
                          samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                          on_event=lambda metadata, payload: seen.append(len(payload)))
        self.assertEqual(seen, [])
        self.assertEqual(budget.snapshot().used_bytes, 24)
        self.assertFalse(service.quarantined)
        inputs = None
        claim.release()
        owner.close()
        service.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_manifest_budget_denial_drains_staged_trace_without_exposing_it(self):
        module, service, budget = self.make_service(budget_bytes=200_000)
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        seen = []
        with self.assertRaises(module.PostForwardCapacityError):
            client.submit(request_id='bounded', inputs=inputs, input_owner=owner, input_claim=claim,
                          samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                          on_event=lambda metadata, payload: seen.append(len(payload)))
        self.assertEqual(seen, [])
        self.assertEqual(budget.snapshot().used_bytes, 24)
        self.assertFalse(service.quarantined)
        inputs = None
        claim.release()
        owner.close()
        service.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_invalid_input_claim_rejection_cleans_unstarted_job(self):
        _, service, budget = self.make_service()
        client = service.client('alpha')
        owner = budget.open_request('input')
        claim = owner.reserve('input', 'input', 16, items=1)
        inputs = {'input_ids': torch.tensor([[1, 2, 3]], dtype=torch.int64)}
        seen = []
        with self.assertRaises(ValueError):
            client.submit(request_id='bad-claim', inputs=inputs, input_owner=owner, input_claim=claim,
                          samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                          on_event=lambda metadata, payload: seen.append(len(payload)))
        self.assertEqual(seen, [])
        self.assertEqual(budget.snapshot().used_bytes, 16)
        self.assertFalse(service.quarantined)
        inputs = None
        claim.release()
        owner.close()
        service.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_receipt_is_recoverable_only_by_its_client_and_requires_acknowledgement(self):
        module, service, budget = self.make_service()
        alpha, beta = service.client('alpha'), service.client('beta')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        receipt = alpha.submit(request_id='same-label', inputs=inputs, input_owner=owner, input_claim=claim,
                               samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                               on_event=lambda metadata, payload: None)
        self.assertIs(alpha.recover('same-label'), receipt)
        with self.assertRaises(module.SharedTraceError):
            beta.recover('same-label')
        with self.assertRaises(module.SharedTraceError):
            beta.acknowledge(receipt)
        with self.assertRaises(module.SharedTraceError):
            alpha.submit(request_id='next', inputs=inputs, input_owner=owner, input_claim=claim,
                         samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                         on_event=lambda metadata, payload: None)
        with self.assertRaises(module.SharedTraceError):
            service.close()
        self.assertIs(alpha.recover('same-label'), receipt)
        self.assertTrue(alpha.acknowledge(receipt))
        self.assertFalse(alpha.acknowledge(receipt))
        inputs = None
        claim.release()
        owner.close()
        service.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_one_owner_cannot_be_bound_to_two_service_lifetimes(self):
        module, service, budget = self.make_service()
        owner, profile, limits = service._owner, service._profile, service._limits
        with self.assertRaises(ValueError):
            module.CpuSharedTraceService(owner=owner, profile=profile, limits=limits,
                                         team_ids=('alpha', 'beta'))
        service.close()
        with self.assertRaises(ValueError):
            module.CpuSharedTraceService(owner=owner, profile=profile, limits=limits,
                                         team_ids=('alpha', 'beta'))
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_external_policy_changes_do_not_expand_the_service_trace_cap(self):
        _, service, budget = self.make_service(trace_bytes=32, max_event_bytes=64)
        object.__setattr__(self.original_limits, 'max_trace_bytes', 64)
        object.__setattr__(self.original_profile.trace_bundle, 'nbytes', 64)
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        seen = []
        with self.assertRaises(ValueError):
            client.submit(request_id='larger', inputs=inputs, input_owner=owner, input_claim=claim,
                          samples=(SampleTrace('row', (0, 1)),), sites=(SiteTrace(0, 'attn'),),
                          on_event=lambda metadata, payload: seen.append(len(payload)))
        self.assertEqual(seen, [])
        self.assertFalse(service.quarantined)
        self.assertEqual(budget.snapshot().used_bytes, 24)
        inputs = None
        claim.release()
        owner.close()
        service.close()

    def test_interrupted_receipt_ack_keeps_a_recoverable_or_acknowledged_outcome(self):
        if sys.gettrace() is not None:
            self.skipTest('preserve an existing trace hook')
        _, service, budget = self.make_service()
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        receipt = client.submit(request_id='ack', inputs=inputs, input_owner=owner, input_claim=claim,
                                samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                                on_event=lambda metadata, payload: None)
        source, start = inspect.getsourcelines(type(client).acknowledge)
        target = start + next(i for i, line in enumerate(source) if '_acknowledged[self._team_id] = receipt' in line)
        code, injected = type(client).acknowledge.__code__, []

        def interrupt(frame, event, _arg):
            if frame.f_code is code and event == 'line' and frame.f_lineno == target:
                injected.append(True)
                sys.settrace(None)
                raise KeyboardInterrupt('lost acknowledgement')
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                client.acknowledge(receipt)
        finally:
            sys.settrace(None)
        self.assertEqual(injected, [True])
        self.assertIs(client.recover('ack'), receipt)
        self.assertTrue(client.acknowledge(receipt))
        self.assertFalse(client.acknowledge(receipt))
        inputs = None
        claim.release()
        owner.close()
        service.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_lost_submit_return_recovers_exact_receipt_without_another_forward(self):
        if sys.gettrace() is not None:
            self.skipTest('preserve an existing trace hook')
        module, service, budget = self.make_service()
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        seen = []
        code = type(service)._submit.__code__
        injected = []

        def interrupt(frame, event, _arg):
            if frame.f_code is code and event == 'return':
                injected.append(True)
                sys.settrace(None)
                raise KeyboardInterrupt('lost submit return')
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                client.submit(request_id='lost', inputs=inputs, input_owner=owner, input_claim=claim,
                              samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                              on_event=lambda metadata, payload: seen.append(len(payload)))
        finally:
            sys.settrace(None)
        self.assertEqual(injected, [True])
        self.assertEqual(seen, [32])
        receipt = client.recover('lost')
        self.assertEqual((receipt.event_count, receipt.trace_bytes), (1, 32))
        with self.assertRaises(module.SharedTraceError):
            service.close()
        self.assertTrue(client.acknowledge(receipt))
        self.assertFalse(client.acknowledge(receipt))
        inputs = None
        claim.release()
        owner.close()
        service.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_same_client_cannot_queue_another_request_before_its_receipt(self):
        module, service, budget = self.make_service()
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        outcomes = []

        def held_sink(_metadata, _payload):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test did not release callback')

        def submit_first():
            outcomes.append(client.submit(request_id='first', inputs=inputs, input_owner=owner,
                                          input_claim=claim, samples=(SampleTrace('row', (0,)),),
                                          sites=(SiteTrace(0, 'attn'),), on_event=held_sink))

        def submit_again():
            try:
                client.submit(request_id='second', inputs=inputs, input_owner=owner, input_claim=claim,
                              samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                              on_event=lambda metadata, payload: None)
            except BaseException as error:
                outcomes.append(error)
            finally:
                finished.set()

        first, second = threading.Thread(target=submit_first), threading.Thread(target=submit_again)
        first.start()
        try:
            self.assertTrue(entered.wait(5))
            second.start()
            self.assertTrue(finished.wait(1), 'duplicate client must reject while first remains active')
        finally:
            release.set()
            first.join(10)
            if second.ident is not None:
                second.join(10)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(len(outcomes), 2)
        error = next(value for value in outcomes if isinstance(value, BaseException))
        receipt = next(value for value in outcomes if type(value) is module.TraceReceipt)
        self.assertIsInstance(error, module.SharedTraceError)
        client.acknowledge(receipt)
        inputs = None
        claim.release()
        owner.close()
        service.close()

    def test_interrupt_after_releasing_active_slot_does_not_leave_stale_thread(self):
        if sys.gettrace() is not None:
            self.skipTest('preserve an existing trace hook')
        _, service, budget = self.make_service()
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        source, start = inspect.getsourcelines(type(service)._release)
        lines = [i for i, line in enumerate(source) if 'self._condition.notify_all()' in line]
        target = start + lines[-1]
        code, injected = type(service)._release.__code__, []

        def interrupt(frame, event, _arg):
            if frame.f_code is code and event == 'line' and frame.f_lineno == target:
                injected.append(True)
                sys.settrace(None)
                raise KeyboardInterrupt('lost release return')
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                client.submit(request_id='first', inputs=inputs, input_owner=owner, input_claim=claim,
                              samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                              on_event=lambda metadata, payload: None)
        finally:
            sys.settrace(None)
        self.assertEqual(injected, [True])
        self.assertFalse(service.quarantined)
        first = client.recover('first')
        client.acknowledge(first)
        second = client.submit(request_id='second', inputs=inputs, input_owner=owner, input_claim=claim,
                               samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                               on_event=lambda metadata, payload: None)
        client.acknowledge(second)
        inputs = None
        claim.release()
        owner.close()
        service.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_retained_preflight_error_does_not_keep_client_input_alive(self):
        _, service, budget = self.make_service()
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        reference = weakref.ref(inputs['input_ids'])
        captured = []
        try:
            client.submit(request_id='bad', inputs=inputs, input_owner=owner, input_claim=claim,
                          samples=(), sites=(SiteTrace(0, 'attn'),), on_event=lambda metadata, payload: None)
        except ValueError as error:
            captured.append(error)
        self.assertEqual(len(captured), 1)
        self.assertFalse(service.quarantined)
        inputs = None
        claim.release()
        owner.close()
        gc.collect()
        self.assertIsNone(reference(), 'a retained preflight exception must not retain the input tensor')
        service.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_failing_callback_keeps_exposed_payload_charged_and_blocks_other_client(self):
        module, service, budget = self.make_service()
        alpha, beta = service.client('alpha'), service.client('beta')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        held = []

        def failing_sink(_metadata, payload):
            held.append(payload)
            raise RuntimeError('simulated callback failure')

        try:
            with self.assertRaises(DeliveryError):
                alpha.submit(request_id='failed', inputs=inputs, input_owner=owner, input_claim=claim,
                             samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                             on_event=failing_sink)
            self.assertTrue(service.quarantined)
            # Allow the underlying delivery to finish its own asynchronous cleanup.
            deadline = time.monotonic() + 5
            while any(c.stage == 'serialization' for c in budget.snapshot().claims) and time.monotonic() < deadline:
                time.sleep(0.01)
            exposure = [c for c in budget.snapshot().claims if c.stage == 'callback_exposure']
            self.assertEqual([c.nbytes for c in exposure], [32])
            self.assertEqual(len(held), 1)
            self.assertEqual(len(held[0]), 32)
            with self.assertRaises(module.SharedTraceError):
                beta.submit(request_id='other', inputs=inputs, input_owner=owner, input_claim=claim,
                            samples=(SampleTrace('other-row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                            on_event=lambda metadata, payload: None)
            with self.assertRaises(module.SharedTraceError):
                service.close()
        finally:
            # A supervisor keeps this isolated test's exact custody rooted until
            # process exit; the callback's retained bytes were not disposed.
            _QUARANTINE_ROOTS.append((service, budget, owner, claim, inputs, held))

    def test_callback_keyboard_interrupt_is_not_suppressed_and_keeps_custody(self):
        _, service, budget = self.make_service()
        client = service.client('alpha')
        owner, claim, inputs = self.input_claim(budget, 'input', [1, 2, 3])
        held = []

        def interrupted(_metadata, payload):
            held.append(payload)
            raise KeyboardInterrupt('cancel')

        try:
            with self.assertRaises(KeyboardInterrupt):
                client.submit(request_id='cancel', inputs=inputs, input_owner=owner, input_claim=claim,
                              samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                              on_event=interrupted)
            self.assertTrue(service.quarantined)
            self.assertEqual([c.nbytes for c in budget.snapshot().claims if c.stage == 'callback_exposure'], [32])
            self.assertEqual(len(held[0]), 32)
        finally:
            _QUARANTINE_ROOTS.append((service, budget, owner, claim, inputs, held))
