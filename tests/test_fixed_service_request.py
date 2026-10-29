"""CPU-only checks for the fixed service's trusted pre-allocation gate."""
import unittest

from looking_glass.ipc import IPCInputRejected
from looking_glass.fixed_service_request import FixedServiceRequestFactory


class _Spoof:
    def __eq__(self, other):
        raise AssertionError('untrusted equality was invoked')

    def __len__(self):
        raise AssertionError('untrusted length was invoked')


class _SpoofKey:
    def __hash__(self):
        return hash('input_ids')

    def __eq__(self, other):
        raise AssertionError('untrusted key equality was invoked')


class FixedServiceRequestTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.result = object()
        def factory(request_id, inputs, seed):
            self.calls.append((request_id, inputs, seed))
            return self.result
        self.factory = FixedServiceRequestFactory(factory)

    def test_exact_cases_reach_the_tensor_factory_once(self):
        for seed in (29, 43, 71):
            inputs = {'input_ids': [[15496, 995]]}
            self.assertIs(self.factory('request-1', inputs, seed), self.result)
            self.assertEqual(self.calls[-1], ('request-1', inputs, seed))
            self.assertIsNot(self.calls[-1][1], inputs)
            self.assertIsNot(self.calls[-1][1]['input_ids'], inputs['input_ids'])
            self.assertIsNot(self.calls[-1][1]['input_ids'][0], inputs['input_ids'][0])
        self.assertEqual(len(self.calls), 3)

    def test_invalid_cases_are_rejected_before_factory_call(self):
        valid = {'input_ids': [[15496, 995]]}
        cases = [
            ('bad space', valid, 29), ('', valid, 29), (_Spoof(), valid, 29),
            ('r', _Spoof(), 29), ('r', {}, 29), ('r', {_SpoofKey(): [[15496, 995]]}, 29),
            ('r', {'input_ids': []}, 29),
            ('r', {'input_ids': [[15496, 995]], 'attention_mask': [[1, 1]]}, 29),
            ('r', {'input_ids': [[15496, 996]]}, 29), ('r', {'input_ids': [[15496, True]]}, 29),
            ('r', {'input_ids': [[15496, _Spoof()]]}, 29), ('r', {'input_ids': [(15496, 995)]}, 29),
            ('r', valid, True), ('r', valid, 17), ('r', valid, _Spoof()),
        ]
        for request_id, inputs, seed in cases:
            with self.subTest(request_id=type(request_id), inputs=type(inputs), seed=type(seed)):
                with self.assertRaises(IPCInputRejected) as caught:
                    self.factory(request_id, inputs, seed)
                self.assertIs(type(caught.exception), IPCInputRejected)
        self.assertEqual(self.calls, [])

    def test_downstream_clean_signal_becomes_uncertain(self):
        class Subclass(IPCInputRejected):
            pass
        for kind in (IPCInputRejected, Subclass):
            with self.subTest(kind=kind):
                retained = []
                def factory(request_id, inputs, seed):
                    retained.append(inputs)
                    raise kind('post-allocation failure')
                wrapper = FixedServiceRequestFactory(factory)
                inputs = {'input_ids': [[15496, 995]]}
                with self.assertRaisesRegex(RuntimeError, 'tensor factory failed') as caught:
                    wrapper('r', inputs, 29)
                self.assertNotIsInstance(caught.exception, IPCInputRejected)
                self.assertEqual(retained[0], inputs)
                self.assertIsNot(retained[0], inputs)

    def test_mutation_after_validation_cannot_change_the_owned_fixed_case(self):
        import sys

        inputs = {'input_ids': [[15496, 995]]}
        seen = []
        def downstream(_request_id, owned_inputs, _seed):
            seen.append(owned_inputs)
            return self.result
        wrapper = FixedServiceRequestFactory(downstream)
        target = FixedServiceRequestFactory.__call__.__code__
        # Trigger on the first line of the try block, after validation and
        # construction of the canonical snapshot and before factory entry.
        previous = sys.gettrace()
        def mutate(frame, event, arg):
            if frame.f_code is target and event == 'line' and 'canonical_inputs' in frame.f_locals:
                inputs['input_ids'][0][1] = 996
            return mutate
        try:
            sys.settrace(mutate)
            self.assertIs(wrapper('r', inputs, 29), self.result)
        finally:
            sys.settrace(previous)
        self.assertEqual(inputs['input_ids'], [[15496, 996]])
        self.assertEqual(seen, [{'input_ids': [[15496, 995]]}])

    def test_noncallable_factory_is_rejected_at_construction(self):
        with self.assertRaises(TypeError):
            FixedServiceRequestFactory(None)

    def test_direct_ipc_rejects_fixed_case_before_allocation_then_runs_valid_peer(self):
        from looking_glass.ipc import IPCClient, IPCRejected, IPCServer, _default_request_factory
        from test_ipc_completion_authority import stop_owner
        from test_owner_completion import make_owner

        model, _, _, _, _, authority, _, broker, _ = make_owner()
        server = IPCServer(broker, team_ids=['team'], completion_authority=authority,
                           request_factory=FixedServiceRequestFactory(_default_request_factory))
        model.release.set()
        try:
            server.start()
            client = IPCClient(server.address, server.team_tokens['team'])
            with self.assertRaises(IPCRejected) as rejected:
                client.submit(team_id='team', request_id='bad', seed=29,
                              inputs={'input_ids': [[15496, 996]]})
            self.assertEqual(rejected.exception.code, 'invalid_request')
            self.assertFalse(authority.unsafe)
            self.assertFalse(server.custody_quarantined)
            result = client.submit(team_id='team', request_id='good', seed=29,
                                   inputs={'input_ids': [[15496, 995]]}).wait(5)
            self.assertEqual(result['status'], 'succeeded')
            self.assertFalse(authority.unsafe)
            self.assertFalse(broker.quarantined)
        finally:
            model.release.set()
            server.close(timeout=2)
            stop_owner(model, broker)

    def test_fair_ipc_retires_rejected_connection_before_valid_peer(self):
        import time
        from looking_glass.ipc import IPCClient, IPCRejected, IPCServer, _default_request_factory
        from test_ipc_completion_authority import make_fair_owner, root_server_for_fault_injection, stop_owner

        model, broker, authority, bridge = make_fair_owner()
        server = IPCServer(bridge, team_ids=['team'], custody_guardian=bridge,
                           custody_epoch='b' * 32, completion_authority=authority,
                           request_factory=FixedServiceRequestFactory(_default_request_factory))
        root_server_for_fault_injection(bridge, server)
        model.release.set()
        try:
            bridge.start()
            server.start()
            client = IPCClient(server.address, server.team_tokens['team'])
            with self.assertRaises(IPCRejected) as rejected:
                client.submit(team_id='team', request_id='bad', seed=29,
                              inputs={'input_ids': [[15496, 996]]})
            self.assertEqual(rejected.exception.code, 'invalid_request')
            deadline = time.monotonic() + 3
            while bridge.committed_bytes and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(bridge.committed_bytes, 0)
            self.assertFalse(authority.unsafe)
            self.assertFalse(bridge.quarantined)
            self.assertFalse(server.custody_quarantined)
            result = client.submit(team_id='team', request_id='good', seed=29,
                                   inputs={'input_ids': [[15496, 995]]}).wait(5)
            self.assertEqual(result['status'], 'succeeded')
        finally:
            model.release.set()
            server.close(timeout=2)
            bridge.close(timeout=2)
            stop_owner(model, broker)

    def test_downstream_clean_signal_quarantines_over_direct_ipc(self):
        from looking_glass.ipc import IPCClient, IPCRejected, IPCServer
        from test_ipc_completion_authority import stop_owner
        from test_owner_completion import make_owner

        model, _, _, _, _, authority, _, broker, _ = make_owner()
        retained = []
        def downstream(_request_id, inputs, _seed):
            retained.append(inputs)
            raise IPCInputRejected('cannot certify post-allocation cleanup')
        server = IPCServer(broker, team_ids=['team'], completion_authority=authority,
                           request_factory=FixedServiceRequestFactory(downstream))
        try:
            server.start()
            client = IPCClient(server.address, server.team_tokens['team'])
            with self.assertRaises(IPCRejected):
                client.submit(team_id='team', request_id='case', seed=29,
                              inputs={'input_ids': [[15496, 995]]})
            self.assertEqual(len(retained), 1)
            self.assertTrue(authority.unsafe)
            self.assertTrue(server.custody_quarantined)
        finally:
            model.release.set()
            server.close(timeout=2)
            stop_owner(model, broker)


if __name__ == '__main__':
    unittest.main()
