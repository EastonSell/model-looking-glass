"""Custody and lifecycle of the bridge's single server construction."""

import gc
import inspect
import sys
import threading
import traceback
import weakref
from unittest.mock import patch

import pytest
import torch

from looking_glass.broker import BrokerUnavailable
from looking_glass.host_budget import HostBudget
from looking_glass.ipc import IPCServer
from looking_glass.ipc_custody import IPCCustodyKey
from looking_glass.ipc_fair_bridge import FairIPCBridge
from test_ipc_fair_bridge import EPOCH, make_stack


@pytest.mark.parametrize('error_type', [MemoryError, KeyboardInterrupt, SystemExit])
def test_interrupted_server_construction_retains_factory_and_private_failure(error_type):
    _, broker, _, bridge = make_stack()
    budget = HostBudget(max_bytes=8, max_items=1, max_requests=1, max_operations=1)
    claim = budget.open_request('factory').reserve('owner', 'factory-data', 8)
    aliases, factories, temporaries, errors = [], [], [], []

    class Factory:
        def __init__(self):
            self.tensor = torch.ones(2)
            self.claim = claim
            aliases.append(self.tensor.detach())
            factories.append(weakref.ref(self))

        def __call__(self, *_args):
            raise AssertionError('factory must not execute during construction')

        def release(self):
            if self.claim is not None:
                self.claim.release()
                self.claim = None

        def __del__(self):
            self.release()

    def fail_token_generation(_nbytes):
        temporary = torch.ones(3)
        temporaries.append(weakref.ref(temporary))
        raise error_type('private construction error')

    try:
        try:
            with patch('looking_glass.ipc.secrets.token_hex', side_effect=fail_token_generation):
                bridge.make_server(request_factory=Factory())
        except BaseException as error:
            errors.append(error)
        assert len(errors) == 1
        public = errors[0]
        assert isinstance(public, BrokerUnavailable if error_type is MemoryError else error_type)
        assert public.__cause__ is None and public.__context__ is None
        traceback.clear_frames(public.__traceback__)
        public.__traceback__ = None
        gc.collect()

        assert bridge.quarantined
        assert bridge._server is None
        state = bridge._server_construction
        assert type(state.server) is IPCServer
        assert state.server.request_factory is factories[0]()
        assert isinstance(state.error, error_type) and state.error is not public
        assert factories[0]() is not None and temporaries[0]() is not None
        assert budget.snapshot().used_bytes == 8 and aliases[0].numel() == 2
        with pytest.raises(BrokerUnavailable):
            bridge.recover_server()
        assert not bridge.close(timeout=0)
    finally:
        aliases.clear()
        factory = factories[0]() if factories else None
        if factory is not None:
            factory.release()
        assert broker.close(timeout=3)


def test_server_construction_blocks_clean_close_but_not_dispatcher_status():
    _, broker, _, bridge = make_stack()
    entered, resume = threading.Event(), threading.Event()
    servers, errors = [], []
    original_init = IPCServer.__init__

    def paused_init(server, *args, **kwargs):
        entered.set()
        if not resume.wait(8):
            raise TimeoutError('test did not release server construction')
        return original_init(server, *args, **kwargs)

    def construct():
        try:
            servers.append(bridge.make_server())
        except BaseException as error:
            errors.append(error)

    builder = threading.Thread(target=construct, daemon=True)
    bridge.start()
    try:
        with patch.object(IPCServer, '__init__', new=paused_init):
            builder.start()
            assert entered.wait(3)
            assert bridge.dispatcher_alive and not bridge.quarantined
            assert bridge.recover_server() is None
            with pytest.raises(BrokerUnavailable):
                bridge.reserve(IPCCustodyKey(EPOCH, 1))
            assert bridge.committed_bytes == 0
            assert not bridge.close(timeout=0)
            assert bridge._accepting
            resume.set()
            builder.join(3)
        assert not builder.is_alive() and not errors and len(servers) == 1
        server = servers[0]
        assert bridge.recover_server() is server
        server.start()
        assert server.close(timeout=3)
    finally:
        resume.set()
        builder.join(3)
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_recorded_constructor_failure_stops_new_reservations_before_latch():
    _, broker, _, bridge = make_stack()
    entered, resume = threading.Event(), threading.Event()
    errors = []
    lines, first_line = inspect.getsourcelines(FairIPCBridge._make_server_guarded)
    target_line = first_line + next(i for i, line in enumerate(lines)
                                    if 'self._quarantined = True' in line)

    def pause_handler(frame, event, _argument):
        if (frame.f_code is FairIPCBridge._make_server_guarded.__code__ and
                event == 'line' and frame.f_lineno == target_line):
            entered.set()
            if not resume.wait(8):
                raise TimeoutError('test did not release failure handler')
        return pause_handler

    def construct():
        try:
            with patch('looking_glass.ipc.secrets.token_hex', side_effect=MemoryError('injected')):
                sys.settrace(pause_handler)
                bridge.make_server()
        except BaseException as error:
            errors.append(error)
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=construct, daemon=True)
    try:
        thread.start()
        assert entered.wait(3)
        assert isinstance(bridge._server_construction.error, MemoryError)
        assert bridge.quarantined
        with pytest.raises(BrokerUnavailable):
            bridge.reserve(IPCCustodyKey(EPOCH, 1))
        assert bridge.committed_bytes == 0
        resume.set()
        thread.join(3)
        assert not thread.is_alive() and len(errors) == 1
        assert isinstance(errors[0], BrokerUnavailable)
        assert not bridge.close(timeout=0)
    finally:
        resume.set()
        thread.join(3)
        assert broker.close(timeout=3)


def test_interruption_during_failure_handler_keeps_private_partial_allocation():
    _, broker, _, bridge = make_stack()
    budget = HostBudget(max_bytes=8, max_items=1, max_requests=1, max_operations=1)
    claim = budget.open_request('partial').reserve('owner', 'partial', 8)
    aliases, owners, errors = [], [], []
    lines, first_line = inspect.getsourcelines(FairIPCBridge._make_server_guarded)
    target_line = first_line + next(i for i, line in enumerate(lines)
                                    if 'self._quarantined = True' in line)

    class Owner:
        def __init__(self):
            self.claim = claim
            owners.append(weakref.ref(self))

        def release(self):
            if self.claim is not None:
                self.claim.release()
                self.claim = None

        def __del__(self):
            self.release()

    def fail_token_generation(_nbytes):
        temporary = torch.ones(2)
        temporary.owner = Owner()
        aliases.append(temporary.detach())
        raise MemoryError('private allocation failure')

    def trace_handler(frame, event, _argument):
        if (frame.f_code is FairIPCBridge._make_server_guarded.__code__ and
                event == 'line' and frame.f_lineno == target_line):
            raise KeyboardInterrupt('injected during failure handling')
        return trace_handler

    def construct():
        try:
            with patch('looking_glass.ipc.secrets.token_hex', side_effect=fail_token_generation):
                sys.settrace(trace_handler)
                bridge.make_server()
        except BaseException as error:
            errors.append(error)
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=construct, daemon=True)
    try:
        thread.start()
        thread.join(3)
        assert not thread.is_alive() and len(errors) == 1
        public = errors[0]
        assert isinstance(public, KeyboardInterrupt)
        assert public.__context__ is None and public.__cause__ is None
        traceback.clear_frames(public.__traceback__)
        public.__traceback__ = None
        gc.collect()
        assert bridge.quarantined and owners[0]() is not None
        assert budget.snapshot().used_bytes == 8 and aliases[0].numel() == 2
        assert isinstance(bridge._server_construction.error, MemoryError)
        assert isinstance(bridge._server_construction.guard_error, KeyboardInterrupt)
        assert not bridge.close(timeout=0)
    finally:
        aliases.clear()
        owner = owners[0]() if owners else None
        if owner is not None:
            owner.release()
        assert broker.close(timeout=3)


@pytest.mark.parametrize('fatal_type', [KeyboardInterrupt, SystemExit])
def test_fatal_constructor_error_survives_interrupted_error_assignment(fatal_type):
    _, broker, _, bridge = make_stack()
    errors = []
    lines, first_line = inspect.getsourcelines(FairIPCBridge._make_server_guarded)
    target_line = first_line + next(i for i, line in enumerate(lines)
                                    if 'state.error = error' in line)

    def fail_assignment(frame, event, _argument):
        if (frame.f_code is FairIPCBridge._make_server_guarded.__code__ and
                event == 'line' and frame.f_lineno == target_line):
            raise RuntimeError('injected failure in error handling')
        return fail_assignment

    def construct():
        try:
            with patch('looking_glass.ipc.secrets.token_hex', side_effect=fatal_type('injected')):
                sys.settrace(fail_assignment)
                bridge.make_server()
        except BaseException as error:
            errors.append(error)
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=construct, daemon=True)
    try:
        thread.start()
        thread.join(3)
        assert not thread.is_alive() and len(errors) == 1
        assert isinstance(errors[0], fatal_type)
        assert errors[0].__context__ is None and errors[0].__cause__ is None
        assert bridge.quarantined
        private = bridge._server_construction.error
        assert isinstance(private, RuntimeError) and isinstance(private.__context__, fatal_type)
        assert not bridge.close(timeout=0)
    finally:
        assert broker.close(timeout=3)


def test_interrupted_failure_classification_keeps_quarantine_and_error_private():
    _, broker, _, bridge = make_stack()
    errors = []
    try:
        try:
            with patch('looking_glass.ipc.secrets.token_hex', side_effect=MemoryError('private')), \
                    patch('looking_glass.ipc_fair_bridge._failure_identity',
                          side_effect=KeyboardInterrupt('classification')):
                bridge.make_server()
        except BaseException as error:
            errors.append(error)
        assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt)
        assert errors[0].__context__ is None and errors[0].__cause__ is None
        assert bridge.quarantined
        assert isinstance(bridge._server_construction.error, MemoryError)
        assert isinstance(bridge._server_construction.classification_error, KeyboardInterrupt)
        assert not bridge.close(timeout=0)
    finally:
        assert broker.close(timeout=3)


def test_interrupted_server_return_can_be_recovered_by_owner():
    _, broker, _, bridge = make_stack()
    errors = []

    def trace_return(frame, event, _argument):
        if frame.f_code is FairIPCBridge.make_server.__code__ and event == 'return':
            raise KeyboardInterrupt('injected at return')
        return trace_return

    def construct():
        try:
            sys.settrace(trace_return)
            bridge.make_server()
        except BaseException as error:
            errors.append(error)
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=construct, daemon=True)
    try:
        thread.start()
        thread.join(3)
        assert not thread.is_alive() and len(errors) == 1
        assert isinstance(errors[0], KeyboardInterrupt)
        assert not bridge.quarantined
        server = bridge.recover_server()
        assert type(server) is IPCServer
        server.start()
        assert server.close(timeout=3)
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


@pytest.mark.parametrize('target', [IPCServer.__init__.__code__, FairIPCBridge._make_server.__code__])
def test_interrupted_inner_return_quarantines_the_rooted_server(target):
    _, broker, _, bridge = make_stack()
    errors = []

    def trace_return(frame, event, _argument):
        if frame.f_code is target and event == 'return':
            raise KeyboardInterrupt('injected at inner return')
        return trace_return

    def construct():
        try:
            sys.settrace(trace_return)
            bridge.make_server()
        except BaseException as error:
            errors.append(error)
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=construct, daemon=True)
    try:
        thread.start()
        thread.join(3)
        assert not thread.is_alive() and len(errors) == 1
        public = errors[0]
        assert isinstance(public, KeyboardInterrupt)
        assert public.__cause__ is None and public.__context__ is None
        assert bridge.quarantined
        state = bridge._server_construction
        assert type(state.server) is IPCServer and state.server.broker is bridge
        assert state.server._listener is None and state.server._accept_thread is None
        assert isinstance(state.error, KeyboardInterrupt) and state.error is not public
        with pytest.raises(BrokerUnavailable):
            bridge.recover_server()
        assert not bridge.close(timeout=0)
    finally:
        assert broker.close(timeout=3)


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf'), True, 10**1000])
def test_invalid_server_timeout_does_not_begin_construction(timeout):
    _, broker, _, bridge = make_stack()
    try:
        with pytest.raises(ValueError):
            bridge.make_server(io_timeout=timeout)
        assert not bridge.quarantined and bridge._server_construction is None
        server = bridge.make_server()
        server.start()
        assert server.close(timeout=3)
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)
