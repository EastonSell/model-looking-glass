"""CPU checks for polling timeouts and result-bearing interruption custody."""

import gc
import inspect
import sys
import weakref
from unittest.mock import patch

import pytest

from looking_glass.broker import BrokerShutdownError, BrokerUnavailable, RequestHandle
from looking_glass.ipc_fair_bridge import FairIPCBridge
from test_ipc_fair_bridge import eventually, make_stack, notify_closed, submit


def test_retained_normal_timeout_does_not_keep_a_later_result_alive():
    worker, broker, adapter, bridge = make_stack()
    worker.release.clear()
    key, proxy = submit(bridge, 1, 'A', 'timeout', 1)
    assert adapter.transfer_next() is not None
    assert worker.started.wait(3)
    handle = adapter.recover_receipt(proxy._record.ticket)
    saved_error = None
    try:
        try:
            proxy.wait(0)
        except TimeoutError as error:
            saved_error = error
        assert saved_error is not None and saved_error.__traceback__ is not None
        worker.release.set()
        assert handle.wait_for_retirement(3)
        logits = weakref.ref(handle._outcome.result.logits)
        handle = None
        notify_closed(bridge, key)
        proxy = None
        bridge.start()
        eventually(lambda: bridge.committed_bytes == 0 and adapter._target.receipt_committed_bytes == 0)
        gc.collect()
        assert logits() is None
        assert saved_error.__traceback__ is not None
    finally:
        worker.release.set()
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


@pytest.mark.parametrize('operation', ['wait', 'cancel'])
def test_timeout_shaped_interruption_retains_receipt_and_connection(operation):
    worker, broker, adapter, bridge = make_stack()
    _, proxy = submit(bridge, 1, 'A', 'interrupted', 1)
    assert adapter.transfer_next() is not None
    handle = adapter.recover_receipt(proxy._record.ticket)
    assert handle.wait_for_retirement(3)
    target = adapter._target
    original = TimeoutError('typed interruption during outcome inspection')

    def fail(*_args, **_kwargs):
        retained_outcome = handle._outcome
        assert retained_outcome is not None
        raise original

    try:
        dependency = (patch.object(handle, 'wait', side_effect=fail) if operation == 'wait'
                      else patch.object(bridge, '_target_locked', side_effect=fail))
        with dependency:
            with pytest.raises(BrokerUnavailable):
                getattr(proxy, operation)(0) if operation == 'wait' else proxy.cancel()
        assert original.__traceback__ is not None
        assert bridge.quarantined
        assert bridge.committed_bytes == 2_000_000
        assert target.receipt_committed_bytes == 128
        assert proxy._record.wait_failure is original or proxy._record.transition_failure is original
    finally:
        worker.release.set()
        assert not bridge.close(timeout=0)
        try:
            broker.close(timeout=3)
        except BrokerShutdownError:
            pass


def test_cancel_terminal_precheck_does_not_call_handle_wait():
    worker, broker, adapter, bridge = make_stack()
    key, proxy = submit(bridge, 1, 'A', 'complete', 1)
    assert adapter.transfer_next() is not None
    handle = adapter.recover_receipt(proxy._record.ticket)
    assert handle.wait_for_retirement(3)
    try:
        with patch.object(handle, 'wait', side_effect=AssertionError('must not poll wait')):
            assert proxy.cancel() is False
        assert not bridge.quarantined
        notify_closed(bridge, key)
        bridge.start()
        eventually(lambda: bridge.committed_bytes == 0 and adapter._target.receipt_committed_bytes == 0)
    finally:
        worker.release.set()
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


@pytest.mark.parametrize('boundary', ['terminal_wait', 'clear_pending_handle'])
def test_timeout_interruption_at_result_boundary_is_quarantined(boundary):
    worker, broker, adapter, bridge = make_stack()
    if boundary == 'clear_pending_handle':
        worker.release.clear()
    _, proxy = submit(bridge, 1, 'A', 'trace-interruption', 1)
    assert adapter.transfer_next() is not None
    assert worker.started.wait(3)
    handle = adapter.recover_receipt(proxy._record.ticket)
    if boundary == 'terminal_wait':
        assert handle.wait_for_retirement(3)
        function = RequestHandle.wait
        line = 'broker = self._broker'
    else:
        function = FairIPCBridge._wait_inner
        line = 'handle = None'
    lines, start = inspect.getsourcelines(function)
    target_line = start + next(i for i, source in enumerate(lines) if source.strip() == line)
    original = TimeoutError('injected at the result boundary')
    fired = []

    def trace(frame, event, _arg):
        if frame.f_code is function.__code__ and event == 'line' and frame.f_lineno == target_line:
            fired.append(True)
            raise original
        return trace

    try:
        sys.settrace(trace)
        try:
            with pytest.raises(BrokerUnavailable):
                proxy.wait(0)
        finally:
            sys.settrace(None)
        assert fired and original.__traceback__ is not None
        assert bridge.quarantined and proxy._record.wait_failure is original
        assert bridge.committed_bytes == 2_000_000
        assert adapter._target.receipt_committed_bytes == 128
    finally:
        sys.settrace(None)
        worker.release.set()
        assert not bridge.close(timeout=0)
        try:
            broker.close(timeout=3)
        except BrokerShutdownError:
            pass
