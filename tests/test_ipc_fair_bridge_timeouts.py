"""Timeout validation at the trusted fair bridge boundary."""

import threading

import pytest

from looking_glass.broker import RequestState
from test_ipc_fair_bridge import eventually, make_stack, notify_closed, submit


def test_invalid_wait_and_close_timeouts_do_not_change_custody():
    _, broker, adapter, bridge = make_stack()
    key, proxy = submit(bridge, 1, 'A', 'timeout', 1)
    invalid = (float('nan'), float('inf'), float('-inf'), -1, True, '3',
               10**10000, threading.TIMEOUT_MAX * 2)
    try:
        for timeout in invalid:
            with pytest.raises(ValueError):
                proxy.wait(timeout)
            with pytest.raises(ValueError):
                bridge.close(timeout)
            assert not proxy.cancel_requested
            assert len(adapter) == 1
            assert bridge.committed_bytes == 2_000_000
            assert bridge._accepting and not bridge.quarantined
        with pytest.raises(ValueError):
            bridge.close(None)
        with pytest.raises(TimeoutError):
            proxy.wait(0)
        assert proxy.cancel()
        assert proxy.wait(0).state is RequestState.CANCELLED
        notify_closed(bridge, key)
        bridge.start()
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_none_wait_and_runtime_maximum_are_accepted():
    _, broker, _, bridge = make_stack()
    key, proxy = submit(bridge, 1, 'A', 'timeout', 1)
    bridge.start()
    try:
        assert proxy.wait(None).state is RequestState.SUCCEEDED
        assert proxy.wait(threading.TIMEOUT_MAX).state is RequestState.SUCCEEDED
        notify_closed(bridge, key)
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        assert bridge.close(timeout=threading.TIMEOUT_MAX)
        assert broker.close(timeout=3)


def test_dispatcher_liveness_does_not_wait_for_bridge_condition():
    _, broker, _, bridge = make_stack()
    seen, finished = [], threading.Event()
    reader = None
    try:
        assert bridge.dispatcher_alive is False
        bridge.start()
        assert bridge.dispatcher_alive is True

        def observe():
            seen.append(bridge.dispatcher_alive)
            finished.set()

        with bridge._condition:
            reader = threading.Thread(target=observe, daemon=True)
            reader.start()
            assert finished.wait(2), 'liveness observation waited for the bridge condition'
            assert seen == [True]
        reader.join(2)
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)
    assert bridge.dispatcher_alive is False
