import threading
import time
import unittest
import inspect
import dis
import io
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

try:
    from looking_glass.service import OwnerStatus, make_gpt2_request_factory, serve_cpu_model, serve_owner
except ModuleNotFoundError as exc:
    if exc.name != 'looking_glass.service':
        raise
    serve_owner = None


class TestServer:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return self

    def close(self, timeout=0):
        self.stopped = True
        return True


class TestBroker:
    def __init__(self):
        self.available = True
        self.failed = False
        self.quarantined = False
        self.waited = None

    def close(self, *, wait=True, timeout=None):
        self.waited = wait
        self.available = False
        return True


class TestWorker:
    def __init__(self):
        self.poisoned = False
        self.quarantined = False

    @property
    def failed(self):
        return self.poisoned or self.quarantined


class OwnerTests(unittest.TestCase):
    def test_trusted_health_provider_avoids_blocking_legacy_properties(self):
        from looking_glass import service

        class LegacyBroker:
            closed = False

            @property
            def available(self):
                raise AssertionError('condition-locked broker availability was read')

            @property
            def failed(self):
                raise AssertionError('condition-locked broker failure was read')

            @property
            def quarantined(self):
                raise AssertionError('legacy broker quarantine was read')

            def close(self, *, timeout=None):
                self.closed = True
                return True

        class LegacyWorker:
            @property
            def quarantined(self):
                raise AssertionError('legacy worker quarantine was read')

            @property
            def poisoned(self):
                raise AssertionError('legacy worker availability was read')

            @property
            def failed(self):
                raise AssertionError('legacy worker failure was read')

        broker, worker, stop, faults = LegacyBroker(), LegacyWorker(), threading.Event(), []

        def health():
            return service.OwnerHealth(False, False, False, False, broker.closed, not broker.closed)

        result = serve_owner(TestServer(), broker, worker, stop, announce=lambda _: stop.set(),
                             verify_drain=lambda: True, health_provider=health,
                             on_health_uncertain=lambda: faults.append(True),
                             poll_interval=.001, shutdown_timeout=.5)
        self.assertEqual(result.status, OwnerStatus.CLEAN)
        self.assertTrue(result.teardown_verified)
        self.assertEqual(faults, [])

    def test_real_broker_health_remains_observable_with_its_condition_held(self):
        from looking_glass import service
        from looking_glass.broker import Broker

        class Cancelled(Exception):
            pass
        class Worker:
            cancellation_exception = Cancelled
            poisoned = False
            failed = False
            quarantined = False
            def close(self):
                pass

        broker, worker = Broker(Worker(), max_pending=1, max_pending_per_team=1), TestWorker()
        stop, ready, unsafe = threading.Event(), threading.Event(), threading.Event()
        notices, results = [], []

        def health():
            value = broker.health_snapshot
            return service.OwnerHealth(unsafe.is_set() or value.quarantined, False, False,
                                       value.failed, value.closed, value.owner_alive)

        def owner():
            results.append(serve_owner(TestServer(), broker, worker, stop,
                announce=lambda _: ready.set(), verify_drain=lambda: True,
                health_provider=health, on_health_uncertain=lambda: notices.append(True),
                poll_interval=.001, shutdown_timeout=.5))

        thread = threading.Thread(target=owner, daemon=True)
        try:
            with broker._condition:
                thread.start()
                self.assertTrue(ready.wait(2))
                unsafe.set()
                thread.join(.5)
                self.assertFalse(thread.is_alive(), 'health observation waited on the broker condition')
                self.assertEqual(results[0].status, OwnerStatus.QUARANTINED)
                self.assertFalse(results[0].teardown_verified)
                self.assertTrue(notices)
        finally:
            thread.join(2)
            broker.close(timeout=2)

    def test_health_provider_and_uncertainty_callback_are_both_required(self):
        server = TestServer()
        for provider, notify in ((lambda: None, None), (None, lambda: None), (1, lambda: None),
                                 (lambda: None, 1)):
            with self.subTest(provider=provider, notify=notify), self.assertRaises(TypeError):
                serve_owner(server, TestBroker(), TestWorker(), threading.Event(),
                            announce=lambda _: None, verify_drain=lambda: True,
                            health_provider=provider, on_health_uncertain=notify)
        self.assertFalse(server.started)

    def test_malformed_or_raising_health_provider_notifies_and_fails_closed(self):
        from looking_glass import service
        malformed = lambda: service.OwnerHealth(1, False, False, False, False, True)
        def raises():
            raise RuntimeError('injected observation failure')
        for provider in (lambda: object(), malformed, raises):
            with self.subTest(provider=provider):
                stop, notices, drains = threading.Event(), [], []
                result = serve_owner(TestServer(), TestBroker(), TestWorker(), stop,
                                     announce=lambda _: stop.set(),
                                     verify_drain=lambda: drains.append(True) or True,
                                     health_provider=provider, on_health_uncertain=lambda: notices.append(True),
                                     poll_interval=.001, shutdown_timeout=.5)
                self.assertEqual(result.status, OwnerStatus.QUARANTINED)
                self.assertFalse(result.teardown_verified)
                self.assertTrue(notices)
                self.assertEqual(drains, [])

    def test_interrupted_health_notification_never_certifies_teardown(self):
        stop, drains = threading.Event(), []
        def interrupted():
            raise KeyboardInterrupt('injected notification interruption')
        with self.assertRaises(KeyboardInterrupt):
            serve_owner(TestServer(), TestBroker(), TestWorker(), stop,
                        announce=lambda _: stop.set(),
                        verify_drain=lambda: drains.append(True) or True,
                        health_provider=lambda: object(), on_health_uncertain=interrupted,
                        poll_interval=.001, shutdown_timeout=.5)
        self.assertEqual(drains, [])

    def test_one_shot_interruption_at_uncertainty_callback_entry_still_marks(self):
        stop, marked, fired = threading.Event(), threading.Event(), []
        def notify():
            marked.set()
        def trace(frame, event, _arg):
            if frame.f_code is notify.__code__ and event == 'call' and not fired:
                fired.append(True)
                raise KeyboardInterrupt('injected at uncertainty callback entry')
            return trace
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            result = serve_owner(TestServer(), TestBroker(), TestWorker(), stop,
                announce=lambda _: None, verify_drain=lambda: True,
                health_provider=lambda: object(), on_health_uncertain=notify,
                poll_interval=.001, shutdown_timeout=.5)
        finally:
            sys.settrace(previous)
        self.assertEqual(fired, [True])
        self.assertTrue(marked.is_set())
        self.assertEqual(result.status, OwnerStatus.QUARANTINED)
        self.assertFalse(result.teardown_verified)

    def test_interruption_before_observation_failure_notification_still_marks(self):
        from looking_glass import service
        target_function = getattr(service, '_serve_owner_impl', service.serve_owner)
        lines, first = inspect.getsourcelines(target_function)
        start = next(i for i, line in enumerate(lines) if line.strip() == 'def observation():')
        target = first + next(i for i, line in enumerate(lines) if i > start and (
            'on_health_uncertain()' in line or '_notify_health_uncertainty(on_health_uncertain)' in line))
        marked, fired = threading.Event(), []

        def provider():
            raise RuntimeError('injected health failure')

        def trace(frame, event, _arg):
            if (frame.f_code.co_name == 'observation' and event == 'line'
                    and frame.f_lineno == target and not fired):
                fired.append(True)
                raise KeyboardInterrupt('injected before uncertainty notification')
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            result = serve_owner(TestServer(), TestBroker(), TestWorker(), threading.Event(),
                announce=lambda _: None, verify_drain=lambda: True,
                health_provider=provider, on_health_uncertain=marked.set,
                poll_interval=.001, shutdown_timeout=.5)
        finally:
            sys.settrace(previous)
        self.assertEqual(fired, [True])
        self.assertTrue(marked.is_set())
        self.assertEqual(result.status, OwnerStatus.QUARANTINED)
        self.assertFalse(result.teardown_verified)

    def test_interruption_at_healthy_observation_return_still_marks(self):
        from looking_glass import service
        marked, fired = threading.Event(), []

        def trace(frame, event, _arg):
            if frame.f_code.co_name == 'observation' and event == 'return' and not fired:
                fired.append(True)
                raise KeyboardInterrupt('injected at health observation return')
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            result = serve_owner(TestServer(), TestBroker(), TestWorker(), threading.Event(),
                announce=lambda _: None, verify_drain=lambda: True,
                health_provider=lambda: service.OwnerHealth(False, False, False, False, False, True),
                on_health_uncertain=marked.set, poll_interval=.001, shutdown_timeout=.5)
        finally:
            sys.settrace(previous)
        self.assertEqual(fired, [True])
        self.assertTrue(marked.is_set())
        self.assertEqual(result.status, OwnerStatus.QUARANTINED)
        self.assertFalse(result.teardown_verified)

    def test_startup_return_failure_marks_before_delegating_blocked_thread_hook(self):
        from looking_glass import service
        marked, hook_entered, release, ready = (threading.Event() for _ in range(4))
        fired, results, offsets = [], [], {}
        previous_trace, previous_hook = threading.gettrace(), threading.excepthook

        def trace(frame, event, _arg):
            if frame.f_code.co_name == 'startup':
                if event == 'call':
                    frame.f_trace_opcodes = True
                    offsets[frame.f_code] = {item.offset for item in dis.get_instructions(frame.f_code)
                                             if item.opname == 'RETURN_VALUE'}
                elif (event == 'opcode' and frame.f_lasti in offsets.get(frame.f_code, ())
                      and frame.f_locals.get('startup_terminal', [False])[0] and not fired):
                    fired.append(True)
                    raise KeyboardInterrupt('injected startup target return')
            return trace

        def blocked_hook(_args):
            hook_entered.set()
            release.wait(3)

        def owner():
            results.append(serve_owner(TestServer(), TestBroker(), TestWorker(), threading.Event(),
                announce=lambda _: ready.set(), verify_drain=lambda: True,
                health_provider=lambda: service.OwnerHealth(False, False, False, False, False, True),
                on_health_uncertain=marked.set, poll_interval=.001, shutdown_timeout=1))

        thread = threading.Thread(target=owner, daemon=True)
        try:
            threading.excepthook = blocked_hook
            threading.settrace(trace)
            thread.start()
            self.assertTrue(ready.wait(2))
            self.assertTrue(hook_entered.wait(2))
            self.assertTrue(marked.is_set(), 'authority was not marked before the original hook')
        finally:
            release.set()
            thread.join(3)
            threading.settrace(previous_trace)
            threading.excepthook = previous_hook
        self.assertEqual(fired, [True])
        self.assertFalse(thread.is_alive())
        self.assertFalse(results[0].teardown_verified)

    def test_unexpected_closed_or_dead_broker_marks_uncertainty_before_drain(self):
        from looking_glass import service
        for closed, alive in ((True, True), (False, False)):
            with self.subTest(closed=closed, alive=alive):
                stop, notices, drains = threading.Event(), [], []
                result = serve_owner(TestServer(), TestBroker(), TestWorker(), stop,
                                     announce=lambda _: None,
                                     verify_drain=lambda: drains.append(True) or True,
                                     health_provider=lambda: service.OwnerHealth(
                                         False, False, False, False, closed, alive),
                                     on_health_uncertain=lambda: notices.append(True),
                                     poll_interval=.001, shutdown_timeout=.5)
                self.assertEqual(result.status, OwnerStatus.QUARANTINED)
                self.assertFalse(result.teardown_verified)
                self.assertTrue(notices)
                self.assertEqual(drains, [])

    def test_quarantine_observed_during_drain_marks_uncertainty(self):
        from looking_glass import service
        stop, marked, notices = threading.Event(), [False], []
        class ClosingBroker(TestBroker):
            closed = False
            def close(self, *, timeout=None):
                self.closed = True
                return True
        broker = ClosingBroker()
        def drain():
            marked[0] = True
            return True
        result = serve_owner(TestServer(), broker, TestWorker(), stop,
                             announce=lambda _: stop.set(), verify_drain=drain,
                             health_provider=lambda: service.OwnerHealth(
                                 marked[0], False, False, False, broker.closed, not broker.closed),
                             on_health_uncertain=lambda: notices.append(True),
                             poll_interval=.001, shutdown_timeout=.5)
        self.assertEqual(result.status, OwnerStatus.QUARANTINED)
        self.assertFalse(result.teardown_verified)
        self.assertTrue(marked[0])
        self.assertTrue(notices)

    def test_broker_death_before_its_shutdown_call_cannot_be_healed(self):
        from looking_glass import service
        entered, release, dead, stop = (threading.Event() for _ in range(4))
        notices, drains, results = [], [], []

        class SlowServer(TestServer):
            def close(self, timeout=0):
                entered.set()
                release.wait(3)
                return True

        class ClosingBroker(TestBroker):
            closed = False
            def close(self, *, timeout=None):
                self.closed = True
                return True

        broker = ClosingBroker()
        def owner():
            results.append(serve_owner(SlowServer(), broker, TestWorker(), stop,
                announce=lambda _: stop.set(), verify_drain=lambda: drains.append(True) or True,
                health_provider=lambda: service.OwnerHealth(False, False, False, False,
                                                            broker.closed, not dead.is_set()),
                on_health_uncertain=lambda: notices.append(True),
                poll_interval=.001, shutdown_timeout=1))
        thread = threading.Thread(target=owner, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            dead.set()
            # Allow the owner to observe the unexpected death before cleanup
            # returns; the teardown result must never heal that observation.
            thread.join(.2)
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].status, OwnerStatus.QUARANTINED)
        self.assertFalse(results[0].teardown_verified)
        self.assertTrue(notices)
        self.assertEqual(drains, [])

    def test_uncertain_shutdown_notifies_authority_before_later_cleanup(self):
        from looking_glass import service
        for failure in ('server_false', 'server_raises', 'broker_false', 'drain_false'):
            with self.subTest(failure=failure):
                events, stop = [], threading.Event()

                class FaultyServer(TestServer):
                    def close(self, timeout=0):
                        events.append('server_close')
                        if failure == 'server_raises':
                            raise RuntimeError('injected server close failure')
                        return failure != 'server_false'

                class FaultyBroker(TestBroker):
                    closed = False
                    def close(self, *, timeout=None):
                        events.append('broker_close')
                        self.closed = failure != 'broker_false'
                        return self.closed

                def drain():
                    events.append('drain')
                    return failure != 'drain_false'

                broker = FaultyBroker()
                result = serve_owner(FaultyServer(), broker, TestWorker(), stop,
                    announce=lambda _: stop.set(), verify_drain=drain,
                    health_provider=lambda: service.OwnerHealth(False, False, False, False,
                                                                broker.closed, not broker.closed),
                    on_health_uncertain=lambda: events.append('unsafe'),
                    poll_interval=.001, shutdown_timeout=.5)
                self.assertFalse(result.teardown_verified)
                self.assertIn('unsafe', events)
                if failure in ('server_false', 'server_raises'):
                    self.assertNotIn('broker_close', events)
                if failure != 'drain_false':
                    self.assertNotIn('drain', events)

    def test_failed_opt_in_startup_notifies_and_never_attempts_verified_drain(self):
        from looking_glass import service
        events, stop = [], threading.Event()
        class FaultyServer(TestServer):
            def start(self):
                events.append('startup')
                raise RuntimeError('injected startup failure')
            def close(self, timeout=0):
                events.append('close')
                return True
        result = serve_owner(FaultyServer(), TestBroker(), TestWorker(), stop,
            announce=lambda _: None, verify_drain=lambda: events.append('drain') or True,
            health_provider=lambda: service.OwnerHealth(False, False, False, False, False, True),
            on_health_uncertain=lambda: events.append('unsafe'),
            poll_interval=.001, shutdown_timeout=.5)
        self.assertFalse(result.teardown_verified)
        self.assertEqual(events[0], 'startup')
        self.assertTrue(events[1:])
        self.assertTrue(all(event == 'unsafe' for event in events[1:]))

    def test_truthy_non_boolean_broker_close_cannot_certify_opt_in_shutdown(self):
        from looking_glass import service
        events, stop = [], threading.Event()
        class Broker(TestBroker):
            def close(self, *, timeout=None):
                events.append('broker_close')
                return 1
        result = serve_owner(TestServer(), Broker(), TestWorker(), stop,
            announce=lambda _: stop.set(), verify_drain=lambda: events.append('drain') or True,
            health_provider=lambda: service.OwnerHealth(False, False, False, False, False, True),
            on_health_uncertain=lambda: events.append('unsafe'),
            poll_interval=.001, shutdown_timeout=.5)
        self.assertFalse(result.teardown_verified)
        self.assertIn('unsafe', events)
        self.assertNotIn('drain', events)

    def test_successful_close_with_live_broker_observation_cannot_certify_shutdown(self):
        from looking_glass import service
        events, stop = [], threading.Event()
        result = serve_owner(TestServer(), TestBroker(), TestWorker(), stop,
            announce=lambda _: stop.set(), verify_drain=lambda: events.append('drain') or True,
            health_provider=lambda: service.OwnerHealth(False, False, False, False, False, True),
            on_health_uncertain=lambda: events.append('unsafe'),
            poll_interval=.001, shutdown_timeout=.5)
        self.assertFalse(result.teardown_verified)
        self.assertIn('unsafe', events)
        self.assertNotIn('drain', events)

    def test_opt_in_quarantine_does_not_start_new_cleanup_threads(self):
        from looking_glass import service
        events, stop = [], threading.Event()
        class Server(TestServer):
            def close(self, timeout=0):
                events.append('server_close')
                return True
        class Broker(TestBroker):
            def close(self, *, wait=True, timeout=None):
                events.append('broker_close')
                return True
        result = serve_owner(Server(), Broker(), TestWorker(), stop,
            announce=lambda _: stop.set(), verify_drain=lambda: events.append('drain') or True,
            health_provider=lambda: service.OwnerHealth(True, False, False, False, False, True),
            on_health_uncertain=lambda: events.append('unsafe'),
            poll_interval=.001, shutdown_timeout=.5)
        self.assertEqual(result.status, OwnerStatus.QUARANTINED)
        self.assertFalse(result.teardown_verified)
        self.assertTrue(events)
        self.assertTrue(all(event == 'unsafe' for event in events))

    def test_preexisting_unsafe_health_rejects_before_listener_start(self):
        from looking_glass import service
        server, stop, events = TestServer(), threading.Event(), []
        result = serve_owner(server, TestBroker(), TestWorker(), stop,
            announce=lambda _: events.append('announce'), verify_drain=lambda: events.append('drain') or True,
            health_provider=lambda: service.OwnerHealth(True, False, False, False, False, True),
            on_health_uncertain=lambda: events.append('unsafe'),
            poll_interval=.001, shutdown_timeout=.5)
        self.assertEqual(result.status, OwnerStatus.QUARANTINED)
        self.assertFalse(result.teardown_verified)
        self.assertFalse(server.started)
        self.assertTrue(events)
        self.assertTrue(all(event == 'unsafe' for event in events))

    def _interrupted_terminal_return(self, name, marker):
        interrupted, offsets = [], {}
        previous = threading.gettrace()

        def trace(frame, event, _arg):
            if frame.f_code.co_name == name:
                if event == 'call':
                    frame.f_trace_opcodes = True
                    offsets[frame.f_code] = {instruction.offset for instruction in dis.get_instructions(frame.f_code)
                                             if instruction.opname == 'RETURN_VALUE'}
                elif (event == 'opcode' and frame.f_lasti in offsets.get(frame.f_code, ())
                      and frame.f_locals.get(marker, [False])[0] and not interrupted):
                    interrupted.append(True)
                    raise KeyboardInterrupt('injected terminal return interruption')
            return trace

        stop = threading.Event()
        try:
            threading.settrace(trace)
            with patch('threading.excepthook', lambda _args: None):
                result = serve_owner(TestServer(), TestBroker(), TestWorker(), stop,
                                     announce=lambda _: stop.set(), verify_drain=lambda: True,
                                     poll_interval=.001, shutdown_timeout=.5)
        finally:
            threading.settrace(previous)
        self.assertEqual(interrupted, [True])
        self.assertFalse(result.teardown_verified)

    def test_startup_return_interruption_cannot_certify_clean_drain(self):
        self._interrupted_terminal_return('startup', 'startup_terminal')

    def test_shutdown_return_interruption_cannot_certify_clean_drain(self):
        self._interrupted_terminal_return('finish', 'finish_terminal')

    def test_interrupted_startup_failure_handler_cannot_certify_clean_drain(self):
        from looking_glass import service

        class FailingServer(TestServer):
            def start(self):
                raise RuntimeError('injected startup failure')

        lines, start = inspect.getsourcelines(service._serve_owner_impl)
        startup_line = next(i for i, line in enumerate(lines) if line.strip() == 'def startup():')
        target = start + next(i for i, line in enumerate(lines)
                              if i > startup_line and line.strip() == 'except BaseException as exc:') + 1
        interrupted = []
        previous = threading.gettrace()

        def trace(frame, event, _arg):
            if (event == 'line' and frame.f_code.co_name == 'startup'
                    and frame.f_lineno == target and not interrupted):
                interrupted.append(True)
                raise KeyboardInterrupt('injected failure-handler interruption')
            return trace

        stop, drains = threading.Event(), []
        stop.set()
        try:
            threading.settrace(trace)
            with patch('threading.excepthook', lambda _args: None):
                result = serve_owner(FailingServer(), TestBroker(), TestWorker(), stop,
                                     announce=lambda _: None,
                                     verify_drain=lambda: drains.append(True) or True,
                                     poll_interval=.001, shutdown_timeout=.5)
        finally:
            threading.settrace(previous)
        self.assertEqual(interrupted, [True])
        self.assertFalse(result.teardown_verified)
        self.assertEqual(result.status, OwnerStatus.SHUTDOWN_INCOMPLETE)
        self.assertEqual(drains, [])

    def test_startup_signal_failure_after_set_is_not_a_clean_drain(self):
        class FailedSignal(threading.Event):
            def set(self):
                super().set()
                if threading.current_thread().name == 'looking-glass-owner-startup':
                    raise RuntimeError('injected startup signal failure')

        stop, drains = threading.Event(), []
        with patch('looking_glass.service.Event', FailedSignal), \
                patch('threading.excepthook', lambda _args: None):
            result = serve_owner(TestServer(), TestBroker(), TestWorker(), stop,
                                 announce=lambda _: stop.set(),
                                 verify_drain=lambda: drains.append(True) or True,
                                 poll_interval=.001, shutdown_timeout=.5)
        self.assertFalse(result.teardown_verified)
        self.assertEqual(result.status, OwnerStatus.SHUTDOWN_INCOMPLETE)
        self.assertEqual(drains, [])

    def test_shutdown_signal_failure_after_set_is_not_a_clean_drain(self):
        class FailedSignal(threading.Event):
            def set(self):
                super().set()
                if threading.current_thread().name == 'looking-glass-owner-shutdown':
                    raise RuntimeError('injected shutdown signal failure')

        stop = threading.Event()
        with patch('looking_glass.service.Event', FailedSignal), \
                patch('threading.excepthook', lambda _args: None):
            result = serve_owner(TestServer(), TestBroker(), TestWorker(), stop,
                                 announce=lambda _: stop.set(), verify_drain=lambda: True,
                                 poll_interval=.001, shutdown_timeout=.5)
        self.assertFalse(result.teardown_verified)
        self.assertEqual(result.status, OwnerStatus.SHUTDOWN_INCOMPLETE)

    def test_unexited_startup_thread_cannot_certify_a_clean_drain(self):
        reached, release = threading.Event(), threading.Event()
        blocked_threads, drains = [], []

        class CompletionGate(threading.Event):
            def set(self):
                super().set()
                if threading.current_thread().name == 'looking-glass-owner-startup':
                    blocked_threads.append(threading.current_thread())
                    reached.set()
                    release.wait(5)

        stop = threading.Event()
        stop.set()
        try:
            with patch('looking_glass.service.Event', CompletionGate):
                result = serve_owner(TestServer(), TestBroker(), TestWorker(), stop,
                                     announce=lambda _: None, verify_drain=lambda: drains.append(True) or True,
                                     poll_interval=.001, shutdown_timeout=.15)
            self.assertTrue(reached.is_set())
            self.assertFalse(result.teardown_verified)
            self.assertEqual(result.status, OwnerStatus.SHUTDOWN_INCOMPLETE)
            self.assertEqual(drains, [])
        finally:
            release.set()
            for thread in blocked_threads:
                thread.join(5)

    def test_unexited_shutdown_thread_cannot_certify_a_clean_drain(self):
        reached, release = threading.Event(), threading.Event()
        blocked_threads = []

        class CompletionGate(threading.Event):
            def set(self):
                super().set()
                if threading.current_thread().name == 'looking-glass-owner-shutdown':
                    blocked_threads.append(threading.current_thread())
                    reached.set()
                    release.wait(5)

        stop = threading.Event()
        stop.set()
        try:
            with patch('looking_glass.service.Event', CompletionGate):
                result = serve_owner(TestServer(), TestBroker(), TestWorker(), stop,
                                     announce=lambda _: None, verify_drain=lambda: True,
                                     poll_interval=.001, shutdown_timeout=.15)
            self.assertTrue(reached.is_set())
            self.assertFalse(result.teardown_verified)
            self.assertEqual(result.status, OwnerStatus.SHUTDOWN_INCOMPLETE)
        finally:
            release.set()
            for thread in blocked_threads:
                thread.join(5)

    def test_broker_quarantine_takes_precedence_over_verified_shutdown(self):
        broker = TestBroker()
        broker.quarantined = True
        drain = []
        stop = threading.Event()
        stop.set()
        result = serve_owner(TestServer(), broker, TestWorker(), stop,
                             announce=lambda _: None, verify_drain=lambda: drain.append(True) or True,
                             poll_interval=0.001, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.QUARANTINED)
        self.assertFalse(result.teardown_verified)
        self.assertEqual(drain, [])

    def test_cpu_cli_announces_private_endpoint_and_handles_shutdown(self):
        from looking_glass.cpu_service import main
        import torch
        from torch import nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(vocab_size=8, n_positions=4)

            def forward(self, input_ids):
                return input_ids.to(torch.float32).unsqueeze(-1) + torch.tensor([0.0, 1.0])

        bundle = SimpleNamespace(model=TinyModel().eval(), metadata={'model_id': 'tiny',
                                                                      'model_revision': 'revision'})
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('looking_glass.cpu_service.load_distilgpt2', return_value=bundle), \
                patch('sys.stdin', io.StringIO('shutdown\n')), \
                patch('sys.stdout', stdout), patch('sys.stderr', stderr):
            code = main(['--cache-dir', 'unused', '--team', 'alpha'])
        self.assertEqual(code, 0, stderr.getvalue())
        endpoint = json.loads(stdout.getvalue())
        self.assertEqual(endpoint['address'][0], '127.0.0.1')
        self.assertEqual(set(endpoint['team_tokens']), {'alpha'})

    def test_cpu_cli_rejects_unsupported_length_before_loading_checkpoint(self):
        from looking_glass.cpu_service import main
        with patch('looking_glass.cpu_service.load_distilgpt2') as load, \
                patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['--cache-dir', 'unused', '--team', 'alpha', '--max-tokens', '3'])
        load.assert_not_called()

    def test_real_cpu_service_serves_one_model_and_stops_cleanly(self):
        import torch
        from torch import nn
        from looking_glass.ipc import IPCClient

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(vocab_size=8, n_positions=4)

            def forward(self, input_ids):
                return input_ids.to(torch.float32).unsqueeze(-1) + torch.tensor([0.0, 1.0])

        stop, ready = threading.Event(), threading.Event()
        server_box, results = [], []

        def announce(server):
            server_box.append(server)
            ready.set()

        owner = threading.Thread(target=lambda: results.append(serve_cpu_model(
            TinyModel().eval(), model_id='tiny-test', team_ids=('alpha', 'beta'),
            stop_event=stop, announce=announce, max_tokens=2)), daemon=True)
        owner.start()
        try:
            self.assertTrue(ready.wait(5))
            server = server_box[0]
            a = IPCClient(server.address, server.team_tokens['alpha']).submit(
                team_id='alpha', request_id='same', inputs={'input_ids': [[1]]})
            b = IPCClient(server.address, server.team_tokens['beta']).submit(
                team_id='beta', request_id='same', inputs={'input_ids': [[2]]})
            self.assertEqual(a.wait(5)['result']['logits'], [1.0, 2.0])
            self.assertEqual(b.wait(5)['result']['logits'], [2.0, 3.0])
        finally:
            stop.set()
            owner.join(5)
        self.assertFalse(owner.is_alive())
        self.assertEqual(results[0].status, OwnerStatus.CLEAN)
        self.assertTrue(results[0].teardown_verified)

    def test_separate_client_receives_ordinary_worker_oom_before_owner_stops(self):
        import torch
        from torch import nn
        from looking_glass.broker import Broker
        from looking_glass.ipc import IPCServer
        from looking_glass.resident_worker import ResidentWorker
        from looking_glass.service import _CpuServiceWorker

        entered, release, ready, stop = (threading.Event() for _ in range(4))

        class GatedOomModel(nn.Module):
            def forward(self, input_ids):
                entered.set()
                if not release.wait(10):
                    raise RuntimeError('test gate timed out')
                raise torch.OutOfMemoryError('simulated CPU OOM')

        resident = ResidentWorker('cpu-failure-test', GatedOomModel().eval(), device='cpu')
        worker = _CpuServiceWorker(resident)
        broker = Broker(worker, max_pending=4, max_pending_per_team=2)
        server = IPCServer(broker, team_ids=['alpha'], job_timeout=10)
        owner_results = []
        owner = threading.Thread(target=lambda: owner_results.append(serve_owner(
            server, broker, worker, stop, announce=lambda _: ready.set(),
            verify_drain=lambda: True, poll_interval=0.001, shutdown_timeout=5)), daemon=True)
        child = None
        owner.start()
        try:
            self.assertTrue(ready.wait(5))
            client_code = '''
import json, sys
from looking_glass.ipc import IPCClient
spec = json.loads(sys.stdin.readline())
handle = IPCClient(tuple(spec['address']), spec['token']).submit(
    team_id='alpha', request_id='oom', inputs={'input_ids': [[1]]})
print(json.dumps(handle.wait(timeout=10)))
'''
            child = subprocess.Popen([sys.executable, '-c', client_code], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            child.stdin.write(json.dumps({'address': server.address,
                                          'token': server.team_tokens['alpha']}) + '\n')
            child.stdin.close()
            child.stdin = None
            self.assertTrue(entered.wait(10))
            release.set()
            stdout, stderr = child.communicate(timeout=15)
            self.assertEqual(child.returncode, 0, stderr)
            outcome = json.loads(stdout)
            self.assertEqual(outcome['status'], 'failed')
            self.assertEqual(outcome['error'], 'WorkerOutOfMemory')
            self.assertEqual(outcome['team_id'], 'alpha')
            self.assertEqual(outcome['request_id'], 'oom')
            owner.join(5)
            self.assertFalse(owner.is_alive())
            self.assertEqual(owner_results[0].status, OwnerStatus.WORKER_FAILED)
            self.assertTrue(owner_results[0].teardown_verified)
        finally:
            release.set()
            stop.set()
            if child is not None and child.poll() is None:
                child.kill()
                child.communicate(timeout=5)
            owner.join(5)
            server.close(timeout=5)
            broker.close(timeout=5)

    def test_model_specific_request_factory_rejects_out_of_range_ids(self):
        factory = make_gpt2_request_factory(vocab_size=5, max_positions=4,
                                            max_batch=1, max_tokens=4)
        request = factory('safe', {'input_ids': [[0, 4]], 'attention_mask': [[1, 1]]}, 0)
        self.assertEqual(request.inputs['input_ids'].tolist(), [[0, 4]])
        invalid = (
            {'input_ids': [[5]]},
            {'input_ids': [[0]], 'token_type_ids': [[5]]},
            {'input_ids': [[0]], 'position_ids': [[4]]},
            {'input_ids': [[0]], 'attention_mask': [[2]]},
            {'input_ids': [[0]], 'attention_mask': [[0]]},
            {'input_ids': [[0, 0, 0, 0, 0]]},
            {'input_ids': [[0]], 'position_ids': [[False]]},
        )
        for inputs in invalid:
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                factory('unsafe', inputs, 0)

    def test_cpu_service_rejects_unencodable_model_length_before_inference(self):
        from torch import nn

        class WideModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(vocab_size=50257, n_positions=128)
                self.calls = 0

            def forward(self, input_ids):
                self.calls += 1
                raise AssertionError('must not execute')

        model = WideModel().eval()
        stop = threading.Event()
        stop.set()
        with self.assertRaises(ValueError):
            serve_cpu_model(model, model_id='wide', team_ids=('alpha',),
                            stop_event=stop, announce=lambda _: None,
                            max_tokens=3)
        self.assertEqual(model.calls, 0)

    def test_clean_stop_drains_before_reporting_clean(self):
        self.assertIsNotNone(serve_owner, 'owner service not implemented')
        server, broker, worker = TestServer(), TestBroker(), TestWorker()
        stop = threading.Event()
        stop.set()
        observed = []

        def verify_drain():
            observed.append((server.stopped, broker.waited))
            return True

        result = serve_owner(server, broker, worker, stop, announce=lambda _: None,
                             verify_drain=verify_drain, poll_interval=0.01, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.CLEAN)
        self.assertTrue(result.teardown_verified)
        self.assertEqual(observed, [(True, True)])

    def test_quarantine_is_observed_without_any_client(self):
        server, broker, worker = TestServer(), TestBroker(), TestWorker()
        stop = threading.Event()
        result = []
        thread = threading.Thread(target=lambda: result.append(
            serve_owner(server, broker, worker, stop, announce=lambda _: None,
                        verify_drain=lambda: True, poll_interval=0.01, shutdown_timeout=1)))
        thread.start()
        deadline = time.monotonic() + 2
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.001)
        worker.quarantined = True
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0].status, OwnerStatus.QUARANTINED)
        self.assertFalse(broker.waited)

    def test_failed_final_device_drain_is_not_clean(self):
        server, broker, worker = TestServer(), TestBroker(), TestWorker()
        stop = threading.Event()
        stop.set()
        result = serve_owner(server, broker, worker, stop, announce=lambda _: None,
                             verify_drain=lambda: False, poll_interval=0.01, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.QUARANTINED)
        self.assertFalse(result.teardown_verified)

    def test_poison_during_shutdown_is_not_reported_clean(self):
        worker = TestWorker()

        class PoisonOnClose(TestServer):
            def close(self, timeout=0):
                worker.poisoned = True
                return super().close(timeout=timeout)

        stop = threading.Event()
        stop.set()
        result = serve_owner(PoisonOnClose(), TestBroker(), worker, stop,
                             announce=lambda _: None, verify_drain=lambda: True,
                             poll_interval=0.01, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.WORKER_FAILED)

    def test_broker_failure_during_shutdown_is_not_reported_clean(self):
        class FailingBroker(TestBroker):
            def close(self, *, wait=True, timeout=None):
                self.failed = True
                return super().close(wait=wait, timeout=timeout)

        stop = threading.Event()
        stop.set()
        result = serve_owner(TestServer(), FailingBroker(), TestWorker(), stop,
                             announce=lambda _: None, verify_drain=lambda: True,
                             poll_interval=0.01, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.WORKER_FAILED)
        self.assertTrue(result.teardown_verified)

    def test_worker_failure_stops_admission_without_clean_attestation(self):
        class DrainServer(TestServer):
            def __init__(self):
                super().__init__()
                self.calls = []

            def stop_accepting(self, timeout=0):
                self.calls.append('stop_accepting')
                return True

            def drain(self, timeout=0):
                self.calls.append('drain')
                return True

            def close(self, timeout=0):
                self.calls.append('close')
                return super().close(timeout=timeout)

        server, broker, worker = DrainServer(), TestBroker(), TestWorker()
        worker.poisoned = True
        stop = threading.Event()
        verified = []
        result = serve_owner(server, broker, worker, stop, announce=lambda _: None,
                             verify_drain=lambda: (verified.append(True), True)[1], poll_interval=0.01,
                             shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.WORKER_FAILED)
        self.assertTrue(result.teardown_verified)
        self.assertTrue(server.stopped)
        self.assertEqual(server.calls, ['stop_accepting', 'drain', 'close'])
        self.assertEqual(verified, [True])

    def test_normal_worker_close_is_not_a_failure(self):
        worker = TestWorker()

        class CloseableWorker(TestWorker):
            @property
            def failed(self):
                return False

        worker = CloseableWorker()

        class CloseWorkerBroker(TestBroker):
            def close(self, *, wait=True, timeout=None):
                worker.poisoned = True  # admission is closed, but not failed
                return super().close(wait=wait, timeout=timeout)

        stop = threading.Event()
        stop.set()
        result = serve_owner(TestServer(), CloseWorkerBroker(), worker, stop,
                             announce=lambda _: None, verify_drain=lambda: True,
                             poll_interval=0.01, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.CLEAN)
        self.assertTrue(result.teardown_verified)

    def test_unfinished_broker_shutdown_cannot_be_attested(self):
        class StuckBroker(TestBroker):
            def close(self, *, wait=True, timeout=None):
                return False

        server, broker, worker = TestServer(), StuckBroker(), TestWorker()
        stop = threading.Event()
        stop.set()
        result = serve_owner(server, broker, worker, stop, announce=lambda _: None,
                             verify_drain=lambda: True, poll_interval=0.01, shutdown_timeout=0.1)
        self.assertEqual(result.status, OwnerStatus.SHUTDOWN_INCOMPLETE)

    def test_quarantine_interrupts_a_slow_shutdown(self):
        class SlowServer(TestServer):
            def __init__(self):
                super().__init__()
                self.closing = threading.Event()
                self.release = threading.Event()

            def close(self, timeout=0):
                self.closing.set()
                self.release.wait()
                return False

        server, broker, worker = SlowServer(), TestBroker(), TestWorker()
        stop = threading.Event()
        stop.set()
        result = []
        thread = threading.Thread(target=lambda: result.append(
            serve_owner(server, broker, worker, stop, announce=lambda _: None,
                        verify_drain=lambda: True, poll_interval=0.01, shutdown_timeout=3)))
        thread.start()
        self.assertTrue(server.closing.wait(2))
        try:
            worker.quarantined = True
            thread.join(0.5)
            self.assertFalse(thread.is_alive(), 'quarantine waited for slow shutdown')
            self.assertEqual(result[0].status, OwnerStatus.QUARANTINED)
        finally:
            server.release.set()
            thread.join(2)

    def test_quarantine_interrupts_a_blocked_startup_announcement(self):
        server, broker, worker = TestServer(), TestBroker(), TestWorker()
        announce_started, announce_release = threading.Event(), threading.Event()
        result = []

        def announce(_):
            announce_started.set()
            announce_release.wait()

        thread = threading.Thread(target=lambda: result.append(
            serve_owner(server, broker, worker, threading.Event(), announce=announce,
                        verify_drain=lambda: True, poll_interval=0.01, shutdown_timeout=3)))
        thread.start()
        self.assertTrue(announce_started.wait(2))
        try:
            worker.quarantined = True
            thread.join(0.5)
            self.assertFalse(thread.is_alive(), 'quarantine waited for blocked announcement')
            self.assertEqual(result[0].status, OwnerStatus.QUARANTINED)
        finally:
            announce_release.set()
            thread.join(2)

    def test_unfinished_startup_never_certifies_teardown(self):
        class SlowStart(TestServer):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def start(self):
                self.entered.set()
                self.release.wait()
                return super().start()

        server, broker, worker = SlowStart(), TestBroker(), TestWorker()
        stop = threading.Event()
        results = []
        owner = threading.Thread(target=lambda: results.append(
            serve_owner(server, broker, worker, stop, announce=lambda _: None,
                        verify_drain=lambda: True, poll_interval=0.01, shutdown_timeout=1)))
        owner.start()
        self.assertTrue(server.entered.wait(2))
        try:
            stop.set()
            owner.join(0.5)
            self.assertFalse(owner.is_alive())
            self.assertEqual(results[0].status, OwnerStatus.SHUTDOWN_INCOMPLETE)
            self.assertFalse(results[0].teardown_verified)
        finally:
            server.release.set()
            owner.join(2)

    def test_health_read_failure_cannot_be_attested(self):
        class UnknownHealth(TestWorker):
            @property
            def quarantined(self):
                raise RuntimeError('health unavailable')

            @quarantined.setter
            def quarantined(self, value):
                pass

        result = serve_owner(TestServer(), TestBroker(), UnknownHealth(), threading.Event(),
                             announce=lambda _: None, verify_drain=lambda: True,
                             poll_interval=0.01, shutdown_timeout=1)
        self.assertEqual(result.status, OwnerStatus.QUARANTINED)
        self.assertEqual(result.error_type, 'RuntimeError')


if __name__ == '__main__':
    unittest.main()
