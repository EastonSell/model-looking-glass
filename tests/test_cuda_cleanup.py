"""CPU-only ordering and failure tests for the supervised cleanup helper."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from looking_glass.cuda_cleanup import clear_cublas_after_drain
from looking_glass.device_lease import DeviceLease
from looking_glass.supervisor import OwnerSession


class CudaCleanupTests(unittest.TestCase):
    def setUp(self):
        self.session = object.__new__(OwnerSession)
        self.session._identity = SimpleNamespace(cuda_device_index=0)
        self.session._runtime_verified = True
        self.lease = object.__new__(DeviceLease)
        self.events = []
        self.allocated, self.reserved = 32, 32
        self.cuda = SimpleNamespace(is_initialized=lambda: self.events.append(('initialized',)) or True,
            synchronize=lambda device: self.events.append(("sync", device)),
            empty_cache=lambda: self.events.append(("empty",)),
            memory_allocated=lambda device: self.events.append(('allocated',)) or self.allocated,
            memory_reserved=lambda device: self.events.append(('reserved',)) or self.reserved)
        self.torch = SimpleNamespace(__version__="2.8.0+cu128", version=SimpleNamespace(cuda="12.8"),
                                    cuda=self.cuda, _C=SimpleNamespace(_cuda_clearCublasWorkspaces=self.clear))
        self.check = patch.object(OwnerSession, "check", autospec=True,
                                  side_effect=lambda session, lease: self.events.append(("check",)))
        self.mock_check = self.check.start()
        self.addCleanup(self.check.stop)

    def clear(self):
        self.events.append(("clear",))
        self.allocated = self.reserved = 0

    def call(self):
        return clear_cublas_after_drain(session=self.session, lease=self.lease, torch=self.torch)

    def test_clear_is_guarded_and_synchronized(self):
        result = self.call()
        self.assertEqual(result["before"], {"allocated_bytes": 32, "reserved_bytes": 32})
        self.assertEqual(result["after"], {"allocated_bytes": 0, "reserved_bytes": 0})
        for index, event in enumerate(self.events):
            if event[0] != 'check':
                self.assertGreater(index, 0)
                self.assertEqual(self.events[index - 1], ('check',))
        self.assertEqual([event for event in self.events if event[0] != 'check'], [
            ('initialized',), ('sync', 0), ('empty',), ('allocated',), ('reserved',),
            ('clear',), ('sync', 0), ('empty',), ('allocated',), ('reserved',),
        ])
        self.assertEqual(self.events[-1], ('check',))

    def test_unverified_session_and_wrong_version_fail_before_cuda(self):
        self.session._runtime_verified = False
        with self.assertRaises(RuntimeError): self.call()
        self.assertEqual(self.events, [("check",)])
        self.events.clear(); self.session._runtime_verified = True
        self.torch.__version__ = "2.9.0+cu128"
        with self.assertRaises(RuntimeError): self.call()
        self.assertEqual(self.events, [("check",)])

    def test_missing_live_lease_or_bad_counter_fails_closed(self):
        with self.assertRaises(ValueError):
            clear_cublas_after_drain(session=self.session, lease=object(), torch=self.torch)
        self.allocated, self.reserved = 33, 32
        with self.assertRaises(RuntimeError): self.call()
        self.assertNotIn(("clear",), self.events)

    def test_clear_error_propagates_without_success_result(self):
        def fail(): raise RuntimeError("simulated CUDA failure")
        self.torch._C._cuda_clearCublasWorkspaces = fail
        with self.assertRaises(RuntimeError): self.call()
        self.assertEqual(self.events[-1], ("check",))

    def test_lost_marker_after_clear_fails_closed(self):
        calls = 0
        def check(session, lease):
            nonlocal calls
            calls += 1
            if self.allocated == 0:
                raise RuntimeError("marker lost")
        self.mock_check.side_effect = check
        with self.assertRaises(RuntimeError): self.call()
        self.assertIn(("clear",), self.events)

    def test_lost_marker_between_counter_reads_stops_next_cuda_call(self):
        lost = False
        def allocated(device):
            nonlocal lost
            self.events.append(('allocated',))
            lost = True
            return self.allocated
        def check(session, lease):
            if lost:
                raise RuntimeError('marker lost')
        self.cuda.memory_allocated = allocated
        self.mock_check.side_effect = check
        with self.assertRaises(RuntimeError): self.call()
        self.assertIn(('allocated',), self.events)
        self.assertNotIn(('reserved',), self.events)
        self.assertNotIn(('clear',), self.events)


if __name__ == "__main__":
    unittest.main()
