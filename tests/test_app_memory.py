"""Automatic host allowance checks; standard-library-only and runnable standalone."""
import unittest
from unittest.mock import patch
from looking_glass.app_memory import automatic_model_limit, available_host_bytes, _limit_to_cgroup, _cgroup_available, GIB

class MemoryBudgetTests(unittest.TestCase):
    def test_budget_uses_half_available_and_caps_at_sixteen_gib(self):
        with patch('looking_glass.app_memory.available_host_bytes', return_value=12*GIB):
            self.assertEqual(automatic_model_limit(), 6*GIB)
        with patch('looking_glass.app_memory.available_host_bytes', return_value=128*GIB):
            self.assertEqual(automatic_model_limit(), 16*GIB)

    def test_unknown_telemetry_keeps_conservative_fallback(self):
        with patch('looking_glass.app_memory.available_host_bytes', return_value=None):
            self.assertEqual(automatic_model_limit(), 2*GIB)

    def test_nested_cgroup_and_parent_caps_are_considered(self):
        values = {'/proc/self/cgroup':'0::/user.slice/app',
            '/sys/fs/cgroup/user.slice/memory.max':str(2*GIB),
            '/sys/fs/cgroup/user.slice/memory.current':str(GIB//2)}
        def read(path, *args, **kwargs):
            if str(path) not in values: raise OSError('missing')
            return values[str(path)]
        with patch('pathlib.Path.read_text', read):
            self.assertEqual(_cgroup_available(), 3*GIB//2)

    def test_container_cap_limits_host_available_memory(self):
        with patch('looking_glass.app_memory._cgroup_available', return_value=GIB):
            self.assertEqual(_limit_to_cgroup(64*GIB), GIB)
        with patch('looking_glass.app_memory.available_host_bytes', return_value=0):
            self.assertEqual(automatic_model_limit(), 1)

    def test_available_ram_is_positive_or_unknown(self):
        value = available_host_bytes()
        self.assertTrue(value is None or (type(value) is int and value >= 0))

if __name__ == '__main__': unittest.main()
