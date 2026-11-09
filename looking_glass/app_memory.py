"""Bounded automatic launch budgets; estimates are not peak-memory guarantees."""
import os

MIB = 1024**2
GIB = 1024**3


def _cgroup_available():
    from pathlib import Path, PurePosixPath
    # Include the process's delegated/systemd cgroup and its ancestors, not just
    # mount-root limits. Namespace-relative mounts still use the root fallback.
    roots = ((Path('/sys/fs/cgroup'), 'memory.max', 'memory.current'),
             (Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes', 'memory.usage_in_bytes'))
    candidates = set(roots)
    try:
        membership = Path('/proc/self/cgroup').read_text().splitlines()
        for line in membership:
            _, controllers, group = line.split(':', 2)
            layout = roots[0] if not controllers else roots[1] if 'memory' in controllers.split(',') else None
            relative = PurePosixPath(group.lstrip('/'))
            if layout is None or '..' in relative.parts:
                continue
            root, maximum, current = layout
            node = root.joinpath(*relative.parts)
            while node.is_relative_to(root):
                candidates.add((node, maximum, current))
                if node == root:
                    break
                node = node.parent
    except (OSError, ValueError):
        pass
    remaining = []
    for root, maximum, current in candidates:
        try:
            limit = int((root / maximum).read_text().strip())
            used = int((root / current).read_text().strip())
            if 0 < limit < 2**60 and used >= 0:
                remaining.append(max(0, limit - used))
        except (OSError, ValueError):
            pass
    return min(remaining) if remaining else None


def _limit_to_cgroup(available):
    remaining = _cgroup_available()
    return min(available, remaining) if remaining is not None else available


def available_host_bytes():
    """Read available physical RAM without a new dependency (Windows and Unix)."""
    try:
        if os.name == 'nt':
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in ('total_phys', 'avail_phys',
                      'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'extended')]]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.avail_phys)
        # Linux MemAvailable accounts for reclaimable cache, unlike free pages.
        try:
            with open('/proc/meminfo', encoding='ascii') as source:
                for line in source:
                    if line.startswith('MemAvailable:'):
                        return _limit_to_cgroup(int(line.split()[1]) * 1024)
        except (OSError, ValueError):
            pass
        return _limit_to_cgroup(int(os.sysconf('SC_AVPHYS_PAGES')) * int(os.sysconf('SC_PAGE_SIZE')))
    except (AttributeError, OSError, ValueError):
        return None


def automatic_model_limit():
    available = available_host_bytes()
    # Leave host RAM for deserialization, tokenization and the rest of the system.
    # This remains a parameter-storage preflight, not a peak RSS enforcement.
    return min(16 * GIB, max(1, available // 2)) if available is not None else 2 * GIB
