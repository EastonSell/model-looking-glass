"""Exercise float32 IPC values with independent thread-local FP modes."""
import importlib.util
import os
import subprocess
import sys
import unittest


_THREAD_RUNNER = r'''
import sys
import threading
program = sys.stdin.read()
errors = []
def run():
    try:
        exec(compile(program, '<serializer-thread>', 'exec'), {'__name__': '__main__'})
    except BaseException as exc:
        errors.append(exc)
thread = threading.Thread(target=run)
thread.start()
thread.join()
if errors:
    raise errors[0]
'''


_ROUND_TRIP = r'''
import socket
import struct
import sys
import time
import torch
from looking_glass.ipc import _ProtocolError, _encode_tensor_result, _receive, _send
from looking_glass.resident_worker import ForwardResult

if sys.argv[2] == 'stdlib':
    # The base package does not require NumPy; the serializer must still work.
    sys.modules['numpy'] = None

def forbid_cuda(*args, **kwargs):
    raise AssertionError("CUDA is forbidden in this CPU test")
torch.cuda._lazy_init = forbid_cuda
if not torch.set_flush_denormal(bool(int(sys.argv[1]))):
    sys.exit(77)

# The expected values are literal IEEE binary32 values expressed as normal
# binary64 numbers. Compare binary64 bits to preserve the sign of zero.
words = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
         0x00800000, 0x80800000, 0x3f800000, 0xbf800000,
         0x7f7fffff, 0xff7fffff]
expected = [0.0, -0.0, 1.401298464324817e-45, -1.401298464324817e-45,
            1.1754942106924411e-38, -1.1754942106924411e-38,
            1.1754943508222875e-38, -1.1754943508222875e-38,
            1.0, -1.0, 3.4028234663852886e38, -3.4028234663852886e38]
signed = [word if word < 2**31 else word - 2**32 for word in words]
source = torch.tensor(signed, dtype=torch.int32).view(torch.float32).reshape(1, 3, 4)

def check(tensor, indices):
    encoded = _encode_tensor_result(ForwardResult('case', tensor), 4 * 1024 * 1024)
    left, right = socket.socketpair()
    try:
        left.setblocking(False)
        right.setblocking(False)
        _send(left, {'version': 1, 'result': encoded}, 4 * 1024 * 1024, time.monotonic() + 5)
        result = _receive(right, 4 * 1024 * 1024, time.monotonic() + 5)['result']
    finally:
        left.close()
        right.close()
    assert result['dtype'] == 'torch.float32'
    assert result['shape'] == list(tensor.shape)
    actual_bits = [struct.pack('>d', value).hex() for value in result['logits']]
    expected_bits = [struct.pack('>d', expected[index]).hex() for index in indices]
    assert actual_bits == expected_bits, (actual_bits, expected_bits)

check(source, list(range(12)))
check(source.transpose(1, 2), [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])
# Both implementations must reject positive/negative infinity and NaN before
# generating JSON, regardless of the serialization thread's flush mode.
for raw in (0x7f800000, -0x00800000, 0x7f800001, -0x007fffff):
    invalid = torch.tensor([raw], dtype=torch.int32).view(torch.float32).reshape(1, 1, 1)
    try:
        _encode_tensor_result(ForwardResult('case', invalid), 4 * 1024 * 1024)
    except _ProtocolError:
        pass
    else:
        raise AssertionError('nonfinite float32 result was not rejected')
assert not torch.cuda.is_initialized()
'''

_OTHER_DTYPES = r'''
import socket
import struct
import sys
import time
import torch
from types import SimpleNamespace
from looking_glass.ipc import IPCServer, _ProtocolError, _encode_tensor_result, _receive, _send
from looking_glass.resident_worker import ForwardResult

def forbid_cuda(*args, **kwargs):
    raise AssertionError("CUDA is forbidden in this CPU test")
torch.cuda._lazy_init = forbid_cuda
if not torch.set_flush_denormal(bool(int(sys.argv[1]))):
    sys.exit(77)

# Integer bit views avoid denormal-sensitive tensor arithmetic when creating
# the inputs. Every normal and signed-zero value remains an ordinary JSON number.
formats = [
    (torch.float16, torch.int16, [0, -32768, 0x3c00, -17408, 0x0400], 6.103515625e-05, 0x03ff),
    (torch.bfloat16, torch.int16, [0, -32768, 0x3f80, -16512, 0x0080], 1.1754943508222875e-38, 0x007f),
    (torch.float64, torch.int64, [0, -(2**63), 0x3ff0000000000000,
                                  -4616189618054758400, 0x0010000000000000],
     2.2250738585072014e-308, 0x000fffffffffffff),
]
for dtype, integer, normal_bits, smallest_normal, largest_fraction in formats:
    normal = torch.tensor(normal_bits, dtype=integer).view(dtype).reshape(1, 1, 5)
    encoded = _encode_tensor_result(ForwardResult('case', normal), 4 * 1024 * 1024)
    left, right = socket.socketpair()
    try:
        left.setblocking(False)
        right.setblocking(False)
        _send(left, {'version': 1, 'result': encoded}, 4 * 1024 * 1024, time.monotonic() + 5)
        result = _receive(right, 4 * 1024 * 1024, time.monotonic() + 5)['result']
    finally:
        left.close()
        right.close()
    expected = [0.0, -0.0, 1.0, -1.0, smallest_normal]
    assert [struct.pack('>d', value) for value in result['logits']] == [struct.pack('>d', value) for value in expected]

    width = 16 if integer == torch.int16 else 64
    # Both positive and negative subnormals must fail, even in a strided view.
    for raw in (1, largest_fraction, -(2**(width - 1)) + 1,
                -(2**(width - 1)) + largest_fraction):
        backing = torch.tensor([0, raw, 0, raw], dtype=integer).view(dtype).reshape(1, 2, 2)
        strided = backing.transpose(1, 2)
        try:
            _encode_tensor_result(ForwardResult('case', strided), 4 * 1024 * 1024)
        except _ProtocolError:
            pass
        else:
            raise AssertionError(f'{dtype} subnormal result was not rejected')
        # Check the production outcome path reports a terminal failure on wire.
        broker_id = 'a' * 32
        server = IPCServer(SimpleNamespace(broker_id=broker_id), io_timeout=5)
        result = ForwardResult('case', strided)
        outcome = SimpleNamespace(broker_id=broker_id, request_id='case', team_id='team',
                                  job_id=1, state='succeeded', result=result,
                                  submitted_ns=1, ready_ns=2, started_ns=3, finished_ns=4)
        left, right = socket.socketpair()
        try:
            left.setblocking(False)
            right.setblocking(False)
            server._send_outcome(left, outcome, SimpleNamespace(job_id=1), 'case', 'team')
            response = _receive(right, 4 * 1024 * 1024, time.monotonic() + 5)
        finally:
            left.close()
            right.close()
        assert response['status'] == 'failed' and response['error'] == 'invalid_result'
        assert 'result' not in response

for dtype, integer, width, exponent in (
        (torch.float16, torch.int16, 16, 0x7c00),
        (torch.bfloat16, torch.int16, 16, 0x7f80),
        (torch.float32, torch.int32, 32, 0x7f800000),
        (torch.float64, torch.int64, 64, 0x7ff0000000000000)):
    for raw in (exponent, -(2**(width - 1)) + exponent, exponent | 1):
        value = torch.tensor([raw], dtype=integer).view(dtype).reshape(1, 1, 1)
        try:
            _encode_tensor_result(ForwardResult('case', value), 4 * 1024 * 1024)
        except _ProtocolError:
            pass
        else:
            raise AssertionError(f'{dtype} nonfinite result was not rejected')
assert not torch.cuda.is_initialized()
'''


class Float32TransportTests(unittest.TestCase):
    def test_finite_float32_bits_survive_wire_round_trip_with_flush_off_and_on(self):
        for backend in ('numpy', 'stdlib'):
            if backend == 'numpy' and importlib.util.find_spec('numpy') is None:
                continue
            for enabled in (False, True):
                with self.subTest(backend=backend, flush=enabled):
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
                    child = subprocess.run(
                        [sys.executable, '-c', _THREAD_RUNNER, str(int(enabled)), backend],
                        input=_ROUND_TRIP, env=env, capture_output=True, text=True, timeout=30)
                    if child.returncode == 77:
                        self.skipTest('this CPU does not expose configurable denormal flushing')
                    self.assertEqual(child.returncode, 0, child.stderr)

    def test_other_dtype_subnormals_fail_without_changing_normal_wire_values(self):
        for enabled in (False, True):
            with self.subTest(flush=enabled):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
                child = subprocess.run([sys.executable, '-c', _THREAD_RUNNER, str(int(enabled))],
                                       input=_OTHER_DTYPES, env=env, capture_output=True, text=True, timeout=30)
                if child.returncode == 77:
                    self.skipTest('this CPU does not expose configurable denormal flushing')
                self.assertEqual(child.returncode, 0, child.stderr)


if __name__ == '__main__':
    unittest.main()
