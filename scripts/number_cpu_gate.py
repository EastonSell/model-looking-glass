"""Wait for the parent owner before starting the CPU study launcher.

This short-lived gate is cooperative: a trusted same-account process must
protect the fresh directory and the parent must assign this process to its
Windows Job before writing the grant. No model or study package is imported
while the grant is absent. This file is executed as a direct Python script.
"""
import hashlib
import hmac
from pathlib import Path
import sys
import time


def main():
    if len(sys.argv) < 6 or len(sys.argv[2]) != 64 or len(sys.argv[4]) != 64:
        raise SystemExit('CPU launcher gate rejected its input')
    grant = Path(sys.argv[1])
    token = sys.argv[2].encode('ascii')
    launcher = Path(sys.argv[3])
    source_digest = sys.argv[4]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with grant.open('rb') as handle:
                value = handle.read(65)
        except FileNotFoundError:
            value = b''
        if len(value) >= len(token):
            if len(value) != len(token) or not hmac.compare_digest(value, token):
                raise SystemExit('CPU launcher gate rejected its grant')
            with launcher.open('rb') as handle:
                source = handle.read(1024 * 1024 + 1)
            if not source or len(source) > 1024 * 1024 or hashlib.sha256(source).hexdigest() != source_digest:
                raise SystemExit('CPU launcher gate rejected its source')
            sys.argv = [str(launcher), *sys.argv[5:]]
            namespace = {'__name__': '__main__', '__file__': str(launcher), '__package__': None}
            exec(compile(source, str(launcher), 'exec', dont_inherit=True), namespace)
            return
        time.sleep(0.05)
    raise SystemExit('CPU launcher gate expired')


if __name__ == '__main__':
    main()
