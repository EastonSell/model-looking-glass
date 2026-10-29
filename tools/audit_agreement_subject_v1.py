"""Source-first entry point for the model-free subject-study replay."""
import hashlib
from pathlib import Path
import re
import sys


def main():
    if len(sys.argv) < 3:
        raise ValueError('an absolute reader path and its independently pinned SHA-256 are required')
    source, expected = Path(sys.argv[1]), sys.argv[2]
    if (not source.is_absolute() or '..' in source.parts or type(expected) is not str
            or re.fullmatch(r'[0-9a-f]{64}', expected) is None):
        raise ValueError('invalid reader path or digest')
    with source.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    if not raw or len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('reader source differs from its independent pin')
    sys.argv = [str(source), *sys.argv[3:]]
    namespace = {'__name__': '__main__', '__file__': str(source), '__package__': None}
    exec(compile(raw, str(source), 'exec', dont_inherit=True, optimize=0), namespace)


if __name__ == '__main__':
    main()
