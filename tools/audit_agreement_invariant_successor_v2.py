"""Source-first entry for the public invariant-noun successor saved-byte replay.

The operator must verify this entry's published bytes before executing it and
independently observe the owner exit, launch bindings and journal digest.
"""
import hashlib
from pathlib import Path
import stat
from types import ModuleType


READER_SHA256 = '0bb87c168fcd464d0d2aa1c4fd4ed85b687d8991ba41c9c1132c0fe56a933fb9'


def load_reader(path):
    path = Path(path)
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
                or not 0 < info.st_size <= 1024 * 1024):
            raise ValueError('reader is not a bounded ordinary source file')
        with path.open('rb') as stream:
            raw = stream.read(1024 * 1024 + 1)
        if (len(raw) != info.st_size or len(raw) > 1024 * 1024
                or hashlib.sha256(raw).hexdigest() != READER_SHA256):
            raise ValueError('reader source differs from the reviewed bytes')
    except OSError as error:
        raise ValueError('reader source is unavailable') from error
    module = ModuleType('_verified_agreement_invariant_reader')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
    return module


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    reader = load_reader(root / 'looking_glass/agreement_invariant_successor_science_v2.py')
    return reader.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
