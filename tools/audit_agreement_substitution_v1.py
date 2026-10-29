"""Source-first entry for the public agreement substitution saved-byte audit.

The caller must first verify this entry's own published bytes and independently
observe the owner exit and journal digest. This entry checks and compiles the
reviewed reader source directly, without importing cached bytecode.
"""
import hashlib
from pathlib import Path
import stat
from types import ModuleType


READER_SHA256 = 'f46f7204c926a05f6de993ad4858ce240ddc7682603bee37f5177f45466987c8'


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
    module = ModuleType('_verified_agreement_substitution_reader')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
    return module


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    reader = load_reader(root / 'looking_glass/agreement_substitution_science_v1.py')
    return reader.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
