"""Observe closed Windows file allocations without copying study evidence.

This records filesystem-reported file allocation sizes, not directory metadata,
alternate streams, a live quota, or proof of process-tree quiescence.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat


class FileStandardInfo(ctypes.Structure):
    _fields_ = [('allocation_size', ctypes.c_longlong), ('end_of_file', ctypes.c_longlong),
                ('number_of_links', ctypes.c_ulong), ('delete_pending', ctypes.c_ubyte),
                ('directory', ctypes.c_ubyte)]


def allocation(path, expected_size):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong,
                                   ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.GetFileInformationByHandleEx.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                     ctypes.c_void_p, ctypes.c_ulong]
    kernel.GetFileInformationByHandleEx.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    handle = kernel.CreateFileW(str(path), 0, 7, None, 3, 0x00200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = FileStandardInfo()
        if not kernel.GetFileInformationByHandleEx(handle, 1, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if (info.directory or info.delete_pending or info.number_of_links != 1
                or info.end_of_file != expected_size or info.allocation_size < 0):
            raise ValueError('file identity, size or allocation changed during observation')
        return info.allocation_size
    finally:
        if not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def measure(root):
    root = Path(root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError('root must be an absolute ordinary directory')
    if getattr(root.lstat(), 'st_file_attributes', 0) & 0x400:
        raise ValueError('root cannot be a reparse point')
    files = dirs = apparent = allocated = 0
    todo = [root]
    while todo:
        parent = todo.pop()
        dirs += 1
        if dirs > 5000:
            raise ValueError('directory bound exceeded')
        with os.scandir(parent) as entries:
            for item in entries:
                info = item.stat(follow_symlinks=False)
                if item.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
                    raise ValueError('reparse points are not supported')
                if stat.S_ISDIR(info.st_mode):
                    todo.append(Path(item.path))
                elif stat.S_ISREG(info.st_mode):
                    files += 1
                    if files > 5000:
                        raise ValueError('file bound exceeded')
                    apparent += info.st_size
                    allocated += allocation(Path(item.path), info.st_size)
                else:
                    raise ValueError('nonregular file encountered')
    return {'files': files, 'directories': dirs, 'apparent_bytes': apparent,
            'allocated_file_bytes': allocated}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    supplied_root = args.root.absolute()
    if (not supplied_root.is_dir() or supplied_root.is_symlink()
            or getattr(supplied_root.lstat(), 'st_file_attributes', 0) & 0x400):
        raise ValueError('root must be an ordinary directory')
    root = supplied_root.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if output.is_relative_to(root) or os.path.lexists(output):
        raise ValueError('observation must be a new file outside the measured root')
    result = {'schema': 'agreement-substitution-allocated-file-observation-v1',
              'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              **measure(root),
              'scope': 'Windows FileStandardInfo allocation sizes for ordinary files after direct-child exit; excludes directory metadata and alternate streams and does not establish descendant drain or an OS quota.'}
    raw = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode()
    with output.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'sha256': hashlib.sha256(raw).hexdigest(), **result}, sort_keys=True))


if __name__ == '__main__':
    main()
