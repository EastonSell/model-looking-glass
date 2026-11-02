"""Create-only publication for trusted, bounded calibration diagnostics.

The destination and its directory must be owned by the cooperative caller.
This helper is not a secure interface to an untrusted or changing directory.
"""

import json
import os
from pathlib import Path


def write_json_new(path, value, *, max_bytes=1024 ** 2):
    """Publish complete JSON under a new name, retaining a partial on failure.

    A same-directory hard link makes final-name creation create-only on the
    supported local filesystem. A cancellation may leave either or both names;
    callers must preserve it and treat the outcome as uncertain. File fsync is
    not a claim of directory or power-loss durability.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    destination = Path(path)
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    if len(data) > max_bytes:
        raise ValueError("serialized diagnostic exceeds its byte limit")
    partial = destination.with_name(destination.name + ".partial")
    if os.path.lexists(destination):
        raise FileExistsError("diagnostic destination already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(partial, flags, 0o600)
    try:
        remaining = memoryview(data)
        while remaining:
            count = os.write(fd, remaining)
            if type(count) is not int or not 0 < count <= len(remaining):
                raise OSError("diagnostic write made invalid progress")
            remaining = remaining[count:]
        os.fsync(fd)
    except BaseException as original:
        try:
            os.close(fd)
        except BaseException as close_error:
            if not isinstance(close_error, Exception):
                raise close_error from original
            try:
                original.add_note("Closing the partial diagnostic also failed: "
                                  + type(close_error).__name__)
            except Exception:
                pass
        raise
    os.close(fd)
    os.link(partial, destination)
    try:
        os.unlink(partial)
    except OSError:
        # The published file is already complete. Keep the extra name if its
        # removal fails instead of turning success into a contradictory result.
        pass
