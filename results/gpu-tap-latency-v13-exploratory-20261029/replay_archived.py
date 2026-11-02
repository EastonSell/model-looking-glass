"""Replay the preserved v1.3 public record against its pinned protocol.

This makes a temporary historical layout; it never loads a model or accesses
CUDA. The original replay verifies the source and evidence digests itself.
"""
from pathlib import Path
import runpy
import shutil
import tempfile

HERE = Path(__file__).resolve().parent

with tempfile.TemporaryDirectory(prefix="tap-v13-public-replay-") as temp:
    root = Path(temp)
    evidence = root / "results" / HERE.name
    evidence.mkdir(parents=True)
    source = root / "looking_glass"
    source.mkdir()
    shutil.copyfile(HERE / "pinned-source" / "tap_repeated_protocol.py",
                    source / "tap_repeated_protocol.py")
    for name in ("replay_public.py", "journal.jsonl", "schedule.json",
                 "session-result.json", "host-context.json"):
        shutil.copyfile(HERE / name, evidence / name)
    runpy.run_path(str(evidence / "replay_public.py"), run_name="__main__")
