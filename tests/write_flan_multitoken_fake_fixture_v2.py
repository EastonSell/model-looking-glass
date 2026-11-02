"""Create an exclusive, reproducible on-disk fake pass for an independent reader.

No checkpoint, tokenizer assets, selected study cases or pretrained model are
loaded. The tiny fake has width 2 and vocabulary 256 and does not establish
production behavior or direct-child custody.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from test_flan_multitoken_runner_v2 import FakeModel, FakeTokenizer, full_groups, one_unit_groups, real_test_journal
from looking_glass.flan_multitoken_runner_v2 import run_fabricated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--scenario", choices=("ordinary", "zero-q", "low-only"), default="ordinary")
    args = parser.parse_args()
    root = Path(args.output)
    if not root.is_absolute() or not root.parent.is_dir():
        raise ValueError("expected an absolute new directory under an existing parent")
    root.mkdir()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    groups = full_groups() if args.full else one_unit_groups()
    group_bytes = (json.dumps(groups, sort_keys=True, separators=(",", ":")) + "\n").encode("utf8")
    with (root / "groups.json").open("xb") as stream:
        stream.write(group_bytes)
    journal = real_test_journal(root / "child")
    result = run_fabricated(FakeModel(scenario=args.scenario).eval(), FakeTokenizer(), groups, journal,
                            subject="story", width=2, vocab_size=256)
    print(json.dumps({"schema": "fabricated-multisubtoken-fixture-v1", "testing_only": True,
                      "scenario": args.scenario,
                      "groups_sha256": hashlib.sha256(group_bytes).hexdigest(),
                      "events_sha256": hashlib.sha256((root / "child/events.jsonl").read_bytes()).hexdigest(),
                      "terminal_sha256": hashlib.sha256((root / "child/terminal.json").read_bytes()).hexdigest(),
                      "components": journal.component_count, "slots": result["slots"]}, sort_keys=True))


if __name__ == "__main__":
    main()
