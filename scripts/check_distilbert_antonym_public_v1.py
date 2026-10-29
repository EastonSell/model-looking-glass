"""Rebuild the named-public antonym input comparison without model access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from looking_glass import distilbert_antonym_public_v1 as public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    if hashlib.sha256(raw).hexdigest() != public.CASE_SHA256:
        raise ValueError('fixed case bytes differ')
    result = public.scan(args.repo, args.commit, json.loads(raw)['cases'])
    payload = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode()
    with args.output.open('xb') as out:
        if out.write(payload) != len(payload):
            raise OSError('short comparison write')
    print('comparison SHA-256', hashlib.sha256(payload).hexdigest(),
          'relations', len(result['related_pairs']))


if __name__ == '__main__':
    main()
