"""Run a pinned, offline DistilGPT2 shared-model service on CPU only."""
import argparse
import json
import re
import sys
from threading import Event, Thread

from .pretrained import load_distilgpt2
from .service import OwnerStatus, serve_cpu_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True, help='existing Hugging Face cache with pinned assets')
    parser.add_argument('--team', action='append', required=True, help='team name; repeat for each team')
    parser.add_argument('--max-tokens', type=int, default=2, help='maximum tokens per request (default: 2)')
    args = parser.parse_args(argv)
    if (not 1 <= len(args.team) <= 128 or len(set(args.team)) != len(args.team)
            or any(re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', team) is None for team in args.team)):
        parser.error('team names must be distinct 1–64 character ASCII identifiers')
    # The pinned vocabulary and bounded full-logit JSON transport support at
    # most two positions. Reject impossible configurations before model load.
    if not 1 <= args.max_tokens <= 2:
        parser.error('max-tokens must be 1 or 2 for the bounded full-logit transport')
    stop = Event()
    try:
        # The loader verifies every pinned asset and never downloads here.
        bundle = load_distilgpt2(cache_dir=args.cache_dir, allow_download=False)
        model_id = f"{bundle.metadata['model_id']}@{bundle.metadata['model_revision']}"

        def control():
            try:
                for line in sys.stdin:
                    if line.strip() == 'shutdown':
                        break
            finally:
                stop.set()

        def announce(server):
            # stdout is an intentionally private owner-to-controller channel.
            endpoint = {'address': list(server.address), 'broker_id': server.broker.broker_id,
                        'team_tokens': dict(server.team_tokens), 'model_id': model_id,
                        'device': 'cpu'}
            print(json.dumps(endpoint, separators=(',', ':')), flush=True)
            Thread(target=control, name='looking-glass-cpu-control', daemon=True).start()

        result = serve_cpu_model(bundle.model, model_id=model_id, team_ids=args.team,
                                 stop_event=stop, announce=announce, max_tokens=args.max_tokens)
    except Exception as exc:
        print(f'CPU service failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    if result.status is not OwnerStatus.CLEAN or not result.teardown_verified:
        print(f'CPU service stopped with status {result.status.value}; '
              f'teardown_verified={result.teardown_verified}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
