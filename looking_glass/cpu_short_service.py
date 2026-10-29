"""Serve the pinned DistilGPT2 model on CPU with 16/32-token buckets."""
import argparse
import json
import re
import sys
from threading import Event, Thread

from .pretrained import load_distilgpt2
from .service import OwnerStatus
from .short_cpu_service import serve_short_cpu_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="existing pinned Hugging Face cache")
    parser.add_argument("--team", action="append", required=True, help="team name; give one or two")
    parser.add_argument("--bucket", type=int, choices=(16, 32), required=True)
    args = parser.parse_args(argv)
    if (not 1 <= len(args.team) <= 2 or len(set(args.team)) != len(args.team)
            or any(re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", team, re.ASCII) is None for team in args.team)):
        parser.error("provide one or two distinct 1–64 character ASCII team IDs")
    stop = Event()
    try:
        bundle = load_distilgpt2(cache_dir=args.cache_dir, allow_download=False)
        model_id = f"{bundle.metadata['model_id']}@{bundle.metadata['model_revision']}"
        def control():
            try:
                for line in sys.stdin:
                    if line.strip() == "shutdown":
                        break
            finally:
                stop.set()
        def announce(server):
            # Private controller channel: credentials are not public output.
            endpoint = {"address": list(server.address), "broker_id": server.broker.broker_id,
                        "team_tokens": dict(server.team_tokens), "model_id": model_id,
                        "device": "cpu", "service": "short-last-token-cpu-v1",
                        "bucket_length": args.bucket, "pad_token_id": bundle.model.config.eos_token_id,
                        "vocab_size": bundle.model.config.vocab_size,
                        "output_scope": "last-active-token-logits"}
            print(json.dumps(endpoint, separators=(",", ":")), flush=True)
            Thread(target=control, name="short-cpu-service-control", daemon=True).start()
        result = serve_short_cpu_model(bundle.model, model_id=model_id, team_ids=args.team,
                                       bucket_length=args.bucket, stop_event=stop, announce=announce)
    except Exception as exc:
        print(f"Short CPU service failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if result.status is not OwnerStatus.CLEAN or not result.teardown_verified:
        print(f"Short CPU service stopped with status {result.status.value}; "
              f"teardown_verified={result.teardown_verified}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
