"""Check the public replay rejects rehashed, numerically equal signed-zero bytes.

Run with the reconstructed public 16-token CPU and GPU evidence and the pinned
manifest. The test makes temporary copies and does not call a model or GPU.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cpu-dir', 'gpu-dir', 'manifest', 'scratch-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.cpu_dir, args.gpu_dir, args.manifest, args.scratch_dir)):
        raise ValueError('absolute paths are required')
    helper = Path(__file__).resolve().parent / 'replay_short_cpu_gpu_public.py'
    spec = importlib.util.spec_from_file_location('short_public_replay', helper)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    with tempfile.TemporaryDirectory(prefix='short-replay-signed-zero-', dir=args.scratch_dir) as temp:
        root = Path(temp)
        names = ['result.json', 'cleanup.json']
        names += [f'forward-{i:03d}-entry.json' for i in range(1, 5)]
        names += [f'memory-{i:03d}.json' for i in range(21)]
        names += [name + '.f32' for name in replay.NAMES]
        for name in names:
            shutil.copyfile(args.gpu_dir / name, root / name)
        result = json.loads((root / 'result.json').read_bytes())
        altered = []
        for name, value in (('reference-alpha', 0.0), ('client-alpha', -0.0)):
            path = root / (name + '.f32')
            raw = bytearray(path.read_bytes())
            struct.pack_into('<f', raw, 0, value)
            path.write_bytes(raw)
            result['measurement']['output_sha256'][name] = replay.sha(raw)
            altered.append(bytes(raw))
        check = replay.comparison(*altered)
        if altered[0] == altered[1] or check['violations'] != 0 or check['max_abs'] != 0:
            raise AssertionError('fixture must be numerically equal and bitwise different')
        raw = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        (root / 'result.json').write_bytes(raw)
        try:
            replay.audit(args.cpu_dir, root, args.manifest,
                         'ca9f5a5e6d4593cfe16c4e1cfa19800f2d6df5e7', 16, replay.sha(raw))
        except ValueError as error:
            if 'client and same-device reference bytes differ' not in str(error):
                raise
        else:
            raise AssertionError('replay accepted a rehashed signed-zero byte difference')
    print('PASS: numerically equal signed-zero byte difference was rejected')


if __name__ == '__main__':
    main()
