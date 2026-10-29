"""Run the public v5 annotation-only preparer once and retain its receipt."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK / 'frozen/vision-recognized-v4-preselection'
RUN = WORK / 'vision-cls-preparation-v5-01'
REVISION = 'c05bc3905de37e1cc92119e5a350492482c6ee13'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RUN.mkdir(parents=False, exist_ok=False)
    cache = RUN / 'empty-bytecode-cache'
    cache.mkdir()
    temporary = RUN / 'temporary'
    temporary.mkdir()
    output = RUN / 'manifest.json'
    python = Path(sys._base_executable)
    site = WORK / '.venv/Lib/site-packages'
    dependency = WORK / 'vision-coco-deps'
    bootstrap = ('import runpy,sys; sys.path.extend([sys.argv[1],sys.argv[2],sys.argv[3]]); '
                 'sys.argv=sys.argv[4:]; runpy.run_path(sys.argv[0],run_name="__main__")')
    command = [str(python), '-I', '-S', '-B', '-X', f'pycache_prefix={cache}', '-c', bootstrap,
               str(site), str(dependency), str(ROOT), str(ROOT / 'tools/prepare_vision_coco_cls.py'),
               '--assets', str(WORK / 'vision-coco-assets'),
               '--annotation-archive', str(WORK / 'vision-recognized-assets/annotations_trainval2017.zip'),
               '--output', str(output), '--selection-revision', REVISION]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TEMP=str(temporary), TMP=str(temporary), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = time.monotonic()
    code, timed_out, error = None, False, None
    with (RUN / 'stdout.txt').open('xb') as stdout, (RUN / 'stderr.txt').open('xb') as stderr:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=1800)
            code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except Exception as exc:
            error = type(exc).__name__
    receipt = {'schema': 'vision-cls-v5-preparation-receipt', 'source_revision': REVISION,
               'exit_code': code, 'timed_out': timed_out, 'error_type': error,
               'duration_seconds': time.monotonic() - start, 'stdout_sha256': sha(RUN / 'stdout.txt'),
               'stderr_sha256': sha(RUN / 'stderr.txt'), 'manifest_sha256': sha(output) if output.exists() else None,
               'status': 'complete' if code == 0 and not timed_out and output.is_file() else 'incomplete',
               'model_forwards': 0, 'scope': 'cooperative annotation/image preparation; no model is loaded'}
    raw = (json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with (RUN / 'receipt.json').open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt))
    return 0 if receipt['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
